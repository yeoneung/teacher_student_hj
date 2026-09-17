"""Budget-matched costs, paired seed decisions and explicit probe accounting."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import torch
from experiments.hj_gridfree.engine import write
from .study import verify


def comparison(a,b,alpha=.05/3,draws=3000):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    logs=np.log(a.mean(1)/b.mean(1));n=len(logs)
    upper=float(np.exp(logs.mean()+student_t.ppf(1-alpha,n-1)*logs.std(ddof=1)/np.sqrt(n)))
    rng=np.random.default_rng(881901);ratios=[]
    for _ in range(draws):
        seeds=rng.integers(n,size=n);states=rng.integers(a.shape[1],size=a.shape[1])
        ratios.append(float(a[seeds][:,states].mean()/b[seeds][:,states].mean()))
    ratio=float(a.mean()/b.mean());ci=np.quantile(ratios,[.025,.975]).tolist()
    def difference_test(margin):
        d=a.mean(1)-margin*b.mean(1);se=d.std(ddof=1)/np.sqrt(n)
        bound=float(d.mean()+student_t.ppf(1-alpha,n-1)*se)
        pvalue=float(student_t.cdf(d.mean()/se,n-1)) if se>0 else (0. if d.mean()<0 else 1.)
        return bound,pvalue
    ni_bound,ni_p=difference_test(1.01);superiority_bound,superiority_p=difference_test(1.)
    return dict(pooled_ratio=ratio,geometric_seed_ratio=float(np.exp(logs.mean())),
        corrected_upper_ratio=upper,seed_ratios=(a.mean(1)/b.mean(1)).tolist(),
        noninferiority_upper_mean_difference=ni_bound,noninferiority_pvalue=ni_p,
        superiority_upper_mean_difference=superiority_bound,superiority_pvalue=superiority_p,
        descriptive_ratio_interval=ci,reduction_percent=100*(1-ratio),
        noninferior=bool(ratio<=1.01 and ni_bound<=0),superior_one_percent=bool(ratio<=.99 and superiority_bound<0))


def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);cfg=verify(root);out=root/'report';out.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    costs=[];contrasts=[];decisions={};train_rows=[];query_rows=[];audit_rows=[];transport_queries=[]
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            folder=root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'
            for spec in task['methods']:
                row=json.loads((folder/(spec['name']+'.json')).read_text())
                if spec.get('parent_only'):continue
                retained=None
                if spec.get('parent'):
                    child=torch.load(folder/(spec['name']+'.pt'),map_location='cpu',weights_only=False)['budgets']['60']['state_dict']
                    parent=torch.load(folder/(spec['parent']+'.pt'),map_location='cpu',weights_only=False)['budgets'][str(cfg['parent_budget_seconds'])]['state_dict']
                    retained=all(torch.equal(v,parent[k]) for k,v in child.items())
                train_rows.append(dict(family=task['family'],seed=seed,method=spec['name'],retains_parent=retained,
                    total_seconds=row['total_compute_seconds'],overrun_seconds=row['overrun_seconds'],
                    setup_seconds=row['setup_seconds'],query_seconds=row['query_seconds'],
                    teacher_pairs=row['teacher_pairs'],refreshes=row['full_refreshes'],probes=row['probe_queries'],
                    fresh_minibatch_updates=row.get('fresh_minibatch_updates',0),fresh_update_seconds=row.get('fresh_update_seconds',0.),
                    teacher_type=row.get('teacher_type','analytic'),
                    accepted_refreshes=row.get('accepted_refreshes',0),rejected_refreshes=row.get('rejected_refreshes',0),
                    teacher_trajectories=row.get('teacher_trajectories',0),
                    clipped_updates=row.get('clipped_updates'),last_attempted_gradient_norm=row.get('last_attempted_gradient_norm'),
                    attempts=row['attempted_updates'],nonfinite_skips=row['nonfinite_updates_skipped']))
                probes=[e for e in row['events'] if e['reason']=='probe']
                finite=[e['metrics']['relative_gradient_error'] for e in probes if e['metrics']['relative_gradient_error'] is not None]
                scaled=[e['metrics'].get('scaled_gradient_error',e['metrics']['relative_gradient_error']) for e in probes if e['metrics'].get('scaled_gradient_error',e['metrics']['relative_gradient_error']) is not None]
                if probes:query_rows.append(dict(family=task['family'],seed=seed,method=spec['name'],checks=len(probes),
                    trigger_fraction=sum(e['trigger'] for e in probes)/len(probes),
                    median_scaled_gradient_error=float(np.median(scaled)) if scaled else None,
                    median_gradient_error=float(np.median(finite)) if finite else None,
                    p90_gradient_error=float(np.quantile(finite,.9)) if finite else None))
                if spec['mode']=='transport':
                    probes=[e for e in row['events'] if e['reason']=='action_probe']
                    values=[e['action_drift'] for e in probes if e['action_drift'] is not None]
                    refresh=[e for e in row['events'] if e['reason']=='transport_refresh']
                    transport_queries.append(dict(family=task['family'],seed=seed,method=spec['name'],
                        action_checks=len(probes),trigger_fraction=sum(e['trigger'] for e in probes)/max(1,len(probes)),
                        median_action_drift=float(np.median(values)) if values else None,
                        accepted_teachers=row['accepted_refreshes'],rejected_teachers=row['rejected_refreshes'],
                        trajectory_batches=row['teacher_trajectories']//cfg['batch'],covector_labels=row['teacher_pairs'],
                        minimum_learning_rate=min(e['learning_rate'] for e in refresh),
                        updates_per_attempted_refresh=row['attempted_updates']/row['full_refreshes']))
    for file in sorted((root/'evaluation').glob('*.pt')):
        raw=torch.load(file,map_location='cpu',weights_only=False)
        meta=json.loads(file.with_suffix('.json').read_text());p=meta['problem'];family=p['family']
        task=next(t for t in cfg['tasks'] if t['family']==family)
        for name,budgets in raw['methods'].items():
            for budget,values in budgets.items():
                costs.append(dict(case=file.stem,family=family,dimension=raw['x'].shape[1],condition=meta['condition'],
                    budget=float(budget),method=name,mean=values.mean().item(),p95=float(np.quantile(values.numpy(),.95))))
                audit_rows.extend(dict(case=file.stem,method=name,budget=float(budget),**a) for a in meta['methods'][name][budget]['audits'])
        for budget in cfg['budgets_seconds']:
            candidate=raw['methods']['adaptive'][str(budget)].numpy()
            for name in raw['methods']:
                if name=='adaptive':continue
                result=comparison(candidate,raw['methods'][name][str(budget)].numpy())
                contrasts.append(dict(case=file.stem,family=family,dimension=raw['x'].shape[1],condition=meta['condition'],
                    budget=budget,candidate='adaptive',reference=name,**result))
        if raw['x'].shape[1]==32 and meta['condition']=='nominal':
            references=task.get('primary_references',cfg['primary_references'])
            primary=[c for c in contrasts if c['case']==file.stem and c['budget']==60 and c['reference'] in references]
            assert len(primary)==len(references)
            decisions[family]=dict(noninferior_to_all_standard=all(c['noninferior'] for c in primary),
                superior_one_percent_to_all_standard=all(c['superior_one_percent'] for c in primary),
                comparisons={c['reference']:c for c in primary},development_selected_reference=task['principal_reference'])
        parent_result=comparison(raw['methods']['adaptive']['60'].numpy(),raw['parent'].numpy())
        contrasts.append(dict(case=file.stem,family=family,dimension=raw['x'].shape[1],condition=meta['condition'],
            budget=60,candidate='adaptive',reference='parent',**parent_result))
    objectives=[row for p in (root/'objective_diagnostics').glob('*.json') for row in json.loads(p.read_text())]
    qps=[json.loads(p.read_text()) for p in (root/'qp_references').glob('*.json')]
    transport_gradients=[row for p in (root/'transport_diagnostics').glob('*.json') for row in json.loads(p.read_text())]
    teacher_replays=[row for p in (root/'teacher_audit').glob('*.json') for row in json.loads(p.read_text())]
    summary=dict(training_seeds=cfg['seeds'],cases=len(list((root/'evaluation').glob('*.pt'))),
        costs=costs,comparisons=contrasts,primary_decisions=decisions,training=train_rows,query_diagnostics=query_rows,
        audits=audit_rows,objective_diagnostics=objectives,qp_references=qps,
        transport_gradient_diagnostics=transport_gradients,transport_query_diagnostics=transport_queries,
        accepted_teacher_replay_diagnostics=teacher_replays,
        statistical_scope='Three family questions; each requires noninferiority to every prospectively selected standard baseline, including matched initial-objective controls. Each primary component tests the paired seed arithmetic mean-cost difference A-1.01B, with one-sided alpha .05/3, df=9. Superiority additionally tests A-B and requires >=1% pooled point reduction. The within-family intersection-union test needs all components. corrected_upper_ratio is a secondary geometric-seed log-ratio bound, not an arithmetic mean-ratio confidence bound or the primary decision. Crossed-bootstrap intervals are descriptive.',
        interpretation_scope='Equal total training allocation, not equal updates. Full-trajectory covector labels and individual prefix-boundary queries have different units and cannot be compared as identical oracle calls. Transport has full-gradient agreement at teacher/student equality under differentiability assumptions; action drift and Adam updates carry no general improvement guarantee.')
    write(out/'summary.json',summary)
    csvwrite(out/'costs.csv',costs);csvwrite(out/'training.csv',train_rows)
    csvwrite(out/'comparisons.csv',[{k:v for k,v in c.items() if not isinstance(v,(list,dict))} for c in contrasts])
    if query_rows:csvwrite(out/'query_diagnostics.csv',query_rows)
    if transport_queries:csvwrite(out/'transport_queries.csv',transport_queries)
    print(json.dumps(dict(primary_decisions=decisions,cases=summary['cases']),indent=2))


if __name__=='__main__':main()
