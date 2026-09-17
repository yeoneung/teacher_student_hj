"""Read-only numerical completion audit and descriptive figures for v9."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
from experiments.hj_gridfree.engine import write
from experiments.hj_proximal.study import sha,utc
from .study import folder,verify
from .evaluate import signature

ROOTS={stage:Path('experiments/results/hj_target_'+stage+'_v9') for stage in ['primary','controls','scaling']}
DEST=Path('build/hj_v9_final')
PROTECTED={
    'build/hj_teacher_transport_v7.pdf':'db2f8ed51bc2f2f1968b8f089ea00612c36e91d8a4bc71ec6555d0b805bd19ca',
    'experiments/results/hj_transport_primary_v7/completion_audit.json':'7a150aaad2da5094285cc978e91428fceeb9b63229c82bf24b290e0387173de2',
    'experiments/hj_adaptive/completion.py':'06f134ca51126118c2af8b8220792305cf9017cd74a5ca5bc9119f8ece44f8b4',
    'experiments/results/hj_transport_primary_v7/training_lock.json':'9b3f670934eb470bd7a7af2c5956012dd958b1f2dba3f851e7a157f6d51b92ae',
    'experiments/results/hj_transport_primary_v7/evaluation_lock.json':'ba181f4d85ec2ee1761886dc6cb70b7c5b356b8397b4b5db61db8ee0506ea3e0',
}


def verify_hashes(path):
    data=json.loads(Path(path).read_text())
    for name,expected in data['hashes'].items():assert sha(name)==expected,name
    return len(data['hashes'])


def training_audit(root,cfg):
    rows=[];selections=0;unique=0;gate_checks=0
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            out=folder(root,task,seed);assert (out/'complete.json').exists()
            parent=torch.load(out/'parent.pt',map_location='cpu',weights_only=False)['budgets']['7.5']
            for spec in task['methods']:
                name=spec['name'];record=json.loads((out/(name+'.json')).read_text())
                saved=torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False)['budgets']
                assert set(record['budgets'])==set(saved)
                latest='7.5' if spec.get('parent_only') else str(max(cfg['budgets_seconds']))
                assert latest in saved,(out,name,latest)
                signatures=set()
                for budget,snap in saved.items():
                    assert snap['available_seconds']<=float(budget)
                    meta=record['budgets'][budget]
                    for key in ['available_seconds','validation_mean','best_step']:
                        assert meta[key]==snap[key],(out,name,budget,key)
                    assert all(torch.isfinite(v).all() for v in snap['state_dict'].values())
                    sig=signature(snap['state_dict']);signatures.add(sig);selections+=1
                    if snap.get('from_parent'):
                        assert sig==signature(parent['state_dict'])
                        assert snap['validation_mean']==parent['validation_mean']
                unique+=len(signatures)
                if spec['mode']=='proximal':
                    events=record['events'];accepted=0;rejected=0
                    for event in events:
                        previous=event['previous_training_mean']
                        if previous is None:continue
                        gate_checks+=1
                        actual=event['candidate_training_mean'];incumbent=event['incumbent_training_mean']
                        assert math.isfinite(incumbent)
                        if event['accepted']:
                            accepted+=1
                            assert actual<=previous+1e-7*(1+abs(previous))+1e-12
                            assert incumbent==actual
                        else:
                            rejected+=1;assert incumbent==previous
                        if spec.get('fixed_regularization'):
                            assert event['regularization']==event['previous_regularization']==16.
                    assert accepted==record['accepted_cycles'] and rejected==record['rejected_cycles']
                rows.append(dict(stage=cfg['stage'],family=task['family'],dimension=task['dimension'],
                    horizon=task['horizon'],seed=seed,method=name,setup_seconds=record['setup_seconds'],
                    query_seconds=record.get('query_seconds',0.),total_compute_seconds=record['total_compute_seconds'],
                    overrun_seconds=record['overrun_seconds'],attempted_updates=record['attempted_updates'],
                    nonfinite_updates_skipped=record['nonfinite_updates_skipped'],
                    accepted_cycles=record.get('accepted_cycles'),rejected_cycles=record.get('rejected_cycles'),
                    final_from_parent=bool(saved[latest].get('from_parent',False)),
                    peak_cuda_allocated_bytes=record['peak_cuda_allocated_bytes'],
                    peak_cuda_reserved_bytes=record['peak_cuda_reserved_bytes']))
    return dict(phases=rows,budget_selections=selections,unique_within_method_selections=unique,gate_checks=gate_checks)


def evaluate_audit(root,cfg,report):
    raw_count=0;missing=[];independent=[];max_mean_error=0.;max_stat_error=0.
    for case in report['cases']:
        raw=torch.load(root/'evaluation'/(case['key']+'.pt'),map_location='cpu',weights_only=False)
        assert raw['x'].shape==(case['count'],case['dimension'])
        assert torch.isfinite(raw['x']).all()
        for method,points in raw['methods'].items():
            for budget,values in points.items():
                assert values.shape==(len(cfg['seeds']),case['count'])
                finite=torch.isfinite(values).all(1);meta=case['methods'][method][budget]
                assert int(finite.sum())==meta['available_seeds']
                raw_count+=int(torch.isfinite(values).sum())
                if finite.all():
                    err=abs(float(values.mean())-meta['mean']);max_mean_error=max(max_mean_error,err);assert err<1e-12
                    assert np.max(np.abs(values.mean(1).numpy()-np.array(meta['seed_means'])))<1e-12
                else:missing.append(dict(case=case['key'],method=method,budget=budget,available_seeds=int(finite.sum())))
        independent.extend(case['audits'])
        if cfg['stage']=='primary' and case['dimension']==32:
            decision=next(x for x in report['primary_decisions'] if x['family']==case['family'])
            candidate=raw['methods']['hjb_target']['30.0'].mean(1).numpy()
            sup=[];noninf=[]
            for name in ['short','dpc','warm']:
                reference=raw['methods'][name]['30.0'].mean(1).numpy()
                for label,margin in [('superiority',1.),('noninferiority',1.01)]:
                    d=candidate-margin*reference;n=len(d)
                    sem=np.sqrt(np.sum((d-d.mean())**2)/(n*(n-1)))
                    upper=float(d.mean()+student_t.isf(.05/3,n-1)*sem)
                    error=abs(upper-decision['references'][name][label]['upper_difference'])
                    max_stat_error=max(max_stat_error,error);assert error<1e-12
                    if label=='superiority':sup.append(upper<0 and candidate.mean()/reference.mean()<=.99)
                    else:noninf.append(upper<=0)
            assert all(sup)==decision['all_reference_superiority_with_observed_1pct']
            assert all(noninf)==decision['all_reference_1pct_noninferiority']
    replay=json.loads((root/'validation_replay.json').read_text())
    assert max(row['error'] for row in replay['rows'])==report['maximum_validation_replay_error']<1e-10
    assert len(replay['rows'])==report['validation_replay_count']
    return dict(raw_finite_costs=raw_count,missing_early_budget_rows=missing,
        maximum_mean_recomputation_error=max_mean_error,maximum_statistic_recomputation_error=max_stat_error,
        independent_policy_checks=len(independent),maximum_feedback_scaled_error=max(x['feedback_scaled_error'] for x in independent),
        validation_replay_count=len(replay['rows']),maximum_validation_replay_error=replay['maximum_error'])


def figures(reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'short':'#4c78a8','dpc':'#b279a2','warm':'#f2a541','hjb_target':'#169b88',
        'cached_hamiltonian':'#4c78a8','fixed_rho':'#e58f42','no_value_signal':'#b279a2'}
    labels={'short':'Short','dpc':'DPC','warm':'Warm DPC','hjb_target':'Explicit targets',
        'cached_hamiltonian':'Cached Hamiltonian','fixed_rho':'Fixed proximal coefficient','no_value_signal':'Zero value covector'}
    destination=Path('paper/figures');destination.mkdir(exist_ok=True)
    primary=[x for x in reports['primary']['cases'] if x['dimension']==32]
    fig,axes=plt.subplots(3,1,figsize=(6.3,8.1))
    for ax,case in zip(axes,primary):
        best=min(case['methods'][n]['30.0']['mean'] for n in ['short','dpc','warm'])
        for name in ['short','dpc','warm','hjb_target']:
            points=case['methods'][name];x=np.array([float(k) for k in points]);y=np.array([v['mean']/best if v['mean'] is not None else np.nan for v in points.values()])
            ax.plot(x,y,'o-',markersize=3,color=colors[name],label=labels[name])
        ax.axhline(1.01,color='.5',ls=':',lw=1);ax.set_title(case['family'].capitalize())
        ax.set_xlabel('Charged wall budget (s)');ax.set_xlim(7,30.5)
        all_values=[v['mean']/best for n in ['short','dpc','warm','hjb_target'] for v in case['methods'][n].values() if v['mean'] is not None]
        if max(all_values)/min(all_values)>3:ax.set_yscale('log')
    for ax in axes:ax.set_ylabel('Cost / final best reference')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.075,1,1))
    for ext in ['pdf','png']:fig.savefig(destination/('hj_v9_budget_curves.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(6.3,7.6))
    for ax,case in zip(axes,primary):
        candidate=np.array(case['methods']['hjb_target']['30.0']['seed_means'])
        for index,name in enumerate(['short','dpc','warm']):
            reference=np.array(case['methods'][name]['30.0']['seed_means'])
            ratios=100*(candidate/reference-1)
            jitter=np.linspace(-.11,.11,len(ratios))
            ax.scatter(index+jitter,ratios,s=22,color=colors[name],alpha=.8)
            aggregate=100*(candidate.mean()/reference.mean()-1)
            ax.plot([index-.18,index+.18],[aggregate]*2,color='black',lw=2)
        ax.axhline(0,color='.4',lw=1);ax.axhline(-1,color='.5',ls=':',lw=1)
        ax.set_xticks(range(3),['Short','DPC','Warm DPC']);ax.set_title(case['family'].capitalize())
    for ax in axes:ax.set_ylabel('Candidate cost difference (%)')
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(destination/('hj_v9_seed_effects.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(6.3,7.6))
    control_names=['warm','cached_hamiltonian','fixed_rho','no_value_signal']
    for ax,case in zip(axes,reports['controls']['cases']):
        target=case['methods']['hjb_target']['30.0']['mean']
        ys=[100*(case['methods'][n]['30.0']['mean']/target-1) for n in control_names]
        ax.barh(range(4),ys,color=[colors[n] for n in control_names]);ax.axvline(0,color='.3',lw=1)
        ax.set_yticks(range(4),['Warm DPC','Cached H','Fixed coefficient','Zero covector']);ax.invert_yaxis()
        ax.set_xlabel('Cost relative to target fitting (%)');ax.set_title(case['family'].capitalize())
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(destination/('hj_v9_components.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    torch.set_num_threads(1);DEST.mkdir(parents=True,exist_ok=True)
    reports={};stages={};hash_count=0
    for stage,root in ROOTS.items():
        cfg=verify(root);report=json.loads((root/'report.json').read_text());reports[stage]=report
        assert json.loads((root/'status.json').read_text())['stage']=='evaluation_complete'
        hash_count+=verify_hashes(root/'training_lock.json')+verify_hashes(root/'evaluation_lock.json')
        stages[stage]=dict(training=training_audit(root,cfg),evaluation=evaluate_audit(root,cfg,report))
    expected={'primary':150,'controls':36,'scaling':30}
    for stage,n in expected.items():assert len(stages[stage]['training']['phases'])==n
    for path,expected_hash in PROTECTED.items():assert sha(path)==expected_hash,path
    for root in ['hj_proximal_dev_v8a','hj_curvature_dev_v8b']:
        for lock in Path('experiments/results',root).glob('*lock.json'):hash_count+=verify_hashes(lock)
    for path in ['build/hj_v9_cpu_audit.json','build/hj_v9_controls_audit.json']:
        assert json.loads(Path(path).read_text())['passed']
    diagnostics=json.loads(Path('build/hj_v9_diagnostics.json').read_text())
    assert len(diagnostics['geometry'])==150 and len(diagnostics['strata'])==7
    audit=dict(completed_utc=utc(),passed=True,new_training_phases=216,verified_hash_entries=hash_count,
        protected_files=PROTECTED,stages=stages,
        scope='Phase totals include parents. Repeated budget policies are not independent training runs. All selected snapshots meet deadlines; final unselected updates may overrun.')
    write(DEST/'completion_audit.json',audit)
    figures(reports)
    print(json.dumps(dict(passed=True,new_phases=216,hash_entries=hash_count,
        raw_cost_occurrences=sum(s['evaluation']['raw_finite_costs'] for s in stages.values()),
        validation_replays=sum(s['evaluation']['validation_replay_count'] for s in stages.values()))),flush=True)


if __name__=='__main__':main()
