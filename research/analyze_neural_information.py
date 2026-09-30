"""Verify all frozen neural mechanism runs; statistics and exact decomposition.

No fitting, checkpoint selection or alteration of numerical source/results.
Seed means are units. Confirmation is reported separately from initial study.
"""
import csv,itertools,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
import torch
import neural_information_transfer as base

HERE=Path(__file__).resolve().parent;SUB=HERE.parent
STAGES=['neural_information_transfer','neural_information_confirmation']
METRICS=['normalized_regret','regret','target_mse','normal_contribution','test_cost',
         'rare_cost','common_cost','rare_axis_span','student_boundary_fraction','weighted_distance','radial_slack']

def paired(x):
    x=np.asarray(x,dtype=float);mean=float(x.mean());half=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x)))
    return dict(mean=mean,ci95=[mean-half,mean+half],wins=int(sum(x<0)),ties=int(sum(x==0)),n=len(x),differences=x.tolist())

def value(row,name):return row.get(name,row['metrics'].get(name))

def analyze(stage):
    folder=HERE/'results'/stage;report=folder/'report.json';j=json.loads(report.read_text());cfg=j['config']
    lock=json.loads((folder/'protocol_lock.json').read_text());assert lock['config']==cfg
    assert lock['source_sha256']==base.digest(Path(base.__file__))
    if stage.endswith('confirmation'):
        plan=json.loads((folder/'confirmation_plan.json').read_text())
        assert plan['config']==cfg and plan['base_source_sha256']==lock['source_sha256']
        assert plan['driver_source_sha256']==base.digest(HERE/'neural_information_confirmation.py')
        assert plan['parent_report_sha256']==base.digest(HERE/'results'/STAGES[0]/'report.json')
    rows=j['rows'];expected=list(itertools.product(cfg['seeds'],cfg['control_radii'],cfg['widths'],cfg['methods']))
    assert len(rows)==len(expected)==len(set((r['seed'],r['radius'],r['width'],r['method']) for r in rows))
    cache={};max_error=0.;equal=0
    for row in rows:
        key=f"s{row['seed']}_r{row['radius']:g}_w{row['width']}_{row['method']}"
        assert json.loads((folder/(key+'.json')).read_text())==row
        cp=folder/(key+'.pt');assert base.digest(cp)==row['checkpoint_sha256']
        assert row['nonfinite_updates']==0 and row['updates']==cfg['updates']
        raw=torch.load(folder/(key+'_evaluation.pt'),map_location='cpu',weights_only=True)
        assert len(raw['cost'])==cfg['test_initials']
        assert abs(float(raw['cost'].mean())-row['test_cost'])<1e-10
        assert abs(float(raw['local_regret'].mean()/raw['local_opportunity'].mean())-row['metrics']['normalized_regret'])<1e-10
        assert float((raw['cost']-raw['teacher_cost']-raw['advantage_sum']).abs().max())<1e-9
        if row['seed'] not in cache:cache[row['seed']]=base.teacher_states(base.initials(cfg['test_initials'],53000000+row['seed'])[0])
        model=base.Student(row['width'],row['radius']).cuda();model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True))
        with torch.no_grad():
            metric,parts=base.local_metrics(model,cache[row['seed']],row['radius'])
            assert abs(metric['normalized_regret']-row['metrics']['normalized_regret'])<1e-10
            lam=parts['normal'].norm(dim=-1)/row['radius'];u=parts['action'];v=parts['target']
            weighted=(base.CURV+lam/2)*(u-v).square().sum(-1)
            slack=lam/2*(row['radius']**2-u.square().sum(-1))
            err=float((parts['regret']-weighted-slack).abs().max());assert err<1e-8;max_error=max(max_error,err)
            row['weighted_distance']=float(weighted.mean());row['radial_slack']=float(slack.mean())
        if row['radius']==32 and row['method']!='point':
            other=folder/f"s{row['seed']}_r32_w{row['width']}_point.pt"
            reference=torch.load(other,map_location='cpu',weights_only=True);state=torch.load(cp,map_location='cpu',weights_only=True)
            assert all(torch.equal(state[k],reference[k]) for k in reference);equal+=1
    cells=[]
    for radius,width in itertools.product(cfg['control_radii'],cfg['widths']):
        group={m:[next(r for r in rows if (r['seed'],r['radius'],r['width'],r['method'])==(s,radius,width,m)) for s in cfg['seeds']] for m in cfg['methods']}
        means={m:{name:float(np.mean([value(r,name) for r in rr])) for name in METRICS} for m,rr in group.items()}
        comparisons={m:{name:paired([value(a,name)-value(b,name) for a,b in zip(group['restored'],group[m])]) for name in METRICS} for m in ['point','weight','shuffled']}
        cells.append(dict(radius=radius,width=width,means=means,restored_minus=comparisons,
            active_fraction=float(np.mean([r['metrics']['active_target_fraction'] for r in group['point']]))))
    pr=2 if stage==STAGES[0] else .5
    primary=next(c for c in cells if c['radius']==pr and c['width']==2)['restored_minus']['point']['normalized_regret']
    return dict(stage=stage,passed=True,training_runs=len(rows),report_sha256=base.digest(report),protocol=cfg,
        primary=primary,primary_benefit_supported=primary['ci95'][1]<0,cells=cells,
        audit=dict(inactive_parameter_equalities=equal,maximum_radial_identity_error=max_error,
            maximum_bellman_identity_error=max(r['metrics']['identity_error'] for r in rows),
            maximum_telescoping_error=max(r['maximum_telescoping_error'] for r in rows),
            total_fitting_seconds=sum(r['fit_seconds'] for r in rows)),rows=rows)

def main():
    torch.set_num_threads(1);results=[analyze(s) for s in STAGES]
    out=dict(passed=True,training_runs=sum(r['training_runs'] for r in results),
        inference='Unadjusted paired t intervals across seed means. The initial primary is retained; independent confirmation targets a selected secondary condition. No pooling or family-wide significance claim.',stages=results)
    base.write(HERE/'results/neural_information_analysis.json',out)
    flat=[]
    for result in results:
        for row in result['rows']:
            flat.append(dict(stage=result['stage'],seed=row['seed'],radius=row['radius'],width=row['width'],method=row['method'],**{m:value(row,m) for m in METRICS}))
    with (HERE/'results/neural_information_all_runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=flat[0].keys());writer.writeheader();writer.writerows(flat)
    print(json.dumps({r['stage']:dict(primary=r['primary'],supported=r['primary_benefit_supported'],audit=r['audit']) for r in results},indent=2))

if __name__=='__main__':main()
