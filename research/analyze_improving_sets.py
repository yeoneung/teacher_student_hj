"""Recompute costs, verify frozen evidence, and report set-supervision statistics."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
from impact_core import HERE,SUB,write,digest
from improving_sets import intersection_project,cache_geometry,upper_model
from experiments.hj_cotangent.systems import problem


def main():
    evidence=[];reports={};float_errors=[]
    for stage in ['pilot','warm','replication']:
        folder=HERE/'results'/('improving_sets_'+stage)
        report=json.loads((folder/'report.json').read_text());reports[stage]=report
        lock=json.loads((folder/'protocol_lock.json').read_text())
        for name,sha in lock['sources'].items():assert digest(HERE/name)==sha,(stage,name)
        for row in report['rows']:
            key=f"{row['family']}_w{row['width']}_s{row['seed']}_{row['method']}"
            assert digest(folder/(key+'.pt'))==row['checkpoint_sha256']
            raw=torch.load(folder/(key+'_evaluation.pt'),map_location='cpu',weights_only=False)
            for field in ['test_cost','teacher_cost','initial_cost']:
                assert abs(float(raw[field].mean())-row[field])<1e-12
            assert row['nonfinite_updates']==0
            selected=row['best_update'];assert selected==0 or any(e['updates']==selected and e['accepted'] for e in row['events'])
            assert row['selected_validation_cost']<=row['initial_validation_cost']+1e-12
            d=raw['diagnostic'];u=d['action'];g=d['geometry'];p=problem(row['family'],16 if row['family']=='mechanical' else 32,160 if row['family']=='mechanical' else 320)
            ru=p.n**.5*p.bound
            pr=intersection_project(u.float(),g['center'].float(),g['radius'].float(),ru).double()
            exact=intersection_project(u,g['center'],g['radius'],ru)
            err=float((pr-exact).norm(dim=-1).max())
            float_errors.append(dict(stage=stage,key=key,max_action_error=err,normalized_error=err/ru))
            # Use alpha=.25 consistently across losses for descriptive post-fit probes.
            common=cache_geometry(d['data'],p,.000550492 if p.mechanical else .000417389,.25)
            proj=intersection_project(u,common['center'],common['radius'],ru)
            dist2=(u-proj).square().sum(-1);point2=(u-common['anchor']).square().sum(-1)
            assert float((dist2-point2).max())<1e-8
            model=upper_model(u,d['data']['old'],d['data']['b'],d['data']['r'],.000550492 if p.mechanical else .000417389)
            inside=model<=-.25*common['gain']+1e-9
            evidence.append(dict(stage=stage,family=row['family'],width=row['width'],seed=row['seed'],method=row['method'],
                point_mse=float(point2.mean()/p.n/p.bound**2),set_mse=float(dist2.mean()/p.n/p.bound**2),
                inside_fraction=float(inside.double().mean()),actual_advantage=float(d['actual'].mean()),
                upper_violation_count=int((d['actual']>model+1e-7).sum()),
                nonimproving_inside_count=int(((d['actual']>1e-7)&inside).sum()),count=len(u)))
    rows=reports['replication']['rows'];means={};comparisons={}
    for method in reports['replication']['config']['methods']:
        group=sorted([r for r in rows if r['method']==method],key=lambda r:r['seed'])
        vals=np.array([r['test_cost'] for r in group])
        probes=[e for e in evidence if e['stage']=='replication' and e['method']==method]
        means[method]=dict(mean_cost=float(vals.mean()),std_cost=float(vals.std(ddof=1)),
            mean_seconds=float(np.mean([r['charged_seconds'] for r in group])),mean_fit_seconds=float(np.mean([r['fit_seconds'] for r in group])),
            selected_initial_count=sum(r['best_update']==0 for r in group),
            mean_point_mse=float(np.mean([p['point_mse'] for p in probes])),mean_set_mse=float(np.mean([p['set_mse'] for p in probes])),
            mean_inside_fraction=float(np.mean([p['inside_fraction'] for p in probes])),
            mean_actual_advantage=float(np.mean([p['actual_advantage'] for p in probes])),
            upper_violation_count=sum(p['upper_violation_count'] for p in probes),nonimproving_inside_count=sum(p['nonimproving_inside_count'] for p in probes))
    setrows={r['seed']:r for r in rows if r['method']=='set25'}
    for method in ['point','adaptive','upper','teacher','initial']:
        control={r['seed']:r for r in rows if r['method']==('set25' if method in ['teacher','initial'] else method)}
        field=method+'_cost' if method in ['teacher','initial'] else 'test_cost'
        diff=np.array([setrows[s]['test_cost']-control[s][field] for s in sorted(setrows)])
        half=float(student_t.ppf(.975,4)*diff.std(ddof=1)/np.sqrt(5))
        baseline=float(np.mean([control[s][field] for s in sorted(setrows)]))
        comparisons[method]=dict(mean_difference=float(diff.mean()),ci95=[float(diff.mean()-half),float(diff.mean()+half)],
            relative_percent=float(100*diff.mean()/baseline),wins=int((diff<0).sum()),paired_differences=diff.tolist(),baseline_cost=baseline)
    result=dict(passed=True,phases=56,cold_pilot=reports['pilot']['decisions'],warm_pilot=reports['warm']['decisions'],
        replication_means=means,comparisons=comparisons,standardized_probe_alpha=.25,probes=evidence,float_projection_checks=float_errors,
        projection_max_relative_error=max(e['normalized_error'] for e in float_errors),
        report_hashes={stage:digest(HERE/'results'/('improving_sets_'+stage)/'report.json') for stage in reports})
    write(HERE/'results/improving_sets_analysis.json',result)
    print(json.dumps(dict(means=means,comparisons=comparisons,max_float_relative_error=result['projection_max_relative_error']),indent=2))


if __name__=='__main__':main()
