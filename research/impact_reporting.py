"""Summarize completed post-review evidence; never train or select policies."""
import collections,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
from impact_core import HERE,SUB,PROJECT,write,digest

def interval(a,b):
    d=np.asarray(a)-np.asarray(b);n=len(d);mean=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(n))
    half=float(student_t.ppf(.975,n-1)*se)
    return dict(mean=mean,lower=mean-half,upper=mean+half,n=n,wins=int((d<0).sum()),
                relative_percent=100*(float(np.mean(a))/float(np.mean(b))-1))


def main():
    reports={name:json.loads((HERE/f'results/{name}/report.json').read_text()) for name in
        ['exact_diagnostic','mechanism','pilot','replication_v3','noise_validation','factorial','building']}
    rows=reports['replication_v3']['rows'];labels={'clone':'Cloning','hamiltonian':'Cached H','selected_fixed':'Selected fixed','adaptive':'Adaptive','guided':'Audit-guided'}
    summary=dict(training_phases=dict(mechanism=len(reports['mechanism']['rows']),pilot=len(reports['pilot']['rows']),replication=len(rows)),replication={},noise={},factorial={},building={})
    comparison_rows=[]
    for f in ['mechanical','reaction']:
        grouped={m:sorted([r for r in rows if r['family']==f and r['comparison_label']==m],key=lambda r:r['seed']) for m in labels}
        assert all(len(v)==5 for v in grouped.values())
        means={m:float(np.mean([r['test_cost'] for r in group])) for m,group in grouped.items()}
        comparisons={m:interval([r['test_cost'] for r in grouped['guided']],[r['test_cost'] for r in grouped[m]]) for m in labels if m!='guided'}
        operations=collections.Counter(e['operation'] for r in grouped['guided'] for e in r['events'])
        summary['replication'][f]=dict(means=means,comparisons=comparisons,operations=dict(operations),
            teacher_cost=float(np.mean([r['teacher_test_cost'] for r in grouped['guided']])),
            initial_student_cost=float(np.mean([r['initial_test_cost'] for r in grouped['guided']])),
            guided_wins_over_teacher=sum(r['test_cost']<r['teacher_test_cost'] for r in grouped['guided']),
            mean_updates={m:float(np.mean([r['updates'] for r in group])) for m,group in grouped.items()},
            mean_common_seconds=float(np.mean([r['common_seconds'] for r in grouped['guided']])))
        comparison_rows.append([f.capitalize(),'Teacher',f"{summary['replication'][f]['teacher_cost']:.6f}",'--','--'])
        comparison_rows.append([f.capitalize(),'Initial student',f"{summary['replication'][f]['initial_student_cost']:.6f}",'--','--'])
        for m in labels:comparison_rows.append([f.capitalize(),labels[m],f'{means[m]:.6f}',
            f"{np.mean([r['updates'] for r in grouped[m]]):.0f}",f"{np.mean([r['audits'] for r in grouped[m]]):.1f}"])
    mechanism_rows=[]
    for f in ['mechanical','reaction']:
        for q in [7.5,15.,30.]:
            group=[r for r in reports['mechanism']['rows'] if r['family']==f and r['teacher_quality_budget']==q and r['noise']=='none']
            by_method={r['method']:r for r in group}
            mechanism_rows.append([f.capitalize(),f'{q:g}',f"{group[0]['teacher_test_cost']:.4f}"]+
                [f"{by_method[m]['test_cost']:.4f}" for m in ['clone','hamiltonian','fixed16','guided']])
    intervals=[]
    for f,v in summary['replication'].items():
        for m,c in v['comparisons'].items():intervals.append([f.capitalize(),labels[m],f"{c['mean']:+.6f}",f"[{c['lower']:+.6f}, {c['upper']:+.6f}]",f"{c['wins']}/5"])
    noise_rows=[]
    for family in ['mechanical','reaction']:
        summary['noise'][family]={}
        for kind,magnitude in reports['noise_validation']['config']['perturbations']:
            group=[r for r in reports['noise_validation']['rows'] if r['family']==family and r['noise']==kind and r['magnitude']==magnitude]
            sums={k:sum(r[k] for r in group) for k in ['points','naive_predictions','naive_false_positives','guarded_predictions','guarded_false_positives','actual_improvements','bound_violations']}
            summary['noise'][family][f'{kind}{magnitude:g}']=sums
            noise_rows.append([family.capitalize(),f'{kind} {magnitude:g}',
                f"{sums['naive_false_positives']}/{sums['naive_predictions']}",
                f"{sums['guarded_false_positives']}/{sums['guarded_predictions']}",
                f"{100*sums['guarded_predictions']/sums['points']:.1f}"])
    factorial_rows=reports['factorial']['rows']
    for model in ['nominal','changed']:
        for initial in ['nominal','shifted']:
            key=model+'/'+initial
            group=[r for r in factorial_rows if r['model']==model and r['initial']==initial and r['base']=='adapted']
            summary['factorial'][key]={}
            for training,method in [('base_only','base'),('transfer','hjb_target'),('direct','hjb_target'),('direct','dpc'),('transfer','dpc')]:
                v=[r['cost'] for r in group if r['training']==training and r['method']==method]
                summary['factorial'][key][training+'/'+method]=float(np.mean(v))
    factorial_table=[]
    for condition,v in summary['factorial'].items():factorial_table.append([condition]+[f'{x:.6f}' for x in v.values()])
    building=reports['building'];qp=building['qp'];br=[]
    for m in ['hjb_target','warm','dpc','short']:
        group=[r for r in building['rows'] if r['method']==m]
        summary['building'][m]={k:float(np.mean([r[k] for r in group])) for k in
            ['cost','qp_gap_percent','tracking_rms_temperature_deviation','mean_integrated_heating_control','gpu_policy_median_seconds','gpu_policy_p95_seconds']}
        v=summary['building'][m];br.append([m.replace('_',' '),f"{v['cost']:.6f}",f"{v['qp_gap_percent']:.4f}",f"{v['tracking_rms_temperature_deviation']:.3f}",f"{v['gpu_policy_median_seconds']*1000:.3f}"])
    summary['building']['qp']=dict(cost=building['reference_cost'],cpu_median_seconds=float(np.median([r['cpu_complete_call_seconds'] for r in qp])),
        tracking_rms_temperature_deviation=float(np.sqrt(np.mean([r['tracking_rms_temperature_deviation']**2 for r in qp]))))
    write(HERE/'results/impact_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
