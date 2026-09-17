"""Summarize completed post-review evidence; never train or select policies."""
import collections,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from impact_core import HERE,SUB,PROJECT,write,digest

SECTIONS=SUB/'source/sections'
FIGURES=SUB/'source/figures'


def table(filename,headers,rows,caption):
    text='\\begin{table}[htbp]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n'
    text+='\\begin{tabular}{@{}'+('l'+'r'*(len(headers)-1))+'@{}}\\toprule\n'
    text+=' & '.join(headers)+'\\\\\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+'\\\\' for row in rows)+'\n'
    text+='\\bottomrule\\end{tabular}\n\\caption{'+caption+'}\n\\end{table}\n'
    (SECTIONS/filename).write_text(text)


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
    table('impact_replication_table.tex',['Task','Method','Cost','Updates','Audits'],comparison_rows,
        r'Five independently fitted students per method and task at a 30-second teaching budget. The frozen strong teachers and common weak student initializations come from archived checkpoints; teacher construction is separate. Common preparation, including covectors also charged to cloning, and all audits are included. Fixed coefficients are selected on pilot validation cost. This is a new exploratory replication, not the original ten-seed endpoint.')
    mechanism_rows=[]
    for f in ['mechanical','reaction']:
        for q in [7.5,15.,30.]:
            group=[r for r in reports['mechanism']['rows'] if r['family']==f and r['teacher_quality_budget']==q and r['noise']=='none']
            by_method={r['method']:r for r in group}
            mechanism_rows.append([f.capitalize(),f'{q:g}',f"{group[0]['teacher_test_cost']:.4f}"]+
                [f"{by_method[m]['test_cost']:.4f}" for m in ['clone','hamiltonian','fixed16','guided']])
    table('impact_teacher_table.tex',['Task','Teacher s','Teacher','Clone','H','Fixed','Audit'],mechanism_rows,
        r'Absolute teacher and student costs without injected noise in the one-seed developmental mechanism study (256 fitting updates). Common initial student costs are 3.026129 and 0.771079. All noise conditions, including failures, remain in the raw report. These developmental results are not independent replications of the selected audit rule.')
    intervals=[]
    for f,v in summary['replication'].items():
        for m,c in v['comparisons'].items():intervals.append([f.capitalize(),labels[m],f"{c['mean']:+.6f}",f"[{c['lower']:+.6f}, {c['upper']:+.6f}]",f"{c['wins']}/5"])
    table('impact_uncertainty_table.tex',['Task','Reference',r'$\Delta$ cost','95\% CI','Wins'],intervals,
        r'Paired arithmetic cost differences across five teacher/student/data replications. Intervals use a two-sided $t$ calculation with four degrees of freedom; comparisons are exploratory and not multiplicity-adjusted.')
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
    table('impact_noise_table.tex',['Task','Noise','Naive FP/pred.','Margin FP/pred.','Coverage (\%)'],noise_rows,
        r'Independent local-action diagnostics over five archived teachers and three checkpoint qualities: 480 action evaluations on 160 state/time points per task and noise condition, with states reused across teacher qualities. A fixed empirical curvature quantile comes from separate development trajectories. Ratios count false positive improvement predictions; zero predictions imply abstention. This calibrated margin is not a certified upper bound, and no student is retrained in this diagnostic.')
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
    table('impact_factorial_table.tex',['Model/initial','Base','T trans.','T direct','DPC direct','DPC trans.'],factorial_table,
        r'Common new 256-dimensional reaction states separate model-coefficient changes from initial-distribution changes. Values average three archived policies; the same underlying 256 initials are scaled by 1.25 for the shifted condition. T denotes explicit targets. Transfer and direct actors had different training dimensions and budgets, so the table is diagnostic, not a matched training comparison.')
    building=reports['building'];qp=building['qp'];br=[]
    for m in ['hjb_target','warm','dpc','short']:
        group=[r for r in building['rows'] if r['method']==m]
        summary['building'][m]={k:float(np.mean([r[k] for r in group])) for k in
            ['cost','qp_gap_percent','tracking_rms_temperature_deviation','mean_integrated_heating_control','gpu_policy_median_seconds','gpu_policy_p95_seconds']}
        v=summary['building'][m];br.append([m.replace('_',' '),f"{v['cost']:.6f}",f"{v['qp_gap_percent']:.4f}",f"{v['tracking_rms_temperature_deviation']:.3f}",f"{v['gpu_policy_median_seconds']*1000:.3f}"])
    summary['building']['qp']=dict(cost=building['reference_cost'],cpu_median_seconds=float(np.median([r['cpu_complete_call_seconds'] for r in qp])),
        tracking_rms_temperature_deviation=float(np.sqrt(np.mean([r['tracking_rms_temperature_deviation']**2 for r in qp]))))
    table('impact_building_table.tex',['Method','Cost','QP gap (\%)','Tracking RMS','GPU ms'],br,
        r'New 64-state nominal building check, averaging ten archived policy seeds. Tracking is temperature deviation RMS. GPU latency measures one resident-input policy call with synchronization; QP CPU complete-call timings are reported separately. The approximate deterministic full-horizon QP is not a receding-horizon disturbance benchmark.')
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(6.6,3.2),constrained_layout=True)
    for ax,(f,v) in zip(axes,summary['replication'].items()):
        for i,(m,c) in enumerate(v['comparisons'].items()):
            ax.errorbar(c['mean'],i,xerr=[[c['mean']-c['lower']],[c['upper']-c['mean']]],fmt='o',capsize=3,color='#286d8e')
        ax.axvline(0,color='grey',lw=.8);ax.set_yticks(range(4),[labels[m] for m in v['comparisons']]);ax.set_title(f.capitalize());ax.set_xlabel('Guided minus reference cost')
    fig.savefig(FIGURES/'impact_replication.pdf');fig.savefig(FIGURES/'impact_replication.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(6.6,6.3),constrained_layout=True)
    for row,f in enumerate(['mechanical','reaction']):
        points=summary['noise'][f];xs=np.arange(len(points));naive=[];guard=[];coverage=[]
        for v in points.values():
            naive.append(v['naive_false_positives']/max(v['naive_predictions'],1));guard.append(v['guarded_false_positives']/max(v['guarded_predictions'],1));coverage.append(v['guarded_predictions']/v['points'])
        axes[row,0].plot(xs,naive,'o-',label='Local quadratic');axes[row,0].plot(xs,guard,'s-',label='Empirical margin')
        axes[row,0].set_ylabel('False positives / predictions');axes[row,0].set_title(f.capitalize())
        axes[row,1].bar(xs,coverage,color='#4e9b89');axes[row,1].set_ylabel('Margin prediction coverage');axes[row,1].set_ylim(0,1)
        for ax in axes[row]:ax.set_xticks(xs,['0','B .5','N .5','B 1.5','N 1.5']);ax.set_xlabel('Projected-signal perturbation')
    axes[0,0].legend(fontsize=8)
    fig.savefig(FIGURES/'impact_noise.pdf');fig.savefig(FIGURES/'impact_noise.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.6,3.4),constrained_layout=True)
    for ax,f in zip(axes,['mechanical','reaction']):
        for mode,marker in [('fresh','o'),('cached','x')]:
            raw=torch.load(HERE/f'results/exact_diagnostic/{f}_{mode}.pt',weights_only=False)['raw']
            ax.scatter(raw['fitted_model'].numpy(),raw['actual'].numpy(),s=9,alpha=.5,marker=marker,label=mode)
        ax.set_xscale('symlog',linthresh=1e-5);ax.set_yscale('symlog',linthresh=1e-5)
        ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7);ax.set_title(f.capitalize());ax.set_xlabel('Predicted teacher advantage');ax.set_ylabel('Exact continuation advantage');ax.legend(fontsize=8)
    fig.savefig(FIGURES/'impact_advantage.pdf');fig.savefig(FIGURES/'impact_advantage.png',dpi=180);plt.close(fig)
    # Focus original primary plot on each family's strongest reference.
    primary=json.loads((PROJECT/'experiments/results/hj_target_primary_v9/report.json').read_text())['cases']
    controls=json.loads((PROJECT/'experiments/results/hj_target_controls_v9/report.json').read_text())['cases']
    fig,axes=plt.subplots(1,2,figsize=(6.6,3.1),constrained_layout=True)
    for i,f in enumerate(['mechanical','reaction','building']):
        case=next(c for c in primary if c['family']==f and c['dimension']==32)
        refs=['short','dpc','warm'];best=min(refs,key=lambda m:case['methods'][m]['30.0']['mean'])
        a=case['methods']['hjb_target']['30.0']['seed_means'];b=case['methods'][best]['30.0']['seed_means'];c=interval(a,b);scale=100/np.mean(b)
        axes[0].errorbar(c['mean']*scale,i,xerr=[[(c['mean']-c['lower'])*scale],[(c['upper']-c['mean'])*scale]],fmt='o',capsize=3)
        cc=next(c for c in controls if c['family']==f);a=np.array(cc['methods']['hjb_target']['30.0']['seed_means']);b=np.array(cc['methods']['cached_hamiltonian']['30.0']['seed_means'])
        axes[1].scatter(100*(a/b-1),[i]*2,s=24)
    for ax in axes:ax.axvline(0,color='grey',lw=.8);ax.set_yticks(range(3),['Mechanical','Reaction','Building']);ax.set_xlabel('Target cost difference (%)')
    axes[0].set_title('Strongest primary reference (n=10)');axes[1].set_title('Matched Hamiltonian (n=2)')
    fig.savefig(FIGURES/'impact_strong_references.pdf');fig.savefig(FIGURES/'impact_strong_references.png',dpi=180);plt.close(fig)
    write(HERE/'results/impact_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
