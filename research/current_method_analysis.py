"""Analyze only the evaluated fixed-teacher audit algorithm; no new training."""
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SUB=HERE.parent


def paired(a,b):
    a=np.asarray(a);b=np.asarray(b);d=a-b
    half=t.ppf(.975,len(d)-1)*d.std(ddof=1)/math.sqrt(len(d))
    return dict(mean=float(d.mean()),ci=[float(d.mean()-half),float(d.mean()+half)],
                wins=int((d<0).sum()),relative_percent=float(100*(a.mean()/b.mean()-1)))


def main():
    rows=json.loads((HERE/'results/replication_v3/report.json').read_text())['rows']
    out={};fig,axes=plt.subplots(1,2,figsize=(6.6,3.3),constrained_layout=True)
    plt.rcParams.update({'font.size':9})
    for ax,f in zip(axes,['mechanical','reaction']):
        groups={m:sorted([r for r in rows if r['family']==f and r['comparison_label']==m],key=lambda r:r['seed']) for m in ['clone','hamiltonian','selected_fixed','adaptive','guided']}
        g=groups['guided'];a=[r['test_cost'] for r in g];teacher=[r['teacher_test_cost'] for r in g];initial=[r['initial_test_cost'] for r in g]
        raw=[torch.load(HERE/f"results/replication_v3/{f}_s{r['seed']}_guided_evaluation.pt",map_location='cpu',weights_only=False)['diagnostic'] for r in g]
        v={k:torch.cat([q[k].reshape(-1) for q in raw]) for k in ['actual','target_model','fitted_model','fitting_term','curvature','sensitivity_error']}
        assert torch.all(v['sensitivity_error']==0)
        assert (v['actual']-v['target_model']-v['fitting_term']-v['curvature']).abs().max()<1e-9
        pred=v['fitted_model']< -1e-8;bad=v['actual']>1e-8
        out[f]=dict(means={m:float(np.mean([r['test_cost'] for r in group])) for m,group in groups.items()},
                    teacher_cost=float(np.mean(teacher)),initial_cost=float(np.mean(initial)),
                    versus_teacher=paired(a,teacher),versus_initial=paired(a,initial),
                    versus_baselines={m:paired(a,[r['test_cost'] for r in group]) for m,group in groups.items() if m!='guided'},
                    teacher_gap_recovered_percent=float(100*(np.mean(initial)-np.mean(a))/(np.mean(initial)-np.mean(teacher))),
                    standardized_probe=dict(eta=16,points=160,means={k:float(z.mean()) for k,z in v.items()},
                        predicted_improvements=int(pred.sum()),false_positives=int((pred&bad).sum()),
                        scope='Post-fit finite-state probe with fresh exact teacher sensitivities; not the actual training target history or a full-trajectory sum.'),
                    seeds=[dict(seed=r['seed'],student=r['test_cost'],teacher=r['teacher_test_cost'],initial=r['initial_test_cost']) for r in g])
        for i,r in enumerate(g):
            q=raw[i];ax.scatter(q['fitted_model'],q['actual'],s=12,alpha=.65,label=f"Replicate {i+1}")
        ax.set_xscale('symlog',linthresh=1e-6);ax.set_yscale('symlog',linthresh=1e-6)
        ax.set_xticks([-1e-1,-1e-4,0,1e-4,1e-1] if f=='mechanical' else [-1e-5,0,1e-5,1e-3])
        ax.tick_params(axis='x',labelsize=8)
        ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7)
        ax.set_title(f.capitalize());ax.set_xlabel('Local quadratic advantage');ax.set_ylabel('Exact continuation advantage')
    axes[0].legend(fontsize=6)
    dest=SUB/'source/figures';dest.mkdir(exist_ok=True)
    for suffix in ['pdf','png']:fig.savefig(dest/f'current_advantage.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.6,3.1),constrained_layout=True)
    for ax,(f,v) in zip(axes,out.items()):
        d=[r['student']-r['teacher'] for r in v['seeds']]
        ax.bar(range(1,6),d,color=['#277c73' if x<0 else '#c17b35' for x in d])
        ax.axhline(0,color='grey',lw=.7);ax.set_xticks(range(1,6));ax.set_xlabel('Teacher/student/data replicate')
        ax.set_ylabel('Student minus teacher cost');ax.set_title(f.capitalize())
    for suffix in ['pdf','png']:fig.savefig(dest/f'current_teacher_gap.{suffix}',dpi=180)
    plt.close(fig)
    (HERE/'results/current_method_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
