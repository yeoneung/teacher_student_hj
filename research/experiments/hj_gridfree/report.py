"""Reproducible paired statistics and publication figures from locked outputs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .engine import write
from .train import verify_lock


def comparison(a,b,seed=12345,repetitions=3000):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.ndim==1: a=a[None]
    if b.ndim==1: b=np.broadcast_to(b,a.shape)
    if b.shape[0]==1: b=np.broadcast_to(b,a.shape)
    assert a.shape==b.shape and np.all(b>0)
    s,n=a.shape; gen=np.random.default_rng(seed); draws=[]
    for start in range(0,repetitions,100):
        count=min(100,repetitions-start)
        si=gen.integers(s,size=(count,s)); xi=gen.integers(n,size=(count,n))
        aa=a[si[:,:,None],xi[:,None,:]].mean((1,2)); bb=b[si[:,:,None],xi[:,None,:]].mean((1,2))
        draws.extend((100*(1-aa/bb)).tolist())
    return dict(reduction_percent=float(100*(1-a.mean()/b.mean())),ci95_percent=np.quantile(draws,[.025,.975]).tolist(),seed_reductions_percent=(100*(1-a.mean(1)/b.mean(1))).tolist(),paired_state_win_fraction=float((a<b).mean()),paired_ratio_p95=float(np.quantile(a/b,.95)),bootstrap_repetitions=repetitions)


def collect(root,cfg):
    summary=dict(configuration=cfg,training={},cases={},timing={},isolated_memory={},statistics='Crossed bootstrap: resample training seeds and common test-state indices independently. Ratio of pooled mean costs; positive reduction is favorable. Descriptive95% intervals, no multiple-comparison correction.')
    for family in cfg['families']:
        records=[json.loads((root/'training'/f'{family}_s{s}'/'results.json').read_text()) for s in cfg['seeds']]
        summary['training'][family]={}
        for mode in cfg['modes']:
            rr=[r['methods'][mode] for r in records]
            total=[r['fit_seconds']+r['setup_seconds']+r['validation_seconds']+r['preparation_seconds'] for r in rr]
            summary['training'][family][mode]=dict(mean_validation_cost=float(np.mean([r['mean'] for r in rr])),fit_seconds=float(np.mean([r['fit_seconds'] for r in rr])),preparation_seconds=float(np.mean([r['preparation_seconds'] for r in rr])),total_seconds=float(np.mean(total)),seed_total_seconds=total,max_recorded_process_gpu_bytes=max(r['peak_gpu_allocated_bytes'] for r in rr),simulated_transitions_per_fit=rr[0]['training_simulated_transitions'],neural_evaluations_per_fit=rr[0]['training_neural_evaluations'])
            mem=json.loads((root/'isolated_memory'/f'{family}_{mode}.json').read_text())
            summary['isolated_memory'][f'{family}_{mode}']=mem
    for f in sorted((root/'evaluation').glob('*.pt')):
        raw=torch.load(f,weights_only=False); meta=json.loads(f.with_suffix('.json').read_text())
        fam=meta['problem']['family']; a=raw['neural']['block16'].numpy()
        case=dict(family=fam,dimension=raw['x'].shape[1],distribution=meta['distribution'],count=len(raw['x']),means={},comparisons={},audit=meta['audit'],gate=meta['methods']['block16_gate'])
        all_audits=[v for v in meta['methods'].values() if 'independent_scaled_cost_error' in v]+[v for z in meta['neural'].values() for v in z['audits']]
        case['audit']=dict(max_independent_scaled_cost_error=max(z['independent_scaled_cost_error'] for z in all_audits),max_budget_ratio=max(z['max_budget_ratio'] for z in all_audits))
        for mode,c in raw['neural'].items():
            case['means'][mode]=c.mean().item()
            case['comparisons'][mode]=comparison(a,c.numpy())
        for method in ['lqr','tuned_pd','cem','adam_multistart_1024','cem_adam512','best_classical']:
            case['means'][method]=raw['classical'][method].mean().item()
            case['comparisons'][method]=comparison(a,raw['classical'][method].numpy())
        nr=torch.load(root/'native_evaluation'/f.name,weights_only=False)
        nm=json.loads((root/'native_evaluation'/f.with_suffix('.json').name).read_text())
        count=len(nr['indices']); two=np.array([min(z['lqr']['cost'],z['zero']['cost']) for z in nm['rows']])
        case['native_subset']=dict(count=count,bellman_mean=float(a[:,:count].mean()),two_start_mean=float(two.mean()),hybrid_mean=nr['cost'].mean().item(),two_start_comparison=comparison(a[:,:count],two),hybrid_comparison=comparison(a[:,:count],nr['cost'].numpy()),cpu_quality_wall_seconds=nm['wall_seconds'])
        t=summary['training'][fam]
        case['targets']=dict(lqr_mean_reduction_at_least_10=case['comparisons']['lqr']['reduction_percent']>=10,
            hj_quality_advantage_at_least_5_over_both_mse_and_dpc=min(case['comparisons']['mse']['reduction_percent'],case['comparisons']['dpc']['reduction_percent'])>=5,
            dpc_quality_within_1_at_half_total_training_time=case['means']['block16']<=1.01*case['means']['dpc'] and t['block16']['total_seconds']<=.5*t['dpc']['total_seconds'])
        case['seed_means']={m:c.mean(1).tolist() for m,c in raw['neural'].items()}
        case['tails']={m:dict(p95_cost=float(np.quantile(c.numpy(),.95)),max_cost=c.max().item()) for m,c in raw['neural'].items()}
        # A feasible envelope is a diagnostic; it is neither a solver nor a known optimum.
        envelope=torch.stack([raw['classical']['best_classical']]+[c.min(0).values for c in raw['neural'].values()]+list(raw['polished'].values())).min(0).values
        case['empirical_envelope']=dict(mean=envelope.mean().item(),bellman_mean_excess_percent=float(100*(a.mean()/envelope.mean().item()-1)),bellman_fraction_over_5_percent=float((raw['neural']['block16']>1.05*envelope).double().mean().item()),scope='Casewise best feasible returned cost across all neural seeds, GPU classical methods, and first-seed neural polishing. Not globally optimal.')
        summary['cases'][f.stem]=case
    assert len(summary['cases'])==16
    for f in sorted((root/'timing').glob('*.json')): summary['timing'][f.stem]=json.loads(f.read_text())
    assert len(summary['timing'])==8
    return summary


def tex_table(headers,rows,alignment):
    return '\\begin{tabular}{'+alignment+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'


def plots(root,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    target=Path('paper/figures'); target.mkdir(exist_ok=True)
    colors={'block16':'#087f8c','dpc':'#793f98','truncated16':'#d98427','best_classical':'#555555','lqr':'#aaa6a1'}
    labels={'block16':'Bellman-16','dpc':'DPC','truncated16':'Truncated16','best_classical':'GPU search','lqr':'LQR'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,fam in zip(axes,['mechanical','reaction']):
        cases=[summary['cases'][f'{fam}_d{d}_nominal'] for d in [32,64,128,256]]
        for mode in colors:
            ax.plot([c['dimension'] for c in cases],[c['means'][mode] for c in cases],marker='o',label=labels[mode],color=colors[mode])
        ax.set_xscale('log',base=2); ax.set_xticks([32,64,128,256],['32','64','128','256']); ax.set_xlabel('State dimension'); ax.set_ylabel('Mean cost'); ax.set_title(fam.capitalize())
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    for ext in ['pdf','png']: fig.savefig(target/f'hj_gridfree_scaling.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,fam in zip(axes,['mechanical','reaction']):
        lower=[]; upper=[]
        for mode in ['block16','dpc','truncated16']:
            rr=[json.loads((root/'training'/f'{fam}_s{s}'/'results.json').read_text())['methods'][mode] for s in summary['configuration']['seeds']]
            times=np.mean([[z['fit_seconds']+r['setup_seconds']+r['preparation_seconds'] for z in r['history']] for r in rr],axis=0)
            costs=np.array([[z['best'] for z in r['history']] for r in rr]); avg=costs.mean(0); sd=costs.std(0)
            ax.plot(times,avg,label=labels[mode],color=colors[mode]); ax.fill_between(times,avg-sd,avg+sd,color=colors[mode],alpha=.12)
            lower.extend(avg-sd); upper.extend(avg+sd)
        low=min(lower); high=max(upper); margin=.08*(high-low)
        ax.set_ylim(max(0,low-margin),high+margin); ax.set_title(fam.capitalize()); ax.set_xlabel('Fit + setup + preparation (s)'); ax.set_ylabel('Best validation mean cost'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    for ext in ['pdf','png']: fig.savefig(target/f'hj_gridfree_learning.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,fam in zip(axes,['mechanical','reaction']):
        for offset,key,label,color in [(-.08,'two_start_comparison','Native two-start','#59656f'),(.08,'hybrid_comparison','Native + GPU search','#087f8c')]:
            cc=[summary['cases'][f'{fam}_d{d}_nominal']['native_subset'][key] for d in [32,64,128,256]]
            y=np.array([c['reduction_percent'] for c in cc]); ci=np.array([c['ci95_percent'] for c in cc])
            ax.errorbar(np.arange(4)+offset,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=3,label=label,color=color)
        ax.axhline(0,color='black',lw=.8); ax.set_xticks(range(4),['32','64','128','256']); ax.set_xlabel('State dimension'); ax.set_ylabel('Bellman cost reduction (%)'); ax.set_title(fam.capitalize()); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    for ext in ['pdf','png']: fig.savefig(target/f'hj_gridfree_native.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.7)); ax.set_xlim(-10,41); ax.set_ylim(-.6,3.2); ax.axis('off')
    for y,label,prefix,tail in [(2.2,'Bellman-16',16,True),(1.1,'DPC',40,False),(0,'Truncated16',16,False)]:
        ax.text(-1,y+.2,label,ha='right',va='center'); ax.add_patch(Rectangle((0,y),prefix,.42,color='#087f8c'))
        if tail: ax.add_patch(Rectangle((prefix,y),40-prefix,.42,color='#9eabb3')); ax.text(28,y+.21,'fixed teacher',ha='center',va='center',color='white')
        ax.text(prefix/2,y+.21,'student',ha='center',va='center',color='white')
        ax.text(prefix,y-.18,'teacher value' if tail else 'terminal cost',ha='center',va='top',fontsize=9)
    ax.text(20,3,'Teacher continuation beyond the neural prefix',ha='center',weight='bold')
    for ext in ['pdf','png']: fig.savefig(target/f'hj_gridfree_interface.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='experiments/results/hj_gridfree_primary_v1')
    args=ap.parse_args(); torch.set_num_threads(1); root=Path(args.root); cfg=verify_lock(root)
    summary=collect(root,cfg); out=root/'report'; out.mkdir(exist_ok=True)
    write(out/'summary.json',summary); plots(root,summary)
    rows=[]
    for key,c in summary['cases'].items():
        for method,v in c['comparisons'].items(): rows.append(dict(case=key,comparator=method,reduction_percent=v['reduction_percent'],ci_low=v['ci95_percent'][0],ci_high=v['ci95_percent'][1]))
    with (out/'paired_comparisons.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(cases=len(summary['cases']),training_fits=50,summary=str(out/'summary.json'))),flush=True)


if __name__=='__main__': main()
