"""Reevaluate all additional checkpoints and generate fair-query evidence."""
import json,statistics
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import minimal_teaching as mt
import minimal_teaching_study as st
import direct_query_controls as dc

HERE=Path(__file__).resolve().parent;OUT=HERE/'results/direct_query_study';SOURCE=HERE.parent/'source'
def read(p):return json.loads(Path(p).read_text())
def savefig(fig,name):
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(SOURCE/f'figures/{name}.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)

def main():
    torch.set_num_threads(1);plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    report=read(OUT/'report.json');lock=read(OUT/'protocol_lock.json')
    assert report['protocol_sha256']==mt.digest(OUT/'protocol_lock.json')
    assert lock['source_sha256']==mt.digest(HERE/'direct_query_study.py') and lock['controls_sha256']==mt.digest(dc.__file__)
    for p,h in lock['dependencies'].items():assert mt.digest(HERE/p)==h
    assert len(report['rows'])==120 and len(report['audits'])==85
    reevaluated=[];fidelity=[];maximum_metric_error=0.
    rows={r['key']:r for r in report['rows']}
    for row in report['rows']:
        key=row['key'];cp=OUT/(key+'.pt');ep=OUT/(key+'_evaluation.pt')
        assert mt.digest(cp)==row['checkpoint_sha256'] and mt.digest(ep)==row['evaluation_sha256']
        seed=row['seed'];d=row['dimension'];m=row['student_dimension'];R=row['radius'];shape=row['shape']
        P=dc.structured_embedding(seed+311) if row['structured'] else mt.embedding(d,m,seed+311)
        model=st.Student(P,R,shape).cuda();model.load_state_dict(torch.load(cp,weights_only=True))
        initials=dc.structured_initials if row['structured'] else mt.initials
        x,_=initials(4096,d,63000000+seed);metrics,raw=st.evaluate(model,x,R,shape)
        err=max(abs(metrics[k]-v) for k,v in row['metrics'].items());assert err<1e-11
        maximum_metric_error=max(maximum_metric_error,err)
        saved=torch.load(ep,weights_only=True)
        for k,v in raw.items():assert torch.allclose(v,saved[k],rtol=0,atol=1e-11)
        reevaluated.append(key)
        if row['method'] not in ['point','full']:
            prefix=f's{seed}_d{d}_m{m}_{shape}_r{R:g}';refkey=prefix+'_full'
            reference=rows[refkey] if row['structured'] else read(HERE/'results/minimal_teaching_study'/(refkey+'.json'))
            dr=metrics['normalized_regret']-reference['metrics']['normalized_regret']
            dcst=metrics['test_cost']-reference['metrics']['test_cost']
            fidelity.append(dict(key=key,structured=row['structured'],regret_difference=dr,cost_difference=dcst,within_tolerance=abs(dr)<=1e-3))
        print('verified',key,flush=True)
    originals=[a for a in report['audits'] if a['key'].startswith('s131')]
    boxes=[a for a in originals if a['shape']=='box'];structured=[a for a in report['audits'] if a not in originals]
    for a in originals:
        assert mt.digest(HERE/'results/minimal_teaching_study'/(a['key']+'_cache.pt'))==a['original_cache_sha256']
        assert all(h['r']==min(h['k'],a['student_dimension']) for h in a['rank_histogram'])
    methods=['rank','face']+dc.METHODS
    totals={method:sum(a['methods'][method]['queries'] for a in boxes) for method in methods}
    aligned={method:sum(a['methods'][method]['queries'] for a in structured) for method in methods}
    timings={family:{method:dict(median_ms=1000*statistics.median(statistics.median(a['methods'][method]['seconds']) for a in cells),
        minimum_ms=1000*min(statistics.median(a['methods'][method]['seconds']) for a in cells),
        maximum_ms=1000*max(statistics.median(a['methods'][method]['seconds']) for a in cells)) for method in methods}
        for family,cells in [('original_box',boxes),('original_ball',[a for a in originals if a['shape']=='ball']),('structured',structured)]}
    grouped=[]
    for structured_flag in [False,True]:
        conditions=sorted(set((r['dimension'],r['student_dimension'],r['shape'],r['radius']) for r in report['rows'] if r['structured']==structured_flag))
        for d,m,shape,R in conditions:
            for method in (['point','rank','face','full']+dc.METHODS if structured_flag else dc.METHODS):
                rr=[r for r in report['rows'] if (r['structured'],r['dimension'],r['student_dimension'],r['shape'],r['radius'],r['method'])==(structured_flag,d,m,shape,R,method)]
                grouped.append(dict(structured=structured_flag,dimension=d,student_dimension=m,shape=shape,radius=R,method=method,n=len(rr),
                    mean_regret=statistics.mean(r['metrics']['normalized_regret'] for r in rr),mean_cost=statistics.mean(r['metrics']['test_cost'] for r in rr)))
    primary=read(HERE/'results/neural_information_analysis.json')['stages'][1]['primary']
    differences=np.array(primary['differences']);heterogeneity=dict(median=float(np.median(differences)),mean=float(differences.mean()),
        largest_two_share=float(np.sort(np.abs(differences))[-2:].sum()/np.abs(differences).sum()),all_negative=bool((differences<0).all()))
    approximate=read(HERE/'results/approximate_feedback/report.json');assert approximate['passed']
    result=dict(passed=True,training_runs=120,reevaluated_policies=len(reevaluated),maximum_reevaluation_metric_error=maximum_metric_error,
        report_sha256=mt.digest(OUT/'report.json'),protocol_sha256=mt.digest(OUT/'protocol_lock.json'),source_sha256=mt.digest(__file__),
        fidelity=fidelity,fidelity_pairs=len(fidelity),fidelity_pairs_within_tolerance=sum(f['within_tolerance'] for f in fidelity),
        maximum_absolute_regret_difference=max(abs(f['regret_difference']) for f in fidelity),
        maximum_absolute_cost_difference=max(abs(f['cost_difference']) for f in fidelity),
        maximum_coefficient_error=max(a['methods'][method]['maximum_coefficient_error'] for a in report['audits'] for method in methods),
        original_box_totals=totals,structured_totals=aligned,original_hybrid_rank_count_equal=True,timings=timings,groups=grouped,
        heterogeneity=heterogeneity,approximate_report_sha256=mt.digest(HERE/'results/approximate_feedback/report.json'))
    mt.write(HERE/'results/direct_query_analysis.json',result)
    # Direct-query plot includes the strongest simple controls.
    labels=['Relevant','Full face','Student basis','Orthogonal','Hybrid','Pruned hybrid']
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.3));colors=['#315f9b','#889299','#889299','#889299','#25876e','#25876e']
    for ax,values,title in zip(axes,[totals,aligned],['Random embeddings: box caches','Shared actuation: constructed caches']):
        divisor=491520 if values is totals else 61440
        yy=np.array([values[m] for m in methods])/divisor
        ax.bar(np.arange(6),yy,color=colors);ax.set_xticks(np.arange(6),labels,rotation=40,ha='right')
        ax.set(ylabel='Queries per cached state',title=title)
        for i,y in enumerate(yy):ax.text(i,y+.035,f'{y:.3f}' if values is totals else f'{y:g}',ha='center',fontsize=8)
        ax.set_ylim(0,max(yy)*1.19);ax.spines[['top','right']].set_visible(False)
    savefig(fig,'direct_query_controls')
    fig,ax=plt.subplots(figsize=(7.4,2.5));ax.scatter(np.arange(1,11),differences,color='#315f9b',zorder=3)
    ax.axhline(0,color='black',lw=.8);ax.axhline(differences.mean(),color='#ae5438',label=f'Mean {differences.mean():.3f}')
    ax.axhline(np.median(differences),color='#25876e',ls='--',label=f'Median {np.median(differences):.3f}')
    ax.set(xticks=np.arange(1,11),xlabel='Independent confirmation seed (fixed order)',ylabel='Restored minus point regret')
    ax.legend(frameon=False,loc='lower left');ax.spines[['top','right']].set_visible(False);savefig(fig,'neural_paired_effects')
    # Correct only the published figure; preserve original numerical source and hashes.
    witness=read(HERE/'results/teaching_information_loss.json');half=next(r for r in witness['joint_sets'] if r['alpha']==.5)
    fig,axes=plt.subplots(1,2,figsize=(8,3.1));tau=np.linspace(1,10,250)
    axes[0].plot(tau,(2*tau-3)/9,color='#ae5438',lw=2);axes[0].axhline(0,color='black',lw=.7);axes[0].axvline(1.5,color='#8b9299',ls=':',lw=1)
    axes[0].set(xlabel=r'First-context minimizer $\tau$',ylabel='Fitted cost minus teacher cost',title='Same targets; opposite cost effects')
    axes[0].text(.06,.88,r'Targets $(1,-1,-1)$; fitted action $-1/3$',transform=axes[0].transAxes,fontsize=9)
    axes[1].hlines(1,half['first_interval'][0],1,color='#315f9b',lw=7);axes[1].hlines(0,-1,half['other_intervals'][1],color='#25876e',lw=7)
    axes[1].axvline(0,color='black',lw=.7);axes[1].axvline(half['best_constant'],color='#ae5438',ls='--',lw=1.5)
    axes[1].set(xlim=(-1.12,1.12),ylim=(-.6,1.65),yticks=[0,1],yticklabels=['Contexts 2, 3','Context 1'],xlabel='Constant student action',title=r'No common feasible fit ($\alpha=1/2$)')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    savefig(fig,'teaching_information_loss_corrected')
    fig,ax=plt.subplots(figsize=(6.8,2.7))
    for noise in [0.,.001]:
        rr=[r for r in approximate['rows'] if r['queries']==1 and r['noise']==noise and r['angle']>0]
        ax.loglog([r['angle'] for r in rr],[max(r['maximum_coefficient_error'],1e-16) for r in rr],'o-',label=f'Observed, noise {noise:g}')
        ax.loglog([r['angle'] for r in rr],[r['coefficient_bound'] for r in rr],'--',label=f'Bound, noise {noise:g}')
    ax.set(xlabel=r'Alignment perturbation $\theta$ (radians)',ylabel='One-query coefficient error');ax.legend(frameon=False,fontsize=8,ncol=2)
    ax.spines[['top','right']].set_visible(False);savefig(fig,'approximate_feedback')
    # Compact complete follow-up means: paired differences retained in machine-readable report.
    lines=[r'\begin{table}[htbp]\centering\small',r'\caption{Additional interface checks: means over five paired seeds. All exact-recovery interfaces agree to the reported precision; full-reference values are shown. These checks reuse four conditions and add a constructed shared-actuation condition.}\label{tab:direct-training}',r'\begin{tabular}{lrrrr}\toprule',r'Condition & Fits & Regret & Cost & Max. $|\Delta\mathcal R|$\\\midrule']
    for d,m,shape,R in [[8,2,'box',.5],[32,2,'box',.5],[32,4,'box',2.],[32,4,'ball',.5],[8,4,'structured',.5]]:
        flag=shape=='structured';actualshape='box' if flag else shape
        subset=[r for r in report['rows'] if (r['structured'],r['dimension'],r['student_dimension'],r['shape'],r['radius'])==(flag,d,m,actualshape,R)]
        refs=[r for r in subset if r['method']=='full'] if flag else [read(HERE/f'results/minimal_teaching_study/s{s}_d{d}_m{m}_{shape}_r{R:g}_full.json') for s in range(13101,13106)]
        fs=[f for f in fidelity if any(r['key']==f['key'] for r in subset)]
        delta=max(abs(f['regret_difference']) for f in fs)
        label='Shared actuation' if flag else f'{shape}, $d={d},m={m},R={R:g}$'
        scientific='0' if delta==0 else f'{delta:.2e}'.split('e')[0]+r'\times10^{'+str(int(f'{delta:.2e}'.split('e')[1]))+'}'
        lines.append(f'{label} & {len(subset)} & {np.mean([r["metrics"]["normalized_regret"] for r in refs]):.6f} & {np.mean([r["metrics"]["test_cost"] for r in refs]):.6f} & ${scientific}$'+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (SOURCE/'current/direct_training_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['training_runs','fidelity_pairs','fidelity_pairs_within_tolerance','maximum_absolute_regret_difference','maximum_absolute_cost_difference','original_box_totals','structured_totals','timings','heterogeneity']},indent=2))

if __name__=='__main__':main()
