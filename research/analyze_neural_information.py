"""Verify all frozen neural mechanism runs; statistics and exact decomposition.

No fitting, checkpoint selection or alteration of numerical source/results.
Seed means are units. Confirmation is reported separately from initial study.
"""
import csv,itertools,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,result,title in zip(axes[0],results,['Initial study (5 seeds)','Independent confirmation (10 seeds)']):
        radii=result['protocol']['control_radii']
        for k,w in enumerate([2,4,16]):
            cc=[next(c for c in result['cells'] if c['radius']==r and c['width']==w) for r in radii]
            pp=[c['restored_minus']['point']['normalized_regret'] for c in cc];yy=np.array([p['mean'] for p in pp]);ee=np.array([(p['ci95'][1]-p['ci95'][0])/2 for p in pp])
            ax.errorbar(np.arange(len(radii))+(k-1)*.13,yy,yerr=ee,marker='o',capsize=3,label=f'Width {w}')
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(radii)),[str(r) for r in radii]);ax.set_xlabel('Control radius (smaller = tighter)');ax.set_ylabel('Restored - point normalized regret');ax.set_title(title);ax.legend(fontsize=8)
    cell=next(c for c in results[1]['cells'] if c['radius']==.5 and c['width']==2)
    methods=base.METHODS;x=np.arange(4);means=cell['means'];ax=axes[1,0]
    first=np.array([base.CURV*base.D*means[m]['target_mse'] for m in methods]);second=np.array([means[m]['normal_contribution'] for m in methods])
    ax.bar(x,first,label='Quadratic target error');ax.bar(x,second,bottom=first,label='Normal contribution');ax.set_xticks(x,['Point','Restored','Weight','Shuffled']);ax.set_ylabel('Exact local Bellman regret');ax.set_title('Confirmation: radius 0.5, width 2');ax.legend(fontsize=8)
    ax=axes[1,1]
    for k,w in enumerate([2,4,16]):
        cc=next(c for c in results[1]['cells'] if c['radius']==.5 and c['width']==w)
        ax.bar(x+(k-1)*.24,[cc['means'][m]['normalized_regret'] for m in methods],width=.24,label=f'Width {w}')
    ax.set_xticks(x,['Point','Restored','Weight','Shuffled']);ax.set_ylabel('Normalized local Bellman regret');ax.set_title('Capacity control at radius 0.5');ax.legend(fontsize=8)
    for ext in ['pdf','png']:fig.savefig(SUB/f'source/figures/neural_information_transfer.{ext}',dpi=180)
    plt.close(fig)
    lines=[r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\caption{Controlled neural information transfer. Mean normalized local Bellman regret; lower is better. $\Delta$ is restored minus point with an unadjusted paired 95\% $t$ interval across seeds. The initial primary is $R=2$, width 2; the independent confirmation primary is $R=0.5$, width 2. All radii and widths are shown, including adverse comparisons.}',r'\label{tab:neural-information}',r'\begin{tabular}{rrrrrrl}',r'\toprule',r'$R$ & Width & Point & Restored & Weight & Shuffled & $\Delta$ [95\% CI]\\',r'\midrule']
    for result in results:
        label='Initial study: 5 seeds, 240 runs' if result['stage']==STAGES[0] else 'Independent confirmation: 10 seeds, 360 runs'
        lines.append(r'\multicolumn{7}{l}{\textit{'+label+r'}}\\')
        for c in result['cells']:
            mm=c['means'];p=c['restored_minus']['point']['normalized_regret'];l,h=p['ci95']
            lines.append(f"{c['radius']:g} & {c['width']} & "+' & '.join(f"{mm[m]['normalized_regret']:.4f}" for m in methods)+f" & {p['mean']:+.4f} [{l:+.4f}, {h:+.4f}]"+r'\\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.extend([r'\end{tabular}',r'\end{table}'])
    (SUB/'source/current/neural_information_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({r['stage']:dict(primary=r['primary'],supported=r['primary_benefit_supported'],audit=r['audit']) for r in results},indent=2))

if __name__=='__main__':main()
