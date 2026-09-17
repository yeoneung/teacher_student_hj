"""Recompute costs, verify frozen evidence, and render set-supervision figures."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from impact_core import HERE,SUB,write,digest
from improving_sets import intersection_project,cache_geometry,upper_model
from experiments.hj_cotangent.systems import problem


def figure_geometry():
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(8,3.3),gridspec_kw={'width_ratios':[1,1.3]})
    a=ax[0];theta=np.linspace(0,2*np.pi,400)
    a.plot(3*np.cos(theta),3*np.sin(theta),color='#85929e',lw=1,label='Feasible actions')
    a.text(-1.35,-2.5,r'Feasible actions $U$',fontsize=9,color='#657184')
    m=np.array([1.3,.5]);rad=np.linalg.norm(m)/np.sqrt(2)
    a.add_patch(Circle(m,rad,color='#38a080',alpha=.22))
    a.plot(m[0]+rad*np.cos(theta),m[1]+rad*np.sin(theta),color='#23866d',lw=1.5)
    source=np.array([-1.7,1.4]);pr=m+rad*(source-m)/np.linalg.norm(source-m)
    for pos,label,color,offset in [(np.zeros(2),'Teacher action','#657184',(-55,-18)),(m,'Point anchor','#bf6c37',(8,-12)),(source,'Student action','#315f9b',(-35,13)),(pr,'Closest acceptable action','#23866d',(-5,35))]:
        a.scatter(*pos,color=color,s=32,zorder=5);a.annotate(label,pos,xytext=offset,textcoords='offset points',fontsize=9)
    a.annotate('',pr,source,arrowprops={'arrowstyle':'->','color':'#315f9b','lw':1.5})
    a.text(-.4,2.4,r'$C_{1/2}(z)$',color='#23866d');a.set(xlim=(-3.4,3.4),ylim=(-3.3,3.4),aspect='equal',title='Teacher defines sufficient improvement')
    a.set_axis_off()
    b=ax[1];labels=['Teacher\nconstant 0','Set-feasible\nconstant 1','Point fit\nconstant 1.5']
    b.bar(labels,[2.5,.5,.25],color=['#85929e','#23866d','#bf6c37'],width=.6)
    for i,y in enumerate([2.5,.5,.25]):b.text(i,y+.04,f'{y:g}',ha='center')
    b.set(ylabel='Mean one-step cost (lower is better)',ylim=(0,2.95),title='Fitting freedom has a performance price')
    b.text(.5,.94,'Zero set loss does not imply the lowest cost',ha='center',transform=b.transAxes,fontsize=9)
    b.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(SUB/f'source/figures/improving_set_geometry.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


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
    write(HERE/'results/improving_sets_analysis.json',result);figure_geometry()
    # Paired seed differences, not individual trajectories as independent replicates.
    fig,ax=plt.subplots(figsize=(6.4,2.8))
    for i,method in enumerate(['point','adaptive','upper','teacher']):
        c=comparisons[method];diff=np.array(c['paired_differences']);ax.scatter(diff,np.full(5,i)+np.linspace(-.12,.12,5),s=20,color='#8198ad')
        ax.errorbar(c['mean_difference'],i,xerr=[[c['mean_difference']-c['ci95'][0]],[c['ci95'][1]-c['mean_difference']]],fmt='o',color='#20587d',capsize=4)
    ax.axvline(0,color='black',lw=.8);ax.set(yticks=range(4),yticklabels=['Point anchor','Adaptive point','Upper model','Teacher'],xlabel='Set student cost minus comparator cost',title='Reaction: five independent paired replications')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(SUB/f'source/figures/improving_set_replication.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(means=means,comparisons=comparisons,max_float_relative_error=result['projection_max_relative_error']),indent=2))


if __name__=='__main__':main()
