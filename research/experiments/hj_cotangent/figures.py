"""Vector manuscript figures from the stated algorithm and complete raw report."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


COLORS=dict(fresh='#087f8c',cache='#d47924',dpc='#75529c',short='#577354',tbptt='#8d8d8d',
            dpc_warm='#b64f5d',cache_promoted='#2566b0')
LABELS=dict(fresh='Fresh teacher',cache='Cached teacher',dpc='DPC',short='Short horizon',
            tbptt='Truncated BPTT',dpc_warm='Warm DPC',cache_promoted='Promoted cache')


def style():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.facecolor':'white'})


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def interface():
    style()
    fig,ax=plt.subplots(figsize=(6.5,3.1))
    ax.set_xlim(0,11);ax.set_ylim(0,4);ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',
                                  linewidth=1.2,edgecolor=color,facecolor=color+'15'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(start,end,text=None,above=.12):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#38444a',lw=1.4))
        if text:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+above,text,ha='center',fontsize=8)
    box(.15,2.3,2.35,.95,'Student prefix\n'+r'$C_\theta,\;Z_\theta$',COLORS['fresh'])
    box(4.0,2.3,2.8,.95,'Evaluate fixed\nfeedback teacher\n'+r'$V^b(a),\;\nabla V^b(a)$',COLORS['dpc'])
    box(8.15,2.3,2.55,.95,'Store local\nknowledge\n'+r'$(a,v,p)$',COLORS['cache'])
    arrow((2.6,2.78),(3.88,2.78))
    ax.text(3.24,3.05,r'$a=Z_\theta$',ha='center',fontsize=8)
    arrow((6.9,2.78),(8.02,2.78))
    box(2.55,.45,5.9,1.08,'Many short student updates\n'+
        r'$C_\theta+v+p^\top(Z_\theta-a)+\frac{\rho}{2d}\|Z_\theta-a\|^2$',COLORS['cache'])
    arrow((9.43,2.2),(8.5,1.02))
    ax.text(9.5,1.35,'reuse',fontsize=9)
    ax.annotate('',xy=(2.05,2.2),xytext=(2.45,1.02),
        arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=180,angleB=90,rad=10',color='#38444a',lw=1.4))
    ax.text(.18,1.02,'refresh at\nnew student\nendpoints',fontsize=8.3)
    ax.text(5.5,3.65,'Teacher knowledge is a local future-cost sensitivity',ha='center',fontsize=10)
    ax.text(5.5,-.08,'Terminal costs stay exact. Refresh changes the query location;\npromotion changes the teacher.',
            ha='center',fontsize=8.5)
    save(fig,Path('paper/figures/hj_cotangent_interface'))


def render_summary(root,summary):
    style()
    target=root/'report'
    cases=summary['cases']
    fig,axes=plt.subplots(3,1,figsize=(6.5,7.5))
    for ax,family,h in zip(axes,['mechanical','reaction','building'],[160,320,192]):
        contrasts=[('cache_vs_fresh','cache','Cached vs. Fresh'),
                   ('fresh_vs_dpc','fresh','Fresh vs. DPC')]
        if family=='mechanical':
            contrasts.append(('fresh_vs_dpc_warm','dpc_warm','Fresh vs. Warm DPC'))
        for contrast,name,label in contrasts:
            rows=[cases[f'{family}_h{h}_d{d}_nominal']['comparisons'][contrast] for d in [32,128,256]]
            values=np.array([r['reduction_percent'] for r in rows])
            bounds=np.array([r['ci95_percent'] for r in rows]).T
            # Percentile bootstrap bounds need not contain the point estimate.
            ax.plot([32,128,256],values,'o-',color=COLORS[name],label=label)
            ax.vlines([32,128,256],bounds[0],bounds[1],color=COLORS[name],lw=1)
        ax.axhline(0,color='#777777',lw=.7)
        ax.set_title(f'{family.capitalize()}, N={h}')
        ax.set_xlabel('State dimension')
        ax.set_ylabel('Mean cost reduction (%)')
        ax.set_xticks([32,128,256]);ax.grid(alpha=.15)
        if family=='mechanical':ax.set_yscale('symlog',linthresh=10)
    axes[0].legend(fontsize=9)
    fig.tight_layout();save(fig,target/'dimension_transfer')

    fig,axes=plt.subplots(3,1,figsize=(6.5,7.5))
    for ax,family,h in zip(axes,['mechanical','reaction','building'],[160,320,192]):
        train=summary['training'][f'{family}_h{h}']
        groups=[('cache','cache')]+([('cache_promoted','cache_promoted')] if 'cache_promoted' in train else [])
        for method,color in groups:
            for pool,marker in [('teacher_states','o'),('broad_states','s')]:
                for index,rows in enumerate(train[method]['jet_diagnostics']):
                    selected=[r for r in rows if r['pool']==pool and r['mean_gradient_cosine'] is not None]
                    ax.plot([r['step'] for r in selected],[r['mean_gradient_cosine'] for r in selected],
                            marker+'-',color=COLORS[color],alpha=.45,markersize=3,
                            label=LABELS[method]+', '+pool.replace('_states','') if index==0 else None)
        ax.axhline(0,color='#777777',lw=.7);ax.set_ylim(-1.05,1.05)
        ax.set_title(f'{family.capitalize()}, N={h}')
        ax.set_xlabel('Attempted student updates')
        ax.set_ylabel('Old/new value-gradient cosine')
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=9,ncol=2)
    fig.tight_layout();save(fig,target/'cache_drift')

    for filename,headers,rows in [
        ('training_seeds.csv',['task','method','seed','validation_cost','seconds','best_step','skipped_updates'],
         [[task,name,seed,history[-1]['best'],seconds,best,skips]
          for task,methods in summary['training'].items() for name,record in methods.items()
          for seed,history,seconds,best,skips in zip(summary['configuration']['seeds'],record['histories'],
              record['seed_seconds'],record['best_steps'],record['nonfinite_updates_skipped'])]),
        ('comparisons.csv',['case','contrast','reduction_percent','ci95_low','ci95_high'],
         [[key,name,row['reduction_percent'],*row['ci95_percent']] for key,case in cases.items()
          for name,row in case['comparisons'].items()]),
        ('seed_costs.csv',['case','method','training_seed','test_states','mean_test_cost','teacher_mean_cost'],
         [[key,name,seed,case['count'],value,case['teacher_mean']] for key,case in cases.items()
          for name,values in case['seed_means'].items()
          for seed,value in zip(summary['configuration']['seeds'],values)]),
        ('classical_quality.csv',['case','method','states','method_mean','classical_mean','reduction_percent','ci95_low','ci95_high'],
         [[key,name,case['classical']['count'],case['classical']['method_means'][name],case['classical']['mean'],
           row['reduction_percent'],*row['ci95_percent']] for key,case in cases.items() if 'classical' in case
          for name,row in case['classical']['comparisons'].items()]),
        ('combined_classical_quality.csv',['case','method','states','method_mean','classical_mean','reduction_percent','ci95_low','ci95_high'],
         [[key,name,case['combined_classical']['count'],case['combined_classical']['method_means'][name],case['combined_classical']['mean'],
           row['reduction_percent'],*row['ci95_percent']] for key,case in cases.items() if 'combined_classical' in case
          for name,row in case['combined_classical']['comparisons'].items()]),
        ('cache_diagnostics.csv',['task','method','seed','step','pool','endpoint_rms','gradient_cosine','value_error','secant_curvature'],
         [[task,name,seed,row['step'],row['pool'],row['mean_boundary_rms'],row['mean_gradient_cosine'],
           row['mean_abs_value_error'],row['mean_secant_curvature']] for task,methods in summary['training'].items()
          for name,record in methods.items() for seed,diagnostics in zip(summary['configuration']['seeds'],record['jet_diagnostics'])
          for row in diagnostics])]:
        with (target/filename).open('w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file);writer.writerow(headers);writer.writerows(rows)


def temporal():
    style()
    fig,ax=plt.subplots(figsize=(6.5,3.6))
    ax.set_xlim(-.29,1.19);ax.set_ylim(-.5,5.25);ax.axis('off')
    cut=.23
    rows=[('DPC',4.35),('Fresh teacher',3.35),('Cached teacher',2.35),('Short',1.35),('TBPTT',.35)]
    def segment(left,right,y,color,alpha=1):
        ax.plot([left,right],[y,y],color=color,lw=11,alpha=alpha,solid_capstyle='butt')
    for name,y in rows:
        ax.text(-.035,y,name,ha='right',va='center',fontsize=9)
        ax.plot([0,1],[y,y],color='#dddddd',lw=.8,zorder=0)
    segment(0,1,4.35,COLORS['fresh'])
    ax.text(.5,4.61,'Full student rollout and derivative',ha='center',fontsize=8.5)
    segment(0,cut,3.35,COLORS['fresh']);segment(cut,1,3.35,COLORS['dpc'])
    ax.text(.62,3.61,'Evaluate and differentiate fixed feedback teacher',ha='center',fontsize=8.5)
    segment(0,cut,2.35,COLORS['fresh'])
    ax.plot(cut,2.35,'D',color=COLORS['cache'],ms=9)
    ax.text(cut+.04,2.35,r'Reuse $(a,v,p)$; teacher queried at refresh',va='center',fontsize=8.5)
    segment(0,cut,1.35,COLORS['fresh'])
    ax.plot(cut,1.35,'s',color=COLORS['short'],ms=9)
    ax.text(cut+.04,1.35,r'Apply original $g(x)$ at the early endpoint',va='center',fontsize=8.5)
    for left in np.arange(0,1,cut):
        segment(left,min(left+cut-.008,1),.35,COLORS['fresh'])
        if left>0:ax.plot([left,left], [.19,.51],color='white',lw=2)
    ax.text(.5,.64,'Full forward return; state derivative stops every 16 steps',ha='center',fontsize=8.5)
    for name,y in [rows[0],rows[1],rows[-1]]:
        ax.text(1.035,y,r'$g(x_N)$',va='center',fontsize=9)
    for pos,label in [(0,r'$k$'),(cut,r'$k+16$'),(1,r'$N$')]:
        ax.plot([pos,pos],[-.01,4.65],ls=':',lw=.7,color='#999999',zorder=0)
        ax.text(pos,4.87,label,ha='center',fontsize=10)
    ax.text(.5,-.28,'Both teacher and student respond to their evolving simulated states.',
            ha='center',fontsize=8.5)
    save(fig,Path('paper/figures/hj_cotangent_temporal'))


def hjb_map():
    style()
    fig,ax=plt.subplots(figsize=(6.5,3.2))
    ax.set_xlim(0,12);ax.set_ylim(0,3.6);ax.axis('off')
    def box(x,y,w,h,label,color,size=9):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',
            lw=1.1,edgecolor=color,facecolor=color+'12'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)
    box(.2,2.15,2.7,.7,'Teacher policy\n'+r'$b(t,x)$',COLORS['dpc'])
    box(4.45,2.15,3.1,.7,'Teacher value field\n'+r'$V^b(t,x),\;p^b=\nabla V^b$',COLORS['fresh'])
    box(9.1,2.15,2.7,.7,'Student policy\n'+r'$\pi(t,x)$',COLORS['cache'])
    for left,right,label in [(3.0,4.3,'evaluate'),(7.7,8.95,'improve')]:
        ax.annotate('',xy=(right,2.5),xytext=(left,2.5),arrowprops=dict(arrowstyle='->',lw=1))
        ax.text((left+right)/2,2.66,label,ha='center',fontsize=8.5)
    ax.plot([10.45,10.45,1.55],[2.96,3.19,3.19],ls='--',color='#555555',lw=.9)
    ax.annotate('',xy=(1.55,2.96),xytext=(1.55,3.19),arrowprops=dict(arrowstyle='->',lw=.9,color='#555555'))
    ax.text(6,3.35,'Promotion requires a new policy evaluation',ha='center',fontsize=9)
    ax.text(6,1.86,'A short student prefix uses the teacher continuation as its boundary value.',ha='center',fontsize=8.5)
    box(.2,.85,11.6,.72,'Performance along the student trajectory\n'+
        r'$J^\pi-J^b=\int (r_b^\pi-\delta_b)\,dt$',COLORS['fresh'],10)
    ax.text(3.1,.45,r'Teacher: $\mathcal{F}[V^b]=-\delta_b$',ha='center',fontsize=9)
    ax.text(9,.45,r'Optimal field: $\mathcal{F}[V^*]=0$',ha='center',fontsize=9)
    ax.text(6,.09,r'$r_b^\pi$: student regret in the teacher field; $\delta_b$: teacher improvement opportunity.',
        ha='center',fontsize=8.5)
    save(fig,Path('paper/figures/hj_cotangent_hjb_map'))


def solvable_example():
    style()
    a=np.linspace(0,3,151)
    occupation=np.ones_like(a)
    np.divide(-np.expm1(-2*a),2*a,out=occupation,where=a!=0)
    cost=.5*(1+a*a)*occupation+.625*np.exp(-2*a)
    regret=.5*(a-1.25)**2*occupation
    opportunity=.5*.75**2*occupation
    assert np.max(np.abs(cost-.625-regret+opportunity))<2e-15
    fig,axes=plt.subplots(2,1,figsize=(6.5,5.8))
    ax=axes[0]
    ax.plot(a,cost,color=COLORS['fresh'],lw=1.8)
    for gain,label,position in [(2,'Teacher:\naction imitation',(1.9,.94)),
                               (1.25,'Hamiltonian\nimprovement',(.63,.75))]:
        ii=(1-np.exp(-2*gain))/(2*gain)
        j=.5*(1+gain*gain)*ii+.625*np.exp(-2*gain)
        ax.plot(gain,j,'o',color=COLORS['cache'],ms=5)
        ax.annotate(label,(gain,j),xytext=position,fontsize=9,
            arrowprops=dict(arrowstyle='-',lw=.7,color='#555555'))
    ax.set_ylabel('Closed-loop cost');ax.set_xlabel(r'Student gain $a$ in $\pi_a(x)=-ax$')
    ax.set_title('Copying teacher actions and reducing cost have different targets',fontsize=10)
    ax.grid(alpha=.17)
    ax=axes[1]
    ax.plot(a,regret,color=COLORS['cache'],label=r'Integrated student regret $r_b^\pi$')
    ax.plot(a,opportunity,color=COLORS['dpc'],label=r'Integrated teacher opportunity $\delta_b$')
    ax.plot(a,regret-opportunity,color=COLORS['fresh'],label=r'Integrated $r_b^\pi-\delta_b$')
    ax.plot(a[::10],(cost-.625)[::10],'x',color='#222222',ms=4,label='Independent cost difference')
    ax.axhline(0,color='#888888',lw=.7)
    ax.set_ylabel('Integrated terms / cost difference');ax.set_xlabel(r'Student gain $a$')
    ax.legend(fontsize=9);ax.grid(alpha=.17)
    fig.tight_layout();save(fig,Path('paper/figures/hj_cotangent_solvable'))


def teacher_curvature():
    style()
    source=Path('experiments/results/hj_cotangent_dev_v6c/teacher_curvature_horizon.json')
    records=json.loads(source.read_text())['records']
    fig,ax=plt.subplots(figsize=(6.5,3.6))
    for gain,color in [(.2,COLORS['cache']),(.9,COLORS['fresh']),(3.,COLORS['dpc'])]:
        rows=sorted([r for r in records if r['dimension']==32 and r['teacher_gain']==gain],key=lambda r:r['horizon'])
        label=rf'$\kappa={gain:g},\ \|DG\|={rows[0]["feedback_operator_norm"]:.3f}$'
        ax.plot([r['horizon'] for r in rows],[r['mean_squared_penalty_curvature'] for r in rows],
            'o-',color=color,label=label,ms=4)
    ax.set_yscale('log');ax.set_xticks([16,40,80,160,320])
    ax.set_xlabel('Remaining teacher steps');ax.set_ylabel(r'Value curvature $dM_0$')
    ax.legend(fontsize=9);ax.grid(alpha=.18)
    fig.tight_layout();save(fig,Path('paper/figures/hj_cotangent_curvature'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root');args=parser.parse_args()
    interface()
    temporal()
    hjb_map()
    solvable_example()
    teacher_curvature()
    if args.root:
        root=Path(args.root);render_summary(root,json.loads((root/'report/summary.json').read_text()))


if __name__=='__main__':main()
