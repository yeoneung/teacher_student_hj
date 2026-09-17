"""Publication figures for the versioned teacher-transport manuscript."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch


def interface():
    out=Path('paper/figures');out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.4,3.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    boxes=[(.015,.49,.275,.43,'#eaf0f8','#284b78','TEACHER EVALUATION',
            'Frozen feedback policy $\\phi$\nFull trajectories from $x_0$\nOne reverse sweep'),
           (.365,.49,.27,.43,'#e8f4ef','#256b55','TEACHING DATA',
            'Teacher states and actions\nValue covectors $p_{t+1}$\nOptional anchor gradient $G_\\phi$'),
           (.71,.49,.275,.43,'#fff1e5','#9a571e','STUDENT LEARNING',
            'Student policy $\\pi_\\theta$\nOne-step advantage updates\nReuse across minibatches')]
    for x,y,w,h,bg,fg,title,body in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006',facecolor=bg,edgecolor=fg,linewidth=1.1))
        ax.text(x+w/2,y+h-.07,title,ha='center',va='center',color=fg,fontsize=9,fontweight='bold')
        ax.text(x+w/2,y+h/2-.03,body,ha='center',va='center',linespacing=1.65,fontsize=9)
    for a,b in [((.293,.705),(.359,.705)),((.641,.705),(.704,.705))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#505050',linewidth=1.4))
    ax.add_patch(FancyBboxPatch((.15,.06),.70,.23,boxstyle='round,pad=.012',facecolor='#f5f5f5',edgecolor='#626262',linewidth=1.1))
    ax.text(.5,.215,'REEVALUATE THE PROPOSED TEACHER',ha='center',va='center',fontsize=9,fontweight='bold')
    ax.text(.5,.12,'Accept by actual training cost, or restore the incumbent',ha='center',va='center',fontsize=9)
    ax.add_patch(FancyArrowPatch((.847,.48),(.78,.30),arrowstyle='-|>',mutation_scale=14,color='#505050',linewidth=1.4))
    ax.add_patch(FancyArrowPatch((.22,.30),(.153,.48),arrowstyle='-|>',mutation_scale=14,color='#505050',linewidth=1.4))
    ax.text(.10,.365,'promote',fontsize=8,ha='center',va='center',rotation=64,color='#505050')
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
    fig.savefig(out/'hj_v7_interface.pdf');fig.savefig(out/'hj_v7_interface.png',dpi=180);plt.close(fig)


def examples():
    out=Path('paper/figures');plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.5))
    t=np.linspace(-1.18,1.,400);j=1-2*t+1.5*t*t;f=1+2*t+1.1*t*t
    ax=axes[0];ax.plot(t,j,color='#284b78',label='Full student cost $J$')
    ax.plot(t,f,color='#a86122',linestyle='--',label='Exact prefix cost $F$')
    ax.scatter([0],[1],color='#444444',s=24,zorder=5)
    ax.annotate("$J'(0)=-2$",xy=(.10,.8),xytext=(.18,2.9),arrowprops={'arrowstyle':'->','color':'#284b78'},color='#284b78')
    ax.annotate("$F'(0)=+2$",xy=(-.12,.78),xytext=(-1.08,2.2),arrowprops={'arrowstyle':'->','color':'#a86122'},color='#a86122')
    ax.set(title='Prefix gradient reversal ($a=-2$)',xlabel='Policy parameter $\\theta$',ylabel='Cost',ylim=(0,5.7));ax.legend(loc='upper right',frameon=False,fontsize=8)
    t=np.linspace(-1.85,.20,400);j=1+4*t+4.2*t*t;s=1+4*t+1.2*t*t
    ax=axes[1];ax.plot(t,j,color='#284b78',label='Full student cost $J$')
    ax.plot(t,s,color='#2a8063',linestyle='--',label='All-time surrogate $S_0$')
    ax.scatter([0,-5/3],[1,6],color='#444444',s=24,zorder=5)
    ax.plot([-5/3,-5/3],[-7/3,6],color='#777777',linestyle=':',linewidth=1)
    ax.annotate('Surrogate minimizer:\ntrue cost rises from 1 to 6',xy=(-5/3,6),xytext=(-1.12,4.3),ha='left',fontsize=8,arrowprops={'arrowstyle':'->','color':'#444444'})
    ax.set(title='Reuse error after matching ($a=1$)',xlabel='Policy parameter $\\theta$',ylim=(-2.8,8.5));ax.legend(loc='lower right',frameon=False,fontsize=8)
    for ax in axes:ax.axvline(0,color='#bbbbbb',linewidth=.7);ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(w_pad=2);fig.savefig(out/'hj_v7_analytic_examples.pdf');fig.savefig(out/'hj_v7_analytic_examples.png',dpi=180);plt.close(fig)


def outcomes(root):
    summary=json.loads((root/'report/summary.json').read_text())
    cfg=json.loads((root/'config.json').read_text())
    out=Path('paper/figures');plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.4,3.3))
    labels={'mechanical':'Mechanical','reaction':'Reaction','building':'Building'}
    refs={'short':'Short','dpc':'DPC','warm':'Warm DPC','dpc_full':'Full DPC','warm_full':'Warm Full DPC',
          'dpc_initial':'Initial DPC','warm_initial':'Initial Warm DPC'}
    for ax,task in zip(axes,cfg['tasks']):
        family=task['family'];ref=task['principal_reference']
        rows=sorted([r for r in summary['comparisons'] if r['family']==family and r['dimension']==32 and
            r['condition']=='nominal' and r['reference']==ref],key=lambda r:r['budget'])
        budgets=np.array([r['budget'] for r in rows]);ratios=np.array([r['pooled_ratio'] for r in rows])
        limits=np.array([r['descriptive_ratio_interval'] for r in rows])
        ax.axhline(1,color='#666666',linewidth=.8);ax.axhline(1.01,color='#a15a2d',linewidth=.8,linestyle=':')
        ax.plot(budgets,ratios,'o-',color='#284b78',markersize=4,linewidth=1.3)
        ax.vlines(budgets,limits[:,0],limits[:,1],color='#284b78',linewidth=1)
        ax.plot(budgets,limits[:,0],linestyle='none',marker='_',color='#284b78',markersize=5)
        ax.plot(budgets,limits[:,1],linestyle='none',marker='_',color='#284b78',markersize=5)
        ax.set(title=labels[family]+'\nReference: '+refs[ref],xlabel='Total allocation (s)',xticks=[15,30,60],xlim=(10,65))
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Teacher transport / reference cost')
    fig.tight_layout(w_pad=1.5);fig.savefig(out/'hj_v7_budget_ratios.pdf');fig.savefig(out/'hj_v7_budget_ratios.png',dpi=180);plt.close(fig)
    evidence=json.loads((root/'report/writeup_evidence.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.3));xx=np.arange(3);width=.33
    for offset,method,color,label in [(-width/2,'fixed','#7895b4','Fixed transport'),(width/2,'adaptive','#27654f','Selected transport')]:
        rows=[next(r for r in evidence['work'] if r['family']==t['family'] and r['method']==method) for t in cfg['tasks']]
        axes[0].bar(xx+offset,[r['mean_query_seconds'] for r in rows],width,color=color,label=label)
        axes[1].bar(xx+offset,[100*r['accepted_fraction'] for r in rows],width,color=color,label=label)
    for ax in axes:
        ax.set_xticks(xx,[labels[t['family']] for t in cfg['tasks']]);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Teacher evaluation / probe seconds');axes[0].legend(frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2)
    axes[1].set_ylabel('Accepted teacher sweeps (%)');axes[1].set_ylim(0,105)
    fig.tight_layout(w_pad=2);fig.savefig(out/'hj_v7_reuse_work.pdf');fig.savefig(out/'hj_v7_reuse_work.png',dpi=180);plt.close(fig)
    print('Generated audited budget and reuse figures')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outcomes',action='store_true')
    parser.add_argument('--root',default='experiments/results/hj_transport_primary_v7');args=parser.parse_args()
    interface();examples()
    if args.outcomes:outcomes(Path(args.root))
