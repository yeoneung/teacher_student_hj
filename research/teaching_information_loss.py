"""Exact witnesses of information loss and incompatible improvement sets.

Analytical constructions, not neural experiments or explanations inferred from
the nonlinear benchmark failures. All stated minima are checked independently
with bounded scalar optimization; the closed forms supply their proofs.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    records=[];joint=[]
    for a in [1.,1.5,2.,4.,10.]:
        coefficients=np.array([a,-1.,-1.]);targets=np.clip(coefficients,-1,1)
        cost=lambda u:float(np.mean((u-coefficients)**2))
        mse=lambda u:float(np.mean((u-targets)**2))
        fit=minimize_scalar(mse,bounds=(-1,1),method='bounded',options={'xatol':1e-14})
        direct=minimize_scalar(cost,bounds=(-1,1),method='bounded',options={'xatol':1e-14})
        c=-1/3;opt=float(np.clip((a-2)/3,-1,1));delta=(2*a-3)/9
        assert np.allclose(targets,[1,-1,-1])
        assert abs(fit.x-c)<1e-7 and abs(direct.x-opt)<1e-7
        assert abs(cost(c)-cost(0)-delta)<1e-12
        assert abs(mse(c)-8/9)<1e-12 and mse(0)==1
        assert np.all((targets-coefficients)**2-coefficients**2<0)
        # Exact Bellman realization: F(s,y,u)=(s,u), c=u^2,
        # g(s,y)=a(s)^2-2*a(s)*y, where a(s) interpolates the coefficients.
        s=np.array([1.,2.,3.]);coef=-1+(a+1)*(s-2)*(s-3)/2
        assert np.allclose(coef,coefficients)
        assert np.allclose(c*c+coef**2-2*coef*c,(c-coef)**2)
        # q(u)-q(target) = squared distance + normal-cone contribution.
        normal=2*(targets-coefficients)
        for u in np.linspace(-1,1,51):
            lhs=(u-coefficients)**2-(targets-coefficients)**2
            assert np.allclose(lhs,(u-targets)**2+normal*(u-targets),atol=1e-12)
        records.append(dict(a=a,targets=targets.tolist(),teacher_cost=cost(0),point_fit=c,
            teacher_mse=mse(0),point_fit_mse=mse(c),point_fit_cost=cost(c),cost_difference=delta,
            direct_constant=opt,direct_cost=cost(opt),normal_gradient=normal.tolist()))
    regularized=[]
    for rho in [0.,2.,32.,512.]:
        beta=1+rho/2;coef=beta*np.array([10.,-1.,-1.])
        target=np.clip(coef/beta,-1,1);fit=float(target.mean())
        diff=float(np.mean((fit-coef)**2-coef**2));formula=(1+16*beta)/9
        assert np.allclose(target,[1,-1,-1]) and abs(diff-formula)<1e-7 and diff>0
        regularized.append(dict(rho=rho,beta=beta,targets=target.tolist(),fitted_cost_difference=diff,formula=formula))
    a=10.;coefficients=np.array([a,-1.,-1.]);cost=lambda u:float(np.mean((u-coefficients)**2))
    for alpha in [.001,.01,.1,.25,.5,.75,1.]:
        # Rationalized root avoids cancellation for very small alpha.
        lower=alpha*(2*a-1)/(a+np.sqrt(a*a-alpha*(2*a-1)))
        upper=-alpha/(1+np.sqrt(1-alpha))
        assert lower>0 and upper<0
        intervals=np.array([[lower,1.],[-1.,upper],[-1.,upper]])
        loss=lambda c:float(np.mean((c-np.clip(c,intervals[:,0],intervals[:,1]))**2))
        expected=(lower+2*upper)/3
        fitted=minimize_scalar(loss,bounds=(-1,1),method='bounded',options={'xatol':1e-14})
        assert abs(fitted.x-expected)<1e-7
        assert loss(expected)>0
        joint.append(dict(alpha=alpha,first_interval=intervals[0].tolist(),other_intervals=intervals[1].tolist(),
            intersection_empty=True,best_constant=expected,minimum_set_loss=loss(expected),
            teacher_set_loss=loss(0),cost=cost(expected),teacher_cost=cost(0),cost_difference=cost(expected)-cost(0)))
    half=next(r for r in joint if r['alpha']==.5)
    assert half['cost_difference']>0 and half['minimum_set_loss']<half['teacher_set_loss']
    report=dict(passed=True,point_family=records,joint_sets=joint,regularized_witnesses=regularized,scope='Exact constructed witnesses; no neural training and no attribution of nonlinear failure causes.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'results/teaching_information_loss.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8,3.1))
    aa=np.linspace(1,10,250)
    axes[0].plot(aa,(2*aa-3)/9,color='#ae5438',lw=2)
    axes[0].axhline(0,color='black',lw=.7);axes[0].axvline(1.5,color='#8b9299',ls=':',lw=1)
    axes[0].set(xlabel=r'Quadratic coefficient $a$',ylabel='Fitted cost minus teacher cost',title='Same action targets; opposite cost effects')
    axes[0].text(.06,.88,r'Targets $(1,-1,-1)$; fitted action $-1/3$',transform=axes[0].transAxes,fontsize=9)
    axes[1].hlines(1,half['first_interval'][0],1,color='#315f9b',lw=7)
    axes[1].hlines(0,-1,half['other_intervals'][1],color='#25876e',lw=7)
    axes[1].axvline(0,color='black',lw=.7)
    axes[1].axvline(half['best_constant'],color='#ae5438',ls='--',lw=1.5)
    axes[1].set(xlim=(-1.12,1.12),ylim=(-.6,1.65),yticks=[0,1],yticklabels=['Contexts 2, 3','Context 1'],xlabel='Constant student action',title=r'No shared action meets all sets ($\alpha=1/2$)')
    axes[1].text(.04,.92,'Each set is nonempty; their intersection is empty',transform=axes[1].transAxes,fontsize=8,bbox={'facecolor':'white','edgecolor':'none','pad':1})
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(HERE.parent/f'source/figures/teaching_information_loss.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(a10=records[-1],alpha_half=half),indent=2))


if __name__=='__main__':main()
