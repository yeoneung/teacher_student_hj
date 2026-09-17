"""Independent geometry checks and exact finite-horizon policy accounting."""
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from improving_sets import ball_project, intersection_project, geometry, upper_model
from impact_core import write,digest,HERE


def projection_checks():
    rng=np.random.default_rng(361042);rows=[]
    for dim in (2,3,8,32):
        for case in range(30):
            # Explicit common feasible point prevents infeasible numerical tests.
            c=rng.normal(size=dim);rad_u=float(rng.uniform(.3,2))
            witness=rng.normal(size=dim);witness*=rad_u*rng.uniform(0,1)/np.linalg.norm(witness)
            rad_b=float(np.linalg.norm(witness-c)+rng.uniform(.001,.5))
            s=4*rng.normal(size=dim)
            result=intersection_project(torch.tensor(s[None]),torch.tensor(c[None]),torch.tensor([[rad_b]],dtype=torch.float64),rad_u)[0].numpy()
            cons=[{'type':'ineq','fun':lambda x:rad_u**2-x@x,'jac':lambda x:-2*x},
                  {'type':'ineq','fun':lambda x:rad_b**2-(x-c)@(x-c),'jac':lambda x:-2*(x-c)}]
            scale=1+np.sum(s*s)
            ref=minimize(lambda x:.5*np.sum((x-s)**2)/scale,witness,jac=lambda x:(x-s)/scale,
                         constraints=cons,method='SLSQP',options={'ftol':1e-11,'maxiter':1000})
            err=np.linalg.norm(result-ref.x)
            violation=max(np.linalg.norm(result)-rad_u,np.linalg.norm(result-c)-rad_b,0.)
            objerr=abs(.5*np.sum((result-s)**2)-ref.fun*scale)
            assert violation<1e-10 and err<2e-5 and objerr<2e-7,(dim,case,err,violation,ref.message)
            # Independent finite difference of squared distance vs 2(s-P_C s).
            numerical=[];h=1e-5
            for j in range(dim):
                e=np.eye(dim)[j]*h
                vals=[]
                for q in (s+e,s-e):
                    pr=intersection_project(torch.tensor(q[None]),torch.tensor(c[None]),torch.tensor([[rad_b]],dtype=torch.float64),rad_u)[0].numpy()
                    vals.append(np.sum((q-pr)**2))
                numerical.append((vals[0]-vals[1])/(2*h))
            graderr=float(np.max(np.abs(np.array(numerical)-2*(s-result))))
            assert graderr<2e-7,graderr
            rows.append(dict(dim=dim,case=case,projection_error=float(err),objective_error=float(objerr),
                             feasibility_violation=float(violation),gradient_error=graderr,scipy_success=bool(ref.success)))
    # Concentric, singleton and tangent edge cases, and GPU agreement.
    edge=[]
    for center,radius,ru in [([0.,0.],.5,1.),([.2,.1],0.,1.),([2.,0.],1.,1.)]:
        s=torch.tensor([[3.,1.]],dtype=torch.float64);c=torch.tensor([center],dtype=torch.float64);r=torch.tensor([[radius]],dtype=torch.float64)
        z=intersection_project(s,c,r,ru)
        assert float((z-c).norm())<=radius+1e-12 and float(z.norm())<=ru+1e-12
        gpu=intersection_project(s.cuda(),c.cuda(),r.cuda(),ru).cpu()
        assert torch.allclose(z,gpu,atol=1e-12,rtol=1e-12)
        edge.append(z.tolist())
    return dict(random_cases=rows,edge_projections=edge)


def exact_lq():
    torch.manual_seed(74103);dtype=torch.float64;device='cuda'
    n=8;H=20;N=128;r=.07;bound=2.;alpha=.5
    I=torch.eye(n,dtype=dtype,device=device)
    A=.86*I+.025*(I.roll(1,0)+I.roll(-1,0));B=.15*I;Q=.1*I;P=[None]*(H+1);P[H]=I
    for t in range(H-1,-1,-1):P[t]=Q+A.T@P[t+1]@A
    x0=torch.randn(N,n,dtype=dtype,device=device)
    teacher_cost=torch.einsum('bi,ij,bj->b',x0,P[0],x0)
    out=[]
    for kind in ('set_projection','shared_constant'):
        x=x0.clone();cost=torch.zeros(N,device=device,dtype=dtype);adv=cost.clone();rhs=cost.clone();sumgain=cost.clone();dist2=cost.clone();L2=cost.clone();viol=[];member=[]
        for t in range(H):
            old=torch.zeros_like(x);b=2*(x@A.T)@P[t+1]@B
            kappa=float(2*torch.linalg.eigvalsh(B.T@P[t+1]@B).max())
            anchor=ball_project(-b/(2*(r+.5*kappa)),bound)
            geo=geometry(old,b,r,kappa,bound,alpha,anchor)
            source=torch.full_like(x,.1)
            projected=intersection_project(source,geo['center'],geo['radius'],bound)
            u=projected if kind=='set_projection' else source
            distance=(u-intersection_project(u,geo['center'],geo['radius'],bound)).norm(dim=-1)
            y=x@A.T+u@B.T
            c=(x*(x@Q)).sum(-1)+r*u.square().sum(-1)
            actual=c+(y*(y@P[t+1])).sum(-1)-(x*(x@P[t])).sum(-1)
            upper=upper_model(u,old,b,r,kappa)
            lipschitz=b.norm(dim=-1)+2*r*bound+2*kappa*bound
            cost+=c;adv+=actual;rhs+=-alpha*geo['gain']+lipschitz*distance
            sumgain+=geo['gain'];dist2+=distance.square();L2+=lipschitz.square()
            viol.append(float((actual-upper).max()));member.append(float((upper<=-alpha*geo['gain']+1e-10).double().mean()));x=y
        cost+=(x*(x@P[H])).sum(-1)
        gap=cost-teacher_cost;cauchy=-alpha*sumgain.mean()+(L2.mean()*dist2.mean()).sqrt()
        assert float((gap-adv).abs().max())<1e-10
        assert float((gap-rhs).max())<1e-10 and float(gap.mean()-cauchy)<1e-10
        assert max(viol)<1e-10
        if kind=='set_projection':assert float(gap.max())<0 and min(member)==1.
        out.append(dict(policy=kind,mean_teacher_cost=float(teacher_cost.mean()),mean_cost=float(cost.mean()),
            mean_cost_difference=float(gap.mean()),mean_distance_bound=float(rhs.mean()),cauchy_bound=float(cauchy),
            mean_set_distance_squared=float(dist2.mean()),max_telescoping_error=float((gap-adv).abs().max()),
            max_upper_bound_violation=max(0.,max(viol)),minimum_membership_fraction=min(member)))
    return dict(dim=n,horizon=H,trajectories=N,teacher='globally feasible zero linear feedback',alpha=alpha,policies=out)


def main():
    torch.set_num_threads(1)
    result=dict(projection=projection_checks(),finite_horizon_lq=exact_lq(),
        one_step=dict(teacher_action=0.,point_targets=[1.,2.],point_best_constant=1.5,
                      point_minimum_mse=.25,set_zero_loss_constant=1.,alpha=.5,
                      teacher_cost=2.5,point_cost=.25,set_cost=.5,
                      interpretation='Strictly smaller approximation error does not imply smaller control cost.'))
    # Verify the analytic one-step example and alpha=1 singleton independently.
    c=torch.tensor([[1.,0.],[2.,0.]],dtype=torch.float64);old=torch.zeros_like(c);b=-2*c
    g=geometry(old,b,1.,0.,3.,.5,c)
    candidate=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
    assert torch.allclose(intersection_project(candidate,g['center'],g['radius'],3.),candidate)
    singleton=geometry(old,b,1.,0.,3.,1.,c)
    assert torch.equal(singleton['radius'],torch.zeros(2,1,dtype=torch.float64))
    result['passed']=True;result['sources']={p.name:digest(p) for p in [Path(__file__),HERE/'improving_sets.py']}
    write(HERE/'results/improving_sets_exact.json',result)
    print('Exact-set verification passed:',len(result['projection']['random_cases']),'projection cases; LQ',result['finite_horizon_lq']['policies'])


if __name__=='__main__':main()
