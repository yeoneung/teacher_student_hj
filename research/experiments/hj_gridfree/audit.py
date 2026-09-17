"""Numerical checks with independent dense linear algebra and finite differences."""
import json
import torch
from .core import Problem,LQR,Actor,step,running,terminal,project,sample,rollout,exact_q,prefix_return
from .engine import AdamGraph,write,hashes


def main():
    torch.set_num_threads(1); report={}
    for fam in ['mechanical','reaction']:
        p=Problem(fam,8,horizon=8); base=LQR(p).double()
        z=torch.zeros(p.dim,device='cuda',dtype=torch.float64)
        u=torch.zeros(p.n,device='cuda',dtype=torch.float64)
        A=torch.autograd.functional.jacobian(lambda y:step(y[None],u[None],p)[0],z)
        B=torch.autograd.functional.jacobian(lambda a:step(z[None],a[None],p)[0],u)
        Q=torch.autograd.functional.hessian(lambda y:running(y[None],u[None],p)[0],z)/2
        R=torch.autograd.functional.hessian(lambda a:running(z[None],a[None],p)[0],u)/2
        P=torch.autograd.functional.hessian(lambda y:terminal(y[None],p)[0],z)/2
        ks=[]
        for _ in range(p.horizon):
            K=torch.linalg.solve(R+B.T@P@B,B.T@P@A)
            P=Q+A.T@P@A-A.T@P@B@K; ks.append(K)
        x=sample(p,4,701).double()*.01
        lqr_error=max((base(x,t)-(-x@k.T)).abs().max().item() for t,k in enumerate(ks[::-1]))
        # Gains were created float32 before double casting, tolerance reflects this.
        assert lqr_error<2e-7,lqr_error
        t=torch.tensor([0,2,4,7],device='cuda')
        x=sample(p,4,702).double(); a=base(x,t).detach().requires_grad_()
        q=exact_q(x,a,t,base,p); grad=torch.autograd.grad(q.sum(),a)[0]
        direction=torch.sin(torch.arange(a.numel(),device='cuda',dtype=torch.float64)).reshape_as(a)
        eps=1e-5
        finite=(exact_q(x,a+eps*direction,t,base,p)-exact_q(x,a-eps*direction,t,base,p))/(2*eps)
        graderr=(finite-(grad*direction).sum(-1)).abs().max().item(); assert graderr<1e-6,graderr
        policy=lambda y,k:project(.9*base(y,k)+.1*torch.sin(y[...,:p.n]),p)
        with torch.no_grad():
            jb=rollout(x,base,p); js,xs,us=rollout(x,policy,p,keep=True)
            delta=torch.zeros_like(jb)
            for k in range(p.horizon):
                tt=torch.full((len(x),),k,device='cuda',dtype=torch.long)
                delta+=exact_q(xs[:,k],us[:,k],tt,base,p)-exact_q(xs[:,k],base(xs[:,k],k),tt,base,p)
            identity=(js-jb-delta).abs().max().item(); assert identity<1e-10,identity
            blockdelta=torch.zeros_like(jb)
            for k in range(0,p.horizon,3):
                tt=torch.full((len(x),),k,device='cuda',dtype=torch.long)
                v=exact_q(xs[:,k],base(xs[:,k],k),tt,base,p)
                blockdelta+=prefix_return(xs[:,k],tt,policy,base,p,3)-v
            blockidentity=(js-jb-blockdelta).abs().max().item(); assert blockidentity<1e-10,blockidentity
            feasibility=(us.square().mean(-1).sqrt()/p.bound).max().item(); assert feasibility<=1+1e-10
        solver=AdamGraph(x.float(),p,kind='q',lr=.1)
        xf=x.float(); a0=base(xf,t).float()
        c1,a1=solver.solve(xf,a0,6,t); solver.solve(xf*.8,a0,4,t)
        c2,a2=solver.solve(xf,a0,6,t)
        reset=(a1-a2).abs().max().item(); assert reset==0,reset
        improvement=(c1-exact_q(xf,a0,t,solver.base,p)).max().item(); assert improvement<=1e-5
        report[fam]=dict(lqr_dense_max_abs=lqr_error,q_gradient_max_abs=graderr,bellman_identity_max_abs=identity,block_identity_max_abs=blockidentity,max_budget_ratio=feasibility,optimizer_reset_max_abs=reset,incumbent_max_cost_increase=improvement)
    report['source_hashes']=hashes(); write('experiments/results/hj_gridfree_audit_v1/audit.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
