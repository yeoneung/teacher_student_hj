"""Independent CPU adjoint and native L-BFGS over radially feasible controls.

This is a classical optimization comparator, not a source of PMP teacher labels.
"""
import time
import numpy as np
from numba import njit
from scipy.optimize import minimize


@njit(cache=True)
def value_gradient(zflat,x0,n,horizon,dt,gamma,budget,mechanical):
    z=zflat.reshape(horizon,n); u=z.copy(); scales=np.ones(horizon)
    for k in range(horizon):
        rms=np.sqrt(np.sum(z[k]*z[k])/n)
        if rms>budget: scales[k]=rms/budget; u[k]/=scales[k]
    dim=2*n if mechanical else n
    xs=np.empty((horizon+1,dim)); xs[0]=x0; cost=0.
    for k in range(horizon):
        x=xs[k]
        for i in range(n):
            prev=(i-1)%n; nxt=(i+1)%n
            if mechanical:
                q=x[i]; v=x[n+i]
                cost+=dt/n*(2*(1-np.cos(q))+.08*v*v+.04*u[k,i]**2+.2*(1-np.cos(q-x[nxt])))
                vn=v+dt*(5*np.sin(q)+gamma*(np.sin(x[prev]-q)+np.sin(x[nxt]-q))+u[k,i]-.2*v)
                xs[k+1,n+i]=vn; xs[k+1,i]=q+dt*vn
            else:
                cost+=dt/n*(x[i]**2+.1*u[k,i]**2+.1*(x[i]-x[nxt])**2)
                xs[k+1,i]=x[i]+dt*(gamma*(x[prev]+x[nxt]-2*x[i])+1.2*x[i]-.4*x[i]**3+u[k,i])
    lam=np.empty(dim)
    for i in range(n):
        if mechanical:
            cost+=(8*(1-np.cos(xs[horizon,i]))+.4*xs[horizon,n+i]**2)/n
            lam[i]=8*np.sin(xs[horizon,i])/n; lam[n+i]=.8*xs[horizon,n+i]/n
        else:
            cost+=3*xs[horizon,i]**2/n; lam[i]=6*xs[horizon,i]/n
    grad=np.empty_like(u)
    for k in range(horizon-1,-1,-1):
        x=xs[k]; new=np.empty_like(lam)
        if mechanical:
            a=lam[n:]+dt*lam[:n]
            for i in range(n):
                prev=(i-1)%n; nxt=(i+1)%n; q=x[i]
                cp=np.cos(q-x[prev]); cn=np.cos(q-x[nxt])
                df=(5*np.cos(q)-gamma*(cp+cn))*a[i]+gamma*(cp*a[prev]+cn*a[nxt])
                cq=dt/n*(2*np.sin(q)+.2*(np.sin(q-x[prev])+np.sin(q-x[nxt])))
                new[i]=lam[i]+dt*df+cq
                new[n+i]=(1-.2*dt)*a[i]+.16*dt/n*x[n+i]
                grad[k,i]=dt*a[i]+.08*dt/n*u[k,i]
        else:
            for i in range(n):
                prev=(i-1)%n; nxt=(i+1)%n
                new[i]=(1+dt*(1.2-1.2*x[i]**2-2*gamma))*lam[i]+dt*gamma*(lam[prev]+lam[nxt])+dt/n*(2*x[i]+.2*(2*x[i]-x[prev]-x[nxt]))
                grad[k,i]=dt*lam[i]+.2*dt/n*u[k,i]
        lam=new
    for k in range(horizon):
        if scales[k]>1:
            dot=np.sum(grad[k]*z[k]); norm=np.sum(z[k]*z[k])
            grad[k]=(grad[k]-z[k]*dot/norm)/scales[k]
    return cost,grad.reshape(-1)


def projected(z,p):
    u=np.asarray(z,dtype=np.float64).reshape(p.horizon,p.n).copy()
    scales=np.maximum(1,np.sqrt(np.mean(u*u,axis=1,keepdims=True))/p.bound)
    return u/scales


def solve(x,initial,p,maxiter=512):
    args=(np.asarray(x,dtype=np.float64),p.n,p.horizon,p.dt,p.gamma,p.bound,p.mechanical)
    init=np.asarray(initial,dtype=np.float64).reshape(-1)
    start=time.perf_counter(); initial_cost,_=value_gradient(init,*args)
    best=[initial_cost,init.copy()]
    def objective(z):
        c,g=value_gradient(z,*args)
        if np.isfinite(c) and c<best[0]: best[0]=c; best[1]=z.copy()
        return c,g
    result=minimize(objective,init,jac=True,method='L-BFGS-B',options=dict(maxiter=maxiter,ftol=1e-11,gtol=1e-7,maxls=30,maxcor=10))
    return dict(cost=best[0],u=projected(best[1],p),seconds=time.perf_counter()-start,nit=int(result.nit),nfev=int(result.nfev),success=bool(result.success),message=str(result.message))


def audit():
    # CPU-only numerical audit can run independently of GPU training.
    import torch
    from .core import Problem,sample,sequence_cost,project
    from .engine import write
    records={}; torch.set_num_threads(1)
    for family in ['mechanical','reaction']:
        p=Problem(family,8,horizon=12); x=sample(p,1,919,device='cpu').double()
        gen=np.random.default_rng(919)
        z=gen.normal(size=(p.horizon,p.n))*p.bound*1.3
        zz=torch.tensor(z[None],dtype=torch.float64,requires_grad=True)
        cost=sequence_cost(x,project(zz,p),p)
        grad=torch.autograd.grad(cost.sum(),zz)[0][0].numpy()
        c,g=value_gradient(z.reshape(-1),x[0].numpy(),p.n,p.horizon,p.dt,p.gamma,p.bound,p.mechanical)
        error=max(abs(c-cost.item()),np.max(np.abs(g.reshape(z.shape)-grad)))
        assert error<1e-9,error
        r=solve(x[0].numpy(),z,p,64); assert r['cost']<=c+1e-10
        records[family]=dict(torch_cost_gradient_max_abs=float(error),initial_cost=float(c),optimized_cost=r['cost'],solve_seconds=r['seconds'])
    write('experiments/results/hj_gridfree_audit_v1/native.json',records)
    print(records,flush=True)


if __name__=='__main__': audit()
