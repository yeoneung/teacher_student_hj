"""Native L-BFGS optimization of residuals around a fixed LQR feedback law."""
from functools import lru_cache
import time

import numpy as np
from numba import njit
from scipy.optimize import minimize

from .conditioning import numpy_step_cost,numpy_terminal


@lru_cache(maxsize=2)
def gain_matrices(p):
    from experiments.hj_gridfree.core import LQR
    gains=LQR(p,device='cpu').gains.double().numpy()
    basis=np.fft.rfft(np.eye(p.n),axis=-1)
    q=np.fft.irfft(gains[:,None,:,0]*basis[None],n=p.n,axis=-1).transpose(0,2,1)
    if p.mechanical:
        v=np.fft.irfft(gains[:,None,:,1]*basis[None],n=p.n,axis=-1).transpose(0,2,1)
    else:v=np.zeros((1,1,1))
    return np.ascontiguousarray(q),np.ascontiguousarray(v)


@njit(cache=True)
def value_gradient(zflat,x0,kq,kv,reference,n,horizon,dt,gamma,budget,mechanical):
    residual=zflat.reshape(horizon,n)
    dim=2*n if mechanical else n
    xs=np.empty((horizon+1,dim));xs[0]=x0
    inputs=np.empty((horizon,n));raw=np.empty((horizon,n));scales=np.ones(horizon)
    cost=0.
    for k in range(horizon):
        x=xs[k]
        if mechanical:
            delta=x[:n]-reference[k,:n]
            wrapped=np.arctan2(np.sin(delta),np.cos(delta))
            z=residual[k]-kq[k]@wrapped-kv[k]@(x[n:]-reference[k,n:])
        else:z=residual[k]-kq[k]@(x-reference[k])
        raw[k]=z
        rms=np.sqrt(np.sum(z*z)/n)
        scale=max(1.,rms/budget);scales[k]=scale
        u=z/scale;inputs[k]=u
        for i in range(n):
            prev=(i-1)%n;nxt=(i+1)%n
            if mechanical:
                q=x[i];v=x[n+i]
                cost+=dt/n*(2*(1-np.cos(q))+.08*v*v+.04*u[i]**2+.2*(1-np.cos(q-x[nxt])))
                vn=v+dt*(5*np.sin(q)+gamma*(np.sin(x[prev]-q)+np.sin(x[nxt]-q))+u[i]-.2*v)
                xs[k+1,n+i]=vn;xs[k+1,i]=q+dt*vn
            else:
                cost+=dt/n*(x[i]**2+.1*u[i]**2+.1*(x[i]-x[nxt])**2)
                xs[k+1,i]=x[i]+dt*(gamma*(x[prev]+x[nxt]-2*x[i])+1.2*x[i]-.4*x[i]**3+u[i])
    lam=np.empty(dim)
    for i in range(n):
        if mechanical:
            cost+=(8*(1-np.cos(xs[horizon,i]))+.4*xs[horizon,n+i]**2)/n
            lam[i]=8*np.sin(xs[horizon,i])/n;lam[n+i]=.8*xs[horizon,n+i]/n
        else:
            cost+=3*xs[horizon,i]**2/n;lam[i]=6*xs[horizon,i]/n
    gradient=np.empty_like(inputs)
    for k in range(horizon-1,-1,-1):
        x=xs[k];u=inputs[k];new=np.empty_like(lam);gu=np.empty(n)
        if mechanical:
            a=lam[n:]+dt*lam[:n]
            for i in range(n):
                prev=(i-1)%n;nxt=(i+1)%n;q=x[i]
                cp=np.cos(q-x[prev]);cn=np.cos(q-x[nxt])
                df=(5*np.cos(q)-gamma*(cp+cn))*a[i]+gamma*(cp*a[prev]+cn*a[nxt])
                cq=dt/n*(2*np.sin(q)+.2*(np.sin(q-x[prev])+np.sin(q-x[nxt])))
                new[i]=lam[i]+dt*df+cq
                new[n+i]=(1-.2*dt)*a[i]+.16*dt/n*x[n+i]
                gu[i]=dt*a[i]+.08*dt/n*u[i]
        else:
            for i in range(n):
                prev=(i-1)%n;nxt=(i+1)%n
                new[i]=(1+dt*(1.2-1.2*x[i]**2-2*gamma))*lam[i]+dt*gamma*(lam[prev]+lam[nxt])+dt/n*(2*x[i]+.2*(2*x[i]-x[prev]-x[nxt]))
                gu[i]=dt*lam[i]+.2*dt/n*u[i]
        gz=gu.copy()
        if scales[k]>1:
            z=raw[k]
            gz=(gu-z*np.sum(gu*z)/np.sum(z*z))/scales[k]
        gradient[k]=gz
        new[:n]-=kq[k].T@gz
        if mechanical:new[n:]-=kv[k].T@gz
        lam=new
    return cost,gradient.reshape(-1)


def forward(x0,residual,p,gains=None,reference=None):
    kq,kv=gains if gains is not None else gain_matrices(p)
    if reference is None:reference=np.zeros((p.horizon,p.dim))
    x=np.asarray(x0,dtype=np.float64).copy();states=[x.copy()];controls=[];cost=0.
    for t in range(p.horizon):
        if p.mechanical:
            delta=x[:p.n]-reference[t,:p.n]
            q=np.arctan2(np.sin(delta),np.cos(delta))
            raw=residual[t]-kq[t]@q-kv[t]@(x[p.n:]-reference[t,p.n:])
        else:raw=residual[t]-kq[t]@(x-reference[t])
        u=raw/max(1.,np.sqrt(np.mean(raw*raw))/p.bound)
        x,stage=numpy_step_cost(x,u,p,t)
        cost+=float(stage);states.append(x.copy());controls.append(u)
    return dict(cost=cost+float(numpy_terminal(x,p)),u=np.stack(controls),states=np.stack(states))


def reference_from_plan(x0,u,p):
    x=np.asarray(x0,dtype=np.float64).copy();states=[]
    for t in range(p.horizon):
        states.append(x.copy())
        x,_=numpy_step_cost(x,u[t],p,t)
    return np.stack(states)


def solve(x0,p,initial_plan=None,maxiter=2048,reference_states=None):
    started=time.perf_counter()
    gains=gain_matrices(p)
    initial=np.zeros((p.horizon,p.n)) if initial_plan is None else np.asarray(initial_plan,dtype=np.float64).copy()
    if reference_states is not None:
        reference=np.ascontiguousarray(reference_states[:p.horizon],dtype=np.float64)
    elif initial_plan is None:reference=np.zeros((p.horizon,p.dim))
    else:reference=reference_from_plan(x0,initial,p)
    args=(np.asarray(x0,dtype=np.float64),*gains,reference,p.n,p.horizon,p.dt,p.gamma,p.bound,p.mechanical)
    init=initial.reshape(-1)
    c,_=value_gradient(init,*args);best=[c,init.copy()]
    def objective(z):
        cost,grad=value_gradient(z,*args)
        if np.isfinite(cost) and cost<best[0]:best[0]=cost;best[1]=z.copy()
        return cost,grad
    optimized=minimize(objective,init,jac=True,method='L-BFGS-B',
        options=dict(maxiter=maxiter,ftol=1e-11,gtol=1e-7,maxls=30,maxcor=10))
    candidate=forward(x0,best[1].reshape(initial.shape),p,gains,reference)
    original=forward(x0,initial,p,gains,reference)
    selected=candidate if candidate['cost']<original['cost'] else original
    selected['residual']=best[1].reshape(initial.shape) if selected is candidate else initial
    selected['tracking_reference']=reference
    selected.update(seconds=time.perf_counter()-started,initial_cost=original['cost'],
        native_best_objective=best[0],native_candidate_replay_cost=candidate['cost'],
        native_vs_feedback_replay_scaled_difference=abs(best[0]-candidate['cost'])/(1+abs(candidate['cost'])),
        retained='optimized' if selected is candidate else 'initial',nit=int(optimized.nit),nfev=int(optimized.nfev),
        success=bool(optimized.success),message=str(optimized.message))
    return selected
