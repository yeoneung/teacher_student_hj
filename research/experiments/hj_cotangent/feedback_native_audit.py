"""CPU audit against full torch autodiff and independent feedback replay."""
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.hj_gridfree.core import Problem,project,step,running,terminal,sample,LQR
from .feedback_native import gain_matrices,value_gradient,forward,solve


def main():
    torch.set_num_threads(1);records=[]
    with threadpool_limits(limits=1):
        for family,reference_mode in [(f,m) for f in ['mechanical','reaction'] for m in ['zero','random']]:
            p=Problem(family,4,horizon=12)
            x=sample(p,1,692101,device='cpu').double()
            q,v=gain_matrices(p);tq,tv=torch.from_numpy(q),torch.from_numpy(v)
            gen=np.random.default_rng(692102)
            r=gen.normal(size=(p.horizon,p.n))*p.bound*1.5
            reference=gen.normal(size=(p.horizon,p.dim))*.2 if reference_mode=='random' else np.zeros((p.horizon,p.dim))
            tref=torch.from_numpy(reference)
            z=torch.tensor(r,requires_grad=True,dtype=torch.float64)
            state=x.clone();cost=torch.zeros(1,dtype=torch.float64)
            for t in range(p.horizon):
                if p.mechanical:
                    delta=state[0,:p.n]-tref[t,:p.n]
                    wrapped=torch.atan2(delta.sin(),delta.cos())
                    raw=z[t]-tq[t]@wrapped-tv[t]@(state[0,p.n:]-tref[t,p.n:])
                else:raw=z[t]-tq[t]@(state[0]-tref[t])
                u=project(raw[None],p)
                cost=cost+running(state,u,p);state=step(state,u,p)
            cost=cost+terminal(state,p)
            grad=torch.autograd.grad(cost.sum(),z)[0].numpy()
            actual,g=value_gradient(r.ravel(),x[0].numpy(),q,v,reference,p.n,p.horizon,p.dt,p.gamma,p.bound,p.mechanical)
            value_error=abs(actual-cost.item());gradient_error=float(np.max(np.abs(g.reshape(r.shape)-grad)))
            assert value_error<1e-10 and gradient_error<1e-9,(family,value_error,gradient_error)
            independent=forward(x[0].numpy(),r,p,reference=reference)
            assert abs(independent['cost']-actual)<1e-10
            base=LQR(p,device='cpu')
            linear=-q[0]@x[0,:p.n].numpy()
            if p.mechanical:
                linear=-q[0]@np.arctan2(np.sin(x[0,:p.n].numpy()),np.cos(x[0,:p.n].numpy()))-v[0]@x[0,p.n:].numpy()
            linear=linear/max(1.,np.sqrt(np.mean(linear*linear))/p.bound)
            gain_error=float(np.max(np.abs(linear-base(x,0)[0].numpy())))
            assert gain_error<1e-12
            optimized=solve(x[0].numpy(),p,maxiter=80)
            assert optimized['cost']<=optimized['initial_cost']+1e-10
            records.append(dict(family=family,reference=reference_mode,value_error=value_error,gradient_error=gradient_error,
                Fourier_vs_dense_feedback_error=gain_error,initial_cost=optimized['initial_cost'],
                optimized_cost=optimized['cost'],seconds=optimized['seconds'],
                native_vs_feedback_replay_scaled_difference=optimized['native_vs_feedback_replay_scaled_difference']))
    target=Path('experiments/results/hj_cotangent_dev_v6c/feedback_native_audit.json')
    target.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
