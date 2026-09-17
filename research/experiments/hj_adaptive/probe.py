"""Inspect the gradient error of a reused teacher criterion on a training batch.

The certificate concerns these exact sampled objectives at the check time.
It does not certify all states, subsequent Adam steps, or deployed policy cost.
"""
import torch
from experiments.hj_cotangent.core import local_return
from experiments.hj_gridfree.engine import capture


class GradientProbe:
    def __init__(self, actor, p, batch, length, rho):
        self.x=torch.zeros(batch,p.dim,device='cuda')
        self.t=torch.zeros(batch,device='cuda',dtype=torch.long)
        self.old=(torch.zeros_like(self.x),torch.zeros(batch,device='cuda'),torch.zeros_like(self.x))
        self.new=tuple(torch.zeros_like(v) for v in self.old)
        params=list(actor.net.parameters())
        for parameter in params:
            if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
        self.cached=torch.zeros(sum(v.numel() for v in params),device='cuda',dtype=torch.float64)
        self.fresh=torch.zeros_like(self.cached)
        self.metrics=torch.zeros(6,device='cuda',dtype=torch.float64)
        def inspect():
            for parameter in params:parameter.grad.zero_()
            old=local_return(self.x,self.t,actor,p,length,*self.old,rho=rho).mean()
            old.backward()
            with torch.no_grad():self.cached.copy_(torch.cat([v.grad.reshape(-1) for v in params]))
            for parameter in params:parameter.grad.zero_()
            new=local_return(self.x,self.t,actor,p,length,*self.new,rho=0.).mean()
            new.backward()
            with torch.no_grad():
                self.fresh.copy_(torch.cat([v.grad.reshape(-1) for v in params]))
                delta=(self.fresh-self.cached).norm()
                cn=self.cached.norm();fn=self.fresh.norm()
                dot=(self.cached*self.fresh).sum()
                self.metrics.copy_(torch.stack((delta/cn.clamp_min(1e-12),
                    dot/(cn*fn).clamp_min(1e-24),delta,cn,fn,new.double()-old.double())))
        self.graph=capture(inspect)

    def __call__(self,x,t,old,new):
        self.x.copy_(x);self.t.copy_(t)
        for dest,src in zip(self.old,old):dest.copy_(src)
        for dest,src in zip(self.new,new):dest.copy_(src)
        self.graph.replay()
        values=self.metrics.cpu().tolist()
        return dict(zip(['relative_gradient_error','gradient_cosine','gradient_error_norm',
            'cached_gradient_norm','fresh_gradient_norm','fresh_minus_surrogate_value'],values))
