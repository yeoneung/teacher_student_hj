"""Full initial-set DPC with gradient accumulation before each Adam update."""
import copy
import torch
from experiments.hj_cotangent.systems import rollout
from experiments.hj_gridfree.engine import capture


class FullBatchGraph:
    def __init__(self,actor,p,batch,accumulation,lr):
        self.batch=batch;self.accumulation=accumulation
        self.x=torch.zeros(batch,p.dim,device='cuda')
        params=list(actor.net.parameters());sizes=[z.numel() for z in params];count=sum(sizes)
        self.partial=torch.zeros(count,device='cuda');self.partial_cost=torch.zeros((),device='cuda')
        self.gradient_bank=torch.zeros(accumulation,count,device='cuda')
        self.cost_bank=torch.zeros(accumulation,device='cuda')
        self.gradient=torch.zeros(count,device='cuda');self.loss=torch.zeros((),device='cuda')
        self.grad_norm=torch.zeros((),device='cuda',dtype=torch.float64)
        self.nonfinite_updates=torch.zeros((),device='cuda');self.clipped_updates=torch.zeros((),device='cuda')
        initial=copy.deepcopy(actor.state_dict())
        for z in params:z.grad=torch.zeros_like(z)
        def backward():
            for z in params:z.grad.zero_()
            loss=rollout(self.x,actor,p).mean();loss.backward()
            with torch.no_grad():
                self.partial.copy_(torch.cat([z.grad.reshape(-1) for z in params]));self.partial_cost.copy_(loss)
        self.backward_graph=capture(backward)
        self.m=[torch.zeros_like(z) for z in params];self.v=[torch.zeros_like(z) for z in params]
        self.b1=torch.ones((),device='cuda');self.b2=torch.ones((),device='cuda')
        def optimize():
            with torch.no_grad():
                self.gradient.copy_(self.gradient_bank.mean(0));self.loss.copy_(self.cost_bank.mean())
                parts=self.gradient.split(sizes)
                norm=torch.stack([g.double().square().sum() for g in parts]).sum().sqrt()
                finite=torch.isfinite(norm)&torch.isfinite(self.loss);self.grad_norm.copy_(norm)
                self.nonfinite_updates.add_((~finite).to(self.nonfinite_updates.dtype))
                self.clipped_updates.add_((finite&(norm>10)).to(self.clipped_updates.dtype))
                beta1=torch.where(finite,.9,1.);beta2=torch.where(finite,.999,1.)
                self.b1.mul_(beta1);self.b2.mul_(beta2);scale=(10/norm.clamp_min(1e-12)).clamp_max(1)
                for z,m,v,part in zip(params,self.m,self.v,parts):
                    g=torch.where(finite,part.reshape_as(z)*scale,torch.zeros_like(z))
                    m.mul_(beta1).add_(g,alpha=.1);v.mul_(beta2).addcmul_(g,g,value=.001)
                    delta=-lr*(m/(1-self.b1).clamp_min(1e-12))/((v/(1-self.b2).clamp_min(1e-12)).sqrt()+1e-8)
                    z.add_(torch.where(finite,delta,torch.zeros_like(z)))
        self.optimizer_graph=capture(optimize);actor.load_state_dict(initial)
        for z in self.m+self.v:z.zero_()
        self.b1.fill_(1);self.b2.fill_(1);self.grad_norm.zero_();self.nonfinite_updates.zero_();self.clipped_updates.zero_()

    def update_full(self,initial):
        assert len(initial)==self.batch*self.accumulation
        for i in range(self.accumulation):
            self.x.copy_(initial[i*self.batch:(i+1)*self.batch]);self.backward_graph.replay()
            self.gradient_bank[i].copy_(self.partial);self.cost_bank[i].copy_(self.partial_cost)
        self.optimizer_graph.replay()
