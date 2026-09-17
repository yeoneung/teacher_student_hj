"""One-step student updates using frozen full-trajectory value covectors."""
import copy
import torch
from experiments.hj_gridfree.engine import capture
from .trajectory import advantage


class TransportGraph:
    def __init__(self,actor,p,batch,lr,teacher=None,variance_reduction=False):
        self.actor=actor
        self.x=torch.zeros(batch,p.dim,device='cuda')
        self.t=torch.zeros(batch,device='cuda',dtype=torch.long)
        self.u=torch.zeros(batch,p.n,device='cuda')
        self.anchor=torch.zeros_like(self.x);self.covector=torch.zeros_like(self.x)
        self.lr=torch.tensor(lr,device='cuda')
        self.loss=torch.zeros((),device='cuda')
        self.grad_norm=torch.zeros((),device='cuda',dtype=torch.float64)
        self.nonfinite_updates=torch.zeros((),device='cuda')
        self.clipped_updates=torch.zeros((),device='cuda')
        params=list(actor.net.parameters());initial=copy.deepcopy(actor.state_dict())
        self.variance_reduction=variance_reduction
        self.full_gradient=torch.zeros(sum(z.numel() for z in params),device='cuda')
        if variance_reduction:
            assert teacher is not None
            for z in teacher.net.parameters():z.requires_grad_(True)
            teacher_params=list(teacher.net.parameters())
        sizes=[z.numel() for z in params]
        self.m=[torch.zeros_like(z) for z in params];self.v=[torch.zeros_like(z) for z in params]
        self.b1=torch.ones((),device='cuda');self.b2=torch.ones((),device='cuda')
        for z in params:z.grad=torch.zeros_like(z)
        def update():
            for z in params:z.grad.zero_()
            loss=p.horizon*advantage(self.x,self.t,self.u,self.anchor,self.covector,actor,p).mean()
            loss.backward()
            if variance_reduction:
                reference=p.horizon*advantage(self.x,self.t,self.u,self.anchor,self.covector,teacher,p).mean()
                old_gradients=torch.autograd.grad(reference,teacher_params)
                with torch.no_grad():
                    for z,old,anchor in zip(params,old_gradients,self.full_gradient.split(sizes)):
                        z.grad.sub_(old).add_(anchor.reshape_as(z))
            with torch.no_grad():
                self.loss.copy_(loss)
                norm=torch.stack([z.grad.double().square().sum() for z in params]).sum().sqrt()
                finite=torch.isfinite(norm)&torch.isfinite(loss)
                self.grad_norm.copy_(norm)
                self.clipped_updates.add_((finite&(norm>10)).to(self.clipped_updates.dtype))
                self.nonfinite_updates.add_((~finite).to(self.nonfinite_updates.dtype))
                beta1=torch.where(finite,.9,1.);beta2=torch.where(finite,.999,1.)
                self.b1.mul_(beta1);self.b2.mul_(beta2)
                scale=(10/norm.clamp_min(1e-12)).clamp_max(1)
                for z,m,v in zip(params,self.m,self.v):
                    g=torch.where(finite,z.grad*scale,torch.zeros_like(z))
                    m.mul_(beta1).add_(g,alpha=.1);v.mul_(beta2).addcmul_(g,g,value=.001)
                    delta=-self.lr*(m/(1-self.b1).clamp_min(1e-12))/((v/(1-self.b2).clamp_min(1e-12)).sqrt()+1e-8)
                    z.add_(torch.where(finite,delta,torch.zeros_like(z)))
        self.graph=capture(update);actor.load_state_dict(initial)
        for z in self.m+self.v:z.zero_()
        self.b1.fill_(1);self.b2.fill_(1);self.nonfinite_updates.zero_();self.clipped_updates.zero_()

    def update(self,x,t,u,anchor,covector):
        for target,source in zip([self.x,self.t,self.u,self.anchor,self.covector],[x,t,u,anchor,covector]):target.copy_(source)
        self.graph.replay()

    def optimizer_snapshot(self):return [z.clone() for z in self.m+self.v+[self.b1,self.b2]]

    def restore_optimizer(self,saved):
        for target,source in zip(self.m+self.v+[self.b1,self.b2],saved):target.copy_(source)


def collect(graph,initial,p,batch):
    costs=[];pools=[[],[],[],[],[]];full=[]
    for offset in range(0,len(initial),batch):
        results=graph(initial[offset:offset+batch]);cost,states,controls,covectors=results[:4]
        if graph.parameter_gradient:full.append(results[4])
        costs.append(cost)
        values=[states[:,:-1].reshape(-1,p.dim),torch.arange(p.horizon,device='cuda').repeat(batch),
            controls.reshape(-1,p.n),states[:,1:].reshape(-1,p.dim),covectors.reshape(-1,p.dim)]
        for dest,value in zip(pools,values):dest.append(value)
    result=(torch.cat(costs),tuple(torch.cat(v).contiguous() for v in pools))
    if full:result=result+(torch.stack(full).mean(0),)
    return result
