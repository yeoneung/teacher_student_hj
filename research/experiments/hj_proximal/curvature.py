"""Matrix-free diagonal Bellman curvature from teacher-feedback perturbations.

An unconstrained additive action perturbation at every time defines a smooth
local extension of the fixed teacher's objective. Its same-time Hessian blocks
are c_uu + F_u^T V_xx^teacher F_u for control-affine dynamics. Hutchinson probes
estimate the diagonal without constructing a state or trajectory Hessian.
"""
import torch
from experiments.hj_gridfree.engine import capture
from experiments.hj_cotangent.systems import running,step,terminal


def perturbed_trajectory(initial,delta,teacher,p):
    x=initial;xs=[];us=[];cost=torch.zeros(len(x),device=x.device,dtype=x.dtype)
    for t in range(p.horizon):
        xs.append(x);u=teacher(x,t)+delta[:,t];us.append(u)
        cost=cost+running(x,u,p,t);x=step(x,u,p,t)
    return cost+terminal(x,p),torch.stack(xs,1),torch.stack(us,1)


class CurvatureGraph:
    def __init__(self,teacher,p,batch,probes=4):
        self.teacher,self.p,self.probes=teacher,p,probes
        for v in teacher.parameters():v.requires_grad_(False)
        self.x=torch.zeros(batch,p.dim,device='cuda')
        self.delta=torch.zeros(batch,p.horizon,p.n,device='cuda',requires_grad=True)
        self.signs=torch.ones(probes,batch,p.horizon,p.n,device='cuda')
        self.cost=torch.zeros(batch,device='cuda');self.states=torch.zeros(batch,p.horizon,p.dim,device='cuda')
        self.actions=torch.zeros(batch,p.horizon,p.n,device='cuda');self.gradient=torch.zeros_like(self.actions)
        self.diagonal=torch.zeros_like(self.actions)
        def evaluate():
            cost,x,u=perturbed_trajectory(self.x,self.delta,teacher,p)
            gradient=torch.autograd.grad(cost.sum(),self.delta,create_graph=True)[0]
            diagonal=torch.zeros_like(gradient)
            for j in range(probes):
                hv=torch.autograd.grad(gradient,self.delta,grad_outputs=self.signs[j],retain_graph=True)[0]
                diagonal=diagonal+hv*self.signs[j]/probes
            with torch.no_grad():
                for dest,source in zip([self.cost,self.states,self.actions,self.gradient,self.diagonal],[cost,x,u,gradient,diagonal]):dest.copy_(source)
        self.graph=capture(evaluate)

    def __call__(self,x,generator):
        self.x.copy_(x)
        self.signs.copy_(2*torch.randint(2,self.signs.shape,device='cuda',generator=generator)-1)
        self.graph.replay()
        return tuple(z.clone() for z in [self.cost,self.states,self.actions,self.gradient,self.diagonal])


def weighted_target(old,gradient,diagonal,p,regularization):
    """Exact diagonal-metric projection, up to 32 bisections for the ball."""
    r=p.dt*({'mechanical':.04,'reaction':.1,'building':.025}[p.family])/p.n
    h=diagonal.clamp_min(2*r);metric=h+regularization*2*r
    a=metric*old-gradient
    unconstrained=a/metric
    if p.family=='building':return unconstrained.clamp(0,p.bound),h
    # ||a/(metric+lambda)||_RMS <= bound. This upper endpoint is feasible.
    low=torch.zeros_like(a[:,:1]);high=a.square().mean(-1,keepdim=True).sqrt()/p.bound
    for _ in range(32):
        mid=(low+high)/2
        outside=(a/(metric+mid)).square().mean(-1,keepdim=True)>p.bound**2
        low=torch.where(outside,mid,low);high=torch.where(outside,high,mid)
    projected=a/(metric+high)
    inside=unconstrained.square().mean(-1,keepdim=True)<=p.bound**2
    return torch.where(inside,unconstrained,projected),h
