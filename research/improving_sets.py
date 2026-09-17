"""HJ upper-model sublevel supervision, exact ball-constrained epsilon=0 case.

No nonlinear global curvature certificate is inferred by this implementation.
The geometry is exact for the supplied quadratic model; its validity as an
advantage upper bound is a separate assumption. Frozen numerical sources are
imported unchanged. Projection is recomputed at the current action per update.
"""
import copy
import torch
from experiments.hj_gridfree.engine import capture
from experiments.hj_proximal.targets import reconstruct, feasible


def ball_project(u, radius):
    return u * (radius / u.norm(dim=-1, keepdim=True).clamp_min(1e-30)).clamp_max(1)


def intersection_project(source, center, radius, control_radius):
    """Euclidean projection onto two intersecting closed balls, dimension >=2.

    Test each singly-active projection; otherwise both spheres are active and
    their intersection is a sphere in a hyperplane orthogonal to center.
    Precondition: nonempty intersection (ensured by the feasible anchor).
    """
    pu = ball_project(source, control_radius)
    pb = center + ball_project(source-center, radius)
    d = center.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    direction = center/d
    height = (control_radius**2+d.square()-radius.square())/(2*d)
    transverse = source-(source*direction).sum(-1, keepdim=True)*direction
    joint = height*direction + (control_radius**2-height.square()).clamp_min(0).sqrt()*transverse/transverse.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.where((pu-center).norm(dim=-1, keepdim=True)<=radius, pu,
                       torch.where(pb.norm(dim=-1, keepdim=True)<=control_radius, pb, joint))


def upper_model(u, old, b, r, kappa):
    delta = u-old
    return ((r*(u+old)+b)*delta).sum(-1) + .5*kappa*delta.square().sum(-1)


def geometry(old, b, r, kappa, control_radius, alpha, anchor):
    """Feasible realized anchor/fallback shared by point and set supervision."""
    value = upper_model(anchor,old,b,r,kappa)
    anchor = torch.where((value < 0)[:,None], anchor, old)
    gain = -upper_model(anchor,old,b,r,kappa)
    a = r+.5*kappa
    grad = 2*r*old+b
    center = old-grad/(2*a)
    radius2 = (grad.square().sum(-1)/(4*a)-alpha*gain)/a
    return dict(anchor=anchor, gain=gain, center=center,
                radius=radius2.clamp_min(0).sqrt()[:,None], a=a)


def cache_geometry(data,p,kappa,alpha,eta=16.):
    r=data['r'];rho=2*r*eta
    w=feasible((rho*data['old']-data['b'])/(2*r+rho),p)
    return geometry(data['old'],data['b'],r,kappa,p.n**.5*p.bound,alpha,w)


class SupervisionGraph:
    """Identical captured Adam for point, current-action set, and upper model."""
    def __init__(self,actor,p,batch,lr,kind):
        self.kind=kind;self.p=p
        self.features=torch.zeros(batch,p.n,19 if p.mechanical else 13,device='cuda')
        self.base=torch.zeros(batch,p.n,device='cuda')
        self.anchor=torch.zeros_like(self.base);self.center=torch.zeros_like(self.base)
        self.radius=torch.ones(batch,1,device='cuda')
        self.loss=torch.zeros((),device='cuda');self.nonfinite_updates=torch.zeros_like(self.loss)
        params=list(actor.net.parameters());initial=copy.deepcopy(actor.state_dict())
        self.m=[torch.zeros_like(z) for z in params];self.v=[torch.zeros_like(z) for z in params]
        self.b1=torch.ones((),device='cuda');self.b2=torch.ones_like(self.b1)
        for z in params:z.grad=torch.zeros_like(z)
        def update():
            for z in params:z.grad.zero_()
            u=reconstruct(actor.net,self.features,self.base,p)
            if kind=='set':
                with torch.no_grad(): target=intersection_project(u,self.center,self.radius,p.n**.5*p.bound)
                loss=((u-target)/p.bound).square().mean()
            elif kind=='upper':loss=((u-self.center)/p.bound).square().mean()
            else:loss=((u-self.anchor)/p.bound).square().mean()
            loss.backward()
            with torch.no_grad():
                self.loss.copy_(loss)
                norm=torch.stack([z.grad.double().square().sum() for z in params]).sum().sqrt()
                finite=torch.isfinite(norm)&torch.isfinite(loss)
                self.nonfinite_updates.add_((~finite).float())
                beta1=torch.where(finite,.9,1.);beta2=torch.where(finite,.999,1.)
                self.b1.mul_(beta1);self.b2.mul_(beta2)
                scale=(10/norm.clamp_min(1e-12)).clamp_max(1)
                for z,m,v in zip(params,self.m,self.v):
                    g=torch.where(finite,z.grad*scale,torch.zeros_like(z))
                    m.mul_(beta1).add_(g,alpha=.1);v.mul_(beta2).addcmul_(g,g,value=.001)
                    delta=-lr*(m/(1-self.b1).clamp_min(1e-12))/((v/(1-self.b2).clamp_min(1e-12)).sqrt()+1e-8)
                    z.add_(torch.where(finite,delta,torch.zeros_like(z)))
        self.graph=capture(update);actor.load_state_dict(initial);self.reset_optimizer()
        self.nonfinite_updates.zero_()

    def reset_optimizer(self):
        for z in self.m+self.v:z.zero_()
        self.b1.fill_(1);self.b2.fill_(1)

    def update(self,features,base,anchor,center,radius):
        self.features.copy_(features);self.base.copy_(base);self.anchor.copy_(anchor)
        self.center.copy_(center);self.radius.copy_(radius);self.graph.replay()
