from dataclasses import dataclass, asdict
import math
import torch
from torch import nn


@dataclass(frozen=True)
class Problem:
    family: str
    n: int
    horizon: int = 40
    coupling: float = -1.
    budget: float = -1.

    @property
    def mechanical(self): return self.family == 'mechanical'
    @property
    def dim(self): return self.n * (2 if self.mechanical else 1)
    @property
    def dt(self): return .08 if self.mechanical else .05
    @property
    def gamma(self): return self.coupling if self.coupling >= 0 else (2.5 if self.mechanical else 2.)
    @property
    def bound(self): return self.budget if self.budget > 0 else (3.2 if self.mechanical else .8)


def project(u, p):
    return u / (u.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt() / p.bound).clamp_min(1.)


def step(x, u, p):
    if p.mechanical:
        q, v = x[..., :p.n], x[..., p.n:]
        force = 5*q.sin() + p.gamma*((q.roll(1, -1)-q).sin()+(q.roll(-1, -1)-q).sin())
        vn = v + p.dt*(force + u - .2*v)
        return torch.cat((q+p.dt*vn, vn), -1)
    lap = x.roll(1, -1) + x.roll(-1, -1) - 2*x
    return x + p.dt*(p.gamma*lap + 1.2*x - .4*x.pow(3) + u)


def running(x, u, p):
    if p.mechanical:
        q,v = x[..., :p.n], x[..., p.n:]
        return p.dt*(2*(1-q.cos()) + .08*v.square() + .04*u.square() + .2*(1-(q-q.roll(-1,-1)).cos())).mean(-1)
    return p.dt*(x.square()+.1*u.square()+.1*(x-x.roll(-1,-1)).square()).mean(-1)


def terminal(x,p):
    if p.mechanical:
        q,v = x[..., :p.n], x[..., p.n:]
        return (8*(1-q.cos())+.4*v.square()).mean(-1)
    return 3*x.square().mean(-1)


def sample(p, count, seed, device='cuda', shift=False):
    gen=torch.Generator(device='cpu').manual_seed(seed)
    x = 2*torch.rand(count,p.dim,generator=gen)-1
    if p.mechanical:
        # Both local disorder and a coherent component, fixed before outcomes.
        common=2*torch.rand(count,1,generator=gen)-1
        x[:,:p.n]=2.2*x[:,:p.n]+.6*common
        x[:,p.n:]*=1.5
        if shift: x[:,:p.n]*=1.15; x[:,p.n:]*=1.3
    else:
        common=2*torch.rand(count,1,generator=gen)-1
        x=1.4*x+1.*common
        if shift: x*=1.25
    return x.to(device)


class LQR(nn.Module):
    """Exact finite-horizon linearized LQR in Fourier coordinates; radial constraints."""
    def __init__(self,p,device='cuda'):
        super().__init__(); self.p=p
        f=torch.arange(p.n//2+1,dtype=torch.float64,device=device)
        lap=2*torch.cos(2*math.pi*f/p.n)-2
        d=2 if p.mechanical else 1
        A=torch.zeros(len(f),d,d,dtype=torch.float64,device=device)
        B=torch.zeros(len(f),d,1,dtype=torch.float64,device=device)
        Q=torch.zeros_like(A); P=torch.zeros_like(A)
        if p.mechanical:
            a=5+p.gamma*lap
            A[:,0,0]=1+p.dt**2*a; A[:,0,1]=p.dt*(1-.2*p.dt)
            A[:,1,0]=p.dt*a; A[:,1,1]=1-.2*p.dt
            B[:,0,0]=p.dt**2; B[:,1,0]=p.dt
            Q[:,0,0]=p.dt*(1-.1*lap); Q[:,1,1]=p.dt*.08
            P[:,0,0]=4; P[:,1,1]=.4; R=p.dt*.04
        else:
            A[:,0,0]=1+p.dt*(1.2+p.gamma*lap); B[:,0,0]=p.dt
            Q[:,0,0]=p.dt*(1-.1*lap); P[:,0,0]=3; R=p.dt*.1
        gains=[]
        for _ in range(p.horizon):
            K=torch.linalg.solve(R+B.mT@P@B,B.mT@P@A)
            P=Q+A.mT@P@A-A.mT@P@B@K
            P=.5*(P+P.mT); gains.append(K[:,0,:])
        self.register_buffer('gains',torch.stack(gains[::-1]).float())

    def forward(self,x,t):
        p=self.p; k=self.gains[t].to(x.dtype)
        if p.mechanical:
            q=x[...,:p.n]; q=torch.atan2(q.sin(),q.cos())
            qf=torch.fft.rfft(q,dim=-1); vf=torch.fft.rfft(x[...,p.n:],dim=-1)
            uf=-(k[...,0]*qf+k[...,1]*vf)
        else: uf=-k[...,0]*torch.fft.rfft(x,dim=-1)
        return project(torch.fft.irfft(uf,n=p.n,dim=-1),p)


class Actor(nn.Module):
    """Node-shared residual policy; same architecture for every learning loss."""
    def __init__(self,p,base,width=64):
        super().__init__(); self.p=p; self.base=base
        features=19 if p.mechanical else 13
        self.net=nn.Sequential(nn.Linear(features,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,t):
        p=self.p; orig=x.dtype; x=x.float(); u0=self.base(x,t)
        # Local, second-neighbor and pooled features; no dimension-specific parameters.
        if p.mechanical:
            q,v=x[...,:p.n],x[...,p.n:]/4
            a=[z for s in (-1,0,1) for z in (q.roll(s,-1).sin(),q.roll(s,-1).cos(),v.roll(s,-1))]
            a += [q.roll(-2,-1).sin(),q.roll(2,-1).sin()]
            a += [q.sin().mean(-1,keepdim=True).expand_as(q),q.cos().mean(-1,keepdim=True).expand_as(q),v.mean(-1,keepdim=True).expand_as(q),v.square().mean(-1,keepdim=True).expand_as(q)]
        else:
            a=[x.roll(s,-1) for s in (-2,-1,0,1,2)]
            a += [x.square(), x.pow(3), x.mean(-1,keepdim=True).expand_as(x),x.square().mean(-1,keepdim=True).expand_as(x)]
        a += [u0/p.bound,u0.square().mean(-1,keepdim=True).expand_as(u0)/p.bound**2,torch.ones_like(u0)*(t[...,None] if isinstance(t,torch.Tensor) and t.ndim else t)/p.horizon, torch.ones_like(u0)*p.bound/(3.2 if p.mechanical else .8)]
        z=self.net(torch.stack(a,-1)).squeeze(-1)
        return project(u0+2*p.bound*z.tanh(),p).to(orig)


def rollout(x,policy,p,t0=0,keep=False):
    scalar=isinstance(t0,int); total=torch.zeros(x.shape[0],device=x.device,dtype=x.dtype)
    xs=[]; us=[]
    for i in range(p.horizon-t0 if scalar else p.horizon):
        t=t0+i if scalar else (t0+i).clamp_max(p.horizon-1)
        u=policy(x,t); c=running(x,u,p); y=step(x,u,p)
        if not scalar:
            mask=t0+i<p.horizon; c=c*mask; y=torch.where(mask[:,None],y,x)
        if keep: xs.append(x); us.append(u)
        total=total+c; x=y
    total=total+terminal(x,p)
    return (total,torch.stack(xs,1),torch.stack(us,1)) if keep else total


def exact_q(x,u,t,base,p):
    total=running(x,u,p); x=step(x,u,p)
    for i in range(1,p.horizon):
        active=t+i<p.horizon
        b=base(x,(t+i).clamp_max(p.horizon-1))
        total=total+active*running(x,b,p)
        x=torch.where(active[:,None],step(x,b,p),x)
    return total+terminal(x,p)


def prefix_return(x,t,student,base,p,length):
    """L-step Bellman target: learned prefix followed by a fixed feasible tail."""
    total=torch.zeros(x.shape[0],device=x.device,dtype=x.dtype)
    for i in range(p.horizon):
        active=t+i<p.horizon; tt=(t+i).clamp_max(p.horizon-1)
        u=student(x,tt) if i<length else base(x,tt)
        total=total+active*running(x,u,p)
        x=torch.where(active[:,None],step(x,u,p),x)
    return total+terminal(x,p)


def truncated_return(x,t,student,p,length):
    """Short-horizon cost with the original terminal penalty, no teacher value."""
    total=torch.zeros(x.shape[0],device=x.device,dtype=x.dtype)
    for i in range(length):
        active=t+i<p.horizon; tt=(t+i).clamp_max(p.horizon-1)
        u=student(x,tt); total=total+active*running(x,u,p)
        x=torch.where(active[:,None],step(x,u,p),x)
    return total+terminal(x,p)


def sequence_cost(x,u,p,t0=0):
    total=torch.zeros(x.shape[0],device=x.device,dtype=x.dtype)
    for i in range(u.shape[1]):
        total=total+running(x,u[:,i],p); x=step(x,u[:,i],p)
    return total+terminal(x,p)
