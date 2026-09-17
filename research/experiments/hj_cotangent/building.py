"""Heterogeneous multi-zone RC heating benchmark, with an exact convex QP.

This is a specified synthetic engineering model, not an identified real building.
Temperatures are represented relative to 22 Celsius; time is measured in hours.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class BuildingProblem:
    family: str = 'building'
    n: int = 32
    horizon: int = 96
    topology_seed: int = 52001
    weather_shift: float = 0.
    budget: float = 5.

    @property
    def dim(self): return self.n

    @property
    def dt(self): return .25

    @property
    def bound(self): return self.budget

    @property
    def mechanical(self): return False


def coefficients(p):
    gen = np.random.default_rng(p.topology_seed)
    capacitance = gen.uniform(2., 4., p.n)
    outside = gen.uniform(.10, .20, p.n)
    efficiency = gen.uniform(.85, 1.05, p.n)
    w = np.zeros((p.n, p.n))
    for i in range(p.n):
        j = (i+1) % p.n
        w[i,j] = w[j,i] = gen.uniform(.08, .16)
    # Extra undirected links remove ring homogeneity; keep bounded mean degree.
    for i in range(p.n):
        j = int(gen.integers(p.n))
        if i != j:
            w[i,j] = w[j,i] = gen.uniform(.03, .08)
    a = np.eye(p.n)+p.dt*(w-np.diag(w.sum(1)+outside))/capacitance[:,None]
    b = p.dt*efficiency/capacitance
    g = p.dt*outside/capacitance
    return dict(a=a, b=b, g=g, w=w, capacitance=capacitance,
                outside=outside, efficiency=efficiency)


class BuildingModel(nn.Module):
    def __init__(self, p, device='cuda'):
        super().__init__()
        self.p = p
        for name, value in coefficients(p).items():
            self.register_buffer(name, torch.tensor(value, device=device, dtype=torch.float64))
        # Exogenous schedules are known constants. All learning methods use
        # these tables, avoiding repeated trigonometric/exp work in long rollouts.
        for dtype,suffix in [(torch.float32,'32'),(torch.float64,'64')]:
            tt=torch.arange(p.horizon+1,device=device,dtype=dtype)*p.dt
            phase=2*math.pi*tt/24
            values=dict(weather=-11.+4.*torch.sin(phase-math.pi/2)+p.weather_shift,
                        price=.12+.38*torch.exp(-((torch.remainder(tt,24)-18)/3).square()),
                        comfort=.3+.7*torch.sigmoid((torch.remainder(tt,24)-7)*2)*torch.sigmoid((21-torch.remainder(tt,24))*2))
            for name,value in values.items():self.register_buffer(name+'_'+suffix,value)

    def schedules(self, t, dtype=torch.float32):
        index=t.long() if isinstance(t,torch.Tensor) else t
        suffix='64' if dtype==torch.float64 else '32'
        return tuple(getattr(self,name+'_'+suffix)[index] for name in ['weather','price','comfort'])

    def step(self, x, u, t):
        weather, _, _ = self.schedules(t, x.dtype)
        return x@self.a.to(x.dtype).mT + self.b.to(x.dtype)*u + self.g.to(x.dtype)*weather[...,None]

    def running(self, x, u, t):
        _, price, comfort = self.schedules(t, x.dtype)
        return self.p.dt*(comfort*x.square().mean(-1)+price*u.mean(-1)+.025*u.square().mean(-1))

    def terminal(self, x):
        return 2.*x.square().mean(-1)


class Thermostat(nn.Module):
    def __init__(self, p, device='cuda'):
        super().__init__()
        self.p = p
        self.model = BuildingModel(p, device)

    def forward(self, x, t):
        weather, _, _ = self.model.schedules(t, x.dtype)
        feedforward = -self.model.outside.to(x.dtype)/self.model.efficiency.to(x.dtype)*weather[...,None]
        return (feedforward-1.2*x).clamp(0, self.p.bound)


class BuildingActor(nn.Module):
    def __init__(self, p, base, width=64):
        super().__init__()
        self.p, self.base = p, base
        self.net = nn.Sequential(nn.Linear(15,width), nn.SiLU(), nn.Linear(width,width), nn.SiLU(), nn.Linear(width,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        original = x.dtype
        x = x.float()
        m, p = self.base.model, self.p
        u0 = self.base(x,t)
        weather, price, comfort = m.schedules(t)
        tt = (t.to(x.dtype) if isinstance(t,torch.Tensor) else torch.ones((),device=x.device,dtype=x.dtype)*t)*p.dt
        neighbor = (x@m.w.float().mT)/m.w.float().sum(-1).clamp_min(1e-8)
        pooled = x.mean(-1,keepdim=True).expand_as(x)
        expand = lambda v: torch.ones_like(x)*v[...,None]
        features = [x/5, x.square()/25, neighbor/5, pooled/5, u0/p.bound,
                    expand(weather/15), expand(price/.5), expand(comfort),
                    expand(torch.sin(2*math.pi*tt/24)), expand(torch.cos(2*math.pi*tt/24)),
                    expand(tt/(p.horizon*p.dt)), m.capacitance.float().expand_as(x)/4,
                    m.outside.float().expand_as(x)/.2, m.efficiency.float().expand_as(x), torch.ones_like(x)*p.bound/5]
        z = self.net(torch.stack(features,-1)).squeeze(-1)
        return (u0+p.bound*z.tanh()).clamp(0,p.bound).to(original)


def sample_building(p, count, seed, device='cuda', shift=False):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    x = 4.*(2*torch.rand(count,p.n,generator=gen)-1)
    x += 1.5*(2*torch.rand(count,1,generator=gen)-1)
    if shift: x *= 1.35
    return x.to(device)


def solve_qp(x0, p, eps=1e-7, return_workspace=False):
    """Sparse all-state/all-input convex QP, including exact affine disturbances."""
    import osqp
    import scipy.sparse as sp
    import time
    start = time.perf_counter()
    c = coefficients(p)
    n, h = p.n, p.horizon
    model = BuildingModel(p, 'cpu')
    times = torch.arange(h, dtype=torch.float64)
    weather, price, comfort = [a.numpy() for a in model.schedules(times, torch.float64)]
    qdiag = np.concatenate((np.repeat(2*p.dt*comfort/n,n), np.full(n,4/n), np.full(h*n,2*p.dt*.025/n)))
    linear = np.concatenate((np.zeros((h+1)*n), np.repeat(p.dt*price/n,n)))
    ax = sp.kron(sp.eye(h+1),-sp.eye(n))+sp.kron(sp.diags(np.ones(h),-1,shape=(h+1,h+1)),sp.csc_matrix(c['a']))
    bu = sp.kron(sp.vstack((sp.csc_matrix((1,h)),sp.eye(h))),sp.diags(c['b']))
    dynamics = sp.hstack((ax,bu),format='csc')
    controls = sp.hstack((sp.csc_matrix((h*n,(h+1)*n)),sp.eye(h*n)),format='csc')
    constraints = sp.vstack((dynamics,controls),format='csc')
    equality = np.concatenate((-np.asarray(x0,dtype=np.float64),-(weather[:,None]*c['g']).ravel()))
    lower = np.concatenate((equality,np.zeros(h*n)))
    upper = np.concatenate((equality,np.full(h*n,p.bound)))
    solver = osqp.OSQP()
    solver.setup(P=sp.diags(qdiag,format='csc'),q=linear,A=constraints,l=lower,u=upper,
                 verbose=False,eps_abs=eps,eps_rel=eps,max_iter=100000,polishing=True)
    setup = time.perf_counter()-start
    start = time.perf_counter()
    solution = solver.solve()
    elapsed = time.perf_counter()-start
    if solution.info.status_val != 1:
        raise RuntimeError(solution.info.status)
    u = solution.x[(h+1)*n:].reshape(h,n)
    # Return feasible clipped controls, independently replayed below.
    feasible = u.clip(0,p.bound)
    states = [np.asarray(x0,dtype=np.float64)]
    cost = 0.
    for t in range(h):
        x = states[-1]
        cost += p.dt*(comfort[t]*np.mean(x*x)+price[t]*np.mean(feasible[t])+.025*np.mean(feasible[t]**2))
        states.append(c['a']@x+c['b']*feasible[t]+c['g']*weather[t])
    cost += 2*np.mean(states[-1]**2)
    result=dict(cost=float(cost), u=feasible, x=np.stack(states), qp_objective=float(solution.info.obj_val),
                objective_replay_error=float(abs(cost-solution.info.obj_val)), setup_seconds=setup, solve_seconds=elapsed,
                iterations=int(solution.info.iter), primal_residual=float(solution.info.prim_res),
                dual_residual=float(solution.info.dual_res), status=solution.info.status)
    if return_workspace:
        result['_workspace']=dict(solver=solver,lower=lower,upper=upper)
    return result
