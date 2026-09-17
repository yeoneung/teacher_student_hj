"""Common interfaces; existing archived ring equations are imported unchanged."""
import torch
from experiments.hj_gridfree import core as rings
from .building import BuildingProblem, BuildingModel, Thermostat, BuildingActor, sample_building


_models = {}


def model(p, device='cuda'):
    key = (p, str(device))
    if key not in _models:
        _models[key] = BuildingModel(p, device)
    return _models[key]


def problem(family, n, horizon, **kwargs):
    return BuildingProblem(family=family, n=n, horizon=horizon, **kwargs) if family=='building' else rings.Problem(family, n, horizon=horizon, **kwargs)


def teacher(p):
    return Thermostat(p) if p.family=='building' else rings.LQR(p)


def actor(p, base, width):
    return BuildingActor(p, base, width) if p.family=='building' else rings.Actor(p, base, width)


def sample(p, count, seed, device='cuda', shift=False):
    return (sample_building if p.family=='building' else rings.sample)(p,count,seed,device,shift)


def running(x, u, p, t=0):
    return model(p,x.device).running(x,u,t) if p.family=='building' else rings.running(x,u,p)


def step(x, u, p, t=0):
    return model(p,x.device).step(x,u,t) if p.family=='building' else rings.step(x,u,p)


def terminal(x, p):
    return model(p,x.device).terminal(x) if p.family=='building' else rings.terminal(x,p)


def rollout(x, policy, p, t0=0, keep=False):
    scalar = isinstance(t0,int)
    total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    xs,us = [],[]
    for i in range(p.horizon-t0 if scalar else p.horizon):
        t = t0+i if scalar else (t0+i).clamp_max(p.horizon-1)
        u = policy(x,t)
        cost, y = running(x,u,p,t), step(x,u,p,t)
        if not scalar:
            active = t0+i < p.horizon
            cost = active*cost
            y = torch.where(active[:,None],y,x)
        if keep:
            xs.append(x)
            us.append(u)
        total = total+cost
        x = y
    total = total+terminal(x,p)
    return (total,torch.stack(xs,1),torch.stack(us,1)) if keep else total
