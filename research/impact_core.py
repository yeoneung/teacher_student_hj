"""Exact teacher continuation and error decomposition on student states."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.hj_gridfree.engine import capture
from experiments.hj_cotangent.systems import problem, teacher, actor, sample, rollout, step, running
from experiments.hj_proximal.targets import coefficients, feasible

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
PROJECT = SUB.parent


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(p, seed, method, budget, root='hj_target_primary_v9'):
    folder = PROJECT / 'experiments/results' / root / 'training' / f'{p.family}_d{p.dim}_h{p.horizon}_s{seed}'
    path = folder / (method + '.pt')
    record = torch.load(path, map_location='cuda', weights_only=False)['budgets'][str(float(budget))]
    policy = actor(p, teacher(p), 64).cuda().eval()
    policy.load_state_dict(record['state_dict'])
    return policy, dict(path=str(path.relative_to(PROJECT)), sha256=digest(path), budget=budget,
                        validation_mean=record['validation_mean'])


class ValueGraph:
    def __init__(self, policy, p, batch=32, gradient=False):
        self.batch, self.gradient = batch, gradient
        for z in policy.parameters(): z.requires_grad_(False)
        self.x = torch.zeros(batch, p.dim, device='cuda', dtype=torch.float64, requires_grad=gradient)
        self.t = torch.zeros(batch, device='cuda', dtype=torch.long)
        self.value = torch.zeros(batch, device='cuda', dtype=torch.float64)
        self.covector = torch.zeros_like(self.x)
        def evaluate():
            if gradient:
                value = rollout(self.x, policy, p, self.t)
                covector = torch.autograd.grad(value.sum(), self.x)[0]
            else:
                with torch.no_grad(): value = rollout(self.x, policy, p, self.t)
            with torch.no_grad():
                self.value.copy_(value)
                if gradient: self.covector.copy_(covector)
        self.graph = capture(evaluate)

    def __call__(self, x, t):
        values, covectors = [], []
        for offset in range(0, len(x), self.batch):
            count = min(self.batch, len(x)-offset)
            with torch.no_grad():
                self.x[:count].copy_(x[offset:offset+count]); self.t[:count].copy_(t[offset:offset+count])
                if count < self.batch:
                    self.x[count:].copy_(x[offset]); self.t[count:].fill_(int(t[offset]))
            self.graph.replay()
            values.append(self.value[:count].clone())
            if self.gradient: covectors.append(self.covector[:count].clone())
        return torch.cat(values), (torch.cat(covectors) if self.gradient else None)


@torch.no_grad()
def perturb_signal(b, r, p, magnitude, kind, generator):
    scale = b.square().mean(-1, keepdim=True).sqrt().clamp_min(2*r*p.bound)
    if magnitude == 0: delta = torch.zeros_like(b)
    elif kind == 'bias': delta = magnitude*scale*torch.ones_like(b)
    else:
        # Zero mean in the generating distribution; realized errors are retained.
        delta = magnitude*scale*torch.randn(b.shape, generator=generator, device=b.device, dtype=b.dtype)
    return b + delta, delta


def exact_probe(teacher_policy, student_policy, p, x, t, jet, value, eta=16., magnitude=0., kind='none', seed=0, cached_b=None):
    x = x.double()
    with torch.no_grad():
        old = teacher_policy(x, t).double(); u = student_policy(x, t).double()
        y0 = step(x, old, p, t)
    v0, cov = jet(y0, (t+1).clamp_max(p.horizon))
    with torch.no_grad():
        r, true_b = coefficients(p, t, cov)
        gen = torch.Generator(device='cuda').manual_seed(seed)
        estimated, noise = perturb_signal(true_b if cached_b is None else cached_b, r, p, magnitude, kind, gen)
        rho = eta*2*r
        target = feasible((rho*old-estimated)/(2*r+rho), p)
        d, e, displacement = target-old, u-target, u-old
        q = lambda z: (r*z.square()+estimated*z).sum(-1)
        qtrue = lambda z: (r*z.square()+true_b*z).sum(-1)
        target_model = q(target)-q(old); fitted_model = q(u)-q(old)
        target_true_model = qtrue(target)-qtrue(old)
        fitted_true_model = qtrue(u)-qtrue(old)
    vu, _ = value(step(x, u, p, t), (t+1).clamp_max(p.horizon))
    vt, _ = value(step(x, target, p, t), (t+1).clamp_max(p.horizon))
    with torch.no_grad():
        actual = running(x,u,p,t)-running(x,old,p,t)+vu-v0
        actual_target = running(x,target,p,t)-running(x,old,p,t)+vt-v0
        noise_term = ((true_b-estimated)*displacement).sum(-1)
        curvature = actual-fitted_true_model
        target_curvature = actual_target-target_true_model
        eps = (estimated-true_b).norm(dim=-1)
        dnorm, enorm, unorm = d.norm(dim=-1), e.norm(dim=-1), displacement.norm(dim=-1)
        kappa = torch.maximum(2*curvature.clamp_min(0)/unorm.square().clamp_min(1e-20),
                              2*target_curvature.clamp_min(0)/dnorm.square().clamp_min(1e-20))
        fitted_bound = -(r+rho)*dnorm.square()+(2*r*target+estimated).norm(dim=-1)*enorm+r*enorm.square()+eps*unorm+.5*kappa*unorm.square()
        # This checks the algebra with realized directional remainders, not prediction.
        assert float((actual-fitted_bound).max()) < 2e-5
        assert float((actual-(target_model+(fitted_model-target_model)+noise_term+curvature)).abs().max()) < 1e-9
    return {k:v.detach() for k,v in dict(actual=actual, actual_target=actual_target,
        target_model=target_model, fitted_model=fitted_model, fitting_term=fitted_model-target_model,
        noise_term=noise_term, curvature=curvature, target_curvature=target_curvature,
        sensitivity_error=eps, dnorm=dnorm, enorm=enorm, unorm=unorm, kappa_observed=kappa,
        fitted_bound_realized=fitted_bound, true_b=true_b, estimated_b=estimated,
        teacher_action=old, student_action=u, target_action=target).items()}


def diagnostic():
    torch.set_num_threads(1)
    out = HERE / 'results/exact_diagnostic'; out.mkdir(parents=True, exist_ok=True)
    config = dict(seed=391001, initials=2, teacher_seed=9501, families=['mechanical','reaction'],
                  eta=16., scope='Development: complete finite student trajectories; no population certificate.')
    write(out/'plan_lock.json', dict(config=config, source_sha256=digest(__file__)))
    rows=[]
    for fi,family in enumerate(config['families']):
        p=problem(family,16 if family=='mechanical' else 32,160 if family=='mechanical' else 320)
        parent, provenance=load_policy(p,9501,'parent',7.5)
        student, sp=load_policy(p,9501,'hjb_target',30.)
        start=time.perf_counter(); jet=ValueGraph(parent,p,32,True); value=ValueGraph(parent,p,32,False)
        initial=sample(p,2,config['seed']+fi).double()
        with torch.no_grad():
            cost,xs,us=rollout(initial,student,p,keep=True); base=rollout(initial,parent,p)
            _,tx,_=rollout(initial,parent,p,keep=True)
            x=xs.reshape(-1,p.dim); t=torch.arange(p.horizon,device='cuda').repeat(2)
        for mode in ['fresh','cached']:
            cached=None
            if mode=='cached':
                with torch.no_grad(): old=parent(tx.reshape(-1,p.dim),t).double()
                _,tp=jet(step(tx.reshape(-1,p.dim),old,p,t),(t+1).clamp_max(p.horizon))
                cached=coefficients(p,t,tp)[1]
            raw=exact_probe(parent,student,p,x,t,jet,value,cached_b=cached)
            difference=float((cost-base).mean()); summation=float(raw['actual'].reshape(2,p.horizon).sum(1).mean())
            error=abs(difference-summation); assert error < 1e-5
            prediction=raw['fitted_model'] < -1e-8
            row=dict(family=family,mode=mode,teacher=provenance,student=sp,
                teacher_cost=float(base.mean()),student_cost=float(cost.mean()),cost_difference=difference,
                sum_teacher_advantages=summation,telescoping_error=error,points=len(x),
                false_positive_predictions=int((prediction & (raw['actual']>1e-8)).sum()),
                predicted_improvements=int(prediction.sum()),
                positive_curvature_fraction=float((raw['curvature']>1e-8).double().mean()),
                median_fit_ratio=float((raw['enorm']/raw['dnorm'].clamp_min(1e-10)).median()),
                kappa_quantiles=torch.quantile(raw['kappa_observed'],torch.tensor([.5,.9,.95,.99],device='cuda',dtype=torch.float64)).tolist(),
                mean_terms={k:float(raw[k].mean()) for k in ['actual','target_model','fitting_term','noise_term','curvature']})
            torch.save(dict(x=x.cpu(),t=t.cpu(),raw={k:v.cpu() for k,v in raw.items()}),out/f'{family}_{mode}.pt')
            rows.append(row); print(json.dumps(row),flush=True)
        del jet,value,parent,student; torch.cuda.empty_cache()
    write(out/'report.json',dict(rows=rows,passed=True,scope=config['scope']))


if __name__=='__main__': diagnostic()
