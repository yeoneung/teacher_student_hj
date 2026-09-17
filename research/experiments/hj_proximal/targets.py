"""Action-space minimization and saturation-aware regression, no fitted critic."""
import copy
import torch
from experiments.hj_gridfree.core import project
from experiments.hj_gridfree.engine import capture
from experiments.hj_cotangent.systems import model


def coefficients(p, t, covector):
    if p.family == 'building':
        m = model(p, covector.device)
        _, price, _ = m.schedules(t, covector.dtype)
        r = p.dt * .025 / p.n
        b = covector * m.b.to(covector.dtype) + p.dt * price[:, None] / p.n
    elif p.mechanical:
        r = p.dt * .04 / p.n
        b = p.dt**2 * covector[:, :p.n] + p.dt * covector[:, p.n:]
    else:
        r = p.dt * .1 / p.n
        b = p.dt * covector
    return r, b


def feasible(u, p):
    return u.clamp(0, p.bound) if p.family == 'building' else project(u, p)


def target_action(old_u, t, covector, p, regularization):
    r, b = coefficients(p, t, covector)
    rho = regularization * 2 * r
    return feasible((rho * old_u - b) / (2 * r + rho), p)


def model_difference(u, old_u, t, covector, p):
    r, b = coefficients(p, t, covector)
    return (r * (u.square() - old_u.square()) + b * (u-old_u)).sum(-1)


def expose(actor, x, t):
    """Expose the unchanged actor's net input and analytic base via a local hook.

    The hook runs only in eager feature collection, never in a captured update.
    Actor features contain no learned parameters. Their cached values therefore
    remain valid for fitting all students on the current teacher-state pool.
    """
    inputs = []
    handle = actor.net.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
    try:
        with torch.no_grad():
            deployed = actor(x.contiguous(), t)
            base = actor.base(x.float().contiguous(), t)
    finally:
        handle.remove()
    assert len(inputs) == 1
    return inputs[0].detach(), base.detach(), deployed.detach()


def reconstruct(net, features, base, p):
    scale = p.bound if p.family == 'building' else 2 * p.bound
    return feasible(base + scale * net(features).squeeze(-1).tanh(), p)


class RegressionGraph:
    def __init__(self, actor, p, batch, lr, kind):
        self.actor, self.p, self.kind = actor, p, kind
        features = 15 if p.family == 'building' else (19 if p.mechanical else 13)
        self.features = torch.zeros(batch, p.n, features, device='cuda')
        self.base = torch.zeros(batch, p.n, device='cuda')
        self.target = torch.zeros_like(self.base)
        self.loss = torch.zeros((), device='cuda')
        self.grad_norm = torch.zeros((), device='cuda', dtype=torch.float64)
        self.nonfinite_updates = torch.zeros((), device='cuda')
        self.clipped_updates = torch.zeros((), device='cuda')
        params = list(actor.net.parameters())
        initial = copy.deepcopy(actor.state_dict())
        self.m = [torch.zeros_like(z) for z in params]
        self.v = [torch.zeros_like(z) for z in params]
        self.b1 = torch.ones((), device='cuda'); self.b2 = torch.ones_like(self.b1)
        for z in params: z.grad = torch.zeros_like(z)
        def update():
            for z in params: z.grad.zero_()
            if kind == 'latent':
                loss = (actor.net(self.features).squeeze(-1)-self.target).square().mean()
            else:
                loss = ((reconstruct(actor.net, self.features, self.base, p)-self.target)/p.bound).square().mean()
            loss.backward()
            with torch.no_grad():
                self.loss.copy_(loss)
                norm = torch.stack([z.grad.double().square().sum() for z in params]).sum().sqrt()
                finite = torch.isfinite(norm) & torch.isfinite(loss)
                self.grad_norm.copy_(norm)
                self.nonfinite_updates.add_((~finite).float())
                self.clipped_updates.add_((finite & (norm > 10)).float())
                beta1=torch.where(finite,.9,1.); beta2=torch.where(finite,.999,1.)
                self.b1.mul_(beta1); self.b2.mul_(beta2)
                scale=(10/norm.clamp_min(1e-12)).clamp_max(1)
                for z,m,v in zip(params,self.m,self.v):
                    g=torch.where(finite,z.grad*scale,torch.zeros_like(z))
                    m.mul_(beta1).add_(g,alpha=.1);v.mul_(beta2).addcmul_(g,g,value=.001)
                    delta=-lr*(m/(1-self.b1).clamp_min(1e-12))/((v/(1-self.b2).clamp_min(1e-12)).sqrt()+1e-8)
                    z.add_(torch.where(finite,delta,torch.zeros_like(z)))
        self.graph = capture(update)
        actor.load_state_dict(initial)
        self.reset_optimizer()
        self.nonfinite_updates.zero_(); self.clipped_updates.zero_()

    def reset_optimizer(self):
        for v in self.m+self.v: v.zero_()
        self.b1.fill_(1); self.b2.fill_(1)

    def update(self, features, base, target):
        self.features.copy_(features); self.base.copy_(base); self.target.copy_(target)
        self.graph.replay()


@torch.no_grad()
def teaching_data(actor, cache, p, regularization, kind, chunk=2048):
    xs, ts, old_u, anchors, covectors = cache
    features, bases = [], []
    for start in range(0, len(xs), chunk):
        f, b, _ = expose(actor, xs[start:start+chunk], ts[start:start+chunk])
        features.append(f); bases.append(b)
    features, base = torch.cat(features), torch.cat(bases)
    target = target_action(old_u, ts, covectors, p, regularization)
    scale = p.bound if p.family == 'building' else 2*p.bound
    correction = (target-base)/scale
    clipping = (correction.abs() > .995).float().mean().item()
    labels = correction.clamp(-.995,.995).atanh() if kind == 'latent' else target
    return (features, base, labels), dict(target_clipped_fraction=clipping,
        target_model_reduction=float(-p.horizon*model_difference(target,old_u,ts,covectors,p).mean()),
        target_action_drift=float(((target-old_u)/p.bound).square().mean().sqrt()))


@torch.no_grad()
def predicted_reduction(actor, data, cache, p, chunk=2048):
    total = torch.zeros((), device='cuda', dtype=torch.float64)
    for start in range(0, len(cache[0]), chunk):
        end = start+chunk
        u = reconstruct(actor.net, data[0][start:end], data[1][start:end], p)
        delta = model_difference(u, cache[2][start:end], cache[1][start:end], cache[4][start:end], p)
        total.add_(-delta.double().sum())
    return float(p.horizon * total / len(cache[0]))
