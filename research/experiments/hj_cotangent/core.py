"""Short student prefixes and local teacher value jets (no fitted critic)."""
import copy
import torch
from .systems import running, terminal, step, rollout
from experiments.hj_gridfree.engine import capture


def prefix(x, t, actor, p, length):
    total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(length):
        active = t + i < p.horizon
        tt = (t + i).clamp_max(p.horizon - 1)
        u = actor(x, tt)
        total = total + active * running(x, u, p, tt)
        x = torch.where(active[:, None], step(x, u, p, tt), x)
    return total, x, (t + length).clamp_max(p.horizon)


def local_return(x, t, actor, p, length, anchor, value, covector, rho=0., gradient_covector=None):
    cost, z, end = prefix(x, t, actor, p, length)
    displacement = z - anchor
    linear = (covector * displacement).sum(-1)
    if gradient_covector is not None:
        alternative = (gradient_covector*displacement).sum(-1)
        linear = linear.detach()+alternative-alternative.detach()
    approximate = value + linear
    approximate = approximate + .5 * rho * displacement.square().mean(-1)
    # The known terminal function stays exact, including its derivative.
    tail = torch.where(end < p.horizon, approximate, terminal(z, p))
    return cost + tail


class JetGraph:
    """Exact value/covector at current actor boundaries; teacher weights frozen."""
    def __init__(self, actor, teacher, p, batch, length):
        self.actor, self.teacher, self.p = actor, teacher, p
        self.x = torch.zeros(batch, p.dim, device='cuda')
        self.t = torch.zeros(batch, device='cuda', dtype=torch.long)
        self.anchor = torch.zeros_like(self.x, requires_grad=True)
        self.anchor.grad = torch.zeros_like(self.anchor)
        self.value = torch.zeros(batch, device='cuda')
        self.end = torch.zeros_like(self.t)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
            param.grad = None
        def evaluate():
            with torch.no_grad():
                _, z, end = prefix(self.x, self.t, actor, p, length)
                self.anchor.copy_(z)
                self.end.copy_(end)
            self.anchor.grad.zero_()
            value = rollout(self.anchor, teacher, p, self.end)
            value.sum().backward()
            with torch.no_grad():
                self.value.copy_(value)
        self.graph = capture(evaluate)

    def __call__(self, x, t):
        self.x.copy_(x)
        self.t.copy_(t)
        self.graph.replay()
        return (self.anchor.detach().clone(), self.value.clone(),
                self.anchor.grad.detach().clone())


class LocalGraph:
    def __init__(self, actor, p, batch, length, rho=0., lr=.002, mode='cache', teacher=None, signal='normal'):
        self.actor, self.p = actor, p
        self.x = torch.zeros(batch, p.dim, device='cuda')
        self.t = torch.zeros(batch, device='cuda', dtype=torch.long)
        self.anchor = torch.zeros_like(self.x)
        self.value = torch.zeros(batch, device='cuda')
        self.covector = torch.zeros_like(self.x)
        self.signal_covector = torch.zeros_like(self.x)
        self.curvature = torch.ones(batch, device='cuda')
        self.loss = torch.zeros((), device='cuda')
        self.grad_norm = torch.zeros((), device='cuda',dtype=torch.float64)
        self.clipped_updates = torch.zeros((), device='cuda')
        self.nonfinite_updates = torch.zeros((), device='cuda')
        params = list(actor.net.parameters())
        initial = copy.deepcopy(actor.state_dict())
        self.m = [torch.zeros_like(z) for z in params]
        self.v = [torch.zeros_like(z) for z in params]
        self.b1 = torch.ones((), device='cuda')
        self.b2 = torch.ones((), device='cuda')
        for z in params:
            z.grad = torch.zeros_like(z)
        def update():
            for z in params:
                z.grad.zero_()
            if mode == 'cache':
                loss = local_return(self.x, self.t, actor, p, length,
                                    self.anchor, self.value, self.covector, rho*self.curvature,
                                    gradient_covector=self.signal_covector if signal!='normal' else None).mean()
            elif mode == 'dpc':
                loss = rollout(self.x, actor, p, self.t).mean()
            elif mode == 'tbptt':
                loss = truncated_backprop(self.x,self.t,actor,p,length).mean()
            elif mode == 'short':
                cost, z, _ = prefix(self.x, self.t, actor, p, length)
                loss = (cost+terminal(z,p)).mean()
            elif mode == 'exact':
                loss = continuation(self.x, self.t, actor, teacher, p, length).mean()
            elif mode == 'bc':
                loss = (actor(self.x,self.t)-teacher(self.x,self.t)).square().mean()/p.bound**2
            else:
                raise ValueError(mode)
            loss.backward()
            with torch.no_grad():
                self.loss.copy_(loss)
                # Avoid overflow when squaring finite long-horizon gradients.
                # All learning objectives use this same clipping implementation.
                norm = torch.stack([z.grad.double().square().sum() for z in params]).sum().sqrt()
                finite=torch.isfinite(norm)&torch.isfinite(loss)
                self.grad_norm.copy_(norm)
                self.clipped_updates.add_((finite&(norm>10)).to(self.clipped_updates.dtype))
                self.nonfinite_updates.add_((~finite).to(self.nonfinite_updates.dtype))
                beta1=torch.where(finite,.9,1.)
                beta2=torch.where(finite,.999,1.)
                self.b1.mul_(beta1)
                self.b2.mul_(beta2)
                scale = (10 / norm.clamp_min(1e-12)).clamp_max(1)
                for z, m, v in zip(params, self.m, self.v):
                    g = torch.where(finite,z.grad*scale,torch.zeros_like(z))
                    m.mul_(beta1).add_(g, alpha=.1)
                    v.mul_(beta2).addcmul_(g, g, value=.001)
                    delta=-lr*(m/(1-self.b1).clamp_min(1e-12))/((v/(1-self.b2).clamp_min(1e-12)).sqrt()+1e-8)
                    z.add_(torch.where(finite,delta,torch.zeros_like(z)))
        self.graph = capture(update)
        actor.load_state_dict(initial)
        for z in self.m + self.v:
            z.zero_()
        self.b1.fill_(1)
        self.b2.fill_(1)
        self.grad_norm.zero_()
        self.clipped_updates.zero_()
        self.nonfinite_updates.zero_()

    def update(self, x, t, anchor=None, value=None, covector=None, curvature=None, signal_covector=None):
        self.x.copy_(x)
        self.t.copy_(t)
        if anchor is not None:
            self.anchor.copy_(anchor)
            self.value.copy_(value)
            self.covector.copy_(covector)
        if curvature is not None:
            self.curvature.copy_(curvature)
        if signal_covector is not None:
            self.signal_covector.copy_(signal_covector)
        self.graph.replay()


def continuation(x, t, student, teacher, p, length):
    total = torch.zeros(x.shape[0],device=x.device,dtype=x.dtype)
    for i in range(p.horizon):
        active = t+i < p.horizon
        tt = (t+i).clamp_max(p.horizon-1)
        u = student(x,tt) if i<length else teacher(x,tt)
        total = total+active*running(x,u,p,tt)
        x = torch.where(active[:,None],step(x,u,p,tt),x)
    return total+terminal(x,p)


def truncated_backprop(x,t,actor,p,length):
    """Full student forward return, with state gradients cut every L steps."""
    total = torch.zeros(x.shape[0],device=x.device,dtype=x.dtype)
    for i in range(p.horizon):
        if i and i % length == 0:
            # Finished samples retain the terminal derivative; masked padding
            # is not an additional physical block of truncated backpropagation.
            x = torch.where((t+i<p.horizon)[:,None],x.detach(),x)
        active = t+i < p.horizon
        tt = (t+i).clamp_max(p.horizon-1)
        u = actor(x,tt)
        total = total+active*running(x,u,p,tt)
        x = torch.where(active[:,None],step(x,u,p,tt),x)
    return total+terminal(x,p)


class EvalGraph:
    def __init__(self, policy, p, batch, dtype=torch.float32):
        self.x = torch.zeros(batch,p.dim,device='cuda',dtype=dtype)
        self.cost = torch.zeros(batch,device='cuda',dtype=dtype)
        def evaluate():
            with torch.no_grad():
                self.cost.copy_(rollout(self.x,policy,p))
        self.graph = capture(evaluate)

    def __call__(self, x):
        self.x.copy_(x)
        self.graph.replay()
        return self.cost.clone()
