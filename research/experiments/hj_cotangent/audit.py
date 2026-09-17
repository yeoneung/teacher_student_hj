"""Gradient identity, cache computation and numerical safety checks."""
import copy
import gc
from pathlib import Path
import torch
from .systems import problem, teacher as make_teacher, actor as make_actor, sample, rollout
from experiments.hj_gridfree.engine import write
from .core import prefix, local_return, JetGraph, LocalGraph, continuation


def grad(value, actor):
    return torch.cat([g.flatten() for g in torch.autograd.grad(value.sum(), actor.net.parameters())])


def main():
    torch.set_num_threads(1)
    report = {}
    for family in ['mechanical', 'reaction', 'building']:
        p = problem(family, 4, horizon=12)
        torch.manual_seed(67001)
        actor = make_actor(p, make_teacher(p), 64).cuda()
        with torch.no_grad():
            actor.net[-1].weight.normal_(0, .03)
        teacher = make_teacher(p)
        x = sample(p, 8, 67002)*.3
        t = torch.tensor([0, 1, 3, 5, 7, 8, 10, 11], device='cuda')
        length = 4
        with torch.no_grad():
            _, anchor, end = prefix(x, t, actor, p, length)
        anchor = anchor.detach().requires_grad_(True)
        value = rollout(anchor, teacher, p, end)
        covector = torch.autograd.grad(value.sum(), anchor)[0]
        exact = continuation(x, t, actor, teacher, p, length)
        approximation = local_return(x, t, actor, p, length, anchor.detach(), value.detach(), covector, 1.)
        forward_error = (exact-approximation).abs().max().item()
        ge = grad(exact, actor)
        ga = grad(approximation, actor)
        gradient_error = (ge-ga).abs().max().item()
        assert forward_error < 1e-5 and gradient_error < 1e-4, (forward_error, gradient_error)
        # Capture fresh module leaves, independently compare returned labels.
        state = copy.deepcopy(actor.state_dict())
        actor = make_actor(p, make_teacher(p), 64).cuda()
        actor.load_state_dict(state)
        jet = JetGraph(actor, teacher, p, len(x), length)
        aa, vv, pp = jet(x, t)
        torch.cuda.synchronize()
        jet_error = max((aa-anchor.detach()).abs().max().item(), (vv-value.detach()).abs().max().item(),
                        (pp-covector).abs().max().item())
        assert jet_error < 1e-5, jet_error
        trainer = LocalGraph(actor, p, len(x), length, rho=1.)
        reset_error = max((v-state[k]).abs().max().item() for k,v in actor.state_dict().items())
        assert reset_error == 0.
        for _ in range(4):
            trainer.update(x, t, aa, vv, pp)
        torch.cuda.synchronize()
        assert all(torch.isfinite(v).all().item() for v in actor.state_dict().values())
        report[family] = dict(anchor_value_error=forward_error, anchor_parameter_gradient_error=gradient_error,
                              captured_jet_error=jet_error, warmup_actor_reset_error=reset_error,
                              initial_gradient_norm=ge.norm().item())
        del actor, teacher, jet, trainer
        gc.collect()
        torch.cuda.empty_cache()
    write('experiments/results/hj_cotangent_dev_v6b/numerical_audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
