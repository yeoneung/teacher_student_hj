"""Check the HJB teacher interface against full policy differentiation."""
import argparse
import copy
import json
import torch
from experiments.hj_cotangent.systems import problem,actor,sample,rollout
from experiments.hj_cotangent.building import Thermostat
from experiments.hj_gridfree.core import LQR
from experiments.hj_gridfree.engine import write
from .trajectory import trajectory_covectors,advantage,TrajectoryGraph


def check(family,horizon,device,perturbed=False):
    p=problem(family,4 if device=='cpu' else (16 if family=='mechanical' else 32),horizon)
    torch.manual_seed(99801)
    base=Thermostat(p,device) if family=='building' else LQR(p,device)
    student=actor(p,base,16 if device=='cpu' else 64).to(device)
    if perturbed:
        with torch.no_grad():student.net[-1].weight.normal_(0,.01);student.net[-1].bias.fill_(.003)
    teaching=copy.deepcopy(student).eval()
    for parameter in teaching.parameters():parameter.requires_grad_(False)
    batch=4 if device=='cpu' else 128
    initial=sample(p,batch,99821,device=device).requires_grad_(True)
    cost,states,controls,covectors=trajectory_covectors(initial,teaching,p)
    states,controls,covectors=states.detach(),controls.detach(),covectors.detach()
    reference_errors=[]
    for t in [0,horizon//2,horizon-1]:
        z=states[:,t+1].clone().requires_grad_(True)
        value=rollout(z,teaching,p,t+1)
        direct=torch.autograd.grad(value.sum(),z)[0]
        reference_errors.append(((direct-covectors[:,t]).norm()/direct.norm().clamp_min(1e-12)).item())
    params=list(student.net.parameters())
    full=torch.cat([g.reshape(-1) for g in torch.autograd.grad(rollout(initial.detach(),student,p).mean(),params)]).double()
    # Preserve batch shape between the two implementations for this identity check.
    surrogate=0.
    for t in range(horizon):
        times=torch.full((batch,),t,device=device,dtype=torch.long)
        surrogate=surrogate+advantage(states[:,t].contiguous(),times,controls[:,t],states[:,t+1],covectors[:,t],student,p).mean()
    local=torch.cat([g.reshape(-1) for g in torch.autograd.grad(surrogate,params)]).double()
    discrepancy=((full-local).norm()/full.norm().clamp_min(1e-12)).item()
    value_error=abs(surrogate.item())
    print(json.dumps(dict(family=family,reference_errors=reference_errors,full_norm=full.norm().item(),local_norm=local.norm().item(),difference_norm=(full-local).norm().item(),relative=discrepancy,value_error=value_error)),flush=True)
    assert max(reference_errors)<5e-5 and discrepancy<2e-4 and value_error<1e-4,(family,reference_errors,discrepancy,value_error)
    graph_error=None
    if device=='cuda':
        graph=TrajectoryGraph(teaching,p,batch);gc,gx,gu,gp=graph(initial.detach())
        graph_error=max((a-b).abs().max().item() for a,b in zip([gc,gx,gu,gp],[cost.detach(),states,controls,covectors]))
        assert graph_error<1e-4,(family,graph_error)
        del graph
    return dict(family=family,horizon=horizon,device=device,perturbed=perturbed,initial_vectors=batch,
        individual_continuation_covector_relative_errors=reference_errors,
        full_policy_gradient_relative_error=discrepancy,anchor_advantage_absolute_value=value_error,
        graph_eager_max_absolute_difference=graph_error)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    tasks=[('mechanical',12),('reaction',12),('building',12)] if args.device=='cpu' else [('mechanical',160),('reaction',320),('building',192)]
    rows=[check(f,h,args.device,perturbed) for perturbed in [False,True] for f,h in tasks]
    result=dict(passed=True,rows=rows,scope='Development-only verification of all-time teacher covectors and the full parameter-gradient identity at teacher/student equality. These are not fresh primary states or a convergence test.')
    write(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
