"""Check the full-gradient teacher interface at the selected trained policies."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import torch
from experiments.hj_cotangent.systems import problem,teacher,actor,sample,rollout
from experiments.hj_gridfree.engine import write
from .trajectory import trajectory_covectors,advantage
from .study import verify


def gradients(net,initial,p):
    frozen=copy.deepcopy(net).eval()
    for z in frozen.parameters():z.requires_grad_(False);z.grad=None
    cost,states,controls,covectors=trajectory_covectors(initial.clone().requires_grad_(True),frozen,p)
    states,controls,covectors=states.detach(),controls.detach(),covectors.detach()
    params=list(net.net.parameters())
    full=torch.cat([g.reshape(-1) for g in torch.autograd.grad(rollout(initial,net,p).mean(),params)]).double()
    local=0.
    for t in range(p.horizon):
        tt=torch.full((len(initial),),t,device='cuda',dtype=torch.long)
        local=local+advantage(states[:,t].contiguous(),tt,controls[:,t],states[:,t+1],covectors[:,t],net,p).mean()
    surrogate=torch.cat([g.reshape(-1) for g in torch.autograd.grad(local,params)]).double()
    finite=bool(torch.isfinite(full).all() and torch.isfinite(surrogate).all())
    error=((full-surrogate).norm()/full.norm().clamp_min(1e-12)).item() if finite else None
    return dict(finite=finite,full_norm=full.norm().item() if finite else None,
        relative_gradient_error=error,anchor_advantage_absolute_value=abs(local.item()) if math.isfinite(local.item()) else None,
        agreement_within_audit_tolerance=finite and error<2e-4,
        training_mean=cost.mean().item() if torch.isfinite(cost).all() else None),dict(full_gradient=full.cpu(),surrogate_gradient=surrogate.cpu())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    torch.set_num_threads(1);root=Path(args.root);cfg=verify(root);out=root/'transport_diagnostics';out.mkdir(exist_ok=True)
    for task in cfg['tasks']:
        if (out/(task['family']+'.json')).exists():continue
        p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
        rows=[];vectors=[]
        for seed in cfg['seeds']:
            initial=sample(p,cfg['training_initial_states'],110000+seed)[:cfg['batch']]
            folder=root/'training'/f'{p.family}_h{p.horizon}_s{seed}'
            for spec in task['methods']:
                if spec['mode']!='transport':continue
                net=actor(p,teacher(p),cfg['width']).cuda()
                saved=torch.load(folder/(spec['name']+'.pt'),map_location='cuda',weights_only=False)['budgets']['60']
                net.load_state_dict(saved['state_dict']);r,v=gradients(net,initial,p)
                row=dict(family=p.family,seed=seed,method=spec['name'],**r,
                    scope='Reevaluate this selected policy as its own teacher on the first128 of the actual512 training initial states. Float32 training model, contiguous replay; no test or checkpoint selection.')
                rows.append(row);vectors.append(dict(seed=seed,method=spec['name'],initial=initial.cpu(),**v))
                print(json.dumps(row),flush=True)
                del net,saved;gc.collect();torch.cuda.empty_cache()
        write(out/(p.family+'.json'),rows);torch.save(vectors,out/(p.family+'.pt'))


if __name__=='__main__':main()
