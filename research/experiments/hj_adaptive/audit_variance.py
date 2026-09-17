"""Verify teacher-anchor control variates and captured optimization."""
import argparse
import copy
import gc
import json
import torch
from experiments.hj_cotangent.systems import problem,actor,sample,rollout
from experiments.hj_cotangent.building import Thermostat
from experiments.hj_gridfree.core import LQR
from experiments.hj_gridfree.engine import write
from .trajectory import trajectory_covectors,advantage,TrajectoryGraph
from .transport import TransportGraph,collect


def vector_grad(loss,net):return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,list(net.net.parameters()))])


def check(family,horizon,device):
    p=problem(family,(16 if family=='mechanical' else 32) if device=='cuda' else 4,horizon)
    batch=128 if device=='cuda' else 16;mini=1024 if device=='cuda' else 64
    torch.manual_seed(99981);base=Thermostat(p,device) if family=='building' else LQR(p,device)
    net=actor(p,base,64 if device=='cuda' else 16).to(device)
    with torch.no_grad():net.net[-1].weight.normal_(0,.01);net.net[-1].bias.fill_(.003)
    frozen=copy.deepcopy(net);x=sample(p,batch,99991,device=device)
    if device=='cuda':
        graph=TrajectoryGraph(frozen,p,batch,parameter_gradient=True)
        costs,pool,g0=collect(graph,x,p,batch)
    else:
        costs,states,controls,covectors,g0=trajectory_covectors(x.clone().requires_grad_(True),frozen,p,True)
        pool=(states[:,:-1].reshape(-1,p.dim).detach(),torch.arange(horizon).repeat(batch),
            controls.reshape(-1,p.n).detach(),states[:,1:].reshape(-1,p.dim).detach(),covectors.reshape(-1,p.dim).detach())
        g0=g0.detach()
    direct=vector_grad(rollout(x,net,p).mean(),net)
    global_error=((direct-g0).double().norm()/direct.double().norm().clamp_min(1e-12)).item()
    assert global_error<2e-4,(family,global_error)
    generator=torch.Generator(device=device).manual_seed(99997);plain=[];corrected=[]
    for _ in range(8):
        idx=torch.randint(len(pool[0]),(mini,),generator=generator,device=device);values=[v[idx] for v in pool]
        current=vector_grad(horizon*advantage(*values,net,p).mean(),net)
        old=vector_grad(horizon*advantage(*values,frozen,p).mean(),frozen)
        cv=g0+(current-old)
        plain.append(((current-g0).double().norm()/g0.double().norm().clamp_min(1e-12)).item())
        corrected.append(((cv-g0).double().norm()/g0.double().norm().clamp_min(1e-12)).item())
    assert max(corrected)<1e-6,(family,corrected)
    errors=[]
    if device=='cuda':
        trainer=TransportGraph(net,p,mini,.0005,frozen,True);trainer.full_gradient.copy_(g0)
        reference=copy.deepcopy(net);params=list(reference.net.parameters());sizes=[z.numel() for z in params]
        m=[torch.zeros_like(z) for z in params];v=[torch.zeros_like(z) for z in params]
        b1=torch.ones((),device=device);b2=torch.ones((),device=device)
        for _ in range(5):
            idx=torch.randint(len(pool[0]),(mini,),generator=generator,device=device);values=[z[idx] for z in pool]
            current=vector_grad(horizon*advantage(*values,reference,p).mean(),reference)
            old=vector_grad(horizon*advantage(*values,frozen,p).mean(),frozen)
            gradient=current-old+g0;norm=gradient.double().norm();assert torch.isfinite(norm)
            with torch.no_grad():
                scale=(10/norm.clamp_min(1e-12)).clamp_max(1);b1.mul_(.9);b2.mul_(.999)
                for z,mi,vi,gi in zip(params,m,v,gradient.split(sizes)):
                    g=gi.reshape_as(z)*scale;mi.mul_(.9).add_(g,alpha=.1);vi.mul_(.999).addcmul_(g,g,value=.001)
                    z.add_(-.0005*(mi/(1-b1).clamp_min(1e-12))/((vi/(1-b2).clamp_min(1e-12)).sqrt()+1e-8))
            trainer.update(*values);torch.cuda.synchronize()
            error=max((a-b).abs().max().item() for a,b in zip(net.net.parameters(),reference.net.parameters()))
            errors.append(error);assert error<2e-6,(family,error)
    assert all(z.grad is None for z in frozen.parameters())
    return dict(family=family,horizon=horizon,device=device,global_gradient_relative_error=global_error,
        anchor_plain_relative_errors=plain,anchor_corrected_relative_errors=corrected,
        five_update_parameter_errors=errors,teacher_parameters_received_no_accumulated_gradients=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1);rows=[]
    for family,horizon in [('mechanical',160),('reaction',320),('building',192)]:
        rows.append(check(family,horizon if args.device=='cuda' else 12,args.device));print(json.dumps(rows[-1]),flush=True)
        gc.collect()
        if args.device=='cuda':torch.cuda.empty_cache()
    write(args.output,dict(passed=True,rows=rows,scope='Independent development identity and update checks; these do not establish primary policy improvement.'))


if __name__=='__main__':main()
