"""Full-set DPC versus teacher global labels and independent eager Adam."""
import argparse
import copy
import gc
import json
import torch
from experiments.hj_cotangent.systems import problem,teacher,actor,sample,rollout
from experiments.hj_gridfree.engine import write
from .trajectory import trajectory_covectors
from .full_batch import FullBatchGraph


def check(family,horizon):
    p=problem(family,16 if family=='mechanical' else 32,horizon);torch.manual_seed(99001)
    net=actor(p,teacher(p),64).cuda()
    with torch.no_grad():net.net[-1].weight.normal_(0,.01)
    frozen=copy.deepcopy(net);initial=sample(p,512,99021);labels=[]
    for offset in range(0,512,128):
        result=trajectory_covectors(initial[offset:offset+128].clone().requires_grad_(True),frozen,p,True)
        labels.append(result[-1].detach());del result
    global_label=torch.stack(labels).mean(0);del frozen,labels;gc.collect();torch.cuda.empty_cache()
    trainer=FullBatchGraph(net,p,128,4,.0005);reference=copy.deepcopy(net)
    params=list(reference.net.parameters());sizes=[z.numel() for z in params]
    m=[torch.zeros_like(z) for z in params];v=[torch.zeros_like(z) for z in params]
    b1=torch.ones((),device='cuda');b2=torch.ones((),device='cuda');errors=[];gradient_error=None
    for iteration in range(3):
        gradients=[]
        for offset in range(0,512,128):
            loss=rollout(initial[offset:offset+128],reference,p).mean()
            gradients.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,params)]))
        gradient=torch.stack(gradients).mean(0)
        if iteration==0:
            gradient_error=((gradient-global_label).double().norm()/global_label.double().norm().clamp_min(1e-12)).item()
            assert gradient_error<2e-4,(family,gradient_error)
        with torch.no_grad():
            parts=gradient.split(sizes);norm=torch.stack([g.double().square().sum() for g in parts]).sum().sqrt()
            assert torch.isfinite(norm);scale=(10/norm.clamp_min(1e-12)).clamp_max(1);b1.mul_(.9);b2.mul_(.999)
            for z,mi,vi,part in zip(params,m,v,parts):
                g=part.reshape_as(z)*scale;mi.mul_(.9).add_(g,alpha=.1);vi.mul_(.999).addcmul_(g,g,value=.001)
                z.add_(-.0005*(mi/(1-b1).clamp_min(1e-12))/((vi/(1-b2).clamp_min(1e-12)).sqrt()+1e-8))
        trainer.update_full(initial);torch.cuda.synchronize()
        error=max((a-b).abs().max().item() for a,b in zip(net.net.parameters(),reference.net.parameters()))
        errors.append(error);assert error<2e-6,(family,error)
    return dict(family=family,horizon=horizon,teacher_global_gradient_relative_error=gradient_error,three_update_parameter_errors=errors)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1);rows=[]
    for family,horizon in [('mechanical',160),('reaction',320),('building',192)]:
        rows.append(check(family,horizon));print(json.dumps(rows[-1]),flush=True);gc.collect();torch.cuda.empty_cache()
    write(args.output,dict(passed=True,rows=rows))


if __name__=='__main__':main()
