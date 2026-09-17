"""Captured local updates and changing frozen teachers versus eager references."""
import argparse
import copy
import gc
import json
from pathlib import Path
import torch
from experiments.hj_cotangent.systems import problem,teacher,actor,sample
from experiments.hj_gridfree.engine import write
from .trajectory import trajectory_covectors,advantage,TrajectoryGraph
from .transport import TransportGraph,collect


def check(family,horizon):
    p=problem(family,16 if family=='mechanical' else 32,horizon);torch.manual_seed(99901)
    net=actor(p,teacher(p),64).cuda()
    with torch.no_grad():net.net[-1].weight.normal_(0,.01)
    frozen=copy.deepcopy(net).eval()
    for z in frozen.parameters():z.requires_grad_(False)
    x=sample(p,128,99921);graph=TrajectoryGraph(frozen,p,128)
    # A captured teacher must read its newly promoted weights on subsequent replays.
    with torch.no_grad():frozen.net[-1].bias.add_(.007)
    direct=trajectory_covectors(x.clone().requires_grad_(True),frozen,p)
    replay=graph(x)
    error=max((a-b.detach()).abs().max().item() for a,b in zip(replay,direct))
    assert error<1e-4,(family,error)
    _,pool=collect(graph,x,p,128)
    trainer=TransportGraph(net,p,1024,.0005)
    reference=copy.deepcopy(net);params=list(reference.net.parameters())
    m=[torch.zeros_like(z) for z in params];v=[torch.zeros_like(z) for z in params]
    b1=torch.ones((),device='cuda');b2=torch.ones((),device='cuda')
    generator=torch.Generator(device='cuda').manual_seed(99931);diffs=[]
    for _ in range(5):
        idx=torch.randint(len(pool[0]),(1024,),generator=generator,device='cuda');batch=[z[idx] for z in pool]
        reference.zero_grad(set_to_none=True)
        loss=p.horizon*advantage(*batch,reference,p).mean();loss.backward()
        with torch.no_grad():
            norm=torch.stack([z.grad.double().square().sum() for z in params]).sum().sqrt()
            assert torch.isfinite(norm)&torch.isfinite(loss)
            scale=(10/norm.clamp_min(1e-12)).clamp_max(1);b1.mul_(.9);b2.mul_(.999)
            for z,mi,vi in zip(params,m,v):
                g=z.grad*scale;mi.mul_(.9).add_(g,alpha=.1);vi.mul_(.999).addcmul_(g,g,value=.001)
                z.add_(-.0005*(mi/(1-b1).clamp_min(1e-12))/((vi/(1-b2).clamp_min(1e-12)).sqrt()+1e-8))
        trainer.update(*batch);torch.cuda.synchronize()
        difference=max((a-b).abs().max().item() for a,b in zip(net.net.parameters(),reference.net.parameters()))
        diffs.append(difference)
        assert difference<2e-6,(family,difference)
    assert all(z.grad is None for z in frozen.parameters())
    return dict(family=family,horizon=horizon,promoted_teacher_graph_max_abs_error=error,
        five_update_max_parameter_errors=diffs,teacher_parameters_received_no_gradients=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1);rows=[]
    for family,horizon in [('mechanical',160),('reaction',320),('building',192)]:
        rows.append(check(family,horizon));gc.collect();torch.cuda.empty_cache()
        print(json.dumps(rows[-1]),flush=True)
    write(args.output,dict(passed=True,rows=rows))


if __name__=='__main__':main()
