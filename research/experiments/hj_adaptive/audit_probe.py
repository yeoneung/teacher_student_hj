"""Development-only independent checks for the v7 cached-gradient monitor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.hj_cotangent.systems import problem,teacher,actor,sample
from experiments.hj_cotangent.core import JetGraph,continuation,local_return
from experiments.hj_gridfree.engine import write
from .probe import GradientProbe


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1);rows=[]
    for family in ['mechanical','reaction','building']:
        torch.manual_seed(97001)
        p=problem(family,4,20);base=teacher(p);net=actor(p,base,16).cuda()
        batch=8;length=4;x=sample(p,batch,97501);t=torch.arange(batch,device='cuda')*2
        jet=JetGraph(net,base,p,batch,length);probe=GradientProbe(net,p,batch,length,1.)
        old=jet(x,t)
        same=probe(x,t,old,old)
        assert same['relative_gradient_error']<1e-5,same
        with torch.no_grad():
            for v in net.net.parameters():v.add_(.002*torch.randn_like(v))
        new=jet(x,t);metrics=probe(x,t,old,new)
        params=list(net.net.parameters())
        fg=torch.cat([v.reshape(-1) for v in torch.autograd.grad(continuation(x,t,net,base,p,length).mean(),params)]).double()
        cg=torch.cat([v.reshape(-1) for v in torch.autograd.grad(local_return(x,t,net,p,length,*old,rho=1.).mean(),params)]).double()
        fresh_error=((fg-probe.fresh).norm()/fg.norm().clamp_min(1e-12)).item()
        cache_error=((cg-probe.cached).norm()/cg.norm().clamp_min(1e-12)).item()
        assert fresh_error<3e-5 and cache_error<3e-6,(fresh_error,cache_error)
        rel=(fg-cg).norm()/cg.norm().clamp_min(1e-12)
        assert abs(rel.item()-metrics['relative_gradient_error'])<1e-4
        if metrics['relative_gradient_error']<.5:assert (fg*cg).sum().item()>0
        rows.append(dict(family=family,anchor_relative_error=same['relative_gradient_error'],
            fresh_vs_direct_continuation_gradient_relative_error=fresh_error,
            cached_vs_direct_surrogate_gradient_relative_error=cache_error,probe=metrics))
        del net,jet,probe;torch.cuda.empty_cache()
    rng=np.random.default_rng(97801)
    g=rng.normal(size=(1000,23));e=rng.normal(size=g.shape)
    e*=.5*np.linalg.norm(g,axis=1,keepdims=True)/np.linalg.norm(e,axis=1,keepdims=True)
    dot=np.sum((g+e)*g,axis=1);bound=.5*np.sum(g*g,axis=1)
    assert np.all(dot>=bound-1e-12)
    result=dict(rows=rows,cauchy_schwarz_cases=1000,passed=True,
        scope='Development checks of exact sampled objective gradients; no primary test data and no global policy certificate.')
    write(Path(args.output),result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
