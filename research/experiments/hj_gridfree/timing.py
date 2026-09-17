"""Isolated deployment timing, separate from parallel training/evaluation work."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import Problem,LQR,Actor,rollout
from .engine import capture,AdamGraph,cem,write
from .evaluate import actor_from
from .native import solve,value_gradient
from .train import verify_lock
from .gate import gated_rollout


class PolicyGraph:
    def __init__(self,policy,p,batch,action=False):
        self.x=torch.zeros(batch,p.dim,device='cuda')
        self.u=torch.zeros((batch,p.n) if action else (batch,p.horizon,p.n),device='cuda')
        self.j=torch.zeros(batch,device='cuda'); self.action=action
        def update():
            with torch.no_grad():
                if action: self.u.copy_(policy(self.x,0))
                else:
                    j,_,u=rollout(self.x,policy,p,keep=True); self.j.copy_(j); self.u.copy_(u)
        self.graph=capture(update)
    def __call__(self,x):
        self.x.copy_(x); self.graph.replay(); return self.j.clone(),self.u.clone()


def measure_gpu(fn,repeats):
    fn(); torch.cuda.synchronize(); wall=[]; device=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); start=time.perf_counter(); a.record(); fn(); b.record(); b.synchronize()
        wall.append((time.perf_counter()-start)*1000); device.append(a.elapsed_time(b))
    return dict(wall_median_ms=float(np.median(wall)),wall_p95_ms=float(np.quantile(wall,.95)),device_median_ms=float(np.median(device)),repetitions=repeats,raw_wall_ms=wall)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='experiments/results/hj_gridfree_primary_v1')
    args=ap.parse_args(); torch.set_num_threads(1); root=Path(args.root); cfg=verify_lock(root)
    out=root/'timing'; out.mkdir(exist_ok=True)
    for family in cfg['families']:
        for dim in cfg['test_dimensions']:
            key=f'{family}_d{dim}'; target=out/f'{key}.json'
            if target.exists(): continue
            p=Problem(family,dim//2 if family=='mechanical' else dim)
            raw=torch.load(root/'evaluation'/f'{key}_nominal.pt',weights_only=False)
            x=raw['x'].cuda(); start=time.perf_counter(); base=LQR(p); actor=actor_from(root,p,'block16',cfg['seeds'][0],cfg['width']); torch.cuda.synchronize()
            construction=time.perf_counter()-start
            result=dict(problem=p.__dict__,seed=cfg['seeds'][0],policy_construction_seconds=construction,neural_parameters=sum(z.numel() for z in actor.net.parameters()),methods={},scope='GPU-resident observations, synchronized end-to-end Python call wall times. Initialization and graph construction measured separately. Native CPU uses two fresh starts. Timed at nominal parameters only.')
            for name,policy in [('lqr',base),('block16',actor)]:
                for action in [True,False]:
                    start=time.perf_counter(); graph=PolicyGraph(policy,p,1,action=action); torch.cuda.synchronize()
                    setup=time.perf_counter()-start
                    result['methods'][name+('_action' if action else '_full_plan')]=dict(setup_seconds=setup,**measure_gpu(lambda:graph(x[:1]),100))
                    del graph; gc.collect(); torch.cuda.empty_cache()
                graph=PolicyGraph(policy,p,256)
                timing=measure_gpu(lambda:graph(x[:256]),30); timing['plans_per_second']=256000/timing['wall_median_ms']
                result['methods'][name+'_batch256']=timing; del graph; gc.collect(); torch.cuda.empty_cache()
            start=time.perf_counter(); graph=PolicyGraph(base,p,1); solver=AdamGraph(x[:1],p,lr=.12*p.bound); torch.cuda.synchronize(); planner_setup=time.perf_counter()-start
            for it in [32,128,1024]:
                result['methods'][f'adam_lqr_{it}']=dict(setup_seconds=planner_setup,**measure_gpu(lambda:solver.solve(x[:1],graph(x[:1])[1],it),5))
            result['methods']['cem512x8']=measure_gpu(lambda:cem(x[:1],graph(x[:1])[1],p,samples=512,iterations=8,seed=123),7)
            del graph,solver; gc.collect(); torch.cuda.empty_cache()
            # A complete gate plan includes every base/candidate tail comparison.
            gatex=torch.zeros_like(x[:1]); gatej=torch.zeros(1,device='cuda')
            def gate_update():
                with torch.no_grad(): gatej.copy_(gated_rollout(gatex,actor,base,p)[0])
            gategraph=capture(gate_update)
            def gate_call(): gatex.copy_(x[:1]); gategraph.replay(); return gatej.clone()
            result['methods']['block16_gate_full_plan']=measure_gpu(gate_call,50)
            del gategraph; gc.collect(); torch.cuda.empty_cache()
            # Serial native timing, with initialization rollout and array conversion included.
            start=time.perf_counter(); cp=LQR(p,device='cpu')
            value_gradient(np.zeros(p.horizon*p.n),raw['x'][0].double().numpy(),p.n,p.horizon,p.dt,p.gamma,p.bound,p.mechanical)
            native_setup=time.perf_counter()-start; native=[]; costs=[]
            for j in range(8):
                xx=raw['x'][j:j+1].double()
                start=time.perf_counter()
                with torch.no_grad(): _,_,u=rollout(xx,cp,p,keep=True)
                a=solve(xx[0].numpy(),u[0].numpy(),p,1024)
                b=solve(xx[0].numpy(),np.zeros((p.horizon,p.n)),p,1024)
                native.append((time.perf_counter()-start)*1000); costs.append(min(a['cost'],b['cost']))
            result['methods']['native_two_start1024']=dict(setup_seconds=native_setup,wall_median_ms=float(np.median(native)),wall_p95_ms=float(np.quantile(native,.95)),repetitions=8,raw_wall_ms=native,costs=costs,initialization_included=True)
            write(target,result); print(json.dumps(dict(case=key,action_ms=result['methods']['block16_action']['wall_median_ms'],plan_ms=result['methods']['block16_full_plan']['wall_median_ms'],native_ms=result['methods']['native_two_start1024']['wall_median_ms'])),flush=True)
            del actor,base; gc.collect(); torch.cuda.empty_cache()
    print('ISOLATED TIMING COMPLETE',flush=True)


if __name__=='__main__': main()
