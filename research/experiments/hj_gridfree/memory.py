"""Run one configuration per fresh process for interpretable training memory."""
import argparse
import json
from pathlib import Path
import time
import torch
from .core import Problem,LQR,Actor,sample
from .engine import TrainGraph,write


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--family',required=True); ap.add_argument('--mode',required=True)
    ap.add_argument('--root',default='experiments/results/hj_gridfree_primary_v1')
    args=ap.parse_args(); torch.set_num_threads(1)
    p=Problem(args.family,16 if args.family=='mechanical' else 32)
    torch.cuda.init(); torch.cuda.empty_cache(); before=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter(); actor=Actor(p,LQR(p)).cuda(); x=sample(p,128,123)
    trainer=TrainGraph(actor,x,p,'mse' if args.mode=='bc' else args.mode)
    t=torch.zeros(128,device='cuda',dtype=torch.long); y=torch.zeros(128,p.n,device='cuda')
    for _ in range(10): trainer.update(x,t,y)
    torch.cuda.synchronize()
    result=dict(family=args.family,mode=args.mode,batch=128,state_dimension=p.dim,width=64,peak_allocated_bytes=torch.cuda.max_memory_allocated(),incremental_peak_bytes=torch.cuda.max_memory_allocated()-before,peak_reserved_bytes=torch.cuda.max_memory_reserved(),setup_and_10_updates_seconds=time.perf_counter()-start,scope='Fresh CUDA process, weights/optimizer/static buffers and captured training graph. Excludes dataset storage, unrelated retained graph pools, CUDA context/driver allocations, and label preparation.')
    write(Path(args.root)/'isolated_memory'/f'{args.family}_{args.mode}.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
