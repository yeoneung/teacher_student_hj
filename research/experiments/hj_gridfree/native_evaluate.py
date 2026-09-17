"""Prespecified CPU optimizer subset; state indices are fixed before test access."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import Problem,LQR,rollout
from .engine import write
from .native import solve
from .train import verify_lock


def worker(task):
    state,initials,description=task; p=Problem(**description)
    result={}
    for name,u in initials.items():
        r=solve(state,u,p,1024)
        result[name]={k:v for k,v in r.items() if k!='u'}
        if 'best' not in result or r['cost']<result['best']['cost']:
            result['best']=dict(cost=r['cost'],u=r['u'],initialization=name)
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='experiments/results/hj_gridfree_primary_v1'); ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args(); torch.set_num_threads(1); root=Path(args.root); cfg=verify_lock(root)
    lock=json.loads((root/'evaluation_lock.json').read_text(encoding='utf-8'))
    for file,digest in lock['additional_source_hashes'].items(): assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    out=root/'native_evaluation'; out.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for source in sorted((root/'evaluation').glob('*.pt')):
            if (out/f'{source.stem}.json').exists(): continue
            description=json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
            p=Problem(**description['problem']); data=torch.load(source,weights_only=False)
            count=128 if description['distribution']=='nominal' else 64
            x=data['x'][:count].double(); base=LQR(p,device='cpu')
            with torch.no_grad(): _,_,u=rollout(x,base,p,keep=True)
            gx=data['best_classical_controls'][:count].double()
            tasks=[(x[i].numpy(),dict(lqr=u[i].numpy(),zero=np.zeros((p.horizon,p.n)),gpu_classical=gx[i].numpy()),asdict(p)) for i in range(count)]
            start=time.perf_counter(); rows=list(pool.map(worker,tasks,chunksize=4)); wall=time.perf_counter()-start
            costs=torch.tensor([r['best']['cost'] for r in rows],dtype=torch.float64)
            controls=torch.tensor(np.stack([r['best']['u'] for r in rows]),dtype=torch.float64)
            # Independent source replay against the separate NumPy implementation.
            from .evaluate import independent_cost
            ji=independent_cost(x,controls,p); error=((costs-ji).abs()/(1+ji.abs())).max().item()
            assert error<1e-9,error
            assert controls.square().mean(-1).sqrt().max().item()/p.bound<1.000000001
            saved=dict(indices=torch.arange(count),cost=ji,u=controls)
            torch.save(saved,out/f'{source.stem}.pt')
            for r in rows: del r['best']['u']
            record=dict(case=source.stem,count=count,indices=f'first {count} independently sampled states, fixed before test access',workers=args.workers,wall_seconds=wall,mean=ji.mean().item(),max_independent_scaled_cost_error=error,rows=rows,
                cost_accounting='Three native starts, one initialized from the full GPU classical search. Parallel worker times are diagnostic and are NOT used as isolated deployment latency.')
            write(out/f'{source.stem}.json',record)
            print(json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)
    assert len(list(out.glob('*.json')))==len(cfg['families'])*len(cfg['test_dimensions'])*2
    print('NATIVE SUBSET EVALUATION COMPLETE',flush=True)


if __name__=='__main__': main()
