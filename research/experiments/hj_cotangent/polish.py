"""Prespecified supplementary headroom diagnostic from learned feasible plans."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.hj_gridfree.native import solve
from experiments.hj_gridfree.engine import write
from .systems import problem
from .evaluation import replay,lock_evaluation
from .study import verify,digest


def worker(item):
    x,u,description,iterations=item
    with threadpool_limits(limits=1):
        return solve(x,u,problem(**description),maxiter=iterations)


def protocol():
    return dict(task_horizons=[['mechanical',160],['mechanical',320],['reaction',320]],dimensions=[32,256],
                condition='nominal',states=8,training_seed_index=0,
                methods=['fresh','cache','dpc','short','dpc_warm'],native_iterations=2048,
                scope='Supplementary headroom diagnostic: native optimization initialized by each specified learned policy on the first eight nominal states. Warm DPC applies only where present. Not a replacement primary comparison or a global optimality certificate.')


def make_lock(root):
    path=root/'polish_protocol_lock.json'
    cfg=protocol()
    sources=[Path(__file__),Path('experiments/hj_gridfree/native.py'),Path('experiments/hj_gridfree/evaluate.py')]
    sources=[p.relative_to(Path.cwd()) if p.is_absolute() else p for p in sources]
    if path.exists():
        lock=json.loads(path.read_text())
        assert lock['configuration']==cfg
        for source,expected in lock['sources'].items():assert digest(source)==expected,source
    else:
        assert not (root/'evaluation_lock.json').exists(),'Supplementary protocol must be recorded before primary test access.'
        write(path,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        configuration=cfg,sources={str(p):digest(p) for p in sources}))
    return cfg


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='experiments/results/hj_cotangent_primary_v6')
    parser.add_argument('--lock-only',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();root=Path(args.root)
    cfg=make_lock(root)
    if args.lock_only:
        print('Supplementary learned-plan headroom diagnostic frozen before primary test access.');return
    primary=verify(root);lock_evaluation(root,primary)
    torch.set_num_threads(1)
    output=root/'learned_plan_polish';output.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for family,h in cfg['task_horizons']:
            for d in cfg['dimensions']:
                key=f'{family}_h{h}_d{d}_nominal'
                target=output/(key+'.json')
                if target.exists():continue
                data=torch.load(root/'evaluation'/(key+'.pt'),weights_only=False)
                metadata=json.loads((root/'evaluation'/(key+'.json')).read_text())
                p=problem(**metadata['problem'])
                x=data['x'][:cfg['states']]
                records={};raw=dict(x=x,methods={})
                for method in cfg['methods']:
                    if method not in data['methods']:continue
                    initial=data['audit_controls'][method][cfg['training_seed_index']]
                    assert len(initial)==len(x)
                    tasks=[(xx.numpy(),uu.numpy(),metadata['problem'],cfg['native_iterations']) for xx,uu in zip(x,initial)]
                    start=time.perf_counter()
                    rows=list(pool.map(worker,tasks,chunksize=1))
                    elapsed=time.perf_counter()-start
                    u=torch.tensor(np.stack([row.pop('u') for row in rows]),dtype=torch.float64)
                    costs=replay(x,u,p)
                    original=data['methods'][method][cfg['training_seed_index'],:len(x)]
                    reported=torch.tensor([row['cost'] for row in rows],dtype=torch.float64)
                    error=((costs-reported).abs()/(1+costs.abs())).max().item()
                    assert error<1e-8
                    raw['methods'][method]=dict(original=original,polished=costs,u=u)
                    records[method]=dict(original_mean=original.mean().item(),polished_mean=costs.mean().item(),
                        mean_reduction_percent=100*(1-costs.mean().item()/original.mean().item()),
                        maximum_scaled_increase=((costs-original)/(1+original.abs())).max().item(),
                        seconds=elapsed,independent_scaled_cost_error=error,rows=rows)
                torch.save(raw,output/(key+'.pt'))
                write(target,dict(problem=metadata['problem'],count=len(x),training_seed=primary['seeds'][cfg['training_seed_index']],
                    workers=args.workers,methods=records,scope=cfg['scope']))
                print(json.dumps(dict(case=key,methods={k:dict(original=v['original_mean'],polished=v['polished_mean']) for k,v in records.items()})),flush=True)
    print('LEARNED-PLAN HEADROOM DIAGNOSTIC COMPLETE',flush=True)


if __name__=='__main__':main()
