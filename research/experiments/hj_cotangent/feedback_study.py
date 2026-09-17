"""Additional feedback planning and learned-policy headroom after primary tests."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiments.hj_gridfree.engine import write
from .feedback_native import solve
from .systems import problem
from .study import verify,digest
from .evaluation import lock_evaluation


def protocol():
    return dict(tasks=[['mechanical',160],['mechanical',320],['reaction',320]],dimensions=[32,256],
        condition='nominal',reference_states=32,reference_starts=['zero_residual','gpu_plan'],
        headroom_states=8,headroom_seed_index=0,headroom_methods=['fresh','cache','dpc','short','dpc_warm'],
        iterations=2048,
        scope='Feedback L-BFGS-B reference: optimize time-indexed feedforward controls with fixed known LQR tracking feedback around the zero or initial-plan reference trajectory. Angular deviations are wrapped relative to that trajectory. Learned starts use the stored neural feedback trajectory; they are separate headroom diagnostics, not free classical baselines.')


def make_lock(root):
    target=root/'feedback_protocol_lock.json'
    cfg=protocol()
    paths=[Path('experiments/hj_cotangent')/name for name in ['feedback_study.py','feedback_native.py','conditioning.py']]
    if target.exists():
        previous=json.loads(target.read_text())
        assert previous['configuration']==cfg
        for path,expected in previous['sources'].items():assert digest(path)==expected,path
    else:
        assert not (root/'evaluation_lock.json').exists()
        write(target,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configuration=cfg,
                          sources={str(p):digest(p) for p in paths}))
    return cfg


def worker(task):
    x,description,initial,reference,iterations=task
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):return solve(x,problem(**description),initial,maxiter=iterations,reference_states=reference)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/results/hj_cotangent_primary_v6')
    parser.add_argument('--lock-only',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();root=Path(args.root);cfg=make_lock(root)
    if args.lock_only:
        print('Feedback reference and corrected headroom protocol frozen before primary test access.');return
    primary=verify(root);lock_evaluation(root,primary)
    target=root/'feedback_classical';target.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for family,h in cfg['tasks']:
            for dim in cfg['dimensions']:
                key=f'{family}_h{h}_d{dim}_nominal';dest=target/(key+'.json')
                if dest.exists():continue
                test=torch.load(root/'evaluation'/(key+'.pt'),weights_only=False)
                meta=json.loads((root/'evaluation'/(key+'.json')).read_text())
                gpu=torch.load(root/'classical_gpu'/(key+'.pt'),weights_only=False)
                x=test['x'][:cfg['reference_states']]
                modes=['zero_residual','gpu_plan']+[m for m in cfg['headroom_methods'] if m in test['methods']]
                arrays={};records={}
                for mode in modes:
                    if mode=='zero_residual':initial=[None for _ in x];states=x;references=[None for _ in x]
                    elif mode=='gpu_plan':initial=gpu['best_u'].double().numpy();states=x;references=[None for _ in x]
                    else:
                        states=x[:cfg['headroom_states']]
                        initial=test['audit_controls'][mode][cfg['headroom_seed_index']].numpy()
                        references=test['audit_states'][mode][cfg['headroom_seed_index']].numpy()
                    tasks=[(xx.numpy(),meta['problem'],u,ref,cfg['iterations']) for xx,u,ref in zip(states,initial,references)]
                    started=time.perf_counter();rows=list(pool.map(worker,tasks,chunksize=1));seconds=time.perf_counter()-started
                    costs=torch.tensor([r['cost'] for r in rows],dtype=torch.float64)
                    assert torch.isfinite(costs).all() and costs.min()>0
                    arrays[mode]=dict(cost=costs,u=torch.tensor(np.stack([r.pop('u') for r in rows])),
                        states=torch.tensor(np.stack([r.pop('states') for r in rows])),
                        residual=torch.tensor(np.stack([r.pop('residual') for r in rows])),
                        tracking_reference=torch.tensor(np.stack([r.pop('tracking_reference') for r in rows])))
                    records[mode]=dict(mean=costs.mean().item(),count=len(costs),wall_seconds=seconds,rows=rows)
                    if mode not in cfg['reference_starts']:
                        original=test['methods'][mode][cfg['headroom_seed_index'],:len(states)]
                        arrays[mode]['original_feedback_cost']=original
                        records[mode]['original_feedback_mean']=original.mean().item()
                        records[mode]['reduction_from_original_feedback_percent']=100*(1-costs.mean().item()/original.mean().item())
                    print(json.dumps(dict(feedback_case=key,mode=mode,mean=costs.mean().item(),seconds=seconds)),flush=True)
                a=arrays['zero_residual']['cost'];b=arrays['gpu_plan']['cost']
                best=torch.minimum(a,b)
                torch.save(dict(x=x,methods=arrays,reference_best=best),target/(key+'.pt'))
                write(dest,dict(problem=meta['problem'],methods=records,count=len(x),reference_mean=best.mean().item(),
                    workers=args.workers,headroom_training_seed=primary['seeds'][cfg['headroom_seed_index']],scope=cfg['scope']))
    write(root/'feedback_complete.json',dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=6,
        reference_optimizations=384,headroom_optimizations=224,headroom_scope='First eight states and first training seed; not additional independent primary data.'))


if __name__=='__main__':main()
