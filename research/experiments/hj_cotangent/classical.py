"""Matched classical quality: GPU multistart search and native optimization/QP."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.hj_gridfree.core import LQR, rollout
from experiments.hj_gridfree.engine import AdamGraph, cem, write
from experiments.hj_gridfree.native import solve
from experiments.hj_gridfree.planning_probe import pd_policy
from .systems import problem
from .building import solve_qp
from .convex_certificate import certificate
from .evaluation import replay, lock_evaluation
from .study import verify


def cpu_worker(task):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        x,starts,description,iterations = task
        p = problem(**description)
        if p.family=='building':
            result = solve_qp(x,p)
            result['convex_certificate'] = certificate(x,result['u'],p)
            assert result['convex_certificate']['first_order_gap']<1e-5*(1+result['cost'])
            result.pop('x')
            return result
        rows={}
        for name,initial in starts.items():
            result=solve(x,initial,p,maxiter=iterations)
            candidate=result['u']
            costs=replay(torch.tensor(np.stack([x,x]),dtype=torch.float64),
                         torch.tensor(np.stack([candidate,initial]),dtype=torch.float64),p)
            native_cost=result['cost']
            if costs[1]<costs[0]:
                result['u']=initial.copy()
                selected=float(costs[1])
                retained='initial'
            else:
                selected=float(costs[0]);retained='optimized'
            result.update(cost=selected,native_objective_cost=native_cost,
                optimized_canonical_replay_cost=float(costs[0]),initial_canonical_replay_cost=float(costs[1]),
                native_vs_replay_scaled_difference=abs(native_cost-float(costs[0]))/(1+abs(float(costs[0]))),
                retained=retained)
            rows[name]=result
        best_name = min(rows,key=lambda name:rows[name]['cost'])
        controls = rows[best_name]['u']
        for record in rows.values():
            record.pop('u')
        return dict(cost=rows[best_name]['cost'],u=controls,selected_start=best_name,starts=rows)


def gpu_case(root,cfg,source):
    key = source.stem
    out = root/'classical_gpu'
    out.mkdir(exist_ok=True)
    if (out/(key+'.json')).exists():
        return
    meta = json.loads(source.with_suffix('.json').read_text())
    p = problem(**meta['problem'])
    if p.family=='building':
        return
    data = torch.load(source,weights_only=False)
    x = data['x'][:cfg['classical_states']].cuda()
    calibration = json.loads(Path('experiments/results/hj_gridfree_primary_v1/pd_calibration.json').read_text())
    quality_started = time.perf_counter()
    started = quality_started
    with torch.no_grad():
        _,_,ub = rollout(x,LQR(p),p,keep=True)
        _,_,up = rollout(x,pd_policy(p,*calibration[p.family]['selected']),p,keep=True)
    solver = AdamGraph(x.float(),p,lr=.12*p.bound)
    torch.cuda.synchronize()
    setup = time.perf_counter()-started
    best = None
    best_u = None
    records = {}
    raw = dict(x=x.cpu(),costs={},lqr_u=ub.cpu(),pd_u=up.cpu())
    for name,initial in [('lqr',ub),('zero',torch.zeros_like(ub)),('half_lqr',.5*ub),('tuned_pd',up)]:
        started = time.perf_counter()
        _,u = solver.solve(x.float(),initial.float(),cfg['classical_steps'])
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        j = replay(x,u.double(),p)
        records[name] = dict(mean=j.mean().item(),seconds=seconds)
        raw['costs'][name] = j
        if best is None:
            best,best_u = j,u
        else:
            better = j<best
            best = torch.minimum(best,j)
            best_u = torch.where(better.cuda()[:,None,None],u,best_u)
    started = time.perf_counter()
    uc = []
    for offset in range(0,len(x),8):
        _,u = cem(x[offset:offset+8].float(),ub[offset:offset+8].float(),p,
                  samples=256,iterations=8,seed=meta['test_seed']+offset,knots=20)
        uc.append(u)
    _,u = solver.solve(x.float(),torch.cat(uc),512)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    j = replay(x,u.double(),p)
    records['cem_adam512'] = dict(mean=j.mean().item(),seconds=seconds)
    raw['costs']['cem_adam512'] = j
    better = j<best
    best_u = torch.where(better.cuda()[:,None,None],u,best_u)
    best = torch.minimum(best,j)
    for name,initial in [('initial_lqr',ub),('initial_zero',torch.zeros_like(ub)),
                         ('initial_half_lqr',.5*ub),('initial_tuned_pd',up)]:
        j=replay(x,initial.double(),p)
        raw['costs'][name]=j
        records[name]=dict(mean=j.mean().item(),seconds=0.,scope='Retained original start, scored in canonical NumPy float64 arithmetic.')
        better=j<best
        best_u=torch.where(better.cuda()[:,None,None],initial.double(),best_u.double())
        best=torch.minimum(best,j)
    assert torch.isfinite(best).all().item()
    raw.update(best=best,best_u=best_u.cpu())
    torch.save(raw,out/(key+'.pt'))
    write(out/(key+'.json'),dict(problem=asdict(p),count=len(x),mean=best.mean().item(),methods=records,
                                setup_seconds=setup,optimization_seconds=setup+sum(r['seconds'] for r in records.values()),
                                quality_seconds=time.perf_counter()-quality_started,
                                scoring='All returned controls and original starts are rescored by the canonical independent NumPy float64 plant; original starts can be retained.'))
    print(json.dumps(dict(gpu_classical=key,mean=best.mean().item())),flush=True)
    del solver,x,data,raw
    gc.collect()
    torch.cuda.empty_cache()


def cpu_cases(root,cfg,workers):
    out = root/'classical_native'
    out.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for source in sorted((root/'evaluation').glob('*_nominal.pt')):
            key = source.stem
            if (out/(key+'.json')).exists():
                continue
            meta = json.loads(source.with_suffix('.json').read_text())
            p = problem(**meta['problem'])
            data = torch.load(source,weights_only=False)
            x = data['x'][:cfg['classical_states']]
            if p.family=='building':
                starts = [{} for _ in x]
            else:
                gpu = torch.load(root/'classical_gpu'/source.name,weights_only=False)
                starts = [dict(lqr=gpu['lqr_u'][i].numpy(),zero=np.zeros((p.horizon,p.n)),
                               tuned_pd=gpu['pd_u'][i].numpy(),gpu_search=gpu['best_u'][i].double().numpy()) for i in range(len(x))]
            tasks = [(x[i].numpy(),starts[i],asdict(p),cfg['native_iterations']) for i in range(len(x))]
            started = time.perf_counter()
            rows = list(pool.map(cpu_worker,tasks,chunksize=1))
            seconds = time.perf_counter()-started
            u = torch.tensor(np.stack([r.pop('u') for r in rows]),dtype=torch.float64)
            costs = replay(x,u,p)
            oldcosts = torch.tensor([r['cost'] for r in rows],dtype=torch.float64)
            error = ((costs-oldcosts).abs()/(1+costs.abs())).max().item()
            assert error<1e-8,(key,error)
            torch.save(dict(x=x,cost=costs,u=u),out/(key+'.pt'))
            write(out/(key+'.json'),dict(problem=asdict(p),count=len(x),mean=costs.mean().item(),
                                        wall_seconds=seconds,workers=workers,independent_scaled_cost_error=error,rows=rows,
                                        scope='Quality search after GPU quality evaluation; ring native four starts include the preceding GPU search. This stage is separate from isolated deployment latency.'))
            print(json.dumps(dict(native_classical=key,mean=costs.mean().item(),seconds=seconds)),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='experiments/results/hj_cotangent_primary_v6')
    parser.add_argument('--stage',choices=['gpu','native'],required=True)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root)
    cfg = verify(root)
    lock_evaluation(root,cfg)
    if args.stage=='gpu':
        for source in sorted((root/'evaluation').glob('*_nominal.pt')):
            gpu_case(root,cfg,source)
    else:
        cpu_cases(root,cfg,args.workers)


if __name__=='__main__':
    main()
