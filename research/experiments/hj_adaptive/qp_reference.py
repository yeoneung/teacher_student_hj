"""Independent building optimum checks on predeclared eight-state subsets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import torch
from experiments.hj_cotangent.building import BuildingProblem,solve_qp
from experiments.hj_cotangent.convex_certificate import certificate
from experiments.hj_gridfree.engine import write
from .study import verify


def worker(payload):
    parameters,x=payload;p=BuildingProblem(**parameters)
    with threadpool_limits(limits=1):
        qp=solve_qp(x,p);u=qp['u'];cert=certificate(x,u,p)
    assert np.isfinite(u).all() and u.min()>=-1e-7 and u.max()<=p.bound+1e-7
    assert abs(qp['cost']-cert['feasible_cost'])<1e-8
    return dict(controls=u,certificate=cert,status=qp.get('status'),cost=qp['cost'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    torch.set_num_threads(1);root=Path(args.root);cfg=verify(root);out=root/'qp_references';out.mkdir(exist_ok=True)
    for file in sorted((root/'evaluation').glob('building*.pt')):
        if (out/(file.stem+'.json')).exists():continue
        meta=json.loads(file.with_suffix('.json').read_text());raw=torch.load(file,map_location='cpu',weights_only=False)
        x=raw['x'][:8].numpy();payload=[(meta['problem'],state) for state in x]
        with ProcessPoolExecutor(max_workers=2) as pool:solutions=list(pool.map(worker,payload))
        optimal=np.array([row['cost'] for row in solutions]);lower=np.array([row['certificate']['convex_lower_bound'] for row in solutions])
        comparisons={}
        for name,budgets in raw['methods'].items():
            values=budgets['60'][:,:8].numpy()
            comparisons[name]=dict(mean=float(values.mean()),excess_over_feasible_percent=100*(float(values.mean()/optimal.mean())-1),
                excess_over_lower_bound_percent=100*(float(values.mean()/lower.mean())-1))
        record=dict(case=file.stem,states=8,problem=meta['problem'],optimal_mean=float(optimal.mean()),
            lower_bound_mean=float(lower.mean()),maximum_first_order_gap=max(row['certificate']['first_order_gap'] for row in solutions),
            methods=comparisons,scope='Matched first eight fresh states, all ten learned seeds, 60-second policies. This is a subset optimum diagnostic; its means are not mixed with the 512-state policy means. No deployment timing claim is made.')
        torch.save(dict(x=torch.from_numpy(x),solutions=solutions),out/(file.stem+'.pt'));write(out/(file.stem+'.json'),record)
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
