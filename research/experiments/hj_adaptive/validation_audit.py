"""Replay selected validation policies without changing any selection."""
import argparse
import gc
import json
from pathlib import Path
import torch
from experiments.hj_cotangent.core import EvalGraph
from experiments.hj_cotangent.systems import problem,teacher,actor
from experiments.hj_gridfree.engine import write
from .study import verify,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    torch.set_num_threads(1);root=Path(args.root);cfg=verify(root);rows=[]
    for task in cfg['tasks']:
        p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
        net=actor(p,teacher(p),cfg['width']).cuda();ev=EvalGraph(net,p,cfg['validation_states'],dtype=torch.float64)
        for seed in cfg['seeds']:
            folder=root/'training'/f'{p.family}_h{p.horizon}_s{seed}'
            data=torch.load(folder/'data.pt',map_location='cuda',weights_only=False)
            for spec in task['methods']:
                path=folder/(spec['name']+'.pt');before=digest(path)
                saved=torch.load(path,map_location='cuda',weights_only=False)['budgets']
                for budget,record in saved.items():
                    net.load_state_dict(record['state_dict']);cost=ev(data['valx'])
                    error=abs(cost.mean().item()-record['validation_mean'])/(1+abs(record['validation_mean']))
                    assert torch.isfinite(cost).all() and error<1e-8,(path,budget,error)
                    assert record['available_seconds']<=float(budget)
                    rows.append(dict(family=p.family,seed=seed,method=spec['name'],budget=float(budget),scaled_mean_replay_error=error))
                assert digest(path)==before
            del data
        del net,ev;gc.collect();torch.cuda.empty_cache()
    record=dict(passed=True,records=rows,maximum_scaled_mean_error=max(r['scaled_mean_replay_error'] for r in rows),
        scope='Saved validation policies replayed on their original full validation batch; no checkpoint or primary choice was altered.')
    write(root/'validation_replay_audit.json',record);print(json.dumps(dict(passed=True,policies=len(rows),maximum_scaled_mean_error=record['maximum_scaled_mean_error'])))


if __name__=='__main__':main()
