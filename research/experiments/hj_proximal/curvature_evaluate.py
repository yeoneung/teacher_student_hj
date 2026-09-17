"""Freeze development artifacts, then evaluate on untouched development states."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem, teacher, actor, sample
from experiments.hj_cotangent.evaluation import PlanGraph
from experiments.hj_cotangent.core import EvalGraph
from experiments.hj_cotangent.conditioning import local_trajectory_audit, independent_feedback_cost
from .study import sha, utc


def evaluate(root):
    cfg=json.loads((root/'config.json').read_text())
    locked=json.loads((root/'development_lock.json').read_text())
    for path,expected in locked['hashes'].items():assert sha(path)==expected,path
    paths=[Path(__file__),root/'config.json']
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            folder=root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'
            assert (folder/'complete.json').exists(),folder
            for spec in task['methods']:
                paths.extend([folder/(spec['name']+'.pt'),folder/(spec['name']+'.json')])
    frozen={str(p):sha(p) for p in paths}
    lockpath=root/'evaluation_lock.json'
    if lockpath.exists():assert json.loads(lockpath.read_text())['hashes']==frozen
    else:write(lockpath,dict(created_utc=utc(),hashes=frozen,scope='Development follow-up evaluation, no claim of confirmatory significance. All methods frozen before new state generation.'))
    output=root/'evaluation';output.mkdir(exist_ok=True)
    rows=[]
    for fi,task in enumerate(cfg['tasks']):
        for dimension,condition,count in [(32,'nominal',256),(256,'nominal',128),(256,'changed',128)]:
            family=task['family'];n=dimension//2 if family=='mechanical' else dimension
            kwargs={}
            if condition=='changed':
                if family=='building':kwargs=dict(weather_shift=-2.,budget=4.5,topology_seed=83011)
                else:
                    standard=problem(family,n,task['horizon']);kwargs=dict(coupling=.8*standard.gamma,budget=.9*standard.bound)
            p=problem(family,n,task['horizon'],**kwargs)
            key=f'{family}_d{dimension}_{condition}';dest=output/key
            if dest.with_suffix('.json').exists():rows.append(json.loads(dest.with_suffix('.json').read_text()));continue
            seed_base=16400000+fi*100000+dimension*10+int(condition=='changed')
            x=sample(p,count,seed_base,shift=condition=='changed').double()
            net=actor(p,teacher(p),cfg['width']).cuda();planner=PlanGraph(net,p,64)
            raw=dict(x=x.cpu(),methods={});summary=dict(family=family,dimension=dimension,condition=condition,count=count,
                test_seed=seed_base,methods={},audits=[])
            for spec in task['methods']:
                name=spec['name'];cost_rows=[]
                for seed in cfg['seeds']:
                    folder=root/'training'/f'{family}_h{p.horizon}_s{seed}'
                    budgets=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=False)['budgets']
                    key_budget='7.5' if spec.get('parent_only') else str(max(cfg['budgets_seconds']))
                    selected=budgets[key_budget]
                    assert selected['available_seconds']<=float(key_budget)
                    net.net.load_state_dict({k[4:]:v for k,v in selected['state_dict'].items() if k.startswith('net.')})
                    costs=[]
                    for start in range(0,count,64):
                        j,u=planner(x[start:start+64]);assert torch.isfinite(j).all()
                        if p.family=='building':assert u.min()>=-1e-7 and u.max()<=p.bound+1e-6
                        else:assert u.square().mean(-1).sqrt().max()<=p.bound+1e-6
                        costs.append(j.cpu())
                        if start==0:
                            local=local_trajectory_audit(planner.states[:8],u[:8],j[:8],p)
                            independent=independent_feedback_cost(x[:64],net,p)
                            error=float(((independent-j.cpu()).abs()/(1+independent.abs())).max())
                            assert error<2e-5,(family,name,seed,error)
                            summary['audits'].append(dict(method=name,seed=seed,feedback_scaled_error=error,**local))
                    cost_rows.append(torch.cat(costs))
                raw['methods'][name]=torch.stack(cost_rows)
                summary['methods'][name]=dict(mean=float(raw['methods'][name].mean()),seed_means=raw['methods'][name].mean(1).tolist())
            refs=['short','dpc','warm']
            best_ref=min(refs,key=lambda name:summary['methods'][name]['mean'])
            summary['best_learning_reference']=best_ref
            for name in ['fixed','prox_action','prox_latent']:
                summary['methods'][name]['ratio_to_best_learning_reference']=summary['methods'][name]['mean']/summary['methods'][best_ref]['mean']
                summary['methods'][name]['paired_seed_ratios_to_best_reference']=(raw['methods'][name].mean(1)/raw['methods'][best_ref].mean(1)).tolist()
            torch.save(raw,dest.with_suffix('.pt'));write(dest.with_suffix('.json'),summary);rows.append(summary)
            print(json.dumps({k:v for k,v in summary.items() if k!='audits'}),flush=True)
            del net,planner,raw;gc.collect();torch.cuda.empty_cache()
    # Replay every available budget using the original validation set. Confirm
    # the saved weights, rather than trusting logged scalar improvements.
    replay=[]
    for task in cfg['tasks']:
        p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
        net=actor(p,teacher(p),cfg['width']).cuda();valgraph=EvalGraph(net,p,cfg['validation_states'],dtype=torch.float64)
        for seed in cfg['seeds']:
            folder=root/'training'/f'{p.family}_h{p.horizon}_s{seed}'
            data=torch.load(folder/'data.pt',map_location='cuda',weights_only=False)
            for spec in task['methods']:
                budgets=torch.load(folder/(spec['name']+'.pt'),map_location='cuda',weights_only=False)['budgets']
                for budget,selected in budgets.items():
                    net.load_state_dict(selected['state_dict']);cost=float(valgraph(data['valx']).mean())
                    error=abs(cost-selected['validation_mean']);assert error<1e-10,(p.family,seed,spec['name'],budget,error)
                    assert selected['available_seconds']<=float(budget)
                    replay.append(dict(family=p.family,seed=seed,method=spec['name'],budget=budget,error=error))
        del net,valgraph;gc.collect();torch.cuda.empty_cache()
    report=dict(completed_utc=utc(),scope='Two-seed development study. These comparisons are descriptive, not confirmatory claims.',
        cases=rows,validation_replay=replay,max_validation_replay_error=max(r['error'] for r in replay))
    write(root/'report.json',report)
    write(root/'status.json',dict(stage='development_and_evaluation_complete',updated_utc=utc()))
    print('COMPLETE: development evaluation and all saved-checkpoint validation replays.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/results/hj_curvature_dev_v8b')
    args=parser.parse_args();torch.set_num_threads(1);evaluate(Path(args.root))


if __name__=='__main__':main()
