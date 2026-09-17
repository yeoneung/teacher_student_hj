"""Hash frozen policies before fresh tests; evaluate only eligible snapshots."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem,teacher,actor,sample
from experiments.hj_cotangent.evaluation import PlanGraph
from experiments.hj_cotangent.core import EvalGraph
from experiments.hj_cotangent.conditioning import local_trajectory_audit,independent_feedback_cost
from experiments.hj_proximal.study import sha,utc
from .study import verify,folder
from .statistics import family_decision


def signature(state):
    digest=hashlib.sha256()
    for k,v in sorted(state.items()):
        if k.startswith('net.'):
            digest.update(k.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze(root,cfg):
    path=root/'evaluation_lock.json'
    if path.exists():
        for name,expected in json.loads(path.read_text())['hashes'].items():assert sha(name)==expected,name
        return
    paths=[root/'config.json',Path(__file__),Path('experiments/hj_target_v9/statistics.py')]
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            out=folder(root,task,seed);assert (out/'complete.json').exists(),out
            paths.append(out/'data.pt')
            for spec in task['methods']:paths.extend([out/(spec['name']+'.pt'),out/(spec['name']+'.json')])
    write(path,dict(created_utc=utc(),hashes={str(p):sha(p) for p in paths},
        scope='All training data, numerical evaluation/statistics code and selected policy bundles frozen before v9 test-state generation.'))


def cases(cfg):
    for index,task in enumerate(cfg['tasks']):
        if cfg['stage']=='primary':conditions=[(32,'nominal',1024),(256,'nominal',512),(256,'changed',512)]
        elif cfg['stage']=='scaling':conditions=[(task['dimension'],'nominal',512),(task['dimension'],'changed',512)]
        else:conditions=[(32,'nominal',512)]
        for dim,condition,count in conditions:
            n=dim//2 if task['family']=='mechanical' else dim;kwargs={}
            if condition=='changed':
                if task['family']=='building':kwargs=dict(weather_shift=-2.,budget=4.5,topology_seed=93011)
                else:
                    base=problem(task['family'],n,task['horizon']);kwargs=dict(coupling=.8*base.gamma,budget=.9*base.bound)
            p=problem(task['family'],n,task['horizon'],**kwargs)
            test_seed=cfg['test_seed_base']+index*100000+dim*10+int(condition=='changed')
            yield task,p,condition,count,test_seed


def evaluate_case(root,cfg,task,p,condition,count,test_seed):
    key=f'{p.family}_train{task["dimension"]}_h{p.horizon}_eval{p.dim}_{condition}'
    out=root/'evaluation';out.mkdir(exist_ok=True);dest=out/key
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    x=sample(p,count,test_seed,shift=condition=='changed').double()
    net=actor(p,teacher(p),cfg['width']).cuda();planner=PlanGraph(net,p,64)
    raw=dict(x=x.cpu(),methods={},state_hashes={});meta=dict(key=key,family=p.family,training_dimension=task['dimension'],
        dimension=p.dim,horizon=p.horizon,condition=condition,count=count,test_seed=test_seed,problem=asdict(p),methods={},audits=[])
    cache={}
    for spec in task['methods']:
        name=spec['name'];budgets=[7.5] if spec.get('parent_only') else cfg['budgets_seconds']
        raw['methods'][name]={};raw['state_hashes'][name]={};meta['methods'][name]={}
        bundles={seed:torch.load(folder(root,task,seed)/(name+'.pt'),map_location='cpu',weights_only=False)['budgets'] for seed in cfg['seeds']}
        for budget in budgets:
            bkey=str(budget);rows=[];hashes=[]
            for seed in cfg['seeds']:
                if bkey not in bundles[seed]:rows.append(torch.full((count,),float('nan'),dtype=torch.float64));hashes.append(None);continue
                saved=bundles[seed][bkey];assert saved['available_seconds']<=budget
                state=saved['state_dict'];sig=signature(state);hashes.append(sig)
                net.net.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('net.')})
                need_audit=budget==max(budgets)
                if sig not in cache:
                    costs=[]
                    for offset in range(0,count,64):
                        j,u=planner(x[offset:offset+64]);assert torch.isfinite(j).all()
                        if p.family=='building':assert u.min()>=-1e-7 and u.max()<=p.bound+1e-6
                        else:assert u.square().mean(-1).sqrt().max()<=p.bound+1e-6
                        costs.append(j.cpu())
                    cache[sig]=torch.cat(costs)
                rows.append(cache[sig])
                if need_audit:
                    j,u=planner(x[:64]);local=local_trajectory_audit(planner.states[:8],u[:8],j[:8],p)
                    independent=independent_feedback_cost(x[:64],net,p)
                    error=float(((independent-j.cpu()).abs()/(1+independent.abs())).max())
                    assert error<2e-5,(key,name,seed,error)
                    assert torch.allclose(j.cpu(),cache[sig][:64],rtol=0,atol=0)
                    meta['audits'].append(dict(method=name,seed=seed,budget=budget,feedback_scaled_error=error,**local))
            values=torch.stack(rows);available=torch.isfinite(values).all(1)
            raw['methods'][name][bkey]=values;raw['state_hashes'][name][bkey]=hashes
            meta['methods'][name][bkey]=dict(mean=float(values.mean()) if available.all() else None,
                seed_means=[float(row.mean()) if finite else None for row,finite in zip(values,available)],
                available_seeds=int(available.sum()),total_seeds=len(available))
    torch.save(raw,dest.with_suffix('.pt'));write(dest.with_suffix('.json'),meta)
    print(json.dumps(dict(evaluation=key,final_costs={name:value[str(7.5 if name=='parent' else max(cfg['budgets_seconds']))]['mean'] for name,value in meta['methods'].items()})),flush=True)
    del net,planner,raw,cache;gc.collect();torch.cuda.empty_cache();return meta


def validation_replay(root,cfg):
    path=root/'validation_replay.json'
    if path.exists():return json.loads(path.read_text())
    rows=[]
    for task in cfg['tasks']:
        p=problem(task['family'],task['nodes'],task['horizon']);net=actor(p,teacher(p),cfg['width']).cuda()
        evaluator=EvalGraph(net,p,cfg['validation_states'],dtype=torch.float64)
        for seed in cfg['seeds']:
            out=folder(root,task,seed);data=torch.load(out/'data.pt',map_location='cuda',weights_only=False)
            for spec in task['methods']:
                bundle=torch.load(out/(spec['name']+'.pt'),map_location='cuda',weights_only=False)['budgets']
                for budget,saved in bundle.items():
                    net.load_state_dict(saved['state_dict']);value=float(evaluator(data['valx']).mean())
                    error=abs(value-saved['validation_mean']);assert error<1e-10,(out,spec['name'],budget,error)
                    assert saved['available_seconds']<=float(budget)
                    rows.append(dict(task=f'{p.family}_d{p.dim}_h{p.horizon}',seed=seed,method=spec['name'],budget=budget,error=error))
        del net,evaluator;gc.collect();torch.cuda.empty_cache()
    result=dict(completed_utc=utc(),rows=rows,maximum_error=max(r['error'] for r in rows));write(path,result);return result


def report(root,cfg,results,replay):
    decisions=[];time_curves=[]
    for case in results:
        final=str(max(cfg['budgets_seconds']));methods=case['methods']
        if cfg['stage']=='primary' and case['dimension']==32 and case['condition']=='nominal':
            means={name:methods[name][final]['seed_means'] for name in ['hjb_target','short','dpc','warm']}
            decisions.append(dict(family=case['family'],**family_decision(means)))
        if cfg['stage']!='controls':
            references={name:methods[name][final]['mean'] for name in ['short','dpc','warm']}
            best=min(references,key=references.get);threshold=references[best]*1.01
            points=[dict(budget=b,mean=methods['hjb_target'][str(b)]['mean'],ratio_to_final_best_reference=methods['hjb_target'][str(b)]['mean']/references[best] if methods['hjb_target'][str(b)]['mean'] is not None else None) for b in cfg['budgets_seconds']]
            reached=[r['budget'] for r in points if r['mean'] is not None and r['mean']<=threshold]
            time_curves.append(dict(case=case['key'],best_reference=best,reference_budget=float(final),threshold=threshold,
                earliest_grid_budget=min(reached) if reached else None,grid_speed_ratio=float(final)/min(reached) if reached else None,
                scope='Descriptive selected-grid threshold; not a simultaneous confidence statement.',points=points))
    result=dict(completed_utc=utc(),stage=cfg['stage'],cases=results,primary_decisions=decisions,
        time_to_quality=time_curves,validation_replay_count=len(replay['rows']),maximum_validation_replay_error=replay['maximum_error'])
    write(root/'report.json',result);write(root/'status.json',dict(stage='evaluation_complete',updated_utc=utc()))
    print(json.dumps(dict(stage=cfg['stage'],decisions=decisions,validation_replay_error=replay['maximum_error'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/results/hj_target_primary_v9')
    args=parser.parse_args();torch.set_num_threads(1);root=Path(args.root);cfg=verify(root);freeze(root,cfg)
    results=[evaluate_case(root,cfg,*case) for case in cases(cfg)]
    replay=validation_replay(root,cfg);report(root,cfg,results,replay)


if __name__=='__main__':main()
