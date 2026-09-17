"""New v7 states and pre-access checkpoint locking, with independent plant checks."""
import argparse
from dataclasses import asdict
import datetime
import gc
import json
from pathlib import Path
import numpy as np
import torch
from experiments.hj_cotangent.systems import problem,teacher,actor,sample
from experiments.hj_cotangent.evaluation import PlanGraph,teacher_evaluation
from experiments.hj_cotangent.conditioning import local_trajectory_audit,independent_feedback_cost
from experiments.hj_gridfree.engine import write
from .study import verify,digest


def folder(root,task,seed):return root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'


def lock(root,cfg):
    paths=[]
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            out=folder(root,task,seed);assert (out/'complete.json').exists()
            for spec in task['methods']:
                paths += [out/(spec['name']+'.pt'),out/(spec['name']+'.json')]
    sources=[Path(p) for p in json.loads((root/'training_lock.json').read_text())['sources']]
    sources += [Path(__file__).resolve().relative_to(Path.cwd())]
    path=root/'evaluation_lock.json'
    if path.exists():
        record=json.loads(path.read_text())
        for p,sha in record['hashes'].items():assert digest(p)==sha,p
    else:
        write(path,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hashes={str(p):digest(p) for p in set(paths+sources)},
            scope='All selected budget bundles, training records and numerical sources fixed before v7 test generation.'))


def cases(cfg):
    for fi,task in enumerate(cfg['tasks']):
        for dim,condition,count in [(32,'nominal',512),(256,'nominal',512),(256,'changed',128)]:
            n=dim//2 if task['family']=='mechanical' else dim
            kwargs={}
            if condition=='changed':
                if task['family']=='building':kwargs=dict(weather_shift=-2.,budget=4.5,topology_seed=73011)
                else:
                    ref=problem(task['family'],n,task['horizon'])
                    kwargs=dict(coupling=ref.gamma*.8,budget=ref.bound*.9)
            p=problem(task['family'],n,task['horizon'],**kwargs)
            seed=cfg['test_seed_base']+fi*100000+dim*10+int(condition=='changed')
            yield task,p,condition,count,seed


def evaluate(root,cfg,task,p,condition,count,seed):
    key=f'{p.family}_h{p.horizon}_d{p.dim}_{condition}';out=root/'evaluation';out.mkdir(exist_ok=True)
    if (out/(key+'.json')).exists():return
    x=sample(p,count,seed,shift=condition=='changed').double()
    net=actor(p,teacher(p),cfg['width']).cuda().eval();planner=PlanGraph(net,p,cfg['evaluation_batch'])
    tc,tr,tm=teacher_evaluation(x,p,cfg['audit_states'])
    raw=dict(x=x.cpu(),teacher=tc,teacher_audit=tr,methods={},audits={})
    meta=dict(problem=asdict(p),condition=condition,count=count,test_seed=seed,methods={},teacher_audit=tm,
        teacher_audit_scope='Analytic baseline controller. A selected frozen-parent teaching policy is represented separately by the parent-policy costs.',
        selected_teacher_type='iteratively_promoted_student' if cfg.get('interface')=='trajectory_transport' else ('frozen_parent' if task.get('teacher_parent_name') else 'analytic'))
    for spec in task['methods']:
        name=spec['name']
        if spec.get('parent_only'):continue
        raw['methods'][name]={};raw['audits'][name]={};meta['methods'][name]={}
        bundles=[torch.load(folder(root,task,ts)/(name+'.pt'),map_location='cpu',weights_only=False)['budgets'] for ts in cfg['seeds']]
        for budget in cfg['budgets_seconds']:
            bkey=str(budget);rows=[];audit_rows=[];stored=[]
            for ts,bundle in zip(cfg['seeds'],bundles):
                assert bkey in bundle,(name,ts,bkey)
                state=bundle[bkey]['state_dict']
                net.net.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('net.')})
                costs=[];max_budget=0.
                for offset in range(0,count,cfg['evaluation_batch']):
                    batch=x[offset:offset+cfg['evaluation_batch']];j,u=planner(batch)
                    costs.append(j.cpu())
                    if p.family=='building':
                        ratio=u.max().item()/p.bound;assert u.min().item()>=-1e-7
                    else:ratio=u.square().mean(-1).sqrt().max().item()/p.bound
                    max_budget=max(max_budget,ratio)
                    assert torch.isfinite(j).all() and ratio<1.00001
                    if offset==0 and budget==max(cfg['budgets_seconds']):
                        ncheck=cfg['audit_states'];states=planner.states[:ncheck].clone();controls=u[:ncheck].clone()
                        local=local_trajectory_audit(states,controls,j[:ncheck],p)
                        independent=independent_feedback_cost(batch,net,p)
                        assert torch.isfinite(independent).all()
                        audit_rows.append(dict(seed=ts,**local,
                            feedback_scaled_difference=((independent-j.cpu()).abs()/(1+independent.abs())).max().item(),
                            feedback_relative_mean_difference=abs(independent.mean().item()/j.mean().item()-1)))
                        stored.append(dict(seed=ts,states=states.cpu(),controls=controls.cpu(),
                            trajectory_costs=j[:ncheck].cpu(),feedback_costs=independent))
                if audit_rows and budget==max(cfg['budgets_seconds']):audit_rows[-1]['max_budget_ratio']=max_budget
                rows.append(torch.cat(costs))
            values=torch.stack(rows);raw['methods'][name][bkey]=values
            raw['audits'][name][bkey]=stored
            meta['methods'][name][bkey]=dict(mean=values.mean().item(),seed_means=values.mean(1).tolist(),
                p95=float(np.quantile(values.numpy(),.95)),audits=audit_rows)
    parent_name=next(s['parent'] for s in task['methods'] if s['name']=='warm')
    if parent_name=='short':
        parent_cost=raw['methods']['short'][str(cfg['parent_budget_seconds'])].clone()
        parent_reused=True
    else:
        parent_rows=[]
        for ts in cfg['seeds']:
            saved=torch.load(folder(root,task,ts)/(parent_name+'.pt'),map_location='cpu',weights_only=False)['budgets'][str(cfg['parent_budget_seconds'])]
            net.net.load_state_dict({k[4:]:v for k,v in saved['state_dict'].items() if k.startswith('net.')})
            values=[]
            for offset in range(0,count,cfg['evaluation_batch']):
                j,u=planner(x[offset:offset+cfg['evaluation_batch']]);assert torch.isfinite(j).all()
                if p.family=='building':assert u.min().item()>=-1e-7 and u.max().item()/p.bound<1.00001
                else:assert u.square().mean(-1).sqrt().max().item()/p.bound<1.00001
                values.append(j.cpu())
            parent_rows.append(torch.cat(values))
        parent_cost=torch.stack(parent_rows);parent_reused=False
    raw['parent']=parent_cost
    meta['parent']=dict(method=parent_name,budget=cfg['parent_budget_seconds'],mean=parent_cost.mean().item(),
        reused_existing_short_budget_costs=parent_reused,scope='Same parent used to initialize the dependent methods; pre-existing parent quality is not a child improvement.')
    torch.save(raw,out/(key+'.pt'));write(out/(key+'.json'),meta)
    print(json.dumps(dict(case=key,means={name:row[str(max(cfg['budgets_seconds']))]['mean'] for name,row in meta['methods'].items()})),flush=True)
    del net,planner,raw,x;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    torch.set_num_threads(1);root=Path(args.root);cfg=verify(root);lock(root,cfg)
    for values in cases(cfg):evaluate(root,cfg,*values)
    print('V7 EVALUATION COMPLETE',flush=True)


if __name__=='__main__':main()
