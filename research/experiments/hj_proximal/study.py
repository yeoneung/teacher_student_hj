"""Matched wall-budget development; archived v7 implementations are imported."""
import argparse
import copy
from dataclasses import asdict
import datetime
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem, teacher as make_teacher, actor as make_actor, sample
from experiments.hj_cotangent.study import make_data
from experiments.hj_cotangent.core import EvalGraph
from experiments.hj_adaptive.trajectory import TrajectoryGraph
from experiments.hj_adaptive.transport import collect
from experiments.hj_adaptive.study import train as train_baseline
from .targets import RegressionGraph, teaching_data, predicted_reduction


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/'config.json').exists()
    old=json.loads(Path('experiments/results/hj_transport_primary_v7/config.json').read_text())
    tasks=[]
    for source in old['tasks']:
        family=source['family']
        methods={x['name']:copy.deepcopy(x) for x in source['methods']}
        parent=copy.deepcopy(methods.get('short_parent', methods['short']))
        parent.update(name='parent',budgets_seconds=[7.5],parent_only=True)
        parent.pop('parent',None)
        children=[]
        for name in ['short','dpc','warm','fixed']:
            spec=methods[name]
            if spec.get('parent'): spec['parent']='parent'
            children.append(spec)
        for kind in ['action','latent']:
            children.append(dict(name='prox_'+kind,mode='proximal',kind=kind,parent='parent',lr=.002,
                student_batch=1024,cycle_updates=256,regularization=16.))
        tasks.append(dict(family=family,horizon=source['horizon'],methods=[parent]+children))
    cfg={k:old[k] for k in ['training_nodes','dataset','training_initial_states','validation_states','batch','width','length','validation_interval_seconds','update_block']}
    cfg.update(development=True,seeds=[9401,9402],validation_seed_base=12400000,
        budgets_seconds=[7.5,15.,30.],parent_budget_seconds=7.5,tasks=tasks,
        scope='New v8 development only, no new confirmatory claim. Independent seeds; no v7 primary evaluation data.')
    write(root/'config.json',cfg)
    sources=[]
    for directory in ['hj_proximal','hj_adaptive','hj_cotangent','hj_gridfree']:
        sources.extend(Path('experiments',directory).glob('*.py'))
    sources += [root/'config.json',Path('docs/hj_v8_improvement_plan.md')]
    write(root/'development_lock.json',dict(created_utc=utc(),hashes={str(p):sha(p) for p in sources}))


def train_proximal(p,cfg,seed,spec,data,out):
    target=out/spec['name']
    if target.with_suffix('.json').exists(): return
    torch.manual_seed(seed);torch.cuda.synchronize();started=time.perf_counter()
    parent=torch.load(out/'parent.pt',map_location='cuda',weights_only=False)['budgets']['7.5']
    actor=make_actor(p,make_teacher(p),cfg['width']).cuda()
    actor.load_state_dict(parent['state_dict'])
    teacher=copy.deepcopy(actor)
    trainer=RegressionGraph(actor,p,spec['student_batch'],spec['lr'],spec['kind'])
    jet=TrajectoryGraph(teacher,p,cfg['batch'])
    evaluator=EvalGraph(actor,p,len(data['valx']),dtype=torch.float64)
    initial=sample(p,cfg['training_initial_states'],110000+seed)
    # The complete parent allocation includes data work; do not charge it twice.
    elapsed=lambda:cfg['parent_budget_seconds']+time.perf_counter()-started
    maximum=max(cfg['budgets_seconds']);budgets=cfg['budgets_seconds']
    selected={str(b):dict(parent,budget_seconds=b,from_parent=True) for b in budgets}
    history=[];events=[];steps=0;cycles=0;accepted=0;rejected=0
    query_seconds=validation_seconds=0.;last_validation=cfg['parent_budget_seconds']
    regularization=spec['regularization'];cache=labels=None;incumbent=None
    incumbent_state=copy.deepcopy(actor.state_dict())
    gen=torch.Generator(device='cuda').manual_seed(410000+seed)
    torch.cuda.synchronize();setup=elapsed()-cfg['parent_budget_seconds']

    def validate():
        nonlocal validation_seconds,last_validation
        start=time.perf_counter()
        costs=evaluator(data['valx']);torch.cuda.synchronize()
        mean=costs.mean().item();finite=bool(torch.isfinite(costs).all())
        state=copy.deepcopy(actor.state_dict());torch.cuda.synchronize()
        available=elapsed();validation_seconds+=time.perf_counter()-start;last_validation=available
        history.append(dict(step=steps,available_seconds=available,mean=mean if finite else None,finite=finite))
        if finite:
            for b in budgets:
                if available<=b and mean<selected[str(b)]['validation_mean']:
                    selected[str(b)]=dict(state_dict=state,validation_mean=mean,best_step=steps,
                        available_seconds=available,budget_seconds=b,from_parent=False)

    while elapsed()<maximum:
        start=time.perf_counter()
        prediction=None if cache is None else predicted_reduction(actor,labels,cache,p)
        teacher.load_state_dict(actor.state_dict())
        costs,new_cache=collect(jet,initial,p,cfg['batch']);torch.cuda.synchronize()
        value=costs.mean().item();finite=bool(torch.isfinite(costs).all())
        previous=incumbent
        actual=None if previous is None else previous-value
        ratio=actual/prediction if prediction is not None and prediction>1e-12 else None
        take=finite and (previous is None or value<=previous+1e-7*(1+abs(previous)))
        old_reg=regularization
        if take:
            accepted+=int(previous is not None);incumbent=value;cache=new_cache
            incumbent_state=copy.deepcopy(actor.state_dict())
            if previous is not None:
                if ratio is None or ratio<.25:regularization=min(4096.,regularization*2)
                elif ratio>.75:regularization=max(.25,regularization/2)
        else:
            rejected+=1
            actor.load_state_dict(incumbent_state);teacher.load_state_dict(incumbent_state)
            regularization=min(4096.,regularization*2)
        trainer.reset_optimizer();cycles+=1
        # Recollecting features is deliberately charged, including failed cycles.
        if cache is not None:
            labels,metrics=teaching_data(actor,cache,p,regularization,spec['kind'])
        else:raise RuntimeError('Initial teacher has nonfinite cost')
        torch.cuda.synchronize();query_seconds+=time.perf_counter()-start
        events.append(dict(cycle=cycles,step=steps,available_seconds=elapsed(),accepted=take,
            candidate_training_mean=value,previous_training_mean=previous,incumbent_training_mean=incumbent,
            predicted_reduction=prediction,actual_reduction=actual,agreement_ratio=ratio,
            previous_regularization=old_reg,regularization=regularization,**metrics))
        if elapsed()<maximum and elapsed()-last_validation>=cfg['validation_interval_seconds']:validate()
        if elapsed()>=maximum:break
        for block in range(spec['cycle_updates']//cfg['update_block']):
            if elapsed()>=maximum:break
            for _ in range(cfg['update_block']):
                idx=torch.randint(len(labels[0]),(spec['student_batch'],),device='cuda',generator=gen)
                trainer.update(*(v[idx] for v in labels));steps+=1
            torch.cuda.synchronize()
        # Students become validation candidates only after their next full-cost
        # acceptance decision. The original parent remains an eligible fallback.
    torch.cuda.synchronize();actual_elapsed=elapsed()
    packed={k:dict(v,state_dict={name:z.detach().cpu() for name,z in v['state_dict'].items()}) for k,v in selected.items()}
    records={k:{a:b for a,b in v.items() if a!='state_dict'} for k,v in selected.items()}
    for key,value in records.items():assert value['available_seconds']<=float(key)
    torch.save(dict(budgets=packed,problem=asdict(p)),target.with_suffix('.pt'))
    result=dict(problem=asdict(p),seed=seed,method=spec,budgets=records,history=history,events=events,
        parent_allocation_seconds=cfg['parent_budget_seconds'],setup_seconds=setup,query_seconds=query_seconds,
        validation_seconds=validation_seconds,total_compute_seconds=actual_elapsed,overrun_seconds=max(0.,actual_elapsed-maximum),
        attempted_updates=steps,cycles=cycles,accepted_cycles=accepted,rejected_cycles=rejected,
        nonfinite_updates_skipped=int(trainer.nonfinite_updates.item()),clipped_updates=int(trainer.clipped_updates.item()),
        scope='Development. Includes parent allocation, setup, labels, every trial teacher sweep, prediction checks, validation and device snapshots; excludes final disk write.')
    write(target.with_suffix('.json'),result)
    print(json.dumps(dict(run=str(target),steps=steps,accepted=accepted,rejected=rejected,budgets=records)),flush=True)
    del actor,teacher,trainer,jet,evaluator,selected,packed,cache,labels,new_cache,incumbent_state
    gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/results/hj_proximal_dev_v8a')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--family');parser.add_argument('--seed',type=int)
    args=parser.parse_args();root=Path(args.root);torch.set_num_threads(1)
    if args.prepare:prepare(root);return
    cfg=json.loads((root/'config.json').read_text())
    lock=json.loads((root/'development_lock.json').read_text())
    for path,expected in lock['hashes'].items():assert sha(path)==expected,path
    completed=[]
    for task in cfg['tasks']:
        if args.family and task['family']!=args.family:continue
        p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
        for seed in cfg['seeds']:
            if args.seed and args.seed!=seed:continue
            out=root/'training'/f'{p.family}_h{p.horizon}_s{seed}';out.mkdir(parents=True,exist_ok=True)
            data=make_data(p,cfg,seed,out)
            children=task['methods'][1:].copy();random.Random(seed).shuffle(children)
            for spec in [task['methods'][0]]+children:
                write(root/'status.json',dict(stage='training',family=p.family,seed=seed,method=spec['name'],updated_utc=utc()))
                if spec['mode']=='proximal':train_proximal(p,cfg,seed,spec,data,out)
                else:train_baseline(p,cfg,seed,spec,data,out)
            write(out/'complete.json',dict(completed_utc=utc(),methods=[s['name'] for s in task['methods']]))
            completed.append(str(out))
    write(root/'status.json',dict(stage='requested_training_complete',folders=completed,updated_utc=utc()))


if __name__=='__main__':main()
