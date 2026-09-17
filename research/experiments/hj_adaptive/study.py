"""Strict available-checkpoint wall budgets, including all teacher-query work."""
import argparse
import copy
from dataclasses import asdict
import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import torch

from experiments.hj_cotangent.systems import problem,teacher as make_teacher,actor as make_actor,sample
from experiments.hj_cotangent.core import LocalGraph,JetGraph,EvalGraph
from experiments.hj_cotangent.study import make_data
from experiments.hj_gridfree.engine import write
from .probe import GradientProbe
from .trajectory import TrajectoryGraph
from .transport import TransportGraph,collect
from .full_batch import FullBatchGraph


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    cfg=json.loads((root/'config.json').read_text())
    if not cfg.get('development'):
        lock=json.loads((root/'training_lock.json').read_text())
        assert digest(root/'config.json')==lock['config_sha256']
        assert digest(root/'protocol.md')==lock['protocol_sha256']
        for path,sha in lock['sources'].items():assert digest(path)==sha,path
    return cfg


def train(p,cfg,seed,spec,data,out):
    name=spec['name'];target=out/name
    if target.with_suffix('.json').exists():return json.loads(target.with_suffix('.json').read_text())
    torch.manual_seed(seed);torch.cuda.synchronize()
    start=time.perf_counter();io=0.;parent_charge=0.
    actor=make_actor(p,make_teacher(p),cfg['width']).cuda()
    teacher=make_teacher(p)
    fallback=[]
    if spec.get('parent'):
        parent_record=json.loads((out/(spec['parent']+'.json')).read_text())
        parent_budget=cfg['parent_budget_seconds'];parent_charge=parent_budget-data['data_seconds']
        parent=torch.load(out/(spec['parent']+'.pt'),map_location='cuda',weights_only=False)
        actor.load_state_dict(parent['budgets'][str(parent_budget)]['state_dict'])
        fallback=[dict(row,from_parent=True) for row in parent_record['history'] if row['available_seconds']<=parent_budget and row.get('finite',True) and row['mean'] is not None]
    if spec.get('teacher_parent'):
        assert spec['teacher_parent']==spec['parent'],'Use a matched parent for this study'
        teacher=make_actor(p,make_teacher(p),cfg['width']).cuda()
        teacher.load_state_dict(parent['budgets'][str(cfg['parent_budget_seconds'])]['state_dict'])
    if spec['mode']=='transport':teacher=copy.deepcopy(actor)
    teacher.eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    assert all(parameter.requires_grad for parameter in actor.net.parameters())
    def elapsed():
        return data['data_seconds']+parent_charge+time.perf_counter()-start-io
    initial_state=copy.deepcopy(actor.state_dict())
    mode=spec['mode'];length=cfg['length'];batch=cfg['batch']
    dx,dt,valx=data['x'],data['t'],data['valx']
    initial_x=None
    if spec.get('initial_objective') or mode=='transport':
        initial_x=sample(p,cfg['training_initial_states'],110000+seed)
    if spec.get('initial_objective'):
        dx=initial_x;dt=torch.zeros(len(dx),device='cuda',dtype=torch.long)
    if spec.get('full_batch'):
        assert mode=='dpc' and spec.get('initial_objective')
        trainer=FullBatchGraph(actor,p,batch,cfg['training_initial_states']//batch,spec['lr'])
    elif mode=='transport':trainer=TransportGraph(actor,p,spec.get('student_batch',1024),spec['lr'],teacher,spec.get('variance_reduction',False))
    else:trainer=LocalGraph(actor,p,batch,length,lr=spec['lr'],mode=mode,rho=spec.get('rho',0.),teacher=teacher)
    jet=probe=cache=None
    if mode=='cache':
        jet=JetGraph(actor,teacher,p,batch,length)
        cache=(torch.zeros_like(dx),torch.zeros(len(dx),device='cuda'),torch.zeros_like(dx))
        if spec.get('adaptive'):
            probe=GradientProbe(actor,p,batch,length,spec.get('rho',0.))
    if mode=='transport':jet=TrajectoryGraph(teacher,p,batch,spec.get('variance_reduction',False))
    evaluator=EvalGraph(actor,p,len(valx),dtype=torch.float64)
    torch.cuda.synchronize();setup=elapsed()-data['data_seconds']-parent_charge
    budgets=sorted(set(spec.get('budgets_seconds',cfg['budgets_seconds']+[cfg['parent_budget_seconds']])))
    maximum=max(budgets)
    states={};records={};history=fallback[:];events=[]
    teacher_pairs=refreshes=probes=steps=0
    query_seconds=validation_seconds=0.
    last_refresh=0;fresh_until=0;fresh_updates=0;fresh_update_seconds=0.;gradient_floor=0.
    incumbent_cost=None;incumbent_state=None;incumbent_optimizer=None;accepted_refreshes=0;rejected_refreshes=0
    best=float('inf');best_state=initial_state;best_available=None;best_step=0
    if fallback:
        row=min(fallback,key=lambda r:r['mean'])
        best=row['mean'];best_available=row['available_seconds'];best_step=row['step']
        for budget in budgets:
            key=str(budget);parent_key=str(min(budget,cfg['parent_budget_seconds']))
            saved=parent['budgets'][parent_key]
            states[key]=saved['state_dict'];records[key]={k:v for k,v in saved.items() if k!='state_dict'}
            records[key].update(from_parent=True,budget_seconds=budget)
    last_validation=-float('inf')
    def validate():
        nonlocal best,best_state,best_step,best_available,validation_seconds,last_validation
        ts=time.perf_counter();cost=evaluator(valx);torch.cuda.synchronize()
        mean=cost.mean().item();finite=bool(torch.isfinite(cost).all().item())
        candidate_state=copy.deepcopy(actor.state_dict())
        torch.cuda.synchronize()
        available=elapsed();validation_seconds+=time.perf_counter()-ts;last_validation=available
        row=dict(step=steps,mean=mean if finite else None,finite=finite,available_seconds=available,
            teacher_pairs=teacher_pairs,refreshes=refreshes,probes=probes)
        history.append(row)
        if finite and mean<best and available<=maximum:
            best,best_state,best_step,best_available=mean,candidate_state,steps,available
        # A model is eligible only after its validation and snapshot finish.
        for budget in budgets:
            key=str(budget)
            if finite and available<=budget and (key not in records or mean<records[key]['validation_mean']):
                states[key]=candidate_state
                records[key]=dict(validation_mean=mean,best_step=steps,available_seconds=available,
                    budget_seconds=budget,teacher_pairs_at_selection=teacher_pairs,from_parent=False)
        return finite
    if elapsed()<maximum:validate()
    initial_validation=history[-1]['mean'] if history else None
    gen=torch.Generator(device='cuda').manual_seed(410000+seed)
    probe_gen=torch.Generator(device='cuda').manual_seed(710000+seed)
    stop_reason='budget'
    def refresh(reason,metrics=None):
        nonlocal teacher_pairs,refreshes,query_seconds,last_refresh
        ts=time.perf_counter()
        for offset in range(0,len(dx),batch):
            labels=jet(dx[offset:offset+batch],dt[offset:offset+batch])
            for dest,src in zip(cache,labels):dest[offset:offset+batch].copy_(src)
        torch.cuda.synchronize();query_seconds+=time.perf_counter()-ts
        teacher_pairs+=len(dx);refreshes+=1;last_refresh=steps
        events.append(dict(step=steps,reason=reason,available_seconds=elapsed(),metrics=metrics))
    def transport_refresh(reason):
        nonlocal cache,teacher_pairs,refreshes,query_seconds,last_refresh,incumbent_cost,incumbent_state,incumbent_optimizer,accepted_refreshes,rejected_refreshes
        ts=time.perf_counter();teacher.load_state_dict(actor.state_dict())
        collected=collect(jet,initial_x,p,batch);costs,candidate=collected[:2]
        torch.cuda.synchronize();value=costs.mean().item()
        accepted=bool(torch.isfinite(costs).all()) and (incumbent_cost is None or value<=incumbent_cost+1e-7*(1+abs(incumbent_cost)))
        previous=incumbent_cost
        if accepted:
            cache=candidate;incumbent_cost=value;incumbent_state=copy.deepcopy(actor.state_dict())
            incumbent_optimizer=trainer.optimizer_snapshot();accepted_refreshes+=1
            if spec.get('variance_reduction'):trainer.full_gradient.copy_(collected[2])
        else:
            assert incumbent_state is not None
            actor.load_state_dict(incumbent_state);teacher.load_state_dict(incumbent_state)
            trainer.restore_optimizer(incumbent_optimizer)
            trainer.lr.copy_((trainer.lr*.5).clamp_min(spec['lr']/64));rejected_refreshes+=1
            assert all(torch.equal(v,incumbent_state[k]) for k,v in actor.state_dict().items())
            assert all(torch.equal(a,b) for a,b in zip(trainer.optimizer_snapshot(),incumbent_optimizer))
        assert all(parameter.grad is None for parameter in teacher.parameters())
        torch.cuda.synchronize();query_seconds+=time.perf_counter()-ts
        teacher_pairs+=len(initial_x)*p.horizon;refreshes+=1;last_refresh=steps
        events.append(dict(step=steps,reason='transport_refresh',trigger_reason=reason,accepted=accepted,
            candidate_training_mean=value if math.isfinite(value) else None,previous_training_mean=previous,
            incumbent_training_mean=incumbent_cost,learning_rate=trainer.lr.item(),available_seconds=elapsed()))
    # The attempted work may cross the deadline by one block or one query.
    # Such work cannot contribute an eligible checkpoint and is reported.
    while elapsed()<maximum:
        if mode=='transport':
            age=steps-last_refresh
            if refreshes==0:transport_refresh('initial')
            elif spec.get('adaptive') and age and steps%spec['probe_every']==0:
                ts=time.perf_counter()
                idx=torch.randint(len(cache[0]),(spec.get('student_batch',1024),),device='cuda',generator=probe_gen)
                with torch.no_grad():drift=((actor(cache[0][idx],cache[1][idx])-cache[2][idx]).square().mean().sqrt()/p.bound).item()
                trigger=not math.isfinite(drift) or drift>spec['action_tolerance'] or age>=spec['max_age']
                probes+=1;query_seconds+=time.perf_counter()-ts
                events.append(dict(step=steps,reason='action_probe',action_drift=drift if math.isfinite(drift) else None,trigger=trigger,available_seconds=elapsed()))
                if trigger and elapsed()<maximum:transport_refresh('action_drift_or_maximum_age')
            elif not spec.get('adaptive') and age>=spec['refresh_every']:transport_refresh('fixed_period')
        if mode=='cache':
            age=steps-last_refresh
            if refreshes==0:
                refresh('initial')
                if spec.get('gradient_floor_fraction',0.)>0:
                    ts=time.perf_counter()
                    idx=torch.randint(len(dx),(batch,),device='cuda',generator=probe_gen)
                    labels=tuple(v[idx] for v in cache)
                    metrics=probe(dx[idx],dt[idx],labels,labels)
                    gradient_floor=spec['gradient_floor_fraction']*metrics['fresh_gradient_norm']
                    query_seconds+=time.perf_counter()-ts
                    events.append(dict(step=steps,reason='initial_gradient_scale',gradient_floor=gradient_floor,
                        available_seconds=elapsed(),metrics=metrics))
            elif fresh_until and steps>=fresh_until:
                refresh('return_to_cache');fresh_until=0
            elif steps<fresh_until:
                pass
            elif spec.get('adaptive') and steps and steps%spec['probe_every']==0:
                ts=time.perf_counter()
                idx=torch.randint(len(dx),(batch,),device='cuda',generator=probe_gen)
                labels=jet(dx[idx],dt[idx]);teacher_pairs+=batch;probes+=1
                metrics=probe(dx[idx],dt[idx],tuple(v[idx] for v in cache),labels)
                torch.cuda.synchronize();query_seconds+=time.perf_counter()-ts
                err=metrics['gradient_error_norm']/max(metrics['cached_gradient_norm'],gradient_floor,1e-12)
                metrics['scaled_gradient_error']=err
                metrics['gradient_floor']=gradient_floor
                trigger=not math.isfinite(err) or err>spec['gradient_tolerance']
                events.append(dict(step=steps,reason='probe',available_seconds=elapsed(),
                    trigger=trigger,metrics={k:v if math.isfinite(v) else None for k,v in metrics.items()}))
                if trigger and elapsed()<maximum:
                    if spec.get('fresh_window'):
                        fresh_until=steps+spec['fresh_window']
                        events.append(dict(step=steps,reason='fresh_minibatch_window',until_step=fresh_until,available_seconds=elapsed()))
                    else:refresh('gradient_error')
                elif age>=spec['max_age'] and elapsed()<maximum:refresh('maximum_age')
            elif not spec.get('adaptive') and age>=spec['refresh_every']:refresh('fixed_period')
        if elapsed()>=maximum:break
        block=min(spec.get('update_block',cfg['update_block']),spec.get('probe_every',cfg['update_block']))
        in_fresh_window=mode=='cache' and steps<fresh_until
        block_start=time.perf_counter()
        for _ in range(block):
            if spec.get('full_batch'):
                trainer.update_full(initial_x);steps+=1;continue
            if mode=='transport':
                idx=torch.randint(len(cache[0]),(spec.get('student_batch',1024),),device='cuda',generator=gen)
                trainer.update(*(v[idx] for v in cache))
                steps+=1;continue
            idx=torch.randint(len(dx),(batch,),device='cuda',generator=gen)
            if in_fresh_window:
                labels=jet(dx[idx],dt[idx]);teacher_pairs+=batch;fresh_updates+=1
                trainer.update(dx[idx],dt[idx],*labels)
            elif mode=='cache':trainer.update(dx[idx],dt[idx],*(v[idx] for v in cache))
            else:trainer.update(dx[idx],dt[idx])
            steps+=1
        if mode=='exact':teacher_pairs+=block*batch
        torch.cuda.synchronize()
        if in_fresh_window:fresh_update_seconds+=time.perf_counter()-block_start
        now=elapsed()
        if now<maximum and now-last_validation>=cfg['validation_interval_seconds']:
            if not validate():stop_reason='nonfinite_validation';break
        if steps>=cfg.get('max_attempts',1000000):stop_reason='attempt_cap';break
    torch.cuda.synchronize();actual=elapsed()
    # No post-deadline validation or post-deadline checkpoint is selected.
    assert str(maximum) in records,'No validated model available within maximum budget'
    packed={}
    for budget in budgets:
        key=str(budget)
        if key in records:
            assert records[key]['available_seconds']<=budget
            packed[key]=dict(records[key],state_dict={k:v.detach().cpu() for k,v in states[key].items()})
    teacher_unchanged=True
    if spec.get('teacher_parent'):
        reference=parent['budgets'][str(cfg['parent_budget_seconds'])]['state_dict']
        teacher_unchanged=all(torch.equal(value,reference[key]) for key,value in teacher.state_dict().items())
        assert teacher_unchanged and all(parameter.grad is None for parameter in teacher.parameters())
    result=dict(problem=asdict(p),seed=seed,method=spec,initial_validation=initial_validation,
        data_seconds=data['data_seconds'],parent_charge_seconds=parent_charge,setup_seconds=setup,
        total_compute_seconds=actual,budget_seconds=maximum,overrun_seconds=max(0.,actual-maximum),
        teacher_pairs=teacher_pairs,full_refreshes=refreshes,probe_queries=probes,query_seconds=query_seconds,
        fresh_minibatch_updates=fresh_updates,fresh_update_seconds=fresh_update_seconds,gradient_floor=gradient_floor,
        validation_seconds=validation_seconds,attempted_updates=steps,
        nonfinite_updates_skipped=int(trainer.nonfinite_updates.item()),stop_reason=stop_reason,
        clipped_updates=int(trainer.clipped_updates.item()),
        last_attempted_gradient_norm=trainer.grad_norm.item() if torch.isfinite(trainer.grad_norm).item() else None,
        budgets=records,history=history,events=events,
        teacher_type='frozen_parent' if spec.get('teacher_parent') else 'analytic',
        teacher_unchanged=teacher_unchanged,
        scope='Includes data, parent allocation, actor/graph setup, every probe/refresh, training, validation and synchronized device snapshots. Disk artifacts are written after completion. Selected checkpoints must be available before each deadline. CUDA context, process startup and final teardown are excluded uniformly. Query counts are scheduled boundary value/sensitivity pairs, including terminal records and masked graph slots; they exclude common teacher trajectories for data generation, whose wall cost is charged. query_seconds covers pool/probe work; fresh_update_seconds covers teacher queries plus student updates in fresh-minibatch windows.')
    if mode=='transport':
        result.update(teacher_type='iteratively_promoted_student',teacher_unchanged=False,
            accepted_refreshes=accepted_refreshes,rejected_refreshes=rejected_refreshes,
            teacher_trajectories=refreshes*len(initial_x),incumbent_training_mean=incumbent_cost,
            teacher_pairs_scope='One state-value covector per decision time of each teacher trajectory, including rejected promotions. All sweeps and action probes are timed. No extra tail query is made for an action-drift probe.')
    torch.save(dict(budgets=packed,problem=asdict(p),final_teacher_state=incumbent_state if mode=='transport' else None,
        final_teacher_gradient=trainer.full_gradient.detach().cpu() if mode=='transport' and spec.get('variance_reduction') else None),target.with_suffix('.pt'))
    write(target.with_suffix('.json'),result)
    print(json.dumps(dict(run=str(target),steps=steps,compute=actual,queries=teacher_pairs,
        refreshes=refreshes,probes=probes,budgets=records)),flush=True)
    del actor,teacher,trainer,jet,probe,cache,evaluator,states,best_state,initial_state
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--family');parser.add_argument('--seed',type=int);args=parser.parse_args()
    torch.set_num_threads(1);root=Path(args.root);cfg=verify(root)
    for task in cfg['tasks']:
        if args.family and task['family']!=args.family:continue
        for seed in cfg['seeds']:
            if args.seed and args.seed!=seed:continue
            out=root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}';out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
            data=make_data(p,cfg,seed,out)
            methods=task['methods'];shift=seed%len(methods);methods=methods[shift:]+methods[:shift]
            methods=[m for m in methods if not m.get('parent')]+[m for m in methods if m.get('parent')]
            for spec in methods:train(p,cfg,seed,spec,data,out)
            write(out/'complete.json',dict(methods=[m['name'] for m in methods],completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            del data;gc.collect();torch.cuda.empty_cache()
    print('V7 TRAINING COMPLETE',flush=True)


if __name__=='__main__':main()
