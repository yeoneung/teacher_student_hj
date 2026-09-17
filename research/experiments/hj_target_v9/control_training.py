"""Copied frozen v8 budget loop; only declared component dependencies differ."""
import copy
from dataclasses import asdict
import gc
import json
import time
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import teacher as make_teacher,actor as make_actor,sample
from experiments.hj_cotangent.core import EvalGraph
from experiments.hj_adaptive.trajectory import TrajectoryGraph
from experiments.hj_proximal.targets import predicted_reduction
from .controls import make_control_trainer,collect_control,control_teaching_data

def train_control(p,cfg,seed,spec,data,out):
    target=out/spec['name']
    if target.with_suffix('.json').exists(): return
    torch.manual_seed(seed);torch.cuda.synchronize();started=time.perf_counter()
    parent=torch.load(out/'parent.pt',map_location='cuda',weights_only=False)['budgets']['7.5']
    actor=make_actor(p,make_teacher(p),cfg['width']).cuda()
    actor.load_state_dict(parent['state_dict'])
    teacher=copy.deepcopy(actor)
    trainer=make_control_trainer(actor,p,spec['student_batch'],spec['lr'],spec['kind'])
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
        costs,new_cache=collect_control(jet,initial,p,cfg['batch'],spec.get('zero_covector',False));torch.cuda.synchronize()
        value=costs.mean().item();finite=bool(torch.isfinite(costs).all())
        previous=incumbent
        actual=None if previous is None else previous-value
        ratio=actual/prediction if prediction is not None and prediction>1e-12 else None
        take=finite and (previous is None or value<=previous+1e-7*(1+abs(previous)))
        old_reg=regularization
        if take:
            accepted+=int(previous is not None);incumbent=value;cache=new_cache
            incumbent_state=copy.deepcopy(actor.state_dict())
            if previous is not None and not spec.get('fixed_regularization'):
                if ratio is None or ratio<.25:regularization=min(4096.,regularization*2)
                elif ratio>.75:regularization=max(.25,regularization/2)
        else:
            rejected+=1
            actor.load_state_dict(incumbent_state);teacher.load_state_dict(incumbent_state)
            regularization=regularization if spec.get('fixed_regularization') else min(4096.,regularization*2)
        trainer.reset_optimizer();cycles+=1
        # Recollecting features is deliberately charged, including failed cycles.
        if cache is not None:
            labels,metrics=control_teaching_data(actor,cache,p,regularization,spec['kind'])
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
