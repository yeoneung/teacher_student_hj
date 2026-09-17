"""Controlled teaching study and prospectively frozen wall-budget replication."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path
import torch
from impact_core import HERE, SUB, PROJECT, ValueGraph, load_policy, write, digest, perturb_signal, exact_probe
from experiments.hj_cotangent.systems import problem, sample, rollout, step, running
from experiments.hj_proximal.targets import coefficients, feasible, expose, reconstruct, RegressionGraph
from experiments.hj_target_v9.controls import CachedHamiltonianGraph

METHODS = ['clone','hamiltonian','fixed16','adaptive','guided']


@torch.no_grad()
def data_at(policy,p,x,t,jet,magnitude,noise,seed):
    old=policy(x,t).double(); y=step(x,old,p,t)
    # A captured graph computes its derivatives independently of this outer context.
    v,cov=jet(y,(t+1).clamp_max(p.horizon));r,b=coefficients(p,t,cov)
    gen=torch.Generator(device='cuda').manual_seed(seed)
    noisy,delta=perturb_signal(b,r,p,magnitude,noise,gen)
    features,base,_=expose(policy,x.float(),t)
    return dict(x=x,t=t,old=old,b=noisy,true_b=b,noise=delta,r=r,
                features=features,base=base)


def labels(data,p,eta,method):
    rho=2*data['r']*eta
    if method=='clone':target=data['old']
    elif method=='hamiltonian':target=(data['b']-rho*data['old'])/(data['r']+rho/2)
    else:target=feasible((rho*data['old']-data['b'])/(2*data['r']+rho),p)
    return target.float()


@torch.no_grad()
def student_states(policy,p,initial,count,gen):
    _,xs,_=rollout(initial,policy,p,keep=True)
    indices=torch.randint(xs.shape[0]*p.horizon,(count,),device='cuda',generator=gen)
    return xs.reshape(-1,p.dim)[indices].double(),indices % p.horizon


@torch.no_grad()
def audit_operation(teacher,student,p,data,eta,value,jet,initial,magnitude,noise,seed):
    """Finite audit rule; fresh sensitivities are charged, never population certified.

    Compare actual target/student advantages on student states. Refresh state
    coverage if a fresh target improves while the nearest cached target fails;
    fit if the fresh target improves but regression erases that improvement;
    regularize if the fresh target itself fails. The source of remaining model
    error (noise versus curvature) is not identified by this test.
    """
    gen=torch.Generator(device='cuda').manual_seed(seed)
    x,t=student_states(student,p,initial[:4],16,gen)
    fresh=data_at(teacher,p,x,t,jet,magnitude,noise,seed+13)
    old=fresh['old'];u=student(x,t).double();r=fresh['r'];rho=2*r*eta
    # Nearest cached state among equal decision times, with time-distance fallback.
    distance=torch.cdist(x.float(),data['x'].float())/p.dim**.5
    distance=distance+10*(t[:,None]-data['t'][None,:]).abs()/p.horizon
    nearest=distance.argmin(1);cached_b=data['b'][nearest]
    target=feasible((rho*old-fresh['b'])/(2*r+rho),p)
    cached_target=feasible((rho*old-cached_b)/(2*r+rho),p)
    tt=(t+1).clamp_max(p.horizon)
    v0,_=value(step(x,old,p,t),tt)
    vs,_=value(step(x,u,p,t),tt)
    vt,_=value(step(x,target,p,t),tt)
    vc,_=value(step(x,cached_target,p,t),tt)
    c0=running(x,old,p,t)
    actual=float((running(x,u,p,t)-c0+vs-v0).mean())
    target_actual=float((running(x,target,p,t)-c0+vt-v0).mean())
    cached_actual=float((running(x,cached_target,p,t)-c0+vc-v0).mean())
    tol=1e-7
    if target_actual >= -tol: operation='regularize'
    elif cached_actual >= -tol: operation='refresh'
    elif actual > .5*target_actual: operation='fit'
    else: operation='refresh'
    return operation,dict(student_advantage=actual,fresh_target_advantage=target_actual,
        cached_target_advantage=cached_actual,mean_state_distance=float(distance.min(1).values.mean()))


def phase(p,teacher,initial_student,data,jet,value,trainx,valx,method,seed,magnitude,noise,
          mode,wall_budget=8.,common_seconds=0.):
    torch.cuda.synchronize();start=time.perf_counter()
    student=copy.deepcopy(initial_student)
    for z in student.net.parameters():z.requires_grad_(True)
    trainer=(CachedHamiltonianGraph if method=='hamiltonian' else RegressionGraph)(student,p,256,.002,'action')
    teacher.eval();gen=torch.Generator(device='cuda').manual_seed(seed)
    current=data;eta=16.;events=[];updates=0;accepted=0;rejected=0;audits=0;cache_age=0
    with torch.no_grad():
        bestval=float(rollout(valx,student,p).mean());incumbent=float(rollout(trainx,student,p).mean())
    best=copy.deepcopy(student.state_dict());accepted_state=copy.deepcopy(best)
    elapsed=lambda:common_seconds+time.perf_counter()-start
    best_time=0.;lastval=bestval;blocks=0
    while (blocks<2 if mode=='mechanism' else elapsed()<wall_budget):
        y=labels(current,p,eta,method)
        before=copy.deepcopy(student.state_dict())
        for _ in range(128):
            if mode!='mechanism' and updates%16==0:
                torch.cuda.synchronize()
                if elapsed()>=wall_budget:break
            ix=torch.randint(len(y),(256,),device='cuda',generator=gen)
            trainer.update(current['features'][ix],current['base'][ix],y[ix]);updates+=1
        torch.cuda.synchronize();blocks+=1;cache_age+=128
        if mode!='mechanism' and elapsed()>=wall_budget:break
        with torch.no_grad():
            costs=rollout(trainx,student,p);trial=float(costs.mean())
            u=student(current['x'],current['t']).double()
            pred=-float((current['r']*(u.square()-current['old'].square())+current['b']*(u-current['old'])).sum(-1).mean())
        take=bool(torch.isfinite(costs).all()) and trial<=incumbent+1e-7*(1+abs(incumbent))
        actual=incumbent-trial;oldeta=eta
        operation='continue';detail={}
        if method=='guided':
            operation,detail=audit_operation(teacher,student,p,current,eta,value,jet,trainx,magnitude,noise,seed+blocks*911)
            audits+=1
            if operation=='regularize':eta=min(256.,eta*2)
            elif operation=='refresh':
                xx,tt=student_states(student,p,trainx[:8],256,gen)
                fresh=data_at(teacher,p,xx,tt,jet,magnitude,noise,seed+blocks*43)
                # Keep half the original state distribution and half recent states.
                current={k:(torch.cat([v[:256],fresh[k]]) if isinstance(v,torch.Tensor) else v) for k,v in data.items()}
                cache_age=0
        elif method=='adaptive':
            ratio=actual/(p.horizon*pred) if pred>1e-12 else None
            if not take or ratio is None or ratio<.25:eta=min(256.,eta*2)
            elif ratio>.75:eta=max(.25,eta/2)
        # Every method retains the same complete-cost gate, rollback and val chances.
        if take:
            accepted+=1;incumbent=trial;accepted_state=copy.deepcopy(student.state_dict())
        else:
            rejected+=1;student.load_state_dict(accepted_state)
        trainer.reset_optimizer()
        with torch.no_grad():lastval=float(rollout(valx,student,p).mean())
        state=copy.deepcopy(student.state_dict());torch.cuda.synchronize();available=elapsed()
        eligible=mode=='mechanism' or available<=wall_budget
        if eligible and lastval<bestval:
            bestval=lastval;best=state;best_time=available
        events.append(dict(block=blocks,updates=updates,accepted=take,training_cost=trial,
            eta_before=oldeta,eta_after=eta,operation=operation,cache_age_updates=cache_age,
            validation_cost=lastval,available_seconds=available,eligible=eligible,**detail))
    torch.cuda.synchronize();seconds=elapsed();student.load_state_dict(best)
    result=dict(method=method,seed=seed,mode=mode,updates=updates,accepted=accepted,rejected=rejected,
        audits=audits,charged_seconds=seconds,common_seconds=common_seconds,
        best_available_seconds=best_time,best_validation_cost=bestval,events=events,
        nonfinite_updates=int(trainer.nonfinite_updates.item()),wall_budget=wall_budget if mode!='mechanism' else None)
    del trainer;gc.collect();torch.cuda.empty_cache()
    return student,result


def run(stage):
    torch.set_num_threads(1);out=HERE/'results'/stage;out.mkdir(parents=True,exist_ok=True)
    if stage=='mechanism':
        seeds=[9911];qualities=[7.5,15.,30.];noise_specs=[('none',0.),('bias',.5),('gaussian',.5),('bias',1.5),('gaussian',1.5)]
        methods=['clone','hamiltonian','fixed16','guided'];mode='mechanism'
    else:
        seeds=[9951,9952,9953,9954,9955];qualities=[30.];noise_specs=[('none',0.),('bias',.5)]
        methods=METHODS;mode='wall'
    cfg=dict(stage=stage,seeds=seeds,qualities=qualities,noise_specs=noise_specs,methods=methods,
        wall_budget_seconds=8.,state_pool=512,training_initials=32,validation_initials=32,test_initials=256,
        student_batch=256,updates_per_block=128,mechanism_blocks=2,
        teacher_policy_source='archived Short checkpoints for mechanical; Warm DPC for reaction',
        student_initialization='same archived 7.5s parent across teacher qualities',
        rule='fresh-target failure: regularize; cached-target failure with fresh success: refresh; fitted-student erases half fresh-target improvement: fit; otherwise refresh',
        noise_floor='2*r*control_bound in per-state projected-signal RMS',
        scope='New exploratory controlled study; wall replication frozen independently of new test outcomes. Teacher construction reported separately, not included in teaching-only wall budget.')
    lock=out/'protocol_lock.json'
    source_hashes={p.name:digest(p) for p in [Path(__file__),HERE/'impact_core.py']}
    if lock.exists():assert json.loads(lock.read_text())['sources']==source_hashes
    else:write(lock,dict(config=cfg,sources=source_hashes))
    rows=[]
    for fi,family in enumerate(['mechanical','reaction']):
        p=problem(family,16 if family=='mechanical' else 32,160 if family=='mechanical' else 320)
        for si,seed in enumerate(seeds):
            teacher_seed=9501 if stage=='mechanism' else 9503+si
            initial_student,ip=load_policy(p,teacher_seed,'parent',7.5)
            teaching,provenance=load_policy(p,teacher_seed,'short' if family=='mechanical' else 'warm',7.5)
            for z in teaching.parameters():z.requires_grad_(False)
            setup=time.perf_counter();jet=ValueGraph(teaching,p,32,True);value=ValueGraph(teaching,p,32,False)
            torch.cuda.synchronize();graph_seconds=time.perf_counter()-setup
            trainx=sample(p,32,40100000+seed+fi*100000).double();valx=sample(p,32,41100000+seed+fi*100000).double()
            poolgen=torch.Generator(device='cuda').manual_seed(seed)
            common_x,common_t=student_states(initial_student,p,trainx,512,poolgen)
            for quality in qualities:
                policy,provenance=load_policy(p,teacher_seed,'short' if family=='mechanical' else 'warm',quality)
                teaching.load_state_dict(policy.state_dict());del policy
                with torch.no_grad():teacher_train=float(rollout(trainx,teaching,p).mean())
                for noise,magnitude in noise_specs:
                    key=f'{family}_s{seed}_q{quality:g}_{noise}{magnitude:g}'
                    setup=time.perf_counter();data=data_at(teaching,p,common_x,common_t,jet,magnitude,noise,seed+17)
                    torch.cuda.synchronize();common_seconds=time.perf_counter()-setup+graph_seconds
                    # Common preparation is attributed in full to every teaching method.
                    group=[]
                    import random
                    order=methods.copy();random.Random(seed+int(quality)*17+int(magnitude*10)).shuffle(order)
                    for method in order:
                        path=out/(key+'_'+method+'.json')
                        if path.exists():group.append(json.loads(path.read_text()));continue
                        write(out/'status.json',dict(stage='training',case=key,method=method))
                        student,result=phase(p,teaching,initial_student,data,jet,value,trainx,valx,method,seed,
                            magnitude,noise,mode,common_seconds=common_seconds)
                        # Lock this selected policy before constructing its test states.
                        checkpoint=out/(key+'_'+method+'.pt');torch.save(student.state_dict(),checkpoint)
                        policy_sha=digest(checkpoint)
                        testx=sample(p,256,42100000+seed+fi*100000).double()
                        with torch.no_grad():
                            test=rollout(testx,student,p);tc=rollout(testx,teaching,p);ic=rollout(testx,initial_student,p)
                        # Fresh student-state diagnostics use disjoint initial states.
                        dx=sample(p,4,43100000+seed+fi*100000).double()
                        gx=torch.Generator(device='cuda').manual_seed(seed+71)
                        xx,tt=student_states(student,p,dx,32,gx)
                        raw=exact_probe(teaching,student,p,xx,tt,jet,value,magnitude=magnitude,kind=noise,seed=seed+97)
                        # In the mechanism study assess fit and remainder without oracle bounds.
                        predicted=raw['fitted_model'] < -1e-8
                        result.update(case=key,family=family,teacher_quality_budget=quality,noise=noise,magnitude=magnitude,
                            teacher_seed=teacher_seed,teacher_provenance=provenance,initial_provenance=ip,
                            teacher_training_cost=teacher_train,teacher_test_cost=float(tc.mean()),
                            initial_student_test_cost=float(ic.mean()),test_cost=float(test.mean()),
                            policy_sha256=policy_sha,test_seed=42100000+seed+fi*100000,
                            teacher_construction_budget_seconds=quality,
                            diagnostic=dict(points=len(xx),predicted_improvements=int(predicted.sum()),
                                false_positive_predictions=int((predicted & (raw['actual']>1e-8)).sum()),
                                terms={k:float(raw[k].mean()) for k in ['actual','target_model','fitting_term','noise_term','curvature']},
                                relative_fit=float((raw['enorm']/raw['dnorm'].clamp_min(1e-8)).median())))
                        torch.save(dict(costs=test.cpu(),teacher_costs=tc.cpu(),initial_costs=ic.cpu(),
                            diagnostic={k:v.cpu() for k,v in raw.items()}),out/(key+'_'+method+'_evaluation.pt'))
                        write(path,result);group.append(result)
                        print(json.dumps({k:result[k] for k in ['case','method','test_cost','teacher_test_cost','updates','audits','charged_seconds']}),flush=True)
                        del student,raw;gc.collect();torch.cuda.empty_cache()
                    rows.extend(group)
            del initial_student,teaching,jet,value;gc.collect();torch.cuda.empty_cache()
    write(out/'report.json',dict(config=cfg,rows=rows,completed=True))
    write(out/'status.json',dict(stage='complete',phases=len(rows)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['mechanism','replication'],required=True)
    run(parser.parse_args().stage)
