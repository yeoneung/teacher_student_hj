"""Second development: retain unaccepted fitting proposals, preserve deployment gate.

The first mechanism study is frozen. Its guided rule gave no advantage; this
version repairs the operational mismatch between 'fit longer' and rollback.
"""
import argparse,copy,gc,json,random,time
from pathlib import Path
import torch
from impact_core import HERE,load_policy,write,digest,ValueGraph,exact_probe
from impact_learning import data_at,labels,student_states,audit_operation as original_audit_operation
from experiments.hj_cotangent.systems import problem,sample,rollout
from experiments.hj_proximal.targets import RegressionGraph
from experiments.hj_target_v9.controls import CachedHamiltonianGraph


def audit_operation(*args, **kwargs):
    operation, detail = original_audit_operation(*args, **kwargs)
    if detail['fresh_target_advantage'] < -1e-7 and detail['student_advantage'] > .5*detail['fresh_target_advantage']:
        operation = 'fit'
    return operation, detail


def train(p,teacher,parent,data,jet,value,trainx,valx,method,seed,common_seconds,budget):
    torch.cuda.synchronize();start=time.perf_counter();elapsed=lambda:common_seconds+time.perf_counter()-start
    student=copy.deepcopy(parent)
    for z in student.net.parameters():z.requires_grad_(True)
    trainer=(CachedHamiltonianGraph if method=='hamiltonian' else RegressionGraph)(student,p,512,.002,'action')
    eta=float(method[5:]) if method.startswith('fixed') else 16.
    gen=torch.Generator(device='cuda').manual_seed(seed)
    current=data;updates=blocks=accepted=rejected=audits=0;events=[];pending=0
    with torch.no_grad():
        incumbent=float(rollout(trainx,student,p).mean());bestval=float(rollout(valx,student,p).mean())
    accepted_state=copy.deepcopy(student.state_dict());best=copy.deepcopy(accepted_state);best_time=0.
    while elapsed()<budget:
        y=labels(current,p,eta,method)
        for _ in range(128):
            if updates%16==0:
                torch.cuda.synchronize()
                if elapsed()>=budget:break
            ix=torch.randint(len(y),(512,),device='cuda',generator=gen)
            trainer.update(current['features'][ix],current['base'][ix],y[ix]);updates+=1
        torch.cuda.synchronize();blocks+=1;pending+=128
        if elapsed()>=budget:break
        if method!='guided' and pending<512:continue
        with torch.no_grad():
            costs=rollout(trainx,student,p);trial=float(costs.mean())
            u=student(current['x'],current['t']).double()
            prediction=-float((current['r']*(u.square()-current['old'].square())+current['b']*(u-current['old'])).sum(-1).mean())*p.horizon
        take=bool(torch.isfinite(costs).all()) and trial<=incumbent+1e-7*(1+abs(incumbent))
        operation='continue';detail={};oldeta=eta
        if method=='guided':
            operation,detail=audit_operation(teacher,student,p,current,eta,value,jet,trainx,0.,'none',seed+blocks*911)
            audits+=1
            if operation=='regularize':eta=min(4096.,eta*2)
            if operation=='refresh':
                xx,tt=student_states(student,p,trainx[:8],64,gen)
                fresh=data_at(teacher,p,xx,tt,jet,0.,'none',seed+blocks*43)
                current={k:(torch.cat([v[64:],fresh[k]]) if isinstance(v,torch.Tensor) else v) for k,v in data.items()}
        elif method=='adaptive':
            ratio=(incumbent-trial)/prediction if prediction>1e-12 else None
            if not take or ratio is None or ratio<.25:eta=min(4096.,eta*2)
            elif ratio>.75:eta=max(.25,eta/2)
        if take:
            accepted+=1;incumbent=trial;accepted_state=copy.deepcopy(student.state_dict())
            with torch.no_grad():validation=float(rollout(valx,student,p).mean())
            state=copy.deepcopy(student.state_dict());torch.cuda.synchronize();available=elapsed()
            if available<=budget and validation<bestval:bestval=validation;best=state;best_time=available
        else:
            rejected+=1;validation=None
        # Fitting can continue internally; only accepted timely policies are eligible.
        carry=method=='guided' and operation=='fit' and pending<1024
        if not carry:
            if not take:student.load_state_dict(accepted_state)
            trainer.reset_optimizer();pending=0
        events.append(dict(block=blocks,updates=updates,accepted=take,operation=operation,
            retained_proposal=carry,eta_before=oldeta,eta_after=eta,training_cost=trial,
            validation_cost=validation,available_seconds=elapsed(),**detail))
    torch.cuda.synchronize();seconds=elapsed();student.load_state_dict(best)
    result=dict(method=method,seed=seed,updates=updates,accepted=accepted,rejected=rejected,audits=audits,
        best_validation_cost=bestval,best_available_seconds=best_time,charged_seconds=seconds,
        common_seconds=common_seconds,budget=budget,events=events,nonfinite_updates=int(trainer.nonfinite_updates.item()))
    del trainer;gc.collect();torch.cuda.empty_cache()
    return student,result


def run(stage):
    torch.set_num_threads(1);out=HERE/'results'/stage;out.mkdir(parents=True,exist_ok=True)
    if stage=='pilot':seeds=[9921];methods=['clone','hamiltonian','fixed16','fixed64','fixed256','adaptive','guided']
    else:seeds=[9951,9952,9953,9954,9955];methods=['clone','hamiltonian','selected_fixed','adaptive','guided']
    if stage=='replication_v3':
        pilot=json.loads((HERE/'results/pilot/report.json').read_text())['rows']
        selected={f:min([r for r in pilot if r['family']==f and r['method'].startswith('fixed')],key=lambda r:r['best_validation_cost'])['method'] for f in ['mechanical','reaction']}
    else:selected={}
    cfg=dict(stage=stage,seeds=seeds,methods=methods,selected_fixed=selected,budget=30.,
        teacher_budget=30.,training_initials=512,validation_initials=256,state_pool=4096,test_initials=512,
        batch=512,scope='Teacher held frozen to isolate teaching operation; same weak initial student and full training gate. Teacher construction is separate. All graphs and cache work charged to teaching budget.')
    sources={p.name:digest(p) for p in [Path(__file__),HERE/'impact_learning.py',HERE/'impact_core.py']}
    lock=out/'protocol_lock.json'
    if lock.exists():assert json.loads(lock.read_text())['sources']==sources
    else:write(lock,dict(config=cfg,sources=sources))
    rows=[]
    for fi,family in enumerate(['mechanical','reaction']):
        p=problem(family,16 if family=='mechanical' else 32,160 if family=='mechanical' else 320)
        for si,seed in enumerate(seeds):
            teacher_seed=9502 if stage=='pilot' else 9503+si
            parent,ip=load_policy(p,teacher_seed,'parent',7.5)
            teacher,tp=load_policy(p,teacher_seed,'short' if family=='mechanical' else 'warm',30.)
            start=time.perf_counter();jet=ValueGraph(teacher,p,32,True);value=ValueGraph(teacher,p,32,False)
            trainx=sample(p,512,45100000+seed+fi*100000).double();valx=sample(p,256,46100000+seed+fi*100000).double()
            gen=torch.Generator(device='cuda').manual_seed(seed)
            x,t=student_states(parent,p,trainx,4096,gen)
            data=data_at(teacher,p,x,t,jet,0.,'none',seed+17)
            torch.cuda.synchronize();common_seconds=time.perf_counter()-start
            order=methods.copy();random.Random(seed+fi*1000).shuffle(order)
            for original_method in order:
                method=selected[family] if original_method=='selected_fixed' else original_method
                path=out/f'{family}_s{seed}_{original_method}.json'
                if path.exists():rows.append(json.loads(path.read_text()));continue
                write(out/'status.json',dict(stage='training',family=family,seed=seed,method=method))
                student,result=train(p,teacher,parent,data,jet,value,trainx,valx,method,seed,common_seconds,30.)
                checkpoint=path.with_suffix('.pt');torch.save(student.state_dict(),checkpoint);policy_sha=digest(checkpoint)
                # Evaluation is independent of threshold/fixed-coefficient selection.
                testx=sample(p,512,47100000+seed+fi*100000).double()
                with torch.no_grad():
                    costs=rollout(testx,student,p);tc=rollout(testx,teacher,p);ic=rollout(testx,parent,p)
                dx=sample(p,4,48100000+seed+fi*100000).double()
                xx,tt=student_states(student,p,dx,32,gen)
                raw=exact_probe(teacher,student,p,xx,tt,jet,value)
                pred=raw['fitted_model'] < -1e-8
                result.update(family=family,comparison_label=original_method,test_cost=float(costs.mean()),
                    teacher_test_cost=float(tc.mean()),initial_test_cost=float(ic.mean()),teacher=tp,initial=ip,
                    policy_sha256=policy_sha,diagnostic=dict(predicted_improvements=int(pred.sum()),
                        false_positive_predictions=int((pred & (raw['actual']>1e-8)).sum())))
                torch.save(dict(costs=costs.cpu(),teacher_costs=tc.cpu(),initial_costs=ic.cpu(),
                    diagnostic={k:v.cpu() for k,v in raw.items()}),path.with_name(path.stem+'_evaluation.pt'))
                write(path,result);rows.append(result)
                print(json.dumps({k:result[k] for k in ['family','seed','method','test_cost','initial_test_cost','updates','audits','common_seconds']}),flush=True)
                del student;gc.collect();torch.cuda.empty_cache()
            del jet,value,teacher,parent;gc.collect();torch.cuda.empty_cache()
    write(out/'report.json',dict(config=cfg,rows=rows,completed=True));write(out/'status.json',dict(stage='complete',phases=len(rows)))


if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--stage',choices=['pilot','replication_v3'],required=True)
    run(args.parse_args().stage)
