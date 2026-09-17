"""Focused pretrained-student control using the exact matched pilot trainer.

The cold-start pilot and every outcome remain preserved. This control uses the
manuscript's pretrained student initialization to distinguish failure of a
short local teaching phase from an inability to bootstrap a distant policy.
Its protocol is saved before any warm-start training or evaluation.
"""
import copy,gc,json,random,time
from pathlib import Path
import torch
from impact_core import HERE,write,digest,load_policy,ValueGraph
from impact_learning import data_at,student_states
from experiments.hj_cotangent.systems import problem,sample,rollout
from improving_sets_study import phase,diagnostics,METHODS,KAPPA


def run():
    torch.set_num_threads(1);out=HERE/'results/improving_sets_warm';out.mkdir(parents=True,exist_ok=True)
    config=dict(methods=METHODS,teacher_seed=9508,seed=10101,width=64,updates=1024,
        initialization='Archived 7.5-second parent policy; paired across all methods.',
        geometry='Same realized feasible proximal anchor, eta=16, empirical kappa, epsilon=0.',
        trainer='Identical improving_sets_study.phase: batch256, gate256, refresh512, same queries and validation chances.',
        train_initials=64,validation_initials=128,test_initials=512,diagnostic_pairs=64,
        selection='Checkpoints and alpha selected only by validation; retain all alpha test results as exploratory.',
        decision='Independent paired replication only if validation-selected sets beat BOTH point and adaptive validation costs by at least 1% in a task.',
        inference='One teacher/student/data pair per task. No population or equal-wall-budget claim. Cold-start pilot fully retained.')
    sources={p.name:digest(p) for p in [Path(__file__),HERE/'improving_sets.py',HERE/'improving_sets_study.py']}
    lock=out/'protocol_lock.json'
    if lock.exists():assert json.loads(lock.read_text())==dict(config=config,sources=sources)
    else:write(lock,dict(config=config,sources=sources))
    rows=[]
    for fi,family in enumerate(['mechanical','reaction']):
        seed=10101+fi*100000;p=problem(family,16 if family=='mechanical' else 32,160 if family=='mechanical' else 320)
        teacher,tp=load_policy(p,9508,'short' if family=='mechanical' else 'warm',30.)
        initial,ip=load_policy(p,9508,'parent',7.5)
        setup=time.perf_counter();jet=ValueGraph(teacher,p,32,True)
        trainx=sample(p,64,40100000+seed).double();valx=sample(p,128,41100000+seed).double()
        gen=torch.Generator(device='cuda').manual_seed(seed)
        xx,tt=student_states(initial,p,trainx,512,gen);data=data_at(teacher,p,xx,tt,jet,0.,'none',seed)
        torch.cuda.synchronize();common_seconds=time.perf_counter()-setup
        value=ValueGraph(teacher,p,32,False)
        order=METHODS.copy();random.Random(seed+64).shuffle(order)
        for method in order:
            key=f'{family}_w64_s{seed}_{method}';path=out/(key+'.json')
            if path.exists():rows.append(json.loads(path.read_text()));continue
            write(out/'status.json',dict(status='training',case=key,completed=len(rows),total=12))
            student,result=phase(teacher,initial,p,data,jet,trainx,valx,method,seed,1024,common_seconds)
            checkpoint=out/(key+'.pt');torch.save(student.state_dict(),checkpoint)
            testx=sample(p,512,42100000+seed).double()
            with torch.no_grad():tc=rollout(testx,teacher,p);ic=rollout(testx,initial,p);sc=rollout(testx,student,p)
            diag,raw=diagnostics(student,teacher,p,jet,value,seed,result['alpha'])
            result.update(family=family,test_cost=float(sc.mean()),teacher_cost=float(tc.mean()),initial_cost=float(ic.mean()),
                teacher_provenance=tp,initial_provenance=ip,checkpoint_sha256=digest(checkpoint),diagnostics=diag)
            torch.save(dict(test_cost=sc,teacher_cost=tc,initial_cost=ic,diagnostic=raw),out/(key+'_evaluation.pt'))
            write(path,result);rows.append(result)
            print(key,round(result['test_cost'],7),'val',round(result['selected_validation_cost'],7),'seconds',round(result['charged_seconds'],2),flush=True)
            del student;gc.collect();torch.cuda.empty_cache()
        del jet,value,teacher,initial;gc.collect();torch.cuda.empty_cache()
    decisions=[]
    for family in ['mechanical','reaction']:
        group={r['method']:r for r in rows if r['family']==family}
        best=min([group[m] for m in ['set25','set50','set75']],key=lambda r:r['selected_validation_cost'])
        threshold=.99*min(group['point']['selected_validation_cost'],group['adaptive']['selected_validation_cost'])
        decisions.append(dict(family=family,selected_method=best['method'],validation_cost=best['selected_validation_cost'],
            replication_trigger=best['selected_validation_cost']<threshold,threshold=threshold))
    write(out/'report.json',dict(config=config,rows=rows,decisions=decisions))
    write(out/'status.json',dict(status='complete',completed=len(rows),total=12,decisions=decisions))
    print(json.dumps(decisions,indent=2),flush=True)


if __name__=='__main__':run()
