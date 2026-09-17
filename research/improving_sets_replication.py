"""Independent five-pair replication of validation-selected alpha=0.25.

Only reaction triggers the prospectively stated warm-pilot validation rule.
The complete warm pilot includes its adverse test result; it is not pooled
with the independent replications or used to select this alpha.
"""
import gc,json,random,time
from pathlib import Path
import torch
from impact_core import HERE,write,digest,load_policy,ValueGraph
from impact_learning import data_at,student_states
from experiments.hj_cotangent.systems import problem,sample,rollout
from improving_sets_study import phase,diagnostics


def run():
    torch.set_num_threads(1);out=HERE/'results/improving_sets_replication';out.mkdir(parents=True,exist_ok=True)
    pilot=json.loads((HERE/'results/improving_sets_warm/report.json').read_text())
    eligible=[d for d in pilot['decisions'] if d['replication_trigger']]
    assert len(eligible)==1 and eligible[0]['family']=='reaction' and eligible[0]['selected_method']=='set25'
    config=dict(family='reaction',methods=['point','adaptive','upper','set25'],width=64,alpha=.25,
        teacher_seeds=[9501,9502,9503,9504,9505],seeds=[10201,10202,10203,10204,10205],updates=1024,
        initialization='Archived 7.5-second parent per teacher seed.',
        trainer='Unchanged improving_sets_study.phase; same update, refresh and gate schedules.',
        train_initials=64,validation_initials=128,test_initials=512,diagnostic_pairs=64,
        selection='Alpha frozen from disjoint warm-pilot validation; select checkpoint by within-replicate validation; generate test initials only after checkpoint save.',
        inference='Five paired seed means, unadjusted two-sided 95% t intervals, df4. Pilot not pooled. Matched updates; charged times reported without equal-wall-budget claim.',
        pilot_sha256=digest(HERE/'results/improving_sets_warm/report.json'))
    sources={p.name:digest(p) for p in [Path(__file__),HERE/'improving_sets.py',HERE/'improving_sets_study.py']}
    lock=out/'protocol_lock.json'
    if lock.exists():assert json.loads(lock.read_text())==dict(config=config,sources=sources)
    else:write(lock,dict(config=config,sources=sources))
    rows=[];p=problem('reaction',32,320)
    for seed,teacher_seed in zip(config['seeds'],config['teacher_seeds']):
        teacher,tp=load_policy(p,teacher_seed,'warm',30.);initial,ip=load_policy(p,teacher_seed,'parent',7.5)
        setup=time.perf_counter();jet=ValueGraph(teacher,p,32,True)
        trainx=sample(p,64,40100000+seed).double();valx=sample(p,128,41100000+seed).double()
        gen=torch.Generator(device='cuda').manual_seed(seed)
        xx,tt=student_states(initial,p,trainx,512,gen);data=data_at(teacher,p,xx,tt,jet,0.,'none',seed)
        torch.cuda.synchronize();common_seconds=time.perf_counter()-setup
        value=ValueGraph(teacher,p,32,False)
        order=config['methods'].copy();random.Random(seed).shuffle(order)
        for method in order:
            key=f'reaction_w64_s{seed}_{method}';path=out/(key+'.json')
            if path.exists():rows.append(json.loads(path.read_text()));continue
            write(out/'status.json',dict(status='training',case=key,completed=len(rows),total=20))
            student,result=phase(teacher,initial,p,data,jet,trainx,valx,method,seed,1024,common_seconds)
            checkpoint=out/(key+'.pt');torch.save(student.state_dict(),checkpoint)
            testx=sample(p,512,42100000+seed).double()
            with torch.no_grad():tc=rollout(testx,teacher,p);ic=rollout(testx,initial,p);sc=rollout(testx,student,p)
            diag,raw=diagnostics(student,teacher,p,jet,value,seed,result['alpha'])
            result.update(family='reaction',teacher_seed=teacher_seed,test_cost=float(sc.mean()),teacher_cost=float(tc.mean()),initial_cost=float(ic.mean()),
                teacher_provenance=tp,initial_provenance=ip,checkpoint_sha256=digest(checkpoint),diagnostics=diag)
            torch.save(dict(test_cost=sc,teacher_cost=tc,initial_cost=ic,diagnostic=raw),out/(key+'_evaluation.pt'))
            write(path,result);rows.append(result)
            print(key,round(result['test_cost'],7),'val',round(result['selected_validation_cost'],7),'seconds',round(result['charged_seconds'],2),flush=True)
            del student;gc.collect();torch.cuda.empty_cache()
        del jet,value,teacher,initial;gc.collect();torch.cuda.empty_cache()
    write(out/'report.json',dict(config=config,rows=rows))
    write(out/'status.json',dict(status='complete',completed=len(rows),total=20))


if __name__=='__main__':run()
