"""Matched-update pilot: point versus set targets from an identical HJ model.

Independent train/validation/test/diagnostic initials; alpha is never selected
on test results. All methods use identical refresh, gates, optimizer resets,
initial deployed policy, teacher and update allocation. Timings include common
teacher queries and graph preparation. This study makes no wall-time superiority
claim: the fixed number of updates is the intervention being controlled.
"""
import argparse,copy,gc,json,random,time
from pathlib import Path
import torch
from impact_core import HERE,write,digest,load_policy,ValueGraph
from impact_learning import data_at,student_states
from experiments.hj_cotangent.systems import problem,actor,sample,rollout,step,running
from improving_sets import SupervisionGraph,cache_geometry,intersection_project,upper_model

KAPPA={'mechanical':.000550492,'reaction':.000417389}
METHODS=['point','upper','adaptive','set25','set50','set75']


@torch.no_grad()
def diagnostics(student,teacher,p,jet,value,seed,alpha):
    initials=sample(p,4,45100000+seed).double()
    gen=torch.Generator(device='cuda').manual_seed(seed+5101)
    x,t=student_states(student,p,initials,64,gen)
    data=data_at(teacher,p,x,t,jet,0.,'none',seed)
    geo=cache_geometry(data,p,KAPPA[p.family],alpha)
    u=student(x,t).double();projected=intersection_project(u,geo['center'],geo['radius'],p.n**.5*p.bound)
    upper=upper_model(u,data['old'],data['b'],data['r'],KAPPA[p.family])
    v0,_=value(step(x,data['old'],p,t),t+1);vu,_=value(step(x,u,p,t),t+1)
    vp,_=value(step(x,projected,p,t),t+1)
    actual=running(x,u,p,t)-running(x,data['old'],p,t)+vu-v0
    projected_actual=running(x,projected,p,t)-running(x,data['old'],p,t)+vp-v0
    point_error=(u-geo['anchor']).square().sum(-1)
    distance=(u-projected).square().sum(-1)
    inside=upper<=-alpha*geo['gain']+1e-9
    return dict(point_mse=float(point_error.mean()/p.n/p.bound**2),set_mse=float(distance.mean()/p.n/p.bound**2),
        alpha=alpha,inside_fraction=float(inside.double().mean()),actual_advantage=float(actual.mean()),
        projected_actual_advantage=float(projected_actual.mean()),model_advantage=float(upper.mean()),
        mean_anchor_gain=float(geo['gain'].mean()),
        upper_violation_count=int((actual>upper+1e-7).sum()),
        projected_upper_violation_count=int((projected_actual>upper_model(projected,data['old'],data['b'],data['r'],KAPPA[p.family])+1e-7).sum()),
        nonimproving_inside_count=int(((actual>1e-7)&inside).sum()),count=len(x)),dict(data=data,geometry=geo,action=u,projection=projected,actual=actual,projected_actual=projected_actual)


def phase(teacher,initial,p,data,jet,trainx,valx,method,seed,updates,common_seconds):
    torch.cuda.synchronize();start=time.perf_counter();student=copy.deepcopy(initial)
    for z in student.net.parameters():z.requires_grad_(True)
    alpha=int(method[3:])/100 if method.startswith('set') else .5
    kind='set' if method.startswith('set') else ('upper' if method=='upper' else 'point')
    trainer=SupervisionGraph(student,p,256,.002,kind)
    gen=torch.Generator(device='cuda').manual_seed(seed+1001)
    eta=16.;current=data;geo=cache_geometry(current,p,KAPPA[p.family],alpha,eta)
    with torch.no_grad():
        incumbent=float(rollout(trainx,student,p).mean());bestval=float(rollout(valx,student,p).mean())
    accepted_state=copy.deepcopy(student.state_dict());best=copy.deepcopy(accepted_state);events=[]
    initial_val=bestval;fit_seconds=0.;refresh_seconds=0.;gate_seconds=0.;best_update=0
    for offset in range(0,updates,256):
        torch.cuda.synchronize();part=time.perf_counter()
        for _ in range(256):
            ix=torch.randint(len(current['old']),(256,),device='cuda',generator=gen)
            trainer.update(current['features'][ix],current['base'][ix],geo['anchor'][ix],geo['center'][ix],geo['radius'][ix])
        torch.cuda.synchronize();fit_seconds+=time.perf_counter()-part;part=time.perf_counter()
        with torch.no_grad():
            costs=rollout(trainx,student,p);trial=float(costs.mean());u=student(current['x'],current['t']).double()
            pred=-p.horizon*float((current['r']*(u.square()-current['old'].square())+current['b']*(u-current['old'])).sum(-1).mean())
        take=bool(torch.isfinite(costs).all()) and trial<=incumbent+1e-7*(1+abs(incumbent))
        ratio=(incumbent-trial)/pred if pred>1e-12 else None
        if method=='adaptive':
            if not take or ratio is None or ratio<.25:eta=min(4096.,eta*2)
            elif ratio>.75:eta=max(.25,eta/2)
        if take:incumbent=trial;accepted_state=copy.deepcopy(student.state_dict())
        else:student.load_state_dict(accepted_state)
        trainer.reset_optimizer()
        with torch.no_grad():val=float(rollout(valx,student,p).mean())
        if val<bestval:bestval=val;best=copy.deepcopy(student.state_dict());best_update=offset+256
        torch.cuda.synchronize();gate_seconds+=time.perf_counter()-part;part=time.perf_counter()
        # Same update-indexed refresh for every loss; teacher remains fixed.
        refresh=(offset+256)%512==0 and offset+256<updates
        if refresh:
            xx,tt=student_states(student,p,trainx[:8],256,gen)
            fresh=data_at(teacher,p,xx,tt,jet,0.,'none',seed+offset)
            current={k:torch.cat([v[:256],fresh[k]]) if isinstance(v,torch.Tensor) else v for k,v in data.items()}
        geo=cache_geometry(current,p,KAPPA[p.family],alpha,eta)
        torch.cuda.synchronize();refresh_seconds+=time.perf_counter()-part
        events.append(dict(updates=offset+256,accepted=take,trial_cost=trial,validation_cost=val,eta=eta,refresh=refresh,
                           charged_seconds=common_seconds+time.perf_counter()-start))
    student.load_state_dict(best);torch.cuda.synchronize()
    result=dict(method=method,alpha=alpha,width=initial.net[0].out_features,seed=seed,updates=updates,
        best_update=best_update,initial_validation_cost=initial_val,selected_validation_cost=bestval,
        charged_seconds=common_seconds+time.perf_counter()-start,common_seconds=common_seconds,
        fit_seconds=fit_seconds,refresh_seconds=refresh_seconds,gate_seconds=gate_seconds,
        nonfinite_updates=int(trainer.nonfinite_updates.item()),events=events)
    del trainer;gc.collect();torch.cuda.empty_cache()
    return student,result


def run(stage='pilot'):
    torch.set_num_threads(1)
    assert json.loads((HERE/'results/improving_sets_exact.json').read_text())['passed']
    out=HERE/'results'/('improving_sets_'+stage);out.mkdir(parents=True,exist_ok=True)
    config=dict(stage=stage,methods=METHODS,widths=[16,64],seeds=[10101],teacher_seeds=[9508],
        updates=1024,batch=256,gate_every=256,refresh_every=512,pool=512,
        train_initials=64,validation_initials=128,test_initials=512,diagnostic_initials=4,diagnostic_pairs=64,
        alpha_candidates=[.25,.5,.75],eta=16.,kappa=KAPPA,
        kappa_status='Fixed empirical directional estimates; not certified bounds for nonlinear systems.',
        initialization='Same deployed analytic LQR; zero final residual layer; random hidden layers paired within width.',
        selection='Within each run select the lowest validation-cost accepted checkpoint, including initialization. Alpha is selected by validation only within each task/width. Test data are generated after saving the selected checkpoint.',
        budget='Matched 1024 updates. All preparation, queries, refresh, projection, gates, validation, copies, and graph capture included in charged seconds; teacher pretraining excluded. No wall-time superiority claim.',
        decision='Run independent paired replications only if validation-selected sets reduce validation cost by at least 1% relative to BOTH point and adaptive in a task/width. Otherwise report exploratory results without a superiority claim.',
        distinction='No continuation-audit branch. Empirical set is not certified. Exact sensitivity is the numerical derivative of the supplied teacher, with floating-point arithmetic.')
    sources={p.name:digest(p) for p in [Path(__file__),HERE/'improving_sets.py',HERE/'impact_core.py',HERE/'impact_learning.py']}
    lock=out/'protocol_lock.json'
    if lock.exists():assert json.loads(lock.read_text())==dict(config=config,sources=sources)
    else:write(lock,dict(config=config,sources=sources))
    rows=[]
    for fi,family in enumerate(['mechanical','reaction']):
        p=problem(family,16 if family=='mechanical' else 32,160 if family=='mechanical' else 320)
        teacher,provenance=load_policy(p,9508,'short' if family=='mechanical' else 'warm',30.)
        setup=time.perf_counter();jet=ValueGraph(teacher,p,32,True)
        torch.cuda.synchronize();jet_seconds=time.perf_counter()-setup
        # Value-only graph is for post-training diagnostics; excluded from training.
        value=ValueGraph(teacher,p,32,False)
        for width in config['widths']:
            seed=10101+fi*100000
            torch.manual_seed(seed);initial=actor(p,copy.deepcopy(teacher.base),width).cuda().eval()
            setup=time.perf_counter()
            trainx=sample(p,64,40100000+seed).double();valx=sample(p,128,41100000+seed).double()
            poolgen=torch.Generator(device='cuda').manual_seed(seed)
            xx,tt=student_states(initial,p,trainx,512,poolgen)
            data=data_at(teacher,p,xx,tt,jet,0.,'none',seed)
            torch.cuda.synchronize();common_seconds=jet_seconds+time.perf_counter()-setup
            order=METHODS.copy();random.Random(seed+width).shuffle(order)
            for method in order:
                key=f'{family}_w{width}_s{seed}_{method}';path=out/(key+'.json')
                if path.exists():rows.append(json.loads(path.read_text()));continue
                write(out/'status.json',dict(status='training',case=key,completed=len(rows),total=24))
                student,result=phase(teacher,initial,p,data,jet,trainx,valx,method,seed,config['updates'],common_seconds)
                checkpoint=out/(key+'.pt');torch.save(student.state_dict(),checkpoint)
                testx=sample(p,512,42100000+seed).double()
                with torch.no_grad():
                    tc=rollout(testx,teacher,p);ic=rollout(testx,initial,p);sc=rollout(testx,student,p)
                diag,raw=diagnostics(student,teacher,p,jet,value,seed,result['alpha'])
                result.update(family=family,test_cost=float(sc.mean()),teacher_cost=float(tc.mean()),initial_cost=float(ic.mean()),
                              teacher_provenance=provenance,checkpoint_sha256=digest(checkpoint),diagnostics=diag)
                torch.save(dict(test_cost=sc,teacher_cost=tc,initial_cost=ic,diagnostic=raw),out/(key+'_evaluation.pt'))
                write(path,result);rows.append(result)
                print(key,round(result['test_cost'],7),'val',round(result['selected_validation_cost'],7),'seconds',round(result['charged_seconds'],2),flush=True)
                del student;gc.collect();torch.cuda.empty_cache()
        del jet,value,teacher;gc.collect();torch.cuda.empty_cache()
    decisions=[]
    for family in ['mechanical','reaction']:
        for width in config['widths']:
            group={r['method']:r for r in rows if r['family']==family and r['width']==width}
            best=min([group[m] for m in ['set25','set50','set75']],key=lambda r:r['selected_validation_cost'])
            threshold=.99*min(group['point']['selected_validation_cost'],group['adaptive']['selected_validation_cost'])
            decisions.append(dict(family=family,width=width,selected_method=best['method'],validation_cost=best['selected_validation_cost'],
                                  replication_trigger=best['selected_validation_cost']<threshold,threshold=threshold))
    write(out/'report.json',dict(config=config,rows=rows,decisions=decisions))
    write(out/'status.json',dict(status='complete',completed=len(rows),total=24,decisions=decisions))
    print(json.dumps(decisions,indent=2),flush=True)


if __name__=='__main__':run()
