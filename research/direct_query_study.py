"""Frozen direct-query controls and constructed rank-deficient neural study.

Original data/seeds are reused for interface fidelity, not independent confirmation.
Structured cases and all endpoints are fixed before any new fitting.
"""
import copy,gc,json,random,time
from pathlib import Path
import torch
import minimal_teaching as mt
import minimal_teaching_study as st
import direct_query_controls as dc

HERE=Path(__file__).resolve().parent;OUT=HERE/'results/direct_query_study'
OLD=HERE/'results/minimal_teaching_study'
CFG=dict(original_conditions=[[8,2,'box',.5],[32,2,'box',.5],[32,4,'box',2.],[32,4,'ball',.5]],
    original_seeds=list(range(13101,13106)),new_interfaces=dc.METHODS,
    structured_seeds=list(range(13201,13206)),structured_methods=['point','rank','face','full']+dc.METHODS,
    structured_geometry='d8,m4,R.5 box; P=[(e1+e2+e3)/sqrt3,e4,e5,e6]H for random orthogonal H. First three initial coordinates independently signed uniform magnitude6..10; others uniform[-.1,.1]. All teacher-cache states have k3,r1; every active normal and every student basis direction is visible.',
    training_runs=120,training=st.CFG,
    primary='Exact projected-coefficient recovery and actual scalar counts for four new interfaces on all80 original caches and5 structural caches. Expected relevant=hybrid=min(k,m) on original caches. Structural counts relevant1,hybrid/pruned3,student4.',
    secondary='Final paired normalized regret and closed-loop cost fidelity versus full information; tolerance1e-3 regret per pair, not population equivalence. All120 fits retained, no superiority hypothesis or checkpoint selection.',
    timing='Synchronized GPU wall time for geometry, feasible-query construction, actual oracle response and decoding; common target preparation excluded. Orthogonal student basis construction included. One warmup plus three repeats in shuffled order per cache; cheap analytic oracle. Not end-to-end acceleration.',
    interpretation='Follow-up conditions chosen after inspecting existing results. Original-cache fits reuse paired seeds, data and schedule, not new independent confirmation. Structural experiment is deliberately constructed shared-actuation geometry, not a naturally observed benchmark.',
    scope='Exact value differences with supplied target/baseline, known isotropic curvature, fixed orthonormal P. Numerical rank uses frozen Gram eigenvalue threshold1e-10. No claim for analytic-gradient access or unknown curvature.')

def load(p):return json.loads(Path(p).read_text())

def audit_cache(x,P,R,shape,seed,key):
    path=OUT/(key+'_feedback.json')
    if path.exists():return load(path)
    o=mt.Oracle(x,R,shape);geo=mt.geometry(o.target,P,R,shape)
    ref=2*mt.A*mt.B*(x@P);methods=['rank','face']+dc.METHODS;records={}
    for method in methods:
        z=dc.transmit(o,P,method,seed+933);delta=z['coefficient']-ref
        expected=geo['rank'] if method=='rank' else geo['face_dim'] if method=='face' else ((geo['face_dim']>0)*P.shape[1] if method.startswith('student') else torch.minimum(geo['face_dim'],torch.full_like(geo['face_dim'],P.shape[1])))
        assert torch.equal(z['queries'],expected)
        error=float(delta.abs().max());assert error<1e-9
        records[method]=dict(queries=int(z['queries'].sum()),maximum_coefficient_error=error,
            relative_coefficient_error=float(delta.square().mean().sqrt()/ref.square().mean().sqrt()),seconds=[])
    for repeat in range(3):
        order=methods.copy();random.Random(seed+repeat).shuffle(order)
        for method in order:
            torch.cuda.synchronize();start=time.perf_counter();dc.transmit(o,P,method,seed+933)
            torch.cuda.synchronize();records[method]['seconds'].append(time.perf_counter()-start)
    pairs,counts=torch.unique(torch.stack([geo['face_dim'],geo['rank']],1),dim=0,return_counts=True)
    row=dict(key=key,states=len(x),dimension=P.shape[0],student_dimension=P.shape[1],shape=shape,radius=R,
        rank_histogram=[dict(k=int(p[0]),r=int(p[1]),count=int(c)) for p,c in zip(pairs,counts)],methods=records)
    mt.write(path,row);return row

def fit_case(seed,d,m,shape,R,structured,rows):
    initials=dc.structured_initials if structured else mt.initials
    trainx,_=initials(1024,d,61000000+seed);valx,_=initials(512,d,62000000+seed)
    pool=mt.states(trainx);training_x=pool.float()
    P=dc.structured_embedding(seed+311) if structured else mt.embedding(d,m,seed+311)
    prefix=f's{seed}_d{d}_m{m}_{shape}_r{R:g}'
    cache=audit_cache(pool,P,R,shape,seed,prefix)
    oracle=mt.Oracle(pool,R,shape)
    torch.manual_seed(seed+107*m+d);initial=st.Student(P,R,shape).cuda()
    methods=CFG['structured_methods'].copy() if structured else dc.METHODS.copy()
    random.Random(seed+991).shuffle(methods)
    for method in methods:
        key=prefix+'_'+method;path=OUT/(key+'.json')
        if path.exists():rows.append(load(path));continue
        data=dc.transmit(oracle,P,method,seed+933)
        mt.write(OUT/'status.json',dict(status='training',case=key,completed=len(rows),total=120))
        model=copy.deepcopy(initial);torch.cuda.synchronize();start=time.perf_counter();trainer=st.Trainer(model)
        gen=torch.Generator(device='cuda').manual_seed(seed+177);curves=[]
        for update in range(1,4097):
            ix=torch.randint(len(pool),(512,),device='cuda',generator=gen)
            trainer.update(training_x,data['coefficient'],ix)
            if update in [1024,2048,4096]:
                vv,_=st.evaluate(model,valx,R,shape)
                curves.append(dict(update=update,validation_normalized_regret=vv['normalized_regret']))
        torch.cuda.synchronize();fit_seconds=time.perf_counter()-start
        cp=OUT/(key+'.pt');torch.save(model.state_dict(),cp)
        testx,rare=initials(4096,d,63000000+seed);metrics,raw=st.evaluate(model,testx,R,shape);raw['rare']=rare
        ep=OUT/(key+'_evaluation.pt');torch.save(raw,ep)
        row=dict(key=key,seed=seed,dimension=d,student_dimension=m,shape=shape,radius=R,structured=structured,method=method,
            metrics=metrics,fit_seconds=fit_seconds,nonfinite_updates=int(trainer.bad),clipped_updates=int(trainer.clips),
            checkpoint_sha256=mt.digest(cp),evaluation_sha256=mt.digest(ep),feedback_sha256=mt.digest(OUT/(prefix+'_feedback.json')),curves=curves)
        assert row['nonfinite_updates']==0;mt.write(path,row);rows.append(row)
        print(key,'regret',round(metrics['normalized_regret'],8),'cost',round(metrics['test_cost'],6),'sec',round(fit_seconds,2),flush=True)
        del model,trainer,raw;gc.collect();torch.cuda.empty_cache()

def run():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(config=CFG,source_sha256=mt.digest(__file__),controls_sha256=mt.digest(dc.__file__),
        dependencies={str(Path(p).relative_to(HERE)):mt.digest(p) for p in [mt.__file__,st.__file__,HERE/'experiments/hj_gridfree/engine.py']},
        original_report_sha256=mt.digest(OLD/'report.json'),checks=dc.checks())
    lock=OUT/'protocol_lock.json'
    if lock.exists():assert load(lock)==protocol
    else:mt.write(lock,protocol)
    audits=[]
    for cp in sorted(OLD.glob('*_cache.pt')):
        key=cp.stem[:-6];tokens=key.split('_');seed=int(tokens[0][1:]);shape=tokens[3];R=float(tokens[4][1:])
        data=torch.load(cp,weights_only=True)
        row=audit_cache(data['x'],data['P'],R,shape,seed,key)
        row['original_cache_sha256']=mt.digest(cp);audits.append(row)
        print('cache',key,flush=True)
    rows=[]
    for seed in CFG['original_seeds']:
        for d,m,shape,R in CFG['original_conditions']:fit_case(seed,d,m,shape,R,False,rows)
    for seed in CFG['structured_seeds']:
        fit_case(seed,8,4,'box',.5,True,rows)
        audits.append(load(OUT/f's{seed}_d8_m4_box_r0.5_feedback.json'))
    assert len(rows)==120 and len(audits)==85
    mt.write(OUT/'report.json',dict(config=CFG,protocol_sha256=mt.digest(lock),audits=audits,rows=rows))
    mt.write(OUT/'status.json',dict(status='complete',completed=120,total=120))

if __name__=='__main__':run()
