"""Audit all frozen minimal-information runs and export manuscript evidence.

Reevaluates every saved neural policy; does not train or select checkpoints.
The seed, not cached states, is the sampling unit in secondary t intervals.
"""
import json,math,itertools
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
import minimal_teaching as mt
import minimal_teaching_study as study

HERE=Path(__file__).resolve().parent;SUB=HERE.parent;OUT=study.OUT
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def describe(a):
    a=np.asarray(a,dtype=float);mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/math.sqrt(len(a)))
    return dict(mean=mean,sd=sd,ci95=[mean-half,mean+half],n=len(a))

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    lock=read(OUT/'protocol_lock.json');report=read(OUT/'report.json');rows=report['rows']
    assert len(rows)==480 and read(OUT/'status.json')['status']=='complete'
    assert lock['source_sha256']==mt.digest(study.__file__) and lock['interface_sha256']==mt.digest(mt.__file__)
    assert lock['config']==report['config']==study.CFG
    exact=read(HERE/'results/minimal_information_exact/report.json');assert exact['passed']
    assert exact['source_sha256']==mt.digest(HERE/'verify_minimal_information.py')
    assert exact['raw_sha256']==mt.digest(HERE/'results/minimal_information_exact/witnesses.npz')
    groups={};seen=set();maximum_reevaluation=0.;maximum_coefficient=0.;parameter_pairs=[]
    audited={};histogram={}
    for row in rows:
        seed,d,m,shape,radius,method=[row[k] for k in ['seed','dimension','student_dimension','shape','radius','method']]
        prefix=f's{seed}_d{d}_m{m}_{shape}_r{radius:g}';key=prefix+'_'+method
        assert key not in seen;seen.add(key)
        assert read(OUT/(key+'.json'))==row and row['nonfinite_updates']==0
        cp=OUT/(key+'.pt');cache=OUT/(prefix+'_cache.pt')
        assert mt.digest(cp)==row['checkpoint_sha256']
        if prefix not in audited:
            digest=mt.digest(cache);content=torch.load(cache,map_location='cuda',weights_only=True)
            geom=mt.geometry(content['target'],content['P'],radius,shape)
            for name,data in content['data'].items():
                if name in ['rank','random']:assert torch.equal(data['queries'],geom['rank'])
                if name=='face':assert torch.equal(data['queries'],geom['face_dim'])
                delta=data['coefficient']-content['data']['full']['coefficient']
                if name in ['rank','face']:
                    maximum_coefficient=max(maximum_coefficient,float(delta.abs().max()))
                    assert float(delta.abs().max())<1e-9
            hkey=f'{shape}_d{d}_m{m}_r{radius:g}'
            hist=torch.bincount(geom['face_dim']*(m+1)+geom['rank'],minlength=(d+1)*(m+1)).reshape(d+1,m+1).cpu().numpy()
            histogram[hkey]=histogram.get(hkey,np.zeros_like(hist))+hist
            audited[prefix]=dict(sha=digest,P=content['P'].clone(),records={name:dict(queries=int(data['queries'].sum()),
                coefficient_error=float((data['coefficient']-content['data']['full']['coefficient']).abs().max())) for name,data in content['data'].items()})
            del content
        assert audited[prefix]['sha']==row['cache_sha256']
        rec=audited[prefix]['records'][method]
        assert rec['queries']==row['feedback']['queries']
        assert abs(rec['coefficient_error']-row['feedback']['maximum_coefficient_error'])<1e-12
        model=study.Student(audited[prefix]['P'],radius,shape).cuda()
        state=torch.load(cp,map_location='cuda',weights_only=True);model.load_state_dict(state)
        x,rare=mt.initials(study.CFG['test_initials'],d,63000000+seed)
        metrics,raw=study.evaluate(model,x,radius,shape)
        stored=torch.load(OUT/(key+'_evaluation.pt'),map_location='cuda',weights_only=True)
        assert torch.equal(stored['rare'],rare)
        for name,value in metrics.items():
            err=abs(value-row['metrics'][name]);maximum_reevaluation=max(maximum_reevaluation,err);assert err<1e-9
        for name,value in raw.items():assert float((value-stored[name]).abs().max())<1e-9
        if method=='rank':
            full=torch.load(OUT/(prefix+'_full.pt'),map_location='cuda',weights_only=True)
            parameter_pairs.append(dict(case=prefix,maximum_parameter_difference=max(float((state[k]-full[k]).abs().max()) for k in state)))
        groups.setdefault((shape,d,m,radius),{}).setdefault(seed,{})[method]=row
        if len(seen)%60==0:print('Audited',len(seen),'of480',flush=True)
        del model,state,stored,raw
    assert len(groups)==16 and len(audited)==80
    summaries=[];fidelity=[]
    for (shape,d,m,radius),seeds in sorted(groups.items()):
        assert sorted(seeds)==study.CFG['seeds'] and all(set(z)==set(mt.METHODS) for z in seeds.values())
        summary=dict(shape=shape,dimension=d,student_dimension=m,radius=radius,methods={},comparisons={})
        for method in mt.METHODS:
            records=[seeds[s][method] for s in sorted(seeds)]
            summary['methods'][method]={name:describe([z['metrics'][name] for z in records]) for name in ['normalized_regret','test_cost','target_mse']}
            summary['methods'][method].update({name:describe([z['feedback'][name] for z in records]) for name in ['queries_per_state','relative_coefficient_error']})
            summary['methods'][method]['total_queries']=sum(z['feedback']['queries'] for z in records)
        for other in ['point','gain','random','face','full']:
            comparisons={}
            for metric in ['normalized_regret','test_cost']:
                diffs=[seeds[s]['rank']['metrics'][metric]-seeds[s][other]['metrics'][metric] for s in sorted(seeds)]
                comparisons[metric]=dict(**describe(diffs),differences=diffs,wins=sum(v<0 for v in diffs))
                if other=='full' and metric=='normalized_regret':fidelity.extend(abs(v) for v in diffs)
            summary['comparisons']['rank_minus_'+other]=comparisons
        qr=summary['methods']['rank']['total_queries'];qf=summary['methods']['face']['total_queries']
        summary['query_reduction_vs_face']=1-qr/qf if qf else 0
        summaries.append(summary)
    rq=sum(z['feedback']['queries'] for z in rows if z['method']=='rank' and z['shape']=='box')
    fq=sum(z['feedback']['queries'] for z in rows if z['method']=='face' and z['shape']=='box')
    result=dict(passed=True,training_runs=480,cells=80,paired_comparison_groups=16,
        report_sha256=mt.digest(OUT/'report.json'),protocol_sha256=mt.digest(OUT/'protocol_lock.json'),
        source_sha256=mt.digest(__file__),exact_report_sha256=mt.digest(HERE/'results/minimal_information_exact/report.json'),
        primary_coefficient_and_query_checks_passed=True,maximum_coefficient_absolute_error=maximum_coefficient,
        reevaluated_policies=len(seen),maximum_reevaluation_metric_error=maximum_reevaluation,
        fidelity_tolerance=.001,fidelity_pairs=len(fidelity),fidelity_pairs_within_tolerance=sum(v<=.001 for v in fidelity),
        maximum_rank_full_normalized_regret_difference=max(fidelity),parameter_pairs=parameter_pairs,
        box_scalar_queries_rank=rq,box_scalar_queries_face=fq,box_aggregate_query_reduction=1-rq/fq,
        total_fit_seconds=sum(z['fit_seconds'] for z in rows),
        maximum_telescoping_error=max(z['metrics']['maximum_telescoping_error'] for z in rows),
        maximum_subspace_cost_error=max(z['metrics']['maximum_subspace_cost_error'] for z in rows),
        groups=summaries,normal_rank_histograms={k:v.tolist() for k,v in histogram.items()})
    mt.write(HERE/'results/minimal_teaching_analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','parameter_pairs','normal_rank_histograms']},indent=2))

if __name__=='__main__':main()
