"""Reevaluate all additional checkpoints and generate fair-query evidence."""
import json,statistics
from pathlib import Path
import torch
import numpy as np
import minimal_teaching as mt
import minimal_teaching_study as st
import direct_query_controls as dc

HERE=Path(__file__).resolve().parent;OUT=HERE/'results/direct_query_study'
def read(p):return json.loads(Path(p).read_text())

def main():
    torch.set_num_threads(1)
    report=read(OUT/'report.json');lock=read(OUT/'protocol_lock.json')
    assert report['protocol_sha256']==mt.digest(OUT/'protocol_lock.json')
    assert lock['source_sha256']==mt.digest(HERE/'direct_query_study.py') and lock['controls_sha256']==mt.digest(dc.__file__)
    for p,h in lock['dependencies'].items():assert mt.digest(HERE/p)==h
    assert len(report['rows'])==120 and len(report['audits'])==85
    reevaluated=[];fidelity=[];maximum_metric_error=0.
    rows={r['key']:r for r in report['rows']}
    for row in report['rows']:
        key=row['key'];cp=OUT/(key+'.pt');ep=OUT/(key+'_evaluation.pt')
        assert mt.digest(cp)==row['checkpoint_sha256'] and mt.digest(ep)==row['evaluation_sha256']
        seed=row['seed'];d=row['dimension'];m=row['student_dimension'];R=row['radius'];shape=row['shape']
        P=dc.structured_embedding(seed+311) if row['structured'] else mt.embedding(d,m,seed+311)
        model=st.Student(P,R,shape).cuda();model.load_state_dict(torch.load(cp,weights_only=True))
        initials=dc.structured_initials if row['structured'] else mt.initials
        x,_=initials(4096,d,63000000+seed);metrics,raw=st.evaluate(model,x,R,shape)
        err=max(abs(metrics[k]-v) for k,v in row['metrics'].items());assert err<1e-11
        maximum_metric_error=max(maximum_metric_error,err)
        saved=torch.load(ep,weights_only=True)
        for k,v in raw.items():assert torch.allclose(v,saved[k],rtol=0,atol=1e-11)
        reevaluated.append(key)
        if row['method'] not in ['point','full']:
            prefix=f's{seed}_d{d}_m{m}_{shape}_r{R:g}';refkey=prefix+'_full'
            reference=rows[refkey] if row['structured'] else read(HERE/'results/minimal_teaching_study'/(refkey+'.json'))
            dr=metrics['normalized_regret']-reference['metrics']['normalized_regret']
            dcst=metrics['test_cost']-reference['metrics']['test_cost']
            fidelity.append(dict(key=key,structured=row['structured'],regret_difference=dr,cost_difference=dcst,within_tolerance=abs(dr)<=1e-3))
        print('verified',key,flush=True)
    originals=[a for a in report['audits'] if a['key'].startswith('s131')]
    boxes=[a for a in originals if a['shape']=='box'];structured=[a for a in report['audits'] if a not in originals]
    for a in originals:
        assert mt.digest(HERE/'results/minimal_teaching_study'/(a['key']+'_cache.pt'))==a['original_cache_sha256']
        assert all(h['r']==min(h['k'],a['student_dimension']) for h in a['rank_histogram'])
    methods=['rank','face']+dc.METHODS
    totals={method:sum(a['methods'][method]['queries'] for a in boxes) for method in methods}
    aligned={method:sum(a['methods'][method]['queries'] for a in structured) for method in methods}
    timings={family:{method:dict(median_ms=1000*statistics.median(statistics.median(a['methods'][method]['seconds']) for a in cells),
        minimum_ms=1000*min(statistics.median(a['methods'][method]['seconds']) for a in cells),
        maximum_ms=1000*max(statistics.median(a['methods'][method]['seconds']) for a in cells)) for method in methods}
        for family,cells in [('original_box',boxes),('original_ball',[a for a in originals if a['shape']=='ball']),('structured',structured)]}
    grouped=[]
    for structured_flag in [False,True]:
        conditions=sorted(set((r['dimension'],r['student_dimension'],r['shape'],r['radius']) for r in report['rows'] if r['structured']==structured_flag))
        for d,m,shape,R in conditions:
            for method in (['point','rank','face','full']+dc.METHODS if structured_flag else dc.METHODS):
                rr=[r for r in report['rows'] if (r['structured'],r['dimension'],r['student_dimension'],r['shape'],r['radius'],r['method'])==(structured_flag,d,m,shape,R,method)]
                grouped.append(dict(structured=structured_flag,dimension=d,student_dimension=m,shape=shape,radius=R,method=method,n=len(rr),
                    mean_regret=statistics.mean(r['metrics']['normalized_regret'] for r in rr),mean_cost=statistics.mean(r['metrics']['test_cost'] for r in rr)))
    primary=read(HERE/'results/neural_information_analysis.json')['stages'][1]['primary']
    differences=np.array(primary['differences']);heterogeneity=dict(median=float(np.median(differences)),mean=float(differences.mean()),
        largest_two_share=float(np.sort(np.abs(differences))[-2:].sum()/np.abs(differences).sum()),all_negative=bool((differences<0).all()))
    approximate=read(HERE/'results/approximate_feedback/report.json');assert approximate['passed']
    result=dict(passed=True,training_runs=120,reevaluated_policies=len(reevaluated),maximum_reevaluation_metric_error=maximum_metric_error,
        report_sha256=mt.digest(OUT/'report.json'),protocol_sha256=mt.digest(OUT/'protocol_lock.json'),source_sha256=mt.digest(__file__),
        fidelity=fidelity,fidelity_pairs=len(fidelity),fidelity_pairs_within_tolerance=sum(f['within_tolerance'] for f in fidelity),
        maximum_absolute_regret_difference=max(abs(f['regret_difference']) for f in fidelity),
        maximum_absolute_cost_difference=max(abs(f['cost_difference']) for f in fidelity),
        maximum_coefficient_error=max(a['methods'][method]['maximum_coefficient_error'] for a in report['audits'] for method in methods),
        original_box_totals=totals,structured_totals=aligned,original_hybrid_rank_count_equal=True,timings=timings,groups=grouped,
        heterogeneity=heterogeneity,approximate_report_sha256=mt.digest(HERE/'results/approximate_feedback/report.json'))
    mt.write(HERE/'results/direct_query_analysis.json',result)
    print(json.dumps({k:result[k] for k in ['training_runs','fidelity_pairs','fidelity_pairs_within_tolerance','maximum_absolute_regret_difference','maximum_absolute_cost_difference','original_box_totals','structured_totals','timings','heterogeneity']},indent=2))

if __name__=='__main__':main()
