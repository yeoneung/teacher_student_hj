"""Independent artifact and arithmetic audit of the completed impact revision."""
import json,math
from pathlib import Path
import torch
from impact_core import HERE,SUB,PROJECT,write,digest


def finite(value):
    if isinstance(value,float):assert math.isfinite(value)
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,list):
        for v in value:finite(v)


def main():
    torch.set_num_threads(1)
    vendor=json.loads((HERE/'vendor_provenance.json').read_text())
    for p,h in vendor.items():assert digest(SUB/p)==h,p
    checked=0;means=0;maximum_mean_error=0.;phases={}
    for stage in ['mechanism','pilot','replication_v3']:
        root=HERE/'results'/stage;lock=json.loads((root/'protocol_lock.json').read_text())
        for p,h in lock['sources'].items():assert digest(HERE/p)==h,(stage,p)
        report=json.loads((root/'report.json').read_text());assert report['completed'];finite(report)
        phases[stage]=len(report['rows'])
        for r in report['rows']:
            if stage=='mechanism':stem=r['case']+'_'+r['method']
            else:stem=f"{r['family']}_s{r['seed']}_{r.get('comparison_label',r['method'])}"
            checkpoint=root/(stem+'.pt');assert digest(checkpoint)==r['policy_sha256']
            weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert all(torch.isfinite(v).all() for v in weights.values());assert r['nonfinite_updates']==0
            saved=torch.load(root/(stem+'_evaluation.pt'),map_location='cpu',weights_only=False)
            assert torch.isfinite(saved['costs']).all()
            error=abs(float(saved['costs'].mean())-r['test_cost']);assert error<1e-12
            maximum_mean_error=max(maximum_mean_error,error);means+=saved['costs'].numel()
            if stage!='mechanism':assert 0<=r['best_available_seconds']<=r['budget']
            for k,v in saved['diagnostic'].items():assert torch.isfinite(v).all(),(stage,stem,k)
            raw=saved['diagnostic'];account=raw['target_model']+raw['fitting_term']+raw['noise_term']+raw['curvature']
            assert float((raw['actual']-account).abs().max())<1e-9
            assert float((raw['actual']-raw['fitted_bound_realized']).max())<2e-5
            checked+=1
    assert phases==dict(mechanism=120,pilot=14,replication_v3=50)
    numerical=0
    for stage,source in [('exact_diagnostic','impact_core.py'),('noise_validation','impact_noise_validation.py'),('factorial','impact_external_checks.py'),('building','impact_external_checks.py')]:
        root=HERE/'results'/stage;lock=json.loads((root/'plan_lock.json').read_text())
        assert digest(HERE/source)==lock['source_sha256']
        report=json.loads((root/'report.json').read_text());finite(report)
        assert report.get('completed',report.get('passed',False));numerical+=1
    exact=json.loads((HERE/'results/exact_diagnostic/report.json').read_text())
    assert max(r['telescoping_error'] for r in exact['rows'])<1e-5
    noise=json.loads((HERE/'results/noise_validation/report.json').read_text())
    assert len(noise['rows'])==150
    for r in noise['rows']:
        key=f"{r['family']}_s{r['teacher_seed']}_q{r['teacher_quality_budget']:g}_{r['noise']}{r['magnitude']:g}"
        bundle=torch.load(HERE/'results/noise_validation'/(key+'.pt'),map_location='cpu',weights_only=False)
        raw=bundle['raw'];bound=bundle['empirical_bound'];naive=raw['target_model'] < -1e-8;guard=bound < -1e-8;bad=raw['actual_target']>1e-8
        assert int((naive&bad).sum())==r['naive_false_positives']
        assert int((guard&bad).sum())==r['guarded_false_positives']
        assert int((raw['actual_target']>bound+1e-7).sum())==r['bound_violations']
    building=json.loads((HERE/'results/building/report.json').read_text());assert len(building['qp'])==64
    assert all(r['status']=='solved' and r['objective_replay_error']<1e-4 for r in building['qp'])
    # Earlier scientific evidence is preserved outside the working submission.
    original=json.loads((PROJECT/'build/hj_v10_final/manifest.json').read_text())
    for p,h in original['files'].items():assert digest(PROJECT/p)==h,p
    result=dict(passed=True,new_training_phases=sum(phases.values()),phases=phases,
        checked_policies=checked,finite_test_costs=means,maximum_mean_recomputation_error=maximum_mean_error,
        local_action_evaluations=150*32,sampled_local_state_time_points=2*5*32,
        local_states_reused_across_quality_and_noise=True,new_qp_states=64,
        original_v10_files_preserved=len(original['files']),vendored_source_files_verified=len(vendor),
        maximum_exact_telescoping_error=max(r['telescoping_error'] for r in exact['rows']),
        scope='Arithmetic, finite-trajectory, policy-eligibility and provenance checks. Empirical curvature violations are counted, not suppressed. No population certificate.')
    write(HERE/'results/completion_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
