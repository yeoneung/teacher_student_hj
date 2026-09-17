"""CPU-only completion audit of raw evidence, locks and preserved older releases."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from scipy.stats import t as student_t

from experiments.hj_gridfree.engine import write
from .study import verify, digest
from .evaluation import train_dir, lock_evaluation
from .systems import problem
from .conditioning import local_trajectory_audit
from .feedback_study import make_lock as verify_feedback_lock
from experiments.hj_gridfree.core import step as ring_step,running as ring_running,terminal as ring_terminal


def tensor_hash(value):
    a=value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def close(a,b,tol=2e-10):
    assert abs(a-b)<=tol*(1+abs(a)+abs(b)),(a,b)


def prior_releases():
    records={}
    archives={}
    for version in range(1,6):
        path=Path(f'build/hj_gridfree_release_v{version}/release_manifest.json')
        saved=json.loads(path.read_text(encoding='utf-8'))
        records[str(version)]=dict(path=str(path),sha256=digest(path))
        def find(obj):
            if isinstance(obj,dict):
                if 'path' in obj and 'sha256' in obj and str(obj['path']).endswith('.zip'):
                    archives[obj['path']]=obj['sha256']
                for value in obj.values():find(value)
            elif isinstance(obj,list):
                for value in obj:find(value)
        find(saved)
    for path,expected in archives.items():
        assert digest(path)==expected,path
    preserved={}
    for path in archives:
        if 'data' not in Path(path).name:
            continue
        with zipfile.ZipFile(path) as archive:
            manifest=json.loads(archive.read('ARCHIVE_MANIFEST.json'))
        for filename,meta in manifest.items():
            source=Path(filename)
            assert source.stat().st_size==meta['bytes'],filename
            assert digest(source)==meta['sha256'],filename
            preserved[filename]=meta['sha256']
    return dict(releases=records,archive_hashes=archives,preserved_numerical_files=len(preserved))


def evidence(root,check_previous=True):
    cfg=verify(root)
    assert (root/'evaluation_lock.json').exists()
    lock_evaluation(root,cfg)
    lock=json.loads((root/'evaluation_lock.json').read_text())
    planned=sum(len(task['methods'])*len(cfg['seeds']) for task in cfg['tasks'])
    assert len(lock['checkpoints'])==planned==250
    selected_initial=[]
    skips=[]
    validation_stops=[]
    phase_records=[]
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            folder=train_dir(root,task,seed)
            complete=json.loads((folder/'complete.json').read_text())
            assert set(complete['methods'])=={s['name'] for s in task['methods']}
            data=torch.load(folder/'data.pt',weights_only=False,map_location='cpu')
            assert data['x'].shape==(cfg['dataset'],32)
            assert data['t'].shape==(cfg['dataset'],)
            assert data['valx'].shape==(cfg['validation_states'],32)
            assert torch.isfinite(data['x']).all() and torch.isfinite(data['valx']).all()
            assert data['t'].min()>=0 and data['t'].max()<task['horizon']
            for spec in task['methods']:
                name=spec['name']
                path=folder/(name+'.json')
                row=json.loads(path.read_text())
                history=row['history']
                assert row['steps_completed']<=cfg['steps']
                assert len(history)==1+row['steps_completed']//cfg['validation_every']
                finite=[r for r in history if r['mean'] is not None]
                best=min(finite,key=lambda r:r['mean'])
                assert row['best_step']==best['step'],path
                close(row['mean'],best['mean'],tol=1e-8)
                valid=torch.load(folder/(name+'_validation.pt'),weights_only=False,map_location='cpu')
                assert valid.shape==(cfg['validation_states'],) and torch.isfinite(valid).all()
                close(row['mean'],valid.mean().item())
                total=sum(row[k] for k in ['data_seconds','parent_seconds','setup_seconds','fit_seconds',
                                           'refresh_seconds','validation_seconds'])
                close(row['compute_seconds'],total)
                if spec.get('warm_start'):
                    parent=json.loads((folder/(spec['warm_start']+'.json')).read_text())
                    close(row['parent_seconds'],parent['compute_seconds']-row['data_seconds'])
                else:
                    assert row['parent_seconds']==0
                last=torch.load(folder/(name+'_history')/f'step_{row["steps_completed"]:06d}.pt',weights_only=False,map_location='cpu')
                final=torch.load(folder/(name+'.pt'),weights_only=False,map_location='cpu')
                assert final['best_step']==row['best_step']==last['best_step']
                assert all(torch.equal(v,last['state_dict'][k]) for k,v in final['state_dict'].items())
                assert all(torch.isfinite(v).all() for v in final['state_dict'].values())
                if row['best_step']==0:selected_initial.append(str(path))
                if row['nonfinite_updates_skipped']:
                    skips.append(dict(path=str(path),count=row['nonfinite_updates_skipped']))
                if row['stopped_on_nonfinite_validation']:
                    validation_stops.append(str(path))
                else:
                    assert row['steps_completed']==cfg['steps'],path
                if spec['mode']=='cache':
                    assert row['refresh_count']==1+(row['steps_completed']-1)//spec['refresh_every']
                phase_records.append(dict(path=str(path),sha256=digest(path),attempts=row['steps_completed']))

    expected={f'{t["family"]}_h{t["horizon"]}_d{d}_{c}':(t,d,c) for t in cfg['tasks']
              for d in cfg['test_dimensions'] for c in cfg['test_conditions']}
    assert set(p.stem for p in (root/'evaluation').glob('*.json'))==set(expected)
    unique={}
    replay_max=budget_max=feedback_max=feedback_mean_max=local_residual_max=open_loop_max=0.
    teacher_local_max=teacher_cost_error=teacher_feedback_max=teacher_feedback_mean_max=0.
    state_horizons=policy_cost_count=control_replays=0
    source_hashes={}
    for key,(task,dim,condition) in expected.items():
        path=root/'evaluation'/(key+'.pt')
        raw=torch.load(path,weights_only=False,map_location='cpu')
        meta=json.loads(path.with_suffix('.json').read_text())
        p=problem(**meta['problem'])
        count=cfg['test_nominal_states'] if condition=='nominal' else cfg['test_shift_states']
        assert raw['x'].shape==(count,dim) and meta['count']==count
        state_horizons+=count
        statekey=(task['family'],dim,condition)
        xhash=tensor_hash(raw['x'])
        if statekey in unique:assert unique[statekey]['hash']==xhash,statekey
        else:unique[statekey]=dict(hash=xhash,count=count)
        assert set(raw['methods'])=={s['name'] for s in task['methods']}
        assert len(meta['audits'])==len(task['methods'])*len(cfg['seeds'])
        close(raw['teacher'].mean().item(),meta['teacher_mean'])
        teacher_raw=raw['teacher_audit'];teacher_meta=meta['teacher_audit']
        assert teacher_raw['states'].shape==(cfg['audit_states'],task['horizon']+1,dim)
        assert teacher_raw['controls'].shape==(cfg['audit_states'],task['horizon'],p.n)
        assert torch.equal(teacher_raw['states'][:,0],raw['x'][:cfg['audit_states']])
        assert teacher_raw['feedback_costs'].shape==(count,) and torch.isfinite(teacher_raw['feedback_costs']).all()
        checked_teacher=local_trajectory_audit(teacher_raw['states'],teacher_raw['controls'],raw['teacher'][:cfg['audit_states']],p)
        for name,value in checked_teacher.items():close(value,teacher_meta[name])
        assert teacher_meta['count']==count and teacher_meta['audit_trajectories']==cfg['audit_states']
        tj=teacher_raw['feedback_costs']
        teacher_difference=((tj-raw['teacher']).abs()/(1+tj.abs())).max().item()
        teacher_mean_difference=abs(tj.mean().item()/raw['teacher'].mean().item()-1)
        close(teacher_difference,teacher_meta['independent_feedback_scaled_cost_difference'])
        close(teacher_mean_difference,teacher_meta['independent_feedback_relative_mean_difference'])
        assert teacher_meta['max_budget_ratio']<1.00001
        if task['family']=='building':
            assert teacher_raw['controls'].min()>=-1e-7 and teacher_raw['controls'].max()<=p.bound*(1+1e-5)
        else:assert teacher_raw['controls'].square().mean(-1).sqrt().max()<=p.bound*(1+1e-5)
        teacher_local_max=max(teacher_local_max,checked_teacher['max_scaled_single_step_residual'])
        teacher_cost_error=max(teacher_cost_error,checked_teacher['independent_scaled_trajectory_cost_error'])
        teacher_feedback_max=max(teacher_feedback_max,teacher_difference)
        teacher_feedback_mean_max=max(teacher_feedback_mean_max,teacher_mean_difference)
        for name,values in raw['methods'].items():
            assert values.shape==(len(cfg['seeds']),count)
            assert torch.isfinite(values).all() and values.min()>0
            close(values.mean().item(),meta['methods'][name]['mean'])
            for a,b in zip(values.mean(1).tolist(),meta['methods'][name]['seed_means']):close(a,b)
            close(float(np.quantile(values.numpy(),.95)),meta['methods'][name]['p95'])
            controls=raw['audit_controls'][name]
            states=raw['audit_states'][name]
            feedback=raw['feedback_replay_costs'][name]
            assert controls.shape[:3]==(len(cfg['seeds']),cfg['audit_states'],task['horizon'])
            assert torch.isfinite(controls).all()
            assert states.shape==(len(cfg['seeds']),cfg['audit_states'],task['horizon']+1,dim)
            assert torch.equal(states[:,:,0],raw['x'][:cfg['audit_states']].expand(len(cfg['seeds']),-1,-1))
            assert feedback.shape==(len(cfg['seeds']),cfg['evaluation_batch']) and torch.isfinite(feedback).all()
            for si,seed in enumerate(cfg['seeds']):
                checked=local_trajectory_audit(states[si],controls[si],values[si,:cfg['audit_states']],p)
                local_residual_max=max(local_residual_max,checked['max_scaled_single_step_residual'])
                record=next(a for a in meta['audits'] if a['method']==name and a['training_seed']==seed)
                fj=feedback[si];original=values[si,:cfg['evaluation_batch']]
                close(((fj-original).abs()/(1+fj.abs())).max().item(),record['independent_feedback_scaled_cost_difference'])
                close(abs(fj.mean().item()/original.mean().item()-1),record['independent_feedback_relative_mean_difference'])
            if task['family']=='building':
                assert controls.min()>=-1e-7 and controls.max()<=p.bound*(1+1e-5)
            else:
                assert controls.square().mean(-1).sqrt().max()<=p.bound*(1+1e-5)
            policy_cost_count+=values.numel()
            control_replays+=len(cfg['seeds'])*cfg['audit_states']
        replay_max=max(replay_max,max(a['independent_scaled_cost_error'] for a in meta['audits']))
        budget_max=max(budget_max,max(a['max_budget_ratio'] for a in meta['audits']))
        feedback_max=max(feedback_max,max(a['independent_feedback_scaled_cost_difference'] for a in meta['audits']))
        feedback_mean_max=max(feedback_mean_max,max(a['independent_feedback_relative_mean_difference'] for a in meta['audits']))
        open_loop_max=max(open_loop_max,max(a['fixed_control_open_loop_scaled_difference'] for a in meta['audits']))
        source_hashes[str(path)]=digest(path)
    assert len(expected)==72 and sum(v['count'] for v in unique.values())==6912 and state_horizons==18432
    assert replay_max<1e-8 and budget_max<1.00001

    nominal={key for key in expected if key.endswith('_nominal')}
    ring_nominal={key for key in nominal if not key.startswith('building')}
    assert set(p.stem for p in (root/'classical_native').glob('*.json'))==nominal
    assert set(p.stem for p in (root/'classical_gpu').glob('*.json'))==ring_nominal
    qp_gap_max=native_replay_max=0.
    for key in sorted(nominal):
        path=root/'classical_native'/(key+'.pt')
        raw=torch.load(path,weights_only=False,map_location='cpu')
        meta=json.loads(path.with_suffix('.json').read_text())
        test=torch.load(root/'evaluation'/(key+'.pt'),weights_only=False,map_location='cpu')
        assert torch.equal(raw['x'],test['x'][:cfg['classical_states']])
        assert torch.isfinite(raw['cost']).all() and raw['cost'].min()>0
        close(raw['cost'].mean().item(),meta['mean'])
        native_replay_max=max(native_replay_max,meta['independent_scaled_cost_error'])
        if key.startswith('building'):
            for row,cost in zip(meta['rows'],raw['cost']):
                cert=row['convex_certificate']
                close(cert['feasible_cost'],cost.item())
                assert -1e-10<=cert['first_order_gap']<1e-5*(1+cost.item())
                assert cert['convex_lower_bound']<=cost.item()+1e-8
                qp_gap_max=max(qp_gap_max,cert['first_order_gap'])
        else:
            gpu=torch.load(root/'classical_gpu'/(key+'.pt'),weights_only=False,map_location='cpu')
            assert torch.equal(gpu['x'],raw['x'])
            assert (raw['cost']<=gpu['best']+1e-7*(1+gpu['best'].abs())).all()
        source_hashes[str(path)]=digest(path)
    expected_memory={f'{t["family"]}_h{t["horizon"]}_{s["name"]}' for t in cfg['tasks'] for s in t['methods']
                     if not (s.get('warm_start') and not s.get('teacher_parent') or s.get('covector_signal'))}
    expected_timing={f'{t["family"]}_h{t["horizon"]}_d{d}' for t in cfg['tasks'] for d in [32,256]}
    assert set(p.stem for p in (root/'isolated_memory').glob('*.json'))==expected_memory
    assert set(p.stem for p in (root/'timing').glob('*.json'))==expected_timing
    precision_max=batch_max=batch_mean_max=0.
    for path in (root/'timing').glob('*.json'):
        row=json.loads(path.read_text())
        assert row['action']['median_seconds']>0 and row['plan']['median_seconds']>0
        assert row['plan_float64']['median_seconds']>0 and row['gpu_preparation_seconds']>0
        assert all(value<1e-6 for value in row['batch_one_graph_vs_eager'].values())
        probe=row['batch_sensitivity']
        assert probe['count']==8 and probe['source_batch_size']==128 and probe['source_seed']==68304
        single,batched=np.asarray(probe['single_costs']),np.asarray(probe['batched_costs'])
        assert single.shape==batched.shape==(8,) and np.isfinite(single).all() and np.isfinite(batched).all()
        close(float(np.max(np.abs(single-batched)/(1+np.abs(batched)))),probe['max_scaled_difference'])
        close(float(single.mean()/batched.mean()-1),probe['relative_mean_difference'])
        batch_max=max(batch_max,probe['max_scaled_difference'])
        batch_mean_max=max(batch_mean_max,abs(probe['relative_mean_difference']))
        if row['problem']['family']!='building':
            reference=row['single_start_feedback']
            assert len(reference['records'])==8 and reference['median_seconds']>0
            assert reference['matched_neural_mean_cost']>0
        precision_max=max(precision_max,row['precision']['max_scaled_difference'])
    summary=json.loads((root/'report/summary.json').read_text())
    assert summary['counts']['planned_phase_endpoints']==250
    assert summary['counts']['actual_test_cases']==72
    for key,case in summary['cases'].items():
        raw=json.loads((root/'evaluation'/(key+'.json')).read_text())
        for name,value in case['means'].items():close(value,raw['methods'][name]['mean'])
    independent_decisions={}
    def quality_bound(key,candidate,control):
        raw=torch.load(root/'evaluation'/(key+'.pt'),weights_only=False,map_location='cpu')['methods']
        a,b=raw[candidate].numpy(),raw[control].numpy()
        logs=np.log(a.mean(axis=1)/b.mean(axis=1))
        upper=np.exp(logs.mean()+student_t.ppf(1-.05/3,4)*np.std(logs,ddof=1)/np.sqrt(5))
        return float(a.mean()/b.mean()),float(upper)
    checks=[]
    for control in ['dpc','dpc_warm']:
        ratio,upper=quality_bound('mechanical_h160_d32_nominal','fresh',control)
        original=summary['primary_targets']['mechanical_quality']['contrasts'][control]
        close(ratio,original['pooled_mean_ratio']);close(upper,original['upper_ratio'])
        checks.append(ratio<=.90 and upper<.90)
    independent_decisions['mechanical_quality']=all(checks)
    for family,horizon in [('reaction',320),('building',192)]:
        ratio,upper=quality_bound(f'{family}_h{horizon}_d32_nominal','cache','fresh')
        t={}
        for method in ['cache','fresh']:
            t[method]=np.mean([json.loads((root/'training'/f'{family}_h{horizon}_s{seed}'/(method+'.json')).read_text())['compute_seconds'] for seed in cfg['seeds']])
        original=summary['primary_targets'][family+'_cache']
        close(ratio,original['quality']['pooled_mean_ratio']);close(upper,original['quality']['upper_ratio'])
        close(t['cache']/t['fresh'],original['total_time_ratio'])
        independent_decisions[family+'_cache']=ratio<=1.01 and upper<1.01 and t['cache']/t['fresh']<=.5
    assert all(bool(value)==summary['primary_targets'][key]['met'] for key,value in independent_decisions.items())
    numerical_ratio_change=0.
    for key,row in summary['primary_numerical_sensitivity'].items():
        raw=torch.load(root/'evaluation'/(row['case']+'.pt'),weights_only=False,map_location='cpu')
        n=row['count'];names=list(row['gpu_means'])
        for name in names:
            close(raw['methods'][name][:,:n].mean().item(),row['gpu_means'][name])
            close(raw['feedback_replay_costs'][name].mean().item(),row['numpy_plant_means'][name])
        a,b=names
        gpu=row['gpu_means'][a]/row['gpu_means'][b]
        independent=row['numpy_plant_means'][a]/row['numpy_plant_means'][b]
        close(abs(gpu-independent),row['absolute_cost_ratio_change'])
        numerical_ratio_change=max(numerical_ratio_change,abs(gpu-independent))
    supplemental=verify_feedback_lock(root)
    fb_expected={f'{f}_h{h}_d{d}_nominal' for f,h in supplemental['tasks'] for d in supplemental['dimensions']}
    assert set(p.stem for p in (root/'feedback_classical').glob('*.json'))==fb_expected
    feedback_local_max=feedback_cost_error=0.
    reference_solves=headroom_solves=0
    for key in fb_expected:
        path=root/'feedback_classical'/(key+'.pt')
        raw=torch.load(path,weights_only=False,map_location='cpu')
        meta=json.loads(path.with_suffix('.json').read_text())
        p=problem(**meta['problem'])
        assert len(raw['x'])==supplemental['reference_states']
        for name,data in raw['methods'].items():
            states,controls=data['states'],data['u']
            assert torch.equal(states[:,0],raw['x'][:len(states)])
            assert torch.isfinite(states).all() and torch.isfinite(controls).all()
            assert controls.square().mean(-1).sqrt().max()<=p.bound*(1+1e-8)
            assert data['tracking_reference'].shape==states[:,:-1].shape
            assert data['residual'].shape==controls.shape
            xx=states[:,:-1].reshape(-1,p.dim);uu=controls.reshape(-1,p.n)
            following=ring_step(xx,uu,p).reshape_as(states[:,1:])
            local=((following-states[:,1:]).abs()/(1+states[:,1:].abs())).max().item()
            costs=ring_running(xx,uu,p).reshape(len(states),p.horizon).sum(1)+ring_terminal(states[:,-1],p)
            error=((costs-data['cost']).abs()/(1+data['cost'].abs())).max().item()
            assert local<1e-10 and error<1e-10,(key,name,local,error)
            feedback_local_max=max(feedback_local_max,local);feedback_cost_error=max(feedback_cost_error,error)
            close(data['cost'].mean().item(),meta['methods'][name]['mean'])
            if name in supplemental['reference_starts']:reference_solves+=len(states)
            else:headroom_solves+=len(states)
        reference=torch.minimum(raw['methods']['zero_residual']['cost'],raw['methods']['gpu_plan']['cost'])
        assert torch.equal(reference,raw['reference_best'])
        old=torch.load(root/'classical_native'/(key+'.pt'),weights_only=False,map_location='cpu')['cost']
        close(torch.minimum(reference,old).mean().item(),summary['cases'][key]['combined_classical']['mean'])
        source_hashes[str(path)]=digest(path)
    assert reference_solves==384 and headroom_solves==224
    analytic=summary['analytic_audits']
    for family,row in analytic['final_numerical_audit.json'].items():
        assert row['tbptt_unpadded_gradient_error']<1e-5
        assert row['ablation_forward_error']<1e-6
        assert row['ablation_gradient_decomposition_error']<1e-5
        assert row['float64_plan_replay_error']<1e-8
        assert all(r['nonfinite_update_skip_and_recovery'] for r in row['modes'].values())
    for case in analytic['quadratic_jet_audit.json']:
        assert all(r['exact_quadratic_identity_error']<1e-12 and r['min_majorization_slack']>-1e-12 for r in case['cases'])
    for case in analytic['cached_performance_identity.json']:
        assert all(r['max_identity_error']<1e-10 and r['max_trajectory_bound_violation']<1e-10 for r in case['rows'])
    for row in analytic['feedback_native_audit.json']:
        assert row['value_error']<1e-10 and row['gradient_error']<1e-9 and row['Fourier_vs_dense_feedback_error']<1e-12
    curvature=analytic['teacher_curvature_horizon.json']
    assert curvature['configurations']==45 and curvature['distinct_initial_vectors']==24
    for row in curvature['records']:
        assert row['independent_rollout_scaled_cost_error']<1e-11
        assert row['recursive_bound_scaled_error']<1e-10
    assert len(curvature['building_structural_checks'])==6
    for row in curvature['building_structural_checks']:
        assert row['min_secant_jacobian_entry']>=0
        assert row['matrix_operator_norm']<=row['positive_vector_spectral_upper_bound']<.99
    teacher_pretest=analytic['teacher_evaluation_cpu_pretest.json']
    assert len(teacher_pretest['records'])==4
    for row in teacher_pretest['records']:
        assert row['original_cost_bitwise_preserved']
        assert row['max_scaled_single_step_residual']<1e-11
        assert row['independent_scaled_trajectory_cost_error']<1e-11
    for path,expected_hash in teacher_pretest['source_hashes'].items():assert digest(path)==expected_hash,path
    pretest=summary['evaluation_preflight']
    assert len(pretest['records'])==4
    for row in pretest['records']:
        assert digest(row['development_checkpoint'])==row['checkpoint_sha256']
        assert row['teacher_original_cost_bitwise_preserved']
        assert row['teacher_audit']['max_scaled_single_step_residual']<1e-11
        assert row['teacher_audit']['independent_scaled_trajectory_cost_error']<1e-11
    for path,expected_hash in pretest['source_hashes'].items():assert digest(path)==expected_hash,path
    # Bind the audited metadata and preceding GPU candidates to the release,
    # as well as the final policy-cost tensors already recorded above.
    for folder in ['evaluation','classical_gpu','classical_native','feedback_classical',
                   'isolated_memory','timing']:
        for path in sorted((root/folder).iterdir()):
            if path.is_file() and path.suffix in ['.json','.pt']:
                source_hashes[str(path)]=digest(path)
    for name in ['config.json','protocol.md','environment.json','evaluation_addendum.md',
                 'pretest_evaluation_audit.json','feedback_protocol_lock.json','feedback_complete.json']:
        path=root/name
        source_hashes[str(path)]=digest(path)
    for folder in (root/'training').iterdir():
        if not folder.is_dir():continue
        for name in ['data.pt','complete.json']:
            path=folder/name
            source_hashes[str(path)]=digest(path)
    for path in sorted(root.glob('host_load_*.json')):
        source_hashes[str(path)]=digest(path)
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_lock_sha256=digest(root/'training_lock.json'),evaluation_lock_sha256=digest(root/'evaluation_lock.json'),
        report_sha256=digest(root/'report/summary.json'),completed_phase_endpoints=len(phase_records),
        attempted_updates=sum(r['attempts'] for r in phase_records),phase_records=phase_records,
        initial_policy_selected=selected_initial,nonfinite_updates_skipped=skips,nonfinite_validation_stops=validation_stops,
        cases=len(expected),distinct_initial_vectors=6912,state_horizon_instances=state_horizons,
        individual_policy_costs=policy_cost_count,independently_replayed_controls=control_replays,
        max_independent_scaled_cost_error=replay_max,max_budget_ratio=budget_max,
        max_single_step_residual=local_residual_max,max_fixed_control_replay_difference=open_loop_max,
        max_independent_feedback_scaled_difference=feedback_max,max_independent_feedback_relative_mean_difference=feedback_mean_max,
        max_primary_contrast_cost_ratio_change=numerical_ratio_change,
        teacher_audit_cases=len(expected),teacher_audit_trajectories=len(expected)*cfg['audit_states'],
        teacher_independent_feedback_state_horizon_instances=state_horizons,
        teacher_max_local_residual=teacher_local_max,teacher_max_independent_cost_error=teacher_cost_error,
        teacher_max_feedback_scaled_difference=teacher_feedback_max,
        teacher_max_feedback_relative_mean_difference=teacher_feedback_mean_max,
        classical_cases=len(nominal),classical_state_horizon_instances=len(nominal)*cfg['classical_states'],
        max_native_scaled_replay_error=native_replay_max,max_building_convex_gap=qp_gap_max,
        memory_probes=len(expected_memory),timing_cases=len(expected_timing),max_float32_scaled_cost_difference=precision_max,
        max_batch_one_vs_128_scaled_cost_difference=batch_max,
        max_batch_one_vs_128_relative_mean_difference=batch_mean_max,
        additional_feedback_reference_cases=len(fb_expected),additional_feedback_reference_solves=reference_solves,
        learned_plan_headroom_solves=headroom_solves,feedback_reference_max_local_residual=feedback_local_max,
        feedback_reference_max_independent_cost_error=feedback_cost_error,
        raw_evidence_hashes=source_hashes,independently_recomputed_primary_decisions={k:bool(v) for k,v in independent_decisions.items()})
    if check_previous:result['preserved_previous']=prior_releases()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='experiments/results/hj_cotangent_primary_v6')
    parser.add_argument('--skip-prior-archives',action='store_true')
    parser.add_argument('--verify-only',action='store_true',help='Recheck evidence against the saved audit without replacing it.')
    args=parser.parse_args()
    torch.set_num_threads(1)
    root=Path(args.root)
    result=evidence(root,not args.skip_prior_archives)
    if args.verify_only:
        saved=json.loads((root/'completion_audit.json').read_text())
        keys=['training_lock_sha256','evaluation_lock_sha256','report_sha256',
              'phase_records','raw_evidence_hashes','independently_recomputed_primary_decisions']
        if not args.skip_prior_archives:keys.append('preserved_previous')
        for key in keys:
            assert result[key]==saved[key],key
        print('Saved evidence verified; the archived audit record was not rewritten.',flush=True)
    else:
        write(root/'completion_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['phase_records','raw_evidence_hashes','initial_policy_selected']},indent=2),flush=True)


if __name__=='__main__':main()
