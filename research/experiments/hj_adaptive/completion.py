"""Audit v7 selected-budget records and independently reconstruct numerical claims."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import torch
from experiments.hj_cotangent.conditioning import local_trajectory_audit
from experiments.hj_gridfree.engine import write
from .study import verify,digest


def audit_development(root):
    """Reconstruct selection from all retained development endpoints and hashes."""
    cfg=verify(root);phases=[];copied=0
    for suffix in ['', 'b','c','d','e','f_controls']:
        dev=Path('experiments/results/hj_adaptive_dev_v7'+suffix)
        dc=json.loads((dev/'config.json').read_text());start=json.loads((dev/'development_start.json').read_text())
        for path,row in start.get('copied_parent_evidence',{}).items():
            assert digest(path)==row['sha256']==digest(row['source']),path
            copied+=1
        for task in dc['tasks']:
            for seed in dc['seeds']:
                folder=dev/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'
                assert (folder/'complete.json').exists()
                for spec in task['methods']:
                    path=folder/(spec['name']+'.json');record=json.loads(path.read_text())
                    selected=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)['budgets']['30']
                    assert record['budgets']['30']['validation_mean']==selected['validation_mean']
                    assert selected['available_seconds']<=30
                    assert all(torch.isfinite(v).all() for v in selected['state_dict'].values())
                    phases.append(dict(root=str(dev),family=task['family'],seed=seed,method=spec['name'],
                        mean=selected['validation_mean'],json_sha256=digest(path),checkpoint_sha256=digest(path.with_suffix('.pt'))))
    assert len(phases)==312
    selection=json.loads((root/'development_selection.json').read_text());options=0
    for family in selection['selections']:
        task=next(t for t in cfg['tasks'] if t['family']==family['family'])
        for name,rows in family['groups'].items():
            for row in rows:
                values=[]
                for path,sha in row['evidence_hashes'].items():
                    assert digest(path)==sha,path
                    values.append(json.loads(Path(path).read_text())['budgets']['30']['validation_mean'])
                assert values==row['seed_means'] and sum(values)/len(values)==row['mean']
                options+=1
            best=min(rows,key=lambda r:(r['mean'],r['method']['name']))
            assert best==family['winners'][name]
            selected=next(s for s in task['methods'] if s['name']==name)
            expected=dict(best['method']);expected['name']=name
            parent=next(s['parent'] for s in task['methods'] if s['name']=='warm')
            if expected.get('parent'):expected['parent']=parent
            if expected.get('teacher_parent'):expected['teacher_parent']=parent
            assert selected==expected
        assert task['primary_references']==family['primary_references']
        principal=min(task['primary_references'],key=lambda name:family['winners'][name]['mean'])
        assert principal==task['principal_reference']
    return dict(passed=True,phases=len(phases),checked_selection_options=options,
        copied_parent_or_data_files=copied,phase_records=phases,
        scope='All six rounds retained. Saved30-second validation means and selected configurations reconstructed without reselecting anything. Shared copied parents/data are verified but are not independent phase repetitions. Development timing records retain their original version-specific implementations.')


def audit(root):
    cfg=verify(root);lock=json.loads((root/'evaluation_lock.json').read_text())
    for path,sha in lock['hashes'].items():assert digest(path)==sha,path
    phases=[];nonfinite=[];maximum_overrun=0.;snapshots=0
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            folder=root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'
            assert (folder/'complete.json').exists()
            for spec in task['methods']:
                r=json.loads((folder/(spec['name']+'.json')).read_text())
                saved=torch.load(folder/(spec['name']+'.pt'),map_location='cpu',weights_only=False)['budgets']
                budget_list=spec.get('budgets_seconds',cfg['budgets_seconds'])
                for budget in budget_list:
                    key=str(budget);record=r['budgets'][key]
                    assert record['available_seconds']<=budget
                    assert record['validation_mean']==saved[key]['validation_mean']
                    assert all(torch.isfinite(v).all() for v in saved[key]['state_dict'].values())
                    eligible=[h for h in r['history'] if h['available_seconds']<=budget and h['mean'] is not None]
                    assert eligible
                    assert abs(min(h['mean'] for h in eligible)-record['validation_mean'])<1e-10
                    snapshots+=1
                if spec.get('parent'):
                    assert abs(r['data_seconds']+r['parent_charge_seconds']-cfg['parent_budget_seconds'])<1e-9
                    parent=torch.load(folder/(spec['parent']+'.pt'),map_location='cpu',weights_only=False)['budgets'][str(cfg['parent_budget_seconds'])]
                    early=saved[str(cfg['parent_budget_seconds'])]
                    assert all(torch.equal(v,parent['state_dict'][k]) for k,v in early['state_dict'].items())
                expected=0
                if spec['mode']=='cache':expected=r['full_refreshes']*cfg['dataset']+(r['probe_queries']+r.get('fresh_minibatch_updates',0))*cfg['batch']
                elif spec['mode']=='exact':expected=r['attempted_updates']*cfg['batch']
                elif spec['mode']=='transport':
                    expected=r['full_refreshes']*cfg['training_initial_states']*task['horizon']
                    events=[e for e in r['events'] if e['reason']=='transport_refresh']
                    assert len(events)==r['full_refreshes']
                    assert sum(e['accepted'] for e in events)==r['accepted_refreshes']
                    assert sum(not e['accepted'] for e in events)==r['rejected_refreshes']
                    incumbent=None
                    for e in events:
                        if e['accepted']:
                            assert e['candidate_training_mean'] is not None
                            if incumbent is not None:assert e['candidate_training_mean']<=incumbent+1e-7*(1+abs(incumbent))
                            incumbent=e['candidate_training_mean']
                        assert incumbent==e['incumbent_training_mean']
                    assert r['teacher_trajectories']==r['full_refreshes']*cfg['training_initial_states']
                assert r['teacher_pairs']==expected
                maximum_overrun=max(maximum_overrun,r['overrun_seconds'])
                if r['nonfinite_updates_skipped']:nonfinite.append(dict(family=task['family'],seed=seed,method=spec['name'],skips=r['nonfinite_updates_skipped']))
                phases.append(dict(path=str(folder/(spec['name']+'.pt')),sha256=digest(folder/(spec['name']+'.pt'))))
    expected_phases=sum(len(task['methods'])*len(cfg['seeds']) for task in cfg['tasks'])
    expected_policies=sum(sum(not spec.get('parent_only',False) for spec in task['methods'])*len(cfg['seeds']) for task in cfg['tasks'])
    assert len(phases)==expected_phases==270
    assert snapshots==sum(sum(len(spec.get('budgets_seconds',cfg['budgets_seconds'])) for spec in task['methods'])*len(cfg['seeds']) for task in cfg['tasks'])
    expected_cases={f"{task['family']}_h{task['horizon']}_d{dim}_{condition}" for task in cfg['tasks'] for dim,condition in [(32,'nominal'),(256,'nominal'),(256,'changed')]}
    assert {p.stem for p in (root/'evaluation').glob('*.pt')}==expected_cases
    costs=states=traces=0;max_local=max_cost=max_feedback=0.;decisions={};state_hashes=set()
    max_primary_feedback_ratio_change=0.
    from experiments.hj_cotangent.systems import problem
    from experiments.hj_cotangent.building import BuildingProblem
    from experiments.hj_gridfree.core import Problem
    for file in (root/'evaluation').glob('*.pt'):
        raw=torch.load(file,map_location='cpu',weights_only=False);meta=json.loads(file.with_suffix('.json').read_text())
        p=(BuildingProblem if meta['problem']['family']=='building' else Problem)(**meta['problem'])
        task=next(t for t in cfg['tasks'] if t['family']==p.family)
        assert set(raw['methods'])=={s['name'] for s in task['methods'] if not s.get('parent_only')}
        expected_count=128 if meta['condition']=='changed' else 512
        assert len(raw['x'])==expected_count
        states+=len(raw['x'])
        for vector in raw['x'].numpy():
            state_hashes.add((len(vector),hashlib.sha256(vector.tobytes()).hexdigest()))
        for name,budgets in raw['methods'].items():
            assert set(budgets)=={str(b) for b in cfg['budgets_seconds']}
            for key,values in budgets.items():
                assert torch.isfinite(values).all() and tuple(values.shape)==(len(cfg['seeds']),expected_count)
                costs+=values.numel()
                assert abs(values.mean().item()-meta['methods'][name][key]['mean'])<1e-10
                for row in raw['audits'][name][key]:
                    check=local_trajectory_audit(row['states'],row['controls'],row['trajectory_costs'],p)
                    max_local=max(max_local,check['max_scaled_single_step_residual']);max_cost=max(max_cost,check['independent_scaled_trajectory_cost_error'])
                    index=cfg['seeds'].index(row['seed']);batch=values[index,:cfg['evaluation_batch']]
                    max_feedback=max(max_feedback,((row['feedback_costs']-batch).abs()/(1+row['feedback_costs'].abs())).max().item())
                    traces+=len(row['states'])
        if raw['x'].shape[1]==32 and meta['condition']=='nominal':
            aa=raw['methods']['adaptive']['60'].numpy();checks={}
            task=next(t for t in cfg['tasks'] if t['family']==p.family)
            for baseline in task.get('primary_references',cfg['primary_references']):
                bb=raw['methods'][baseline]['60'].numpy();z=np.log(aa.mean(1)/bb.mean(1))
                upper=float(np.exp(z.mean()+student_t.ppf(1-.05/3,len(z)-1)*z.std(ddof=1)/np.sqrt(len(z))))
                ratio=float(aa.mean()/bb.mean());d=aa.mean(1)-1.01*bb.mean(1)
                difference_upper=float(d.mean()+student_t.ppf(1-.05/3,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
                strict_difference=aa.mean(1)-bb.mean(1)
                strict_upper=float(strict_difference.mean()+student_t.ppf(1-.05/3,len(d)-1)*strict_difference.std(ddof=1)/np.sqrt(len(d)))
                checks[baseline]=dict(pooled_ratio=ratio,corrected_upper_ratio=upper,
                    noninferiority_upper_mean_difference=difference_upper,noninferior=bool(ratio<=1.01 and difference_upper<=0),
                    superiority_upper_mean_difference=strict_upper,superior_one_percent=bool(ratio<=.99 and strict_upper<0))
                independent_a=torch.stack([r['feedback_costs'] for r in raw['audits']['adaptive']['60']]).numpy()
                independent_b=torch.stack([r['feedback_costs'] for r in raw['audits'][baseline]['60']]).numpy()
                original_ratio=float(aa[:,:cfg['evaluation_batch']].mean()/bb[:,:cfg['evaluation_batch']].mean())
                independent_ratio=float(independent_a.mean()/independent_b.mean())
                max_primary_feedback_ratio_change=max(max_primary_feedback_ratio_change,abs(independent_ratio-original_ratio))
            decisions[p.family]=dict(noninferior_to_all_standard=all(c['noninferior'] for c in checks.values()),
                superior_one_percent_to_all_standard=all(c['superior_one_percent'] for c in checks.values()),comparisons=checks)
    summary=json.loads((root/'report/summary.json').read_text())
    assert states==len(state_hashes)==3456
    assert costs==expected_policies*1152*len(cfg['budgets_seconds'])==898560
    assert traces==expected_policies*3*cfg['audit_states']==3120
    assert summary['cases']==len(expected_cases)==9
    for family,row in decisions.items():
        assert row['noninferior_to_all_standard']==summary['primary_decisions'][family]['noninferior_to_all_standard']
        assert row['superior_one_percent_to_all_standard']==summary['primary_decisions'][family]['superior_one_percent_to_all_standard']
        for b,c in row['comparisons'].items():
            for key in ['noninferior','superior_one_percent']:
                assert c[key]==summary['primary_decisions'][family]['comparisons'][b][key]
            for key in ['pooled_ratio','corrected_upper_ratio','noninferiority_upper_mean_difference','superiority_upper_mean_difference']:
                assert abs(c[key]-summary['primary_decisions'][family]['comparisons'][b][key])<1e-12
    objective_checks=[]
    for file in (root/'objective_diagnostics').glob('*.pt'):
        raw=torch.load(file,map_location='cpu',weights_only=False);rows=json.loads(file.with_suffix('.json').read_text())
        assert len(raw)==len(rows)
        for data,row in zip(raw,rows):
            assert (data['seed'],data['method'],data['batch'])==(row['seed'],row['method'],row['batch'])
            g,h=data['fresh_gradient'].numpy(),data['full_gradient'].numpy()
            finite=bool(np.isfinite(g).all() and np.isfinite(h).all())
            assert row['finite_gradients']==finite
            if finite:
                gn=float(np.linalg.norm(g));hn=float(np.linalg.norm(h));dot=float(np.dot(g,h))
                reconstructed=dict(fresh_gradient_norm=gn,full_gradient_norm=hn,
                    gradient_cosine=dot/max(gn*hn,1e-24),
                    relative_to_full_gradient_error=float(np.linalg.norm(g-h))/max(hn,1e-12))
                for key,value in reconstructed.items():assert np.isclose(value,row[key],rtol=1e-10,atol=1e-10),(file,key)
                assert row['fresh_negative_gradient_is_full_descent']==(dot>0)
            objective_checks.append(row)
    assert len(objective_checks)==2*expected_policies==520
    validation=json.loads((root/'validation_replay_audit.json').read_text());assert validation['passed']
    qp_gaps=[];qp_comparisons=0
    from experiments.hj_cotangent.convex_certificate import certificate
    for file in (root/'qp_references').glob('*.pt'):
        raw=torch.load(file,map_location='cpu',weights_only=False);meta=json.loads(file.with_suffix('.json').read_text())
        p=BuildingProblem(**meta['problem'])
        feasible=[];lower=[]
        for state,solution in zip(raw['x'].numpy(),raw['solutions']):
            checked=certificate(state,solution['controls'],p)
            for key,value in checked.items():assert abs(value-solution['certificate'][key])<1e-10
            qp_gaps.append(checked['first_order_gap'])
            feasible.append(checked['feasible_cost']);lower.append(checked['convex_lower_bound'])
        feasible_mean=float(np.mean(feasible));lower_mean=float(np.mean(lower))
        assert abs(feasible_mean-meta['optimal_mean'])<1e-8 and abs(lower_mean-meta['lower_bound_mean'])<1e-10
        evaluated=torch.load(root/'evaluation'/file.name,map_location='cpu',weights_only=False)
        assert torch.equal(raw['x'],evaluated['x'][:8])
        assert set(meta['methods'])==set(evaluated['methods'])
        for name,budgets in evaluated['methods'].items():
            mean=float(budgets['60'][:,:8].numpy().mean());published=meta['methods'][name]
            assert abs(mean-published['mean'])<1e-10
            assert abs(100*(mean/feasible_mean-1)-published['excess_over_feasible_percent'])<1e-7
            assert abs(100*(mean/lower_mean-1)-published['excess_over_lower_bound_percent'])<1e-10
            qp_comparisons+=1
    assert len(qp_gaps)==24
    transport_gradient_rows=[]
    for file in (root/'transport_diagnostics').glob('*.pt'):
        vectors=torch.load(file,map_location='cpu',weights_only=False);rows=json.loads(file.with_suffix('.json').read_text())
        assert len(rows)==len(vectors)
        for row,vector in zip(rows,vectors):
            g,h=vector['full_gradient'],vector['surrogate_gradient']
            if row['finite']:
                assert torch.isfinite(g).all() and torch.isfinite(h).all()
                error=((g-h).norm()/g.norm().clamp_min(1e-12)).item()
                assert abs(error-row['relative_gradient_error'])<1e-10
            transport_gradient_rows.append(row)
    assert len(transport_gradient_rows)==sum(sum(s['mode']=='transport' for s in t['methods'])*len(cfg['seeds']) for t in cfg['tasks'])
    teacher_checks=[]
    max_teacher_cost_replay=max_teacher_gradient_replay=0.
    for file in (root/'teacher_audit').glob('*.pt'):
        raw=torch.load(file,map_location='cpu',weights_only=False);rows=json.loads(file.with_suffix('.json').read_text())
        assert len(raw)==len(rows)
        for data,row in zip(raw,rows):
            assert (data['seed'],data['method'])==(row['seed'],row['method'])
            task=next(t for t in cfg['tasks'] if t['family']==row['family'])
            record_path=root/'training'/f'{row["family"]}_h{task["horizon"]}_s{row["seed"]}'/(row['method']+'.json')
            recorded=json.loads(record_path.read_text())['incumbent_training_mean']
            assert torch.isfinite(data['costs']).all()
            # The saved audit mean uses a CUDA float32 reduction. Independently
            # reduce the raw costs in CPU float64 without requiring identical
            # reduction order, then verify the published scalar calculation.
            raw_mean=data['costs'].double().mean().item()
            mean_error=abs(raw_mean-recorded)/(1+abs(recorded))
            scalar_error=abs(row['training_mean']-recorded)/(1+abs(recorded))
            assert mean_error<1e-6 and abs(raw_mean-row['training_mean'])/(1+abs(raw_mean))<1e-6
            assert abs(scalar_error-row['scaled_mean_replay_error'])<1e-12
            max_teacher_cost_replay=max(max_teacher_cost_replay,mean_error)
            if data['saved_global_gradient'] is not None:
                g,h=data['global_gradient'].double(),data['saved_global_gradient'].double()
                assert torch.isfinite(g).all() and torch.isfinite(h).all()
                error=((g-h).norm()/(1+h.norm())).item()
                assert error<1e-5 and abs(error-row['scaled_global_gradient_replay_error'])<1e-12
                max_teacher_gradient_replay=max(max_teacher_gradient_replay,error)
            teacher_checks.append(row)
    assert len(teacher_checks)==len(transport_gradient_rows) and all(r['passed'] for r in teacher_checks)
    batch_lock=json.loads((root/'batch_gradient_audit_lock.json').read_text())
    assert batch_lock['original_training_lock_sha256']==digest(root/'training_lock.json')
    for path,sha in batch_lock['sources'].items():assert digest(path)==sha,path
    batch_checks=[];batch_shape_checks=[]
    for file in (root/'batch_gradient_diagnostics').glob('*.pt'):
        raw=torch.load(file,map_location='cpu',weights_only=False);rows=json.loads(file.with_suffix('.json').read_text())
        assert len(raw)==len(rows)
        for data,row in zip(raw,rows):
            assert (data['seed'],data['method'])==(row['seed'],row['method'])
            task=next(t for t in cfg['tasks'] if t['family']==row['family'])
            checkpoint=root/'training'/f'{row["family"]}_h{task["horizon"]}_s{row["seed"]}'/(row['method']+'.pt')
            assert digest(checkpoint)==row['checkpoint_sha256']
            reference=data['full_gradient'].numpy()
            for batch,vector in data['mean_local_gradients'].items():
                check=row['uncorrected_local_gradient_checks'][batch]
                current=vector.numpy();finite=bool(np.isfinite(reference).all() and np.isfinite(current).all())
                assert finite==check['finite']
                if finite:
                    norm=float(np.linalg.norm(reference));error=float(np.linalg.norm(current-reference))
                    relative=error/max(norm,1e-12);scaled=error/(1+norm)
                    assert abs(relative-check['relative_error'])<1e-10 and abs(scaled-check['scaled_error'])<1e-10
                    assert check['passed']==(relative<batch_lock['relative_tolerance'])
                else:assert not check['passed']
                batch_checks.append(dict(family=row['family'],seed=row['seed'],method=row['method'],batch=int(batch),**check))
            assert set(data['mean_local_gradients'])=={'128','1024'}
            assert row['all_checks_passed']==all(r['passed'] for r in row['uncorrected_local_gradient_checks'].values())
            a=data['mean_local_gradients']['1024'].numpy();b=data['mean_local_gradients']['128'].numpy()
            shape=row['batch_shape_difference'];finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
            assert shape['finite']==finite
            if finite:
                norm=float(np.linalg.norm(b));error=float(np.linalg.norm(a-b))
                relative=error/max(norm,1e-12);scaled=error/(1+norm)
                assert abs(relative-shape['relative_error'])<1e-10 and abs(scaled-shape['scaled_error'])<1e-10
                assert shape['passed']==(relative<batch_lock['relative_tolerance'])
            else:assert not shape['passed']
            batch_shape_checks.append(dict(family=row['family'],seed=row['seed'],method=row['method'],**shape))
    assert len(batch_checks)==2*len(transport_gradient_rows)
    prefix_lock=json.loads((root/'prefix_equality_audit_lock.json').read_text())
    assert prefix_lock['original_training_lock_sha256']==digest(root/'training_lock.json')
    for path,sha in prefix_lock['sources'].items():assert digest(path)==sha,path
    prefix_checks=[]
    for file in (root/'prefix_equality_diagnostics').glob('*.pt'):
        raw=torch.load(file,map_location='cpu',weights_only=False);rows=json.loads(file.with_suffix('.json').read_text())
        assert len(raw)==len(rows)
        for data,row in zip(raw,rows):
            assert (data['seed'],data['method'])==(row['seed'],row['method'])
            task=next(t for t in cfg['tasks'] if t['family']==row['family'])
            checkpoint=root/'training'/f'{row["family"]}_h{task["horizon"]}_s{row["seed"]}'/(row['method']+'.pt')
            assert digest(checkpoint)==row['checkpoint_sha256']
            g,h=data['prefix_gradient'].numpy(),data['full_gradient'].numpy()
            finite=bool(np.isfinite(g).all() and np.isfinite(h).all())
            assert row['finite_gradients']==finite
            if finite:
                gn=float(np.linalg.norm(g));hn=float(np.linalg.norm(h));dot=float(np.dot(g,h))
                reconstructed=dict(prefix_gradient_norm=gn,full_gradient_norm=hn,
                    gradient_cosine=dot/max(gn*hn,1e-24),
                    relative_to_full_gradient_error=float(np.linalg.norm(g-h))/max(hn,1e-12))
                for key,value in reconstructed.items():assert np.isclose(value,row[key],rtol=1e-10,atol=1e-10),(file,key)
                assert row['prefix_negative_gradient_is_full_descent']==(dot>0)
            a,b=data['prefix_batch_values'].numpy(),data['full_batch_values'].numpy()
            finite_values=bool(np.isfinite(a).all() and np.isfinite(b).all())
            assert finite_values==row['finite_values']
            if finite_values:
                error=float(np.max(np.abs(a-b)/(1+np.abs(b))))
                assert abs(error-row['max_scaled_batch_value_difference'])<1e-12
                assert row['equal_values_within_tolerance']==(error<prefix_lock['value_scaled_tolerance'])
            else:assert not row['equal_values_within_tolerance']
            prefix_checks.append(row)
    assert len(prefix_checks)==prefix_lock['policies']==len(transport_gradient_rows)
    paths=[p for folder in ['training','evaluation','objective_diagnostics','transport_diagnostics','teacher_audit','batch_gradient_diagnostics','prefix_equality_diagnostics','qp_references'] for p in (root/folder).rglob('*') if p.is_file()]
    paths += [root/p for p in ['config.json','protocol.md','training_lock.json','evaluation_lock.json','report/summary.json','environment.json','development_selection.json','validation_replay_audit.json']]
    paths += [root/'batch_gradient_audit_lock.json']+[Path(p) for p in batch_lock['sources']]
    paths += [root/'prefix_equality_audit_lock.json']+[Path(p) for p in prefix_lock['sources']]
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phases=len(phases),budget_snapshots=snapshots,
        completion_source_sha256=digest(Path(__file__)),
        development=audit_development(root),
        distinct_primary_states=len(state_hashes),primary_state_instances=states,individual_policy_costs=costs,independently_checked_trajectories=traces,
        max_local_residual=max_local,max_independent_trajectory_cost_error=max_cost,max_feedback_scaled_difference=max_feedback,
        maximum_recorded_compute_overrun_seconds=maximum_overrun,nonfinite_updates=nonfinite,primary_decisions=decisions,
        max_primary_feedback_subset_ratio_change=max_primary_feedback_ratio_change,qp_reference_states=len(qp_gaps),maximum_qp_first_order_gap=max(qp_gaps),
        independently_reconstructed_qp_method_comparisons=qp_comparisons,
        maximum_validation_replay_scaled_mean_error=validation['maximum_scaled_mean_error'],
        independently_reconstructed_objective_gradient_checks=len(objective_checks),
        trained_transport_gradient_checks=len(transport_gradient_rows),
        accepted_teacher_replay_checks=len(teacher_checks),
        max_accepted_teacher_cost_replay_error=max_teacher_cost_replay,
        max_accepted_teacher_global_gradient_replay_error=max_teacher_gradient_replay,
        actual_batch_gradient_checks=batch_checks,
        actual_batch_gradient_failures=[row for row in batch_checks if not row['passed']],
        actual_batch_shape_checks=batch_shape_checks,
        prefix_equality_gradient_checks=prefix_checks,
        prefix_equality_value_failures=[r for r in prefix_checks if not r['equal_values_within_tolerance']],
        trained_transport_gradient_failures=[r for r in transport_gradient_rows if not r['agreement_within_audit_tolerance']],
        phase_records=phases,evidence_hashes={str(p):digest(p) for p in paths})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    root=Path(args.root);result=audit(root);target=root/'completion_audit.json'
    if args.verify_only:
        saved=json.loads(target.read_text())
        for key in result:
            if key!='created_utc':assert saved[key]==result[key],key
    else:
        assert not target.exists(),'Use --verify-only for an existing audit'
        write(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['phase_records','evidence_hashes','development','actual_batch_gradient_checks','actual_batch_shape_checks','prefix_equality_gradient_checks']},indent=2))


if __name__=='__main__':main()
