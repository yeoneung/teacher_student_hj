"""Paired statistics and prospective targets; never selects training settings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_gridfree.report import comparison
from .study import verify
from .evaluation import train_dir,lock_evaluation


def seed_interval(a,b,alpha=.05/3):
    a,b=np.asarray(a),np.asarray(b)
    ratio=a.mean(1)/b.mean(1)
    log=np.log(ratio)
    margin=student_t.ppf(1-alpha,len(log)-1)*log.std(ddof=1)/np.sqrt(len(log))
    return dict(pooled_mean_ratio=float(a.mean()/b.mean()),geometric_seed_ratio=float(np.exp(log.mean())),
                upper_ratio=float(np.exp(log.mean()+margin)),one_sided_alpha=alpha,degrees_of_freedom=len(log)-1,
                seed_ratios=ratio.tolist(),scope='Paired seed-level t approximation on log ratios; state mean within each seed. Bonferroni alpha=0.05/3 for three primary questions.')


def collect(root,cfg):
    summary=dict(configuration=cfg,training={},cases={},primary_targets={},memory={},timing={},feedback={},
                 counts=dict(planned_phase_endpoints=sum(len(t['methods'])*len(cfg['seeds']) for t in cfg['tasks']),
                             independent_training_seeds=len(cfg['seeds']),
                             planned_test_cases=len(cfg['tasks'])*len(cfg['test_dimensions'])*len(cfg['test_conditions'])))
    for task in cfg['tasks']:
        key=f'{task["family"]}_h{task["horizon"]}'
        summary['training'][key]={}
        for spec in task['methods']:
            rows=[json.loads((train_dir(root,task,seed)/(spec['name']+'.json')).read_text()) for seed in cfg['seeds']]
            summary['training'][key][spec['name']]=dict(
                mean_seconds=float(np.mean([r['compute_seconds'] for r in rows])),
                seed_seconds=[r['compute_seconds'] for r in rows],
                mean_validation_cost=float(np.mean([r['mean'] for r in rows])),
                mean_refresh_seconds=float(np.mean([r['refresh_seconds'] for r in rows])),
                steps=[r['steps_completed'] for r in rows],
                best_steps=[r['best_step'] for r in rows],
                initial_seed_costs=[r['initial_mean'] for r in rows],
                nonfinite_updates_skipped=[r['nonfinite_updates_skipped'] for r in rows],
                nonfinite_validation_stops=[r['stopped_on_nonfinite_validation'] for r in rows],
                refresh_counts=[r['refresh_count'] for r in rows],
                jet_diagnostics=[r['jet_diagnostics'] for r in rows],
                seed_accounting=[{k:r[k] for k in ['data_seconds','setup_seconds','fit_seconds','refresh_seconds',
                    'validation_seconds','parent_seconds','compute_seconds','checkpoint_io_seconds']} for r in rows],
                final_clipped_fractions=[r['history'][-1].get('clipped_update_fraction') for r in rows],
                histories=[r['history'] for r in rows])
    for source in sorted((root/'evaluation').glob('*.pt')):
        meta=json.loads(source.with_suffix('.json').read_text())
        raw=torch.load(source,weights_only=False)
        arrays={k:v.numpy() for k,v in raw['methods'].items()}
        record=dict(problem=meta['problem'],condition=meta['condition'],count=meta['count'],
                    means={k:float(a.mean()) for k,a in arrays.items()},teacher_mean=raw['teacher'].mean().item(),
                    teacher_audit=meta['teacher_audit'],
                    seed_means={k:a.mean(1).tolist() for k,a in arrays.items()},comparisons={},
                    tails={k:dict(p95=float(np.quantile(a,.95)),maximum=float(a.max())) for k,a in arrays.items()},
                    max_replay_error=max(a['independent_scaled_cost_error'] for a in meta['audits']),
                    max_single_step_residual=max(a['max_scaled_single_step_residual'] for a in meta['audits']),
                    max_fixed_control_replay_difference=max(a['fixed_control_open_loop_scaled_difference'] for a in meta['audits']),
                    max_feedback_replay_difference=max(a['independent_feedback_scaled_cost_difference'] for a in meta['audits']),
                    max_feedback_mean_difference=max(a['independent_feedback_relative_mean_difference'] for a in meta['audits']),
                    max_budget_ratio=max(a['max_budget_ratio'] for a in meta['audits']))
        for candidate in ['fresh','cache']+(['cache_promoted'] if 'cache_promoted' in arrays else []):
            record['comparisons'][candidate+'_vs_teacher']=comparison(arrays[candidate],raw['teacher'].numpy(),repetitions=cfg['bootstrap_draws'])
            for control in arrays:
                if candidate==control:
                    continue
                record['comparisons'][candidate+'_vs_'+control]=comparison(arrays[candidate],arrays[control],repetitions=cfg['bootstrap_draws'])
        native=root/'classical_native'/source.name
        if native.exists():
            reference=torch.load(native,weights_only=False)
            n=len(reference['cost'])
            nm=json.loads(native.with_suffix('.json').read_text())
            record['classical']=dict(count=n,mean=reference['cost'].mean().item(),comparisons={},
                                     method_means={k:float(a[:,:n].mean()) for k,a in arrays.items()},
                                     native_quality_wall_seconds=nm['wall_seconds'],
                                     native_workers=nm['workers'])
            gpu_meta=root/'classical_gpu'/source.with_suffix('.json').name
            if gpu_meta.exists():
                gm=json.loads(gpu_meta.read_text())
                record['classical']['preceding_gpu_quality_seconds']=gm['quality_seconds']
                record['classical']['full_search_wall_seconds']=gm['quality_seconds']+nm['wall_seconds']
            for name,a in arrays.items():
                record['classical']['comparisons'][name]=comparison(a[:,:n],reference['cost'].numpy(),repetitions=cfg['bootstrap_draws'])
            if meta['problem']['family']=='building':
                lower=np.array([row['convex_certificate']['convex_lower_bound'] for row in nm['rows']])
                record['classical']['convex_certificate']=dict(mean_lower_bound=float(lower.mean()),
                    max_first_order_gap=max(row['convex_certificate']['first_order_gap'] for row in nm['rows']),
                    policy_mean_excess_percent={k:float(100*(a[:,:n].mean()/lower.mean()-1)) for k,a in arrays.items()})
            feedback_source=root/'feedback_classical'/source.name
            if feedback_source.exists():
                fb=torch.load(feedback_source,weights_only=False)
                fm=json.loads(feedback_source.with_suffix('.json').read_text())
                for method,method_data in fb['methods'].items():
                    if 'original_feedback_cost' not in method_data:continue
                    original=method_data['original_feedback_cost'].numpy()
                    initials=np.array([r['initial_cost'] for r in fm['methods'][method]['rows']])
                    final=method_data['cost'].numpy()
                    fm['methods'][method].update(initial_tracking_mean=float(initials.mean()),
                        max_initial_tracking_scaled_difference=float(np.max(np.abs(initials-original)/(1+np.abs(original)))),
                        optimization_reduction_percent=float(100*(1-final.mean()/initials.mean())))
                combined=np.minimum(reference['cost'].numpy(),fb['reference_best'].numpy())
                record['combined_classical']=dict(count=n,mean=float(combined.mean()),
                    original_classical_mean=record['classical']['mean'],feedback_reference_mean=fm['reference_mean'],
                    method_means={k:float(a[:,:n].mean()) for k,a in arrays.items()},
                    comparisons={k:comparison(a[:,:n],combined,repetitions=cfg['bootstrap_draws']) for k,a in arrays.items()},
                    feedback_quality_wall_seconds=sum(fm['methods'][k]['wall_seconds'] for k in ['zero_residual','gpu_plan']),
                    preceding_search_wall_seconds=record['classical']['full_search_wall_seconds'],
                    scope='Feasible minimum of the original classical search and two feedback tracking starts; learned-policy starts excluded.')
                summary['feedback'][source.stem]=fm
        summary['cases'][source.stem]=record
    mechanical=torch.load(root/'evaluation/mechanical_h160_d32_nominal.pt',weights_only=False)['methods']
    contrasts={name:seed_interval(mechanical['fresh'].numpy(),mechanical[name].numpy()) for name in ['dpc','dpc_warm']}
    summary['primary_targets']['mechanical_quality']=dict(contrasts=contrasts,threshold_ratio=.90,
        met=all(c['pooled_mean_ratio']<=.90 and c['upper_ratio']<.90 for c in contrasts.values()))
    for family,horizon in [('reaction',320),('building',192)]:
        data=torch.load(root/'evaluation'/f'{family}_h{horizon}_d32_nominal.pt',weights_only=False)['methods']
        interval=seed_interval(data['cache'].numpy(),data['fresh'].numpy())
        times=summary['training'][f'{family}_h{horizon}']
        ratio=times['cache']['mean_seconds']/times['fresh']['mean_seconds']
        summary['primary_targets'][family+'_cache']=dict(quality=interval,total_time_ratio=ratio,
            met=interval['pooled_mean_ratio']<=1.01 and interval['upper_ratio']<1.01 and ratio<=.5)
    summary['primary_numerical_sensitivity']={}
    for family,horizon,candidate,control in [('mechanical',160,'fresh','dpc'),
            ('mechanical',160,'fresh','dpc_warm'),('reaction',320,'cache','fresh'),
            ('building',192,'cache','fresh')]:
        key=f'{family}_h{horizon}_d32_nominal'
        raw=torch.load(root/'evaluation'/(key+'.pt'),weights_only=False)
        n=cfg['evaluation_batch']
        original={m:raw['methods'][m][:,:n].mean().item() for m in [candidate,control]}
        independent={m:raw['feedback_replay_costs'][m].mean().item() for m in [candidate,control]}
        gpu_ratio=original[candidate]/original[control]
        numpy_ratio=independent[candidate]/independent[control]
        summary['primary_numerical_sensitivity'][family+'_'+candidate+'_vs_'+control]=dict(
            case=key,count=n,gpu_means=original,numpy_plant_means=independent,
            gpu_cost_ratio=gpu_ratio,numpy_plant_cost_ratio=numpy_ratio,
            absolute_cost_ratio_change=abs(gpu_ratio-numpy_ratio),
            scope='Same first 128 states and all five seeds under both plants. This diagnoses numerical sensitivity, not an extra primary decision or independent test population.')
    for folder,target in [('isolated_memory','memory'),('timing','timing')]:
        for file in (root/folder).glob('*.json'):
            summary[target][file.stem]=json.loads(file.read_text())
    summary['statistics']='Crossed bootstrap intervals are descriptive secondary analyses. Primary decisions use the prespecified paired seed log-ratio t bounds. All old v1/v4 criteria remain unchanged.'
    unique={}
    for case in summary['cases'].values():
        p=case['problem']
        unique[(p['family'],p['n'],case['condition'])]=case['count']
    summary['counts'].update(actual_test_cases=len(summary['cases']),distinct_initial_vectors=sum(unique.values()),
                             state_horizon_instances=sum(c['count'] for c in summary['cases'].values()),
                             primary_targets_met=sum(t['met'] for t in summary['primary_targets'].values()),
                             classical_cases=sum('classical' in c for c in summary['cases'].values()),
                             classical_state_horizon_instances=sum(c['classical']['count'] for c in summary['cases'].values() if 'classical' in c))
    audit_root=Path('experiments/results/hj_cotangent_dev_v6c')
    summary['analytic_audits']={name:json.loads((audit_root/name).read_text()) for name in
        ['final_numerical_audit.json','quadratic_jet_audit.json','cached_performance_identity.json','cpu_conditioning_audit.json','feedback_native_audit.json','teacher_curvature_horizon.json','teacher_evaluation_cpu_pretest.json']}
    summary['evaluation_preflight']=json.loads((root/'pretest_evaluation_audit.json').read_text())
    summary['feedback_counts']=json.loads((root/'feedback_complete.json').read_text())
    return summary


def render(root,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=root/'report'
    target.mkdir(exist_ok=True)
    colors=dict(fresh='#087f8c',cache='#d47924',dpc='#75529c',short='#577354',tbptt='#8d8d8d',dpc_warm='#b64f5d',cache_promoted='#2566b0')
    labels=dict(fresh='Fresh teacher',cache='Cached teacher',dpc='DPC',short='Short horizon',tbptt='Truncated BPTT',dpc_warm='Warm DPC',cache_promoted='Promoted cached teacher')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,1,figsize=(6.5,7.5))
    for ax,family,horizons in zip(axes,['mechanical','reaction','building'],[[40,160,320],[40,160,320],[96,192]]):
        for name in ['fresh','cache','dpc','short','tbptt']:
            y=[summary['cases'][f'{family}_h{h}_d32_nominal']['means'][name] for h in horizons]
            ax.plot(horizons,y,'o-',label=labels[name],color=colors[name])
        ax.set_title(family.capitalize())
        ax.set_xlabel('Physical simulation steps')
        ax.set_ylabel('Mean full-trajectory cost')
        if family=='mechanical':ax.set_yscale('log')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9,ncol=3)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(target/f'horizon_quality.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(6.5,7.5))
    for ax,family,horizon in zip(axes,['mechanical','reaction','building'],[160,320,192]):
        training=summary['training'][f'{family}_h{horizon}']
        for name,record in training.items():
            if name not in colors:
                continue
            for index,history in enumerate(record['histories']):
                ax.plot([r['compute_seconds'] for r in history],[r['best'] for r in history],
                        color=colors[name],alpha=.4,label=labels[name] if index==0 else None)
        ax.set_title(f'{family.capitalize()}, H={horizon}')
        ax.set_xlabel('Total observed training compute (s)')
        ax.set_ylabel('Best validation cost')
        ax.set_xscale('log')
        if family=='mechanical':ax.set_yscale('log')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9,ncol=2)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(target/f'learning_compute.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    with (target/'costs.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['case','method','mean','teacher_mean'])
        for key,case in summary['cases'].items():
            for name,value in case['means'].items():writer.writerow([key,name,value,case['teacher_mean']])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='experiments/results/hj_cotangent_primary_v6')
    args=parser.parse_args()
    root=Path(args.root)
    cfg=verify(root)
    lock_evaluation(root,cfg)
    summary=collect(root,cfg)
    write(root/'report/summary.json',summary)
    render(root,summary)
    from .figures import render_summary
    render_summary(root,summary)
    print(json.dumps(summary['primary_targets'],indent=2),flush=True)


if __name__=='__main__':main()
