"""Lock final transport choices after all development and before primary work."""
import copy
import datetime
import json
from pathlib import Path
import shutil
import sys
import torch
from experiments.hj_gridfree.engine import write
from .prepare import A,B,complete,options
from .study import digest


def main():
    C=Path('experiments/results/hj_adaptive_dev_v7c');D=Path('experiments/results/hj_adaptive_dev_v7d');E=Path('experiments/results/hj_adaptive_dev_v7e')
    F=Path('experiments/results/hj_adaptive_dev_v7f_controls')
    acfg=complete(A)
    for path in [B,C,D,E,F]:complete(path)
    for path in [E,F]:assert json.loads((path/'validation_replay_audit.json').read_text())['passed']
    teacher_audits=list((E/'teacher_audit').glob('*.json'))
    assert len(teacher_audits)==3
    teacher_rows=[row for path in teacher_audits for row in json.loads(path.read_text())]
    assert len(teacher_rows)==48 and all(row['passed'] for row in teacher_rows)
    for path in [E/'variance_cuda_audit.json',E/'statistical_audit.json',F/'full_batch_audit.json']:
        assert json.loads(path.read_text())['passed'],path
    root=Path('experiments/results/hj_transport_primary_v7');root.mkdir(parents=True,exist_ok=True)
    assert not (root/'config.json').exists(),'A primary design already exists; do not replace it'
    tasks=[];selections=[]
    for task in acfg['tasks']:
        groups={
            'adaptive':options(C,task,['transport_adaptive_'])+options(D,task,['warm_transport_adaptive_'])+options(E,task,['cv_']),
            'dpc':options(A,task,['dpc_'])+options(C,task,['dpc_initial_']),
            'warm':options(A,task,['warm_'])+options(C,task,['warm_initial_']),
            'short':options(A,task,['short_']),
            'fresh':options(A,task,['fresh_'])+options(B,task,['promoted_fresh_'])}
        groups['dpc_full']=options(F,task,['dpc_full_']);groups['warm_full']=options(F,task,['warm_full_'])
        winners={key:min(rows,key=lambda r:(r['mean'],r['method']['name'])) for key,rows in groups.items()}
        warm=bool(winners['adaptive']['method'].get('parent'))
        groups['fixed']=options(D if warm else C,task,['warm_transport_fixed_' if warm else 'transport_fixed_'])
        winners['fixed']=min(groups['fixed'],key=lambda r:(r['mean'],r['method']['name']))
        names=['short','dpc','warm','fresh','fixed','adaptive','dpc_full','warm_full'];references=['dpc','warm','short','dpc_full','warm_full']
        for key in ['dpc','warm']:
            if not winners[key]['method'].get('initial_objective'):
                name=key+'_initial';groups[name]=options(C,task,[name+'_'])
                winners[name]=min(groups[name],key=lambda r:(r['mean'],r['method']['name']))
                names.append(name);references.append(name)
        parent_name='short' if winners['short']['method']['lr']==.002 else 'short_parent'
        specs=[]
        for name in names:
            spec=copy.deepcopy(winners[name]['method']);spec['name']=name
            if spec.get('parent'):spec['parent']=parent_name
            if spec.get('teacher_parent'):spec['teacher_parent']=parent_name
            specs.append(spec)
        if parent_name=='short_parent':specs.append(dict(name=parent_name,mode='short',lr=.002,parent_only=True,budgets_seconds=[15]))
        principal=min(references,key=lambda name:winners[name]['mean'])
        tasks.append(dict(family=task['family'],horizon=task['horizon'],methods=specs,
            principal_reference=principal,primary_references=references,
            transport_initialization='quarter_budget_short' if warm else 'analytic_residual',
            teacher_parent_name=parent_name if winners['fresh']['method'].get('teacher_parent') else None))
        selections.append(dict(family=task['family'],groups=groups,winners=winners,primary_references=references))
    cfg={k:v for k,v in acfg.items() if k not in ['tasks','development','seeds','budgets_seconds','parent_budget_seconds']}
    cfg.update(development=False,interface='trajectory_transport',seeds=list(range(9301,9311)),tasks=tasks,
        budgets_seconds=[15,30,60],parent_budget_seconds=15,validation_seed_base=9300000,test_seed_base=29000000,
        evaluation_batch=128,audit_states=4,primary_references=['dpc','warm','short'],
        primary_noninferiority_margin=1.01,primary_alpha=.05/3)
    write(root/'config.json',cfg);shutil.copyfile('docs/hj_v7_primary_protocol.md',root/'protocol.md')
    write(root/'development_selection.json',dict(selected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rule='Lowest mean30-second validation cost over the two development seeds; name breaks ties. Fixed transport matches candidate initialization. Matched initial-objective standard controls remain primary even when mixed-objective controls win selection.',
        selections=selections,development_new_phases=312,notes='All six rounds retained, including the baseline-only final round. Offline tuning budgets are not equal; no primary training or tests used for selection.'))
    sources=[p for p in Path('experiments/hj_adaptive').glob('*.py') if p.name not in ['completion.py','writeup.py','release.py','visual.py','observer.py','figures.py']]
    sources+=list(Path('experiments/hj_cotangent').glob('*.py'))
    sources += [Path('experiments/hj_gridfree')/p for p in ['core.py','engine.py','evaluate.py','native.py','planning_probe.py','report.py']]
    for dev in [A,B,C,D,E,F]:
        sources += [dev/'config.json',dev/'protocol.md',dev/'development_start.json']
        sources += list((dev/'source_at_development_start').glob('*.py'))
    sources += [A/'probe_audit.json',B/'fresh_window_audit.json',C/'transport_cuda_expanded_audit.json',C/'transport_update_audit.json',E/'variance_cuda_audit.json',E/'statistical_audit.json',root/'development_selection.json']
    sources += [F/'full_batch_audit.json',F/'validation_replay_audit.json',E/'validation_replay_audit.json']+list((E/'teacher_audit').glob('*.json'))
    sources.append(E/'analytic_examples.json')
    write(root/'training_lock.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_sha256=digest(root/'config.json'),protocol_sha256=digest(root/'protocol.md'),sources={str(p):digest(p) for p in sources}))
    write(root/'environment.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,
        executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),
        cpu_thread_policy=1,host_scope='Shared host; unrelated CPU workers were not changed. GPU numerical jobs run serially.'))
    print(json.dumps(dict(root=str(root),phases=sum(len(t['methods'])*len(cfg['seeds']) for t in tasks),tasks=tasks),indent=2))


if __name__=='__main__':main()
