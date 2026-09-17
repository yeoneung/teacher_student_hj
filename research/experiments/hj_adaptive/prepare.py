"""Preserve development provenance and freeze choices before new primary tests."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch
from experiments.hj_gridfree.engine import write
from .study import digest


A=Path('experiments/results/hj_adaptive_dev_v7')
B=Path('experiments/results/hj_adaptive_dev_v7b')


def complete(root):
    cfg=json.loads((root/'config.json').read_text())
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            folder=root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'
            assert (folder/'complete.json').exists(),folder
            for spec in task['methods']:assert (folder/(spec['name']+'.json')).exists()
    return cfg


def development_b():
    cfg=complete(A);copied={}
    assert json.loads((B/'fresh_window_audit.json').read_text())['passed']
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            name=f'{task["family"]}_h{task["horizon"]}_s{seed}';dest=B/'training'/name;dest.mkdir(parents=True,exist_ok=True)
            for file in ['data.pt','short_002.pt','short_002.json']:
                source=A/'training'/name/file;target=dest/file
                if target.exists():assert digest(target)==digest(source)
                else:shutil.copyfile(source,target)
                copied[str(target)]=dict(source=str(source),sha256=digest(source))
    snap=B/'source_at_development_start';snap.mkdir(exist_ok=True)
    for p in Path('experiments/hj_adaptive').glob('*.py'):
        target=snap/p.name
        if target.exists():assert digest(target)==digest(p)
        else:shutil.copyfile(p,target)
    write(B/'development_start.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        copied_parent_evidence=copied,sources={str(p):digest(p) for p in snap.glob('*.py')},
        note='Parents and shared data copied unchanged from first development round. Their full allocation is charged to each dependent variant. These are not new independent parent runs.'))
    print('SECOND DEVELOPMENT ROOT PREPARED')


def options(root,task,prefixes):
    cfg=json.loads((root/'config.json').read_text());rows=[]
    actual=next(t for t in cfg['tasks'] if t['family']==task['family'])
    for spec in actual['methods']:
        if not any(spec['name'].startswith(p) for p in prefixes):continue
        paths=[root/'training'/f'{task["family"]}_h{task["horizon"]}_s{s}'/(spec['name']+'.json') for s in cfg['seeds']]
        values=[json.loads(p.read_text())['budgets']['30']['validation_mean'] for p in paths]
        rows.append(dict(root=str(root),method=copy.deepcopy(spec),mean=sum(values)/len(values),seed_means=values,
            evidence_hashes={str(p):digest(p) for p in paths}))
    assert rows
    return rows


def frozen_primary():
    acfg=complete(A);complete(B)
    root=Path('experiments/results/hj_adaptive_primary_v7');root.mkdir(parents=True,exist_ok=True)
    assert not (root/'config.json').exists(),'Primary configuration already exists'
    selections=[];tasks=[]
    for task in acfg['tasks']:
        groups={}
        for key,prefix in [('short','short_'),('dpc','dpc_'),('warm','warm_')]:groups[key]=options(A,task,[prefix])
        groups['adaptive']=options(A,task,['adaptive_'])+options(B,task,['adaptive_','promoted_adaptive_'])
        winners={key:min(rows,key=lambda r:(r['mean'],r['method']['name'])) for key,rows in groups.items()}
        promoted=bool(winners['adaptive']['method'].get('teacher_parent'))
        groups['fresh']=options(B if promoted else A,task,['promoted_fresh_' if promoted else 'fresh_'])
        groups['fixed']=options(B if promoted else A,task,['promoted_fixed_' if promoted else 'fixed_'])
        winners.update({key:min(groups[key],key=lambda r:(r['mean'],r['method']['name'])) for key in ['fresh','fixed']})
        parent_name='short' if winners['short']['method']['lr']==.002 else 'short_parent'
        specs=[]
        for name in ['short','dpc','warm','fresh','fixed','adaptive']:
            spec=copy.deepcopy(winners[name]['method']);spec['name']=name
            if spec.get('parent'):spec['parent']=parent_name
            if spec.get('teacher_parent'):spec['teacher_parent']=parent_name
            specs.append(spec)
        if parent_name=='short_parent':specs.append(dict(name=parent_name,mode='short',lr=.002,parent_only=True,budgets_seconds=[15]))
        principal=min(['short','dpc','warm'],key=lambda name:winners[name]['mean'])
        tasks.append(dict(family=task['family'],horizon=task['horizon'],methods=specs,
            principal_reference=principal,teacher_parent_name=parent_name if promoted else None))
        selections.append(dict(family=task['family'],groups=groups,winners=winners,principal_reference=principal,
            teacher_type='frozen_short_parent' if promoted else 'analytic'))
    cfg={k:v for k,v in acfg.items() if k not in ['tasks','development','seeds','budgets_seconds','parent_budget_seconds']}
    cfg.update(development=False,seeds=list(range(9301,9311)),tasks=tasks,budgets_seconds=[15,30,60],
        parent_budget_seconds=15,validation_seed_base=9300000,test_seed_base=29000000,evaluation_batch=128,audit_states=4,
        primary_references=['dpc','warm','short'],primary_noninferiority_margin=1.01,primary_alpha=.05/3)
    write(root/'config.json',cfg);shutil.copyfile('docs/hj_v7_protocol.md',root/'protocol.md')
    write(root/'development_selection.json',dict(selected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rule='Lowest mean 30-second development validation cost over the two fixed seeds; name breaks an exact tie. Fresh/fixed controls match the selected adaptive teacher and parent. All three standard controls remain primary references.',
        selections=selections,notes='The adaptive family has six development variants versus two learning-rate/period settings per matched baseline. Offline tuning budgets are not asserted equal. All choices precede primary training and test access.'))
    sources=[]
    for name in ['study.py','probe.py','evaluation.py','report.py','diagnostics.py','validation_audit.py','qp_reference.py','audit_probe.py','audit_fresh_window.py','prepare.py']:
        sources.append(Path('experiments/hj_adaptive')/name)
    sources+=list(Path('experiments/hj_cotangent').glob('*.py'))
    sources += [Path('experiments/hj_gridfree')/p for p in ['core.py','engine.py','evaluate.py','native.py','planning_probe.py','report.py']]
    for dev in [A,B]:
        sources += [dev/'config.json',dev/'protocol.md',dev/'development_start.json']
        sources += list((dev/'source_at_development_start').glob('*.py'))
    sources += [A/'probe_audit.json',B/'fresh_window_audit.json',root/'development_selection.json']
    write(root/'training_lock.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_sha256=digest(root/'config.json'),protocol_sha256=digest(root/'protocol.md'),
        sources={str(p):digest(p) for p in sources}))
    write(root/'environment.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,
        executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),
        cpu_thread_policy=1,host_scope='Shared host; unrelated CPU workers were not stopped or reprioritized.'))
    print(json.dumps(dict(root=str(root),phases=sum(len(t['methods'])*len(cfg['seeds']) for t in tasks),
        selected=[dict(family=t['family'],methods=t['methods']) for t in tasks]),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development-b','freeze-primary']);args=parser.parse_args()
    if args.stage=='development-b':development_b()
    else:frozen_primary()


if __name__=='__main__':main()
