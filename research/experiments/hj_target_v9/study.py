"""Frozen v8 candidate; new seeds, budgets and direct-training dimensions."""
import argparse
import copy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem
from experiments.hj_cotangent.study import make_data
from experiments.hj_adaptive.study import train as train_baseline
from experiments.hj_proximal.study import train_proximal,sha,utc


def folder(root,task,seed):
    return root/'training'/f'{task["family"]}_d{task["dimension"]}_h{task["horizon"]}_s{seed}'


def prepare(root,stage):
    old=json.loads(Path('experiments/results/hj_proximal_dev_v8a/config.json').read_text())
    cfg={k:copy.deepcopy(old[k]) for k in ['dataset','training_initial_states','validation_states','batch','width','length','validation_interval_seconds','update_block','training_nodes']}
    cfg.update(stage=stage,development=stage!='primary',parent_budget_seconds=7.5,
        seeds=list(range(9501,9511)) if stage=='primary' else [9701,9702,9703],
        validation_seed_base=22400000 if stage=='primary' else 23400000,
        test_seed_base=25400000 if stage=='primary' else 26400000,
        budgets_seconds=[7.5,10.,15.,20.,25.,30.] if stage=='primary' else [7.5,15.,30.,60.],
        role='Frozen independent-training replication' if stage=='primary' else 'Exploratory direct-dimensional and horizon scaling')
    tasks=[]
    sources=old['tasks'] if stage=='primary' else [copy.deepcopy(old['tasks'][1]),copy.deepcopy(old['tasks'][1])]
    for index,source in enumerate(sources):
        methods=[]
        for name in ['parent','short','dpc','warm','prox_action']:
            spec=copy.deepcopy(next(s for s in source['methods'] if s['name']==name))
            if name=='prox_action':spec['name']='hjb_target'
            methods.append(spec)
        dimension=32 if stage=='primary' or index==1 else 256
        horizon=source['horizon'] if stage=='primary' or index==0 else 640
        tasks.append(dict(family=source['family'],horizon=horizon,dimension=dimension,
            nodes=dimension//2 if source['family']=='mechanical' else dimension,
            batch=32 if dimension==256 else 128,methods=methods))
    cfg['tasks']=tasks
    root.mkdir(parents=True,exist_ok=True);assert not (root/'config.json').exists()
    write(root/'config.json',cfg)
    paths=[root/'config.json',Path('docs/hj_v9_protocol.md')]
    for directory in ['hj_target_v9','hj_proximal','hj_adaptive','hj_cotangent','hj_gridfree']:
        paths.extend(Path('experiments',directory).glob('*.py'))
    write(root/'training_lock.json',dict(created_utc=utc(),role=cfg['role'],hashes={str(p):sha(p) for p in paths}))
    write(root/'status.json',dict(stage='prepared',updated_utc=utc()))


def verify(root):
    cfg=json.loads((root/'config.json').read_text())
    lock=json.loads((root/'training_lock.json').read_text())
    for path,expected in lock['hashes'].items():assert sha(path)==expected,path
    return cfg


def run(root):
    cfg=verify(root)
    assert torch.cuda.is_available()
    write(root/'environment.json',dict(started_utc=utc(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),concurrent_timed_gpu_jobs=1))
    for ti,task in enumerate(cfg['tasks']):
        local=copy.deepcopy(cfg);local['batch']=task['batch']
        p=problem(task['family'],task['nodes'],task['horizon'])
        for seed in cfg['seeds']:
            out=folder(root,task,seed);out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            data=make_data(p,local,seed,out)
            children=task['methods'][1:].copy();random.Random(seed+10000*ti).shuffle(children)
            for spec in [task['methods'][0]]+children:
                target=out/spec['name']
                if target.with_suffix('.json').exists():continue
                write(root/'status.json',dict(stage='training',family=p.family,dimension=p.dim,horizon=p.horizon,
                    seed=seed,method=spec['name'],updated_utc=utc()))
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                if spec['mode']=='proximal':train_proximal(p,local,seed,spec,data,out)
                else:train_baseline(p,local,seed,spec,data,out)
                result=json.loads(target.with_suffix('.json').read_text())
                result.update(study_role=cfg['role'],study_stage=cfg['stage'],
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                    numerical_kernel='Unchanged frozen v8 target kernel' if spec['mode']=='proximal' else 'Unchanged archived v7 baseline kernel')
                write(target.with_suffix('.json'),result)
            write(out/'complete.json',dict(completed_utc=utc(),methods=[s['name'] for s in task['methods']],problem=asdict(p)))
            print(json.dumps(dict(completed_task=p.family,dimension=p.dim,horizon=p.horizon,seed=seed)),flush=True)
            gc.collect();torch.cuda.empty_cache()
    write(root/'status.json',dict(stage='training_complete',updated_utc=utc()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/results/hj_target_primary_v9')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--stage',choices=['primary','scaling'],default='primary')
    args=parser.parse_args();torch.set_num_threads(1);root=Path(args.root)
    if args.prepare:prepare(root,args.stage)
    else:run(root)


if __name__=='__main__':main()
