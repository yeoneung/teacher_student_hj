"""Curvature-specific data interface and isolated additional-development runner."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem
from .study import sha,utc
from .targets import expose,reconstruct
from .curvature import weighted_target


def collect_curvature(graph,initial,p,batch):
    if not hasattr(graph,'generator'):
        graph.generator=torch.Generator(device='cuda').manual_seed(610000+torch.initial_seed())
    costs=[];columns=[[],[],[],[],[]]
    for start in range(0,len(initial),batch):
        cost,x,u,g,h=graph(initial[start:start+batch],graph.generator)
        costs.append(cost)
        values=[x.flatten(0,1),torch.arange(p.horizon,device='cuda').repeat(batch),u.flatten(0,1),g.flatten(0,1),h.flatten(0,1)]
        for dest,value in zip(columns,values):dest.append(value)
    return torch.cat(costs),tuple(torch.cat(column).contiguous() for column in columns)


@torch.no_grad()
def curvature_teaching_data(actor,cache,p,regularization,kind,chunk=2048):
    assert kind=='action'
    x,t,old,g,diagonal=cache;features=[];bases=[]
    for start in range(0,len(x),chunk):
        f,b,_=expose(actor,x[start:start+chunk],t[start:start+chunk]);features.append(f);bases.append(b)
    target,h=weighted_target(old,g,diagonal,p,regularization)
    delta=target-old;predicted=-(g*delta+.5*h*delta.square()).sum(-1)
    return (torch.cat(features),torch.cat(bases),target),dict(target_clipped_fraction=0.,
        target_model_reduction=float(p.horizon*predicted.mean()),
        target_action_drift=float((delta/p.bound).square().mean().sqrt()),
        diagonal_negative_fraction=float((diagonal<0).float().mean()),
        positive_curvature_mean=float(h.mean()),raw_curvature_mean=float(diagonal.mean()))


@torch.no_grad()
def curvature_predicted_reduction(actor,data,cache,p,chunk=2048):
    x,t,old,g,diagonal=cache
    r=p.dt*({'mechanical':.04,'reaction':.1,'building':.025}[p.family])/p.n
    total=torch.zeros((),device='cuda',dtype=torch.float64)
    for start in range(0,len(x),chunk):
        end=start+chunk;u=reconstruct(actor.net,data[0][start:end],data[1][start:end],p)
        delta=u-old[start:end];h=diagonal[start:end].clamp_min(2*r)
        total.add_(-(g[start:end]*delta+.5*h*delta.square()).double().sum())
    return float(p.horizon*total/len(x))


def prepare(root):
    previous=Path('experiments/results/hj_proximal_dev_v8a')
    cfg=json.loads((previous/'config.json').read_text())
    cfg['tasks']=[t for t in cfg['tasks'] if t['family']!='building']
    cfg['scope']='Additional two-seed development, four new curvature phases and explicitly reused v8a baseline artifacts.'
    root.mkdir(parents=True,exist_ok=True);assert not (root/'config.json').exists()
    copies={}
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            folder=f'{task["family"]}_h{task["horizon"]}_s{seed}'
            source=previous/'training'/folder;dest=root/'training'/folder;dest.mkdir(parents=True,exist_ok=True)
            assert (source/'complete.json').exists()
            files=['data.pt']+[s['name']+extension for s in task['methods'] for extension in ['.pt','.json']]
            for name in files:
                original=source/name;target=dest/name;assert not target.exists()
                shutil.copyfile(original,target);copies[str(target)]=dict(source=str(original),sha256=sha(original))
        task['methods'].append(dict(name='curvature_action',mode='proximal',kind='action',parent='parent',lr=.002,
            student_batch=1024,cycle_updates=256,regularization=16.,curvature_probes=4))
    write(root/'config.json',cfg)
    sources=[]
    for folder in ['hj_proximal','hj_adaptive','hj_cotangent','hj_gridfree']:sources.extend(Path('experiments',folder).glob('*.py'))
    sources += [root/'config.json',Path('docs/hj_v8_curvature_plan.md')]
    write(root/'development_lock.json',dict(created_utc=utc(),hashes={str(p):sha(p) for p in sources},copied_artifacts=copies))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/results/hj_curvature_dev_v8b');parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();root=Path(args.root);torch.set_num_threads(1)
    if args.prepare:prepare(root);return
    from .curvature_study import train_curvature
    cfg=json.loads((root/'config.json').read_text());lock=json.loads((root/'development_lock.json').read_text())
    for path,expected in lock['hashes'].items():assert sha(path)==expected,path
    for path,row in lock['copied_artifacts'].items():assert sha(path)==row['sha256'],path
    assert json.loads(Path('build/hj_v8_curvature_preflight.json').read_text())['passed']
    for task in cfg['tasks']:
        p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
        for seed in cfg['seeds']:
            out=root/'training'/f'{p.family}_h{p.horizon}_s{seed}'
            data=torch.load(out/'data.pt',map_location='cuda',weights_only=False)
            write(root/'status.json',dict(stage='training',family=p.family,seed=seed,updated_utc=utc()))
            train_curvature(p,cfg,seed,task['methods'][-1],data,out)
            write(out/'complete.json',dict(completed_utc=utc(),new_methods=['curvature_action'],copied_methods=[s['name'] for s in task['methods'][:-1]]))
    write(root/'status.json',dict(stage='additional_training_complete',updated_utc=utc()))


if __name__=='__main__':main()
