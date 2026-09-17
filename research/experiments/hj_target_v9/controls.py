"""Matched cached-Hamiltonian and teacher-signal component tests."""
import argparse
import copy
import gc
import json
from pathlib import Path
import random
import torch
from experiments.hj_gridfree.engine import capture,write
from experiments.hj_cotangent.systems import problem
from experiments.hj_cotangent.study import make_data
from experiments.hj_adaptive.study import train as train_baseline
from experiments.hj_adaptive.transport import collect
from experiments.hj_proximal.study import train_proximal,sha,utc
from experiments.hj_proximal.targets import RegressionGraph,coefficients,reconstruct,teaching_data,predicted_reduction
from .study import folder


class CachedHamiltonianGraph:
    def __init__(self,actor,p,batch,lr,kind):
        self.actor,self.p=actor,p
        nfeatures=15 if p.family=='building' else (19 if p.mechanical else 13)
        self.features=torch.zeros(batch,p.n,nfeatures,device='cuda')
        self.base=torch.zeros(batch,p.n,device='cuda')
        # c/a is the coefficient of u in the normalized proximal quadratic.
        self.target=torch.zeros_like(self.base)
        self.loss=torch.zeros((),device='cuda');self.grad_norm=torch.zeros((),device='cuda',dtype=torch.float64)
        self.nonfinite_updates=torch.zeros((),device='cuda');self.clipped_updates=torch.zeros((),device='cuda')
        params=list(actor.net.parameters());initial=copy.deepcopy(actor.state_dict())
        self.m=[torch.zeros_like(z) for z in params];self.v=[torch.zeros_like(z) for z in params]
        self.b1=torch.ones((),device='cuda');self.b2=torch.ones_like(self.b1)
        for z in params:z.grad=torch.zeros_like(z)
        def update():
            for z in params:z.grad.zero_()
            u=reconstruct(actor.net,self.features,self.base,p)
            loss=(u.square()+self.target*u).mean()/p.bound**2
            loss.backward()
            with torch.no_grad():
                self.loss.copy_(loss)
                norm=torch.stack([z.grad.double().square().sum() for z in params]).sum().sqrt()
                finite=torch.isfinite(norm)&torch.isfinite(loss);self.grad_norm.copy_(norm)
                self.nonfinite_updates.add_((~finite).float());self.clipped_updates.add_((finite&(norm>10)).float())
                beta1=torch.where(finite,.9,1.);beta2=torch.where(finite,.999,1.)
                self.b1.mul_(beta1);self.b2.mul_(beta2)
                scale=(10/norm.clamp_min(1e-12)).clamp_max(1)
                for z,m,v in zip(params,self.m,self.v):
                    g=torch.where(finite,z.grad*scale,torch.zeros_like(z))
                    m.mul_(beta1).add_(g,alpha=.1);v.mul_(beta2).addcmul_(g,g,value=.001)
                    delta=-lr*(m/(1-self.b1).clamp_min(1e-12))/((v/(1-self.b2).clamp_min(1e-12)).sqrt()+1e-8)
                    z.add_(torch.where(finite,delta,torch.zeros_like(z)))
        self.graph=capture(update);actor.load_state_dict(initial);self.reset_optimizer()
        self.nonfinite_updates.zero_();self.clipped_updates.zero_()

    def reset_optimizer(self):
        for z in self.m+self.v:z.zero_()
        self.b1.fill_(1);self.b2.fill_(1)

    def update(self,features,base,target):
        self.features.copy_(features);self.base.copy_(base);self.target.copy_(target);self.graph.replay()


def make_control_trainer(actor,p,batch,lr,kind):
    return CachedHamiltonianGraph(actor,p,batch,lr,kind) if kind=='hamiltonian' else RegressionGraph(actor,p,batch,lr,'action')


@torch.no_grad()
def control_teaching_data(actor,cache,p,regularization,kind):
    data,metrics=teaching_data(actor,cache,p,regularization,'action')
    if kind=='hamiltonian':
        r,b=coefficients(p,cache[1],cache[4]);rho=regularization*2*r
        data=(data[0],data[1],(b-rho*cache[2])/(r+rho/2))
    return data,metrics


def collect_control(jet,initial,p,batch,zero_covector=False):
    costs,cache=collect(jet,initial,p,batch)
    if zero_covector:cache=(*cache[:4],torch.zeros_like(cache[4]))
    return costs,cache


def prepare(root):
    old=json.loads(Path('experiments/results/hj_proximal_dev_v8a/config.json').read_text())
    cfg=copy.deepcopy(old);cfg.update(stage='controls',seeds=[9601,9602],validation_seed_base=24400000,
        test_seed_base=27400000,role='Descriptive component controls, no candidate retuning')
    for task in cfg['tasks']:
        old_methods={s['name']:s for s in task['methods']}
        candidate=copy.deepcopy(old_methods['prox_action']);candidate['name']='hjb_target'
        ham=dict(candidate,name='cached_hamiltonian',kind='hamiltonian')
        fixed=dict(candidate,name='fixed_rho',fixed_regularization=True)
        zero=dict(candidate,name='no_value_signal',zero_covector=True)
        task.update(dimension=32,nodes=16 if task['family']=='mechanical' else 32,batch=128,
            methods=[old_methods['parent'],candidate,old_methods['warm'],ham,fixed,zero])
    root.mkdir(parents=True,exist_ok=True);assert not (root/'config.json').exists();write(root/'config.json',cfg)
    paths=[root/'config.json',Path('docs/hj_v9_protocol.md'),Path('docs/hj_v9_controls_plan.md')]
    for directory in ['hj_target_v9','hj_proximal','hj_adaptive','hj_cotangent','hj_gridfree']:
        paths.extend(Path('experiments',directory).glob('*.py'))
    write(root/'training_lock.json',dict(created_utc=utc(),hashes={str(p):sha(p) for p in paths},role=cfg['role']))


def run(root):
    from .control_training import train_control
    from .study import verify
    cfg=verify(root)
    assert json.loads(Path('build/hj_v9_controls_audit.json').read_text())['passed']
    for ti,task in enumerate(cfg['tasks']):
        p=problem(task['family'],task['nodes'],task['horizon'])
        for seed in cfg['seeds']:
            out=folder(root,task,seed);out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            data=make_data(p,cfg,seed,out);children=task['methods'][1:].copy();random.Random(seed+ti*10000).shuffle(children)
            for spec in [task['methods'][0]]+children:
                if (out/(spec['name']+'.json')).exists():continue
                write(root/'status.json',dict(stage='training',family=p.family,seed=seed,method=spec['name'],updated_utc=utc()))
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                if spec['mode']!='proximal':train_baseline(p,cfg,seed,spec,data,out)
                elif spec['name']=='hjb_target':train_proximal(p,cfg,seed,spec,data,out)
                else:train_control(p,cfg,seed,spec,data,out)
                path=out/(spec['name']+'.json');result=json.loads(path.read_text())
                result.update(study_role=cfg['role'],peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                write(path,result)
            write(out/'complete.json',dict(completed_utc=utc(),methods=[s['name'] for s in task['methods']]))
    write(root/'status.json',dict(stage='training_complete',updated_utc=utc()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/results/hj_target_controls_v9');parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.prepare:prepare(Path(args.root))
    else:run(Path(args.root))


if __name__=='__main__':main()
