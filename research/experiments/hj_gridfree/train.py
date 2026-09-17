"""Primary training runner. Config and source lock required before execution."""
import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from .core import Problem,LQR,Actor,sample,rollout,step,exact_q
from .engine import AdamGraph,TrainGraph,EvalGraph,write


def verify_lock(root):
    lock=json.loads((root/'lock.json').read_text(encoding='utf-8'))
    for name,digest in lock['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((root/'config.json').read_bytes()).hexdigest()==lock['config_sha256']
    return json.loads((root/'config.json').read_text(encoding='utf-8'))


def train_one(root,cfg,family,seed):
    p=Problem(family,cfg['training_nodes'][family]); out=root/'training'/f'{family}_s{seed}'
    if (out/'complete.json').exists(): return
    out.mkdir(parents=True,exist_ok=True); wall=time.perf_counter(); torch.manual_seed(seed)
    base=LQR(p); trainx=sample(p,cfg['training_initial_states'],100000+seed)
    valx=sample(p,cfg['validation_states'],200000+(0 if p.mechanical else 10000))
    gen=torch.Generator(device='cuda').manual_seed(300000+seed)
    start=time.perf_counter()
    with torch.no_grad():
        _,xs,ub=rollout(trainx,base,p,keep=True); flat=xs.flatten(0,1)
        index=torch.randperm(len(flat),device='cuda',generator=gen)[:cfg['dataset']]
        datax=flat[index]; datat=index%p.horizon
        jbase=rollout(valx.double(),base,p)
    torch.cuda.synchronize(); data_seconds=time.perf_counter()-start
    torch.save(dict(x=datax.cpu(),t=datat.cpu(),valx=valx.cpu()),out/'data.pt')
    batch=cfg['batch']; labels=None; bcdata=None; prep={}
    if 'mse' in cfg['modes']:
        start=time.perf_counter(); solver=AdamGraph(datax[:batch],p,kind='q',lr=.08*p.bound); ys=[]
        for j in range(0,len(datax),batch):
            xx,tt=datax[j:j+batch],datat[j:j+batch]
            with torch.no_grad(): a=base(xx,tt)
            _,u=solver.solve(xx,a,cfg['q_label_steps'],tt); ys.append(u)
        labels=torch.cat(ys); torch.cuda.synchronize(); prep['mse']=time.perf_counter()-start
        torch.save(labels.cpu(),out/'q_labels.pt'); del solver; gc.collect(); torch.cuda.empty_cache()
    if 'bc' in cfg['modes']:
        start=time.perf_counter(); solver=AdamGraph(trainx[:batch],p,lr=.12*p.bound); allstates=[]; allu=[]
        for j in range(0,len(trainx),batch):
            xx=trainx[j:j+batch]; init=ub[j:j+batch]
            c,u=solver.solve(xx,init,cfg['bc_teacher_steps'])
            cz,uz=solver.solve(xx,torch.zeros_like(init),cfg['bc_teacher_steps'])
            u=torch.where((cz<c)[:,None,None],uz,u)
            with torch.no_grad():
                sx=[]; x=xx
                for k in range(p.horizon): sx.append(x); x=step(x,u[:,k],p)
            allstates.append(torch.stack(sx,1)); allu.append(u)
        bx=torch.cat(allstates).flatten(0,1); by=torch.cat(allu).flatten(0,1)
        bt=torch.arange(p.horizon,device='cuda').repeat(len(trainx))
        bcdata=(bx,bt,by); torch.cuda.synchronize(); prep['bc']=time.perf_counter()-start
        torch.save(dict(x=bx.cpu(),t=bt.cpu(),u=by.cpu()),out/'bc_labels.pt')
        del solver; gc.collect(); torch.cuda.empty_cache()
    result=dict(problem=asdict(p),seed=seed,data_seconds=data_seconds,base_mean=jbase.mean().item(),preparation=prep,methods={})
    print(json.dumps(dict(training=out.name,preparation=prep)),flush=True)
    for mode in cfg['modes']:
        torch.manual_seed(seed); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        actor=Actor(p,LQR(p),width=cfg['width']).cuda()
        trainer=TrainGraph(actor,datax[:batch],p,'mse' if mode=='bc' else mode,lr=cfg['lr'])
        evaluator=EvalGraph(actor,p,len(valx)); torch.cuda.synchronize(); setup=time.perf_counter()-start
        gen=torch.Generator(device='cuda').manual_seed(400000+seed)
        xx,tt,yy=bcdata if mode=='bc' else (datax,datat,labels)
        best=jbase.mean().item(); state=copy.deepcopy(actor.state_dict()); best_step=0
        fit=0.; validate=0.; history=[]
        steps=cfg['imitation_steps'] if mode in ('mse','bc') else cfg['steps']
        every=cfg['validation_every']; start=time.perf_counter()
        for k in range(1,steps+1):
            idx=torch.randint(len(xx),(batch,),device='cuda',generator=gen)
            trainer.update(xx[idx],tt[idx],yy[idx] if mode in ('mse','bc') else None)
            if k%every==0 or k==steps:
                torch.cuda.synchronize(); fit+=time.perf_counter()-start; startv=time.perf_counter()
                cost=evaluator(valx); torch.cuda.synchronize(); mean=cost.mean().item()
                assert torch.isfinite(cost).all().item(),(family,seed,mode,k)
                if mean<best:
                    best=mean; best_step=k; state=copy.deepcopy(actor.state_dict())
                validate+=time.perf_counter()-startv
                history.append(dict(step=k,mean=mean,best=best,fit_seconds=fit))
                if k%800==0 or k==steps: print(json.dumps(dict(family=family,seed=seed,mode=mode,**history[-1])),flush=True)
                start=time.perf_counter()
        actor.load_state_dict(state)
        with torch.no_grad(): final=rollout(valx.double(),actor,p)
        torch.cuda.synchronize()
        ckpt=dict(state_dict=state,problem=asdict(p),mode=mode,seed=seed,best_step=best_step)
        torch.save(ckpt,out/f'{mode}.pt')
        torch.save(final.cpu(),out/f'{mode}_validation.pt')
        result['methods'][mode]=dict(mean=final.mean().item(),best_step=best_step,fit_seconds=fit,setup_seconds=setup,validation_seconds=validate,preparation_seconds=prep.get(mode,0)+data_seconds,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),history=history,
            training_simulated_transitions=0 if mode in ('mse','bc') else steps*batch*(int(mode[9:]) if mode.startswith('truncated') else p.horizon),
            training_neural_evaluations=steps*batch*(1 if mode in ('mse','bc','bellman') else int(mode[5:]) if mode.startswith('block') else int(mode[9:]) if mode.startswith('truncated') else p.horizon))
        write(out/'results.json',result)
        del actor,trainer,evaluator,state; gc.collect(); torch.cuda.empty_cache()
    result['wall_seconds']=time.perf_counter()-wall; write(out/'results.json',result)
    write(out/'complete.json',dict(wall_seconds=result['wall_seconds']))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='experiments/results/hj_gridfree_primary_v1')
    args=ap.parse_args(); torch.set_num_threads(1); root=Path(args.root); cfg=verify_lock(root)
    for family in cfg['families']:
        for seed in cfg['seeds']: train_one(root,cfg,family,seed)
    print('TRAINING COMPLETE',flush=True)


if __name__=='__main__': main()
