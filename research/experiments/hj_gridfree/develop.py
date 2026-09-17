"""Logged development runs; validation only, not the independent test set."""
import argparse
import copy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import torch
from .core import Problem,LQR,Actor,sample,rollout,exact_q
from .engine import AdamGraph,TrainGraph,EvalGraph,cem,write,hashes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--family',default='mechanical'); ap.add_argument('--n',type=int,default=8)
    ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--seed',type=int,default=801)
    ap.add_argument('--dataset',type=int,default=4096); ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--label-steps',type=int,default=80); ap.add_argument('--tag',default='v1')
    ap.add_argument('--modes',nargs='+',default=['bellman','dpc','mse'])
    args=ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(args.seed)
    p=Problem(args.family,args.n); root=Path(f'experiments/results/hj_gridfree_dev_{args.tag}/{p.family}_d{p.dim}_s{args.seed}')
    root.mkdir(parents=True,exist_ok=True)
    manifest=dict(args=vars(args),problem=asdict(p),source_hashes=hashes(),gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    write(root/'manifest.json',manifest); print(json.dumps({'start':str(root)}),flush=True)
    wall=time.perf_counter(); base=LQR(p)
    trainx=sample(p,512,11000+args.seed); valx=sample(p,128,21000+args.seed)
    with torch.no_grad():
        _,states,_=rollout(trainx,base,p,keep=True)
        flat=states.reshape(-1,p.dim)
        gen=torch.Generator(device='cuda').manual_seed(31000+args.seed)
        index=torch.randperm(len(flat),device='cuda',generator=gen)[:args.dataset]
        datax=flat[index]; datat=index%p.horizon
        basecost,_,baseu=rollout(valx,base,p,keep=True)
    torch.cuda.synchronize(); data_seconds=time.perf_counter()-wall
    results={'base_mean':basecost.mean().item(),'data_seconds':data_seconds,'methods':{}}
    raw={'valx':valx.cpu(),'base':basecost.cpu()}
    # Strong feasible open-loop development references. Validation states only.
    start=time.perf_counter(); solver=AdamGraph(valx,p,lr=.12*p.bound)
    torch.cuda.synchronize(); setup=time.perf_counter()-start
    for it in [32,128,512]:
        start=time.perf_counter(); c,u=solver.solve(valx,baseu,it)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        results['methods'][f'adam{it}']={'mean':c.mean().item(),'seconds':elapsed,'setup_seconds':setup}
        raw[f'adam{it}']=c.cpu()
    del solver; gc.collect(); torch.cuda.empty_cache()
    start=time.perf_counter()
    cc=[]
    for j in range(0,len(valx),16):
        c,_=cem(valx[j:j+16],baseu[j:j+16],p,samples=256,iterations=6,seed=args.seed+j)
        cc.append(c)
    c=torch.cat(cc); torch.cuda.synchronize()
    results['methods']['cem256x6']={'mean':c.mean().item(),'seconds':time.perf_counter()-start}; raw['cem256x6']=c.cpu()
    print(json.dumps({'baselines':results}),flush=True); write(root/'results.json',results)
    labels=None; label_seconds=0
    if 'mse' in args.modes:
        start=time.perf_counter(); solver=AdamGraph(datax[:args.batch],p,kind='q',lr=.08*p.bound)
        ys=[]; costs=[]; qbase=[]
        for j in range(0,args.dataset,args.batch):
            xx,tt=datax[j:j+args.batch],datat[j:j+args.batch]
            with torch.no_grad(): a=base(xx,tt); q0=exact_q(xx,a,tt,base,p)
            c,u=solver.solve(xx,a,args.label_steps,tt); ys.append(u); costs.append(c); qbase.append(q0)
        labels=torch.cat(ys); torch.cuda.synchronize(); label_seconds=time.perf_counter()-start
        results['label_seconds']=label_seconds
        results['label_mean_q_reduction']=(torch.cat(qbase)-torch.cat(costs)).mean().item()
        print(json.dumps({'labels':{k:v for k,v in results.items() if k.startswith('label')}}),flush=True)
        del solver; gc.collect(); torch.cuda.empty_cache()
    # The first mechanism screen uses a common frozen state pool. Subsequent
    # on-policy distribution refresh is an explicit separate ablation.
    for mode in args.modes:
        torch.manual_seed(args.seed); torch.cuda.reset_peak_memory_stats()
        actor=Actor(p,LQR(p)).cuda(); start=time.perf_counter()
        trainer=TrainGraph(actor,datax[:args.batch],p,mode)
        evaluator=EvalGraph(actor,p,len(valx)); torch.cuda.synchronize()
        setup=time.perf_counter()-start
        gen=torch.Generator(device='cuda').manual_seed(41000+args.seed)
        best=basecost.mean().item(); best_state=copy.deepcopy(actor.state_dict()); best_step=0
        history=[]; fit=0.; validation=0.
        for k in range(1,args.steps+1):
            if (k-1)%200==0: torch.cuda.synchronize(); start=time.perf_counter()
            idx=torch.randint(args.dataset,(args.batch,),device='cuda',generator=gen)
            trainer.update(datax[idx],datat[idx],None if labels is None else labels[idx])
            if k%200==0 or k==args.steps:
                torch.cuda.synchronize(); fit+=time.perf_counter()-start
                startv=time.perf_counter(); c=evaluator(valx); torch.cuda.synchronize()
                mean=c.mean().item(); validation+=time.perf_counter()-startv
                if mean<best:
                    best=mean; best_step=k; best_state=copy.deepcopy(actor.state_dict())
                h=dict(step=k,mean=mean,best_mean=best,fit_seconds=fit,loss=trainer.loss.item())
                history.append(h); print(json.dumps({'mode':mode,**h}),flush=True)
        actor.load_state_dict(best_state)
        with torch.no_grad(): c=rollout(valx.double(),actor,p)
        torch.cuda.synchronize()
        results['methods'][mode]=dict(mean=c.mean().item(),best_step=best_step,fit_seconds=fit,setup_seconds=setup,validation_seconds=validation,label_seconds=label_seconds if mode=='mse' else 0.,peak_gpu_bytes=torch.cuda.max_memory_allocated(),history=history)
        raw[mode]=c.cpu(); torch.save(dict(state_dict=best_state,problem=asdict(p),args=vars(args),mode=mode),root/f'{mode}.pt')
        write(root/'results.json',results); torch.save(raw,root/'validation.pt')
        del trainer,evaluator,actor,best_state; gc.collect(); torch.cuda.empty_cache()
    results['wall_seconds']=time.perf_counter()-wall
    write(root/'results.json',results); torch.save(raw,root/'validation.pt')
    print(json.dumps({'finished':str(root),'wall_seconds':results['wall_seconds']}),flush=True)


if __name__=='__main__': main()
