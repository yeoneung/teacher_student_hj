"""Independent high-dimensional transfer and strong grid-free planning comparisons."""
import argparse
from dataclasses import asdict
import datetime
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import Problem,LQR,Actor,sample,rollout,sequence_cost
from .engine import AdamGraph,cem,write
from .planning_probe import pd_policy
from .train import verify_lock
from .gate import gated_rollout


def independent_cost(x,u,p):
    """Independent NumPy float64 replay; no BLAS or shared model functions."""
    z=x.detach().cpu().numpy().astype(np.float64).copy()
    a=u.detach().cpu().numpy().astype(np.float64); j=np.zeros(len(z))
    for k in range(p.horizon):
        vinput=a[:,k]
        if p.mechanical:
            q=z[:,:p.n]; v=z[:,p.n:]
            j+=p.dt*np.mean(2*(1-np.cos(q))+.08*v*v+.04*vinput*vinput+.2*(1-np.cos(q-np.roll(q,-1,axis=1))),axis=1)
            vn=v+p.dt*(5*np.sin(q)+p.gamma*(np.sin(np.roll(q,1,axis=1)-q)+np.sin(np.roll(q,-1,axis=1)-q))+vinput-.2*v)
            z=np.concatenate((q+p.dt*vn,vn),axis=1)
        else:
            j+=p.dt*np.mean(z*z+.1*vinput*vinput+.1*(z-np.roll(z,-1,axis=1))**2,axis=1)
            z=z+p.dt*(p.gamma*(np.roll(z,1,axis=1)+np.roll(z,-1,axis=1)-2*z)+1.2*z-.4*z**3+vinput)
    if p.mechanical: j+=np.mean(8*(1-np.cos(z[:,:p.n]))+.4*z[:,p.n:]**2,axis=1)
    else: j+=3*np.mean(z*z,axis=1)
    return torch.from_numpy(j)


def actor_from(root,p,mode,seed,width):
    state=torch.load(root/'training'/f'{p.family}_s{seed}'/f'{mode}.pt',map_location='cuda',weights_only=False)['state_dict']
    actor=Actor(p,LQR(p),width=width).cuda()
    actor.net.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('net.')})
    actor.eval(); return actor


def audit_controls(x,u,j,p):
    ji=independent_cost(x,u,p)
    error=((j.cpu().double()-ji).abs()/(1+ji.abs())).max().item()
    budget=u.double().square().mean(-1).sqrt().max().item()/p.bound
    assert torch.isfinite(ji).all().item() and torch.isfinite(u).all().item()
    assert error<.003,(p,error)
    assert budget<1.000003,(p,budget)
    return ji,dict(independent_scaled_cost_error=error,max_budget_ratio=budget)


def solve_path(solver,x,init,steps):
    """One run with retained incumbents at prespecified iteration budgets."""
    start=time.perf_counter(); solver.solve(x,init,0)
    output={}
    last=0
    for target in steps:
        for _ in range(target-last): solver.graph.replay()
        with torch.no_grad(): solver.retain(solver.cost())
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        output[target]=(solver.best.clone(),solver.best_u.clone(),elapsed)
        last=target
    return output


def calibrate_pd(root,cfg):
    path=root/'pd_calibration.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    result={}
    for family in cfg['families']:
        p=Problem(family,cfg['training_nodes'][family]); x=sample(p,512,550801+(0 if p.mechanical else 10000))
        settings=itertools.product([0.,1.,3.,6.,12.],[.5,2.,5.],[-5.,0.,5.]) if p.mechanical else itertools.product([.5,1.,2.,4.,8.],[0.],[-.4,0.,.4])
        start=time.perf_counter(); records=[]
        with torch.no_grad():
            for triple in settings:
                cost=rollout(x,pd_policy(p,*triple),p).mean().item()
                records.append(dict(parameters=triple,mean=cost))
        best=min(records,key=lambda z:z['mean']); torch.cuda.synchronize()
        result[family]=dict(selected=best['parameters'],calibration_seconds=time.perf_counter()-start,records=records)
    write(path,result); return result


def test_case(root,cfg,p,distribution,calibration):
    key=f'{p.family}_d{p.dim}_{distribution}'; out=root/'evaluation'; out.mkdir(parents=True,exist_ok=True)
    if (out/f'{key}.json').exists(): return
    shift=distribution=='shift'; count=cfg['test_shift_states'] if shift else cfg['test_nominal_states']
    seed=cfg['test_seed_base']+(0 if p.mechanical else 10000)+p.dim*10+int(shift)
    x=sample(p,count,seed,shift=shift); base=LQR(p)
    record=dict(problem=asdict(p),distribution=distribution,count=count,seed=seed,methods={},neural={})
    raw=dict(x=x.cpu(),classical={},neural={},polished={}); maxerr=0.; maxbudget=0.
    with torch.no_grad():
        jb,_,ub=rollout(x.double(),base,p,keep=True)
        jp,_,up=rollout(x.double(),pd_policy(p,*calibration[p.family]['selected']),p,keep=True)
    for name,j,u in [('lqr',jb,ub),('tuned_pd',jp,up)]:
        ji,audit=audit_controls(x,u,j,p); raw['classical'][name]=ji
        record['methods'][name]=dict(mean=ji.mean().item(),**audit)
        maxerr=max(maxerr,audit['independent_scaled_cost_error']); maxbudget=max(maxbudget,audit['max_budget_ratio'])
    starts={'lqr':ub.float(),'zero':torch.zeros_like(ub).float(),'half_lqr':ub.float()*.5,'tuned_pd':up.float()}
    torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter(); solver=AdamGraph(x,p,lr=.12*p.bound)
    torch.cuda.synchronize(); setup=time.perf_counter()-begin
    best_by_steps={}; best_u=None; best_cost=None
    for name in cfg['planning_initializations']:
        path=solve_path(solver,x,starts[name],cfg['planning_steps'])
        for it,(j,u,seconds) in path.items():
            ji,audit=audit_controls(x,u,j,p); label=f'adam_{name}_{it}'
            raw['classical'][label]=ji; record['methods'][label]=dict(mean=ji.mean().item(),seconds=seconds,setup_seconds=setup,**audit)
            maxerr=max(maxerr,audit['independent_scaled_cost_error']); maxbudget=max(maxbudget,audit['max_budget_ratio'])
            best_by_steps[it]=ji if it not in best_by_steps else torch.minimum(best_by_steps[it],ji)
            if it==max(cfg['planning_steps']):
                if best_cost is None: best_cost=ji; best_u=u.clone()
                else:
                    win=ji<best_cost; best_cost=torch.minimum(best_cost,ji); best_u=torch.where(win.cuda()[:,None,None],u,best_u)
        del path
    for it,j in best_by_steps.items():
        name=f'adam_multistart_{it}'; raw['classical'][name]=j
        record['methods'][name]=dict(mean=j.mean().item(),seconds=sum(record['methods'][f'adam_{s}_{it}']['seconds'] for s in starts),setup_seconds=setup)
    record['adam_peak_gpu_bytes']=torch.cuda.max_memory_allocated()
    # Strong sampling initialization, followed by the same direct optimizer.
    jc=[]; uc=[]; start=time.perf_counter()
    for i in range(0,count,8):
        j,u=cem(x[i:i+8],ub[i:i+8].float(),p,samples=cfg['cem_samples'],iterations=cfg['cem_iterations'],seed=seed+i)
        jc.append(j); uc.append(u)
    j=torch.cat(jc); u=torch.cat(uc); torch.cuda.synchronize(); ctime=time.perf_counter()-start
    ji,audit=audit_controls(x,u,j,p); raw['classical']['cem']=ji
    record['methods']['cem']=dict(mean=ji.mean().item(),seconds=ctime,**audit)
    start=time.perf_counter(); j,u=solver.solve(x,u,512); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    ji,audit=audit_controls(x,u,j,p); raw['classical']['cem_adam512']=ji
    record['methods']['cem_adam512']=dict(mean=ji.mean().item(),seconds=ctime+elapsed,**audit)
    win=ji<best_cost; best_cost=torch.minimum(best_cost,ji); best_u=torch.where(win.cuda()[:,None,None],u,best_u)
    raw['best_classical_controls']=best_u.cpu(); raw['classical']['best_classical']=best_cost
    record['methods']['best_classical']=dict(mean=best_cost.mean().item(),meaning='casewise minimum of four-start Adam1024 and CEM+Adam512; includes all search cost')
    # Each model receives its own full float64 closed-loop deployment, with FP32 neural inference.
    for mode in cfg['modes']:
        rows=[]; audits=[]
        for trainseed in cfg['seeds']:
            actor=actor_from(root,p,mode,trainseed,cfg['width'])
            with torch.no_grad(): j,_,u=rollout(x.double(),actor,p,keep=True)
            ji,audit=audit_controls(x,u,j,p); rows.append(ji); audits.append(audit)
            maxerr=max(maxerr,audit['independent_scaled_cost_error']); maxbudget=max(maxbudget,audit['max_budget_ratio'])
            # A learned warm start is reported as a hybrid, not a purely classical competitor.
            if trainseed==cfg['seeds'][0] and mode in ('block16','dpc'):
                start=time.perf_counter(); jr,ur=solver.solve(x,u.float(),cfg['neural_polish_steps']); torch.cuda.synchronize()
                elapsed=time.perf_counter()-start; jr,ar=audit_controls(x,ur,jr,p)
                raw['polished'][mode]=jr
                record['methods'][f'{mode}_adam{cfg["neural_polish_steps"]}']=dict(mean=jr.mean().item(),polishing_seconds=elapsed,**ar)
            if trainseed==cfg['seeds'][0] and mode=='block16':
                with torch.no_grad(): jg,ug,choices,advantages=gated_rollout(x.double(),actor,base,p)
                jg,ag=audit_controls(x,ug,jg,p)
                assert (jg-raw['classical']['lqr']).max().item()<1e-8
                raw['gate']=dict(cost=jg,choices=choices.cpu(),candidate_advantages=advantages.cpu())
                record['methods']['block16_gate']=dict(mean=jg.mean().item(),student_block_fraction=choices.double().mean().item(),max_cost_increase_vs_base=(jg-raw['classical']['lqr']).max().item(),**ag)
            del actor; gc.collect()
        values=torch.stack(rows); raw['neural'][mode]=values
        record['neural'][mode]=dict(mean=values.mean().item(),seed_means=values.mean(1).tolist(),audits=audits)
    record['audit']=dict(max_independent_scaled_cost_error=maxerr,max_budget_ratio=maxbudget)
    torch.save(raw,out/f'{key}.pt'); write(out/f'{key}.json',record)
    print(json.dumps(dict(case=key,classical=best_cost.mean().item(),neural={m:z['mean'] for m,z in record['neural'].items()},audit=record['audit'])),flush=True)
    del solver; gc.collect(); torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='experiments/results/hj_gridfree_primary_v1')
    args=ap.parse_args(); torch.set_num_threads(1); root=Path(args.root); cfg=verify_lock(root)
    assert all((root/'training'/f'{f}_s{s}'/'complete.json').exists() for f in cfg['families'] for s in cfg['seeds'])
    lockpath=root/'evaluation_lock.json'
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checkpoints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'training').glob('*/*.pt')) if p.stem in cfg['modes']}
    additional={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('experiments/hj_gridfree/native.py'),Path('experiments/hj_gridfree/native_evaluate.py'),Path('experiments/hj_gridfree/gate.py'),Path('experiments/hj_gridfree/timing.py')]}
    if not lockpath.exists(): write(lockpath,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),evaluation_source_sha256=source,additional_source_hashes=additional,checkpoints=checkpoints,protocol='Independent test states generated only after this lock. GPU classical methods evaluated on all test states. Native CPU L-BFGS uses the first128 nominal / first64 shifted states per case, three starts (LQR, zero, best full GPU classical plan),1024 iterations per start. Native subset indices and budgets fixed before test access.'))
    else:
        lock=json.loads(lockpath.read_text(encoding='utf-8')); assert source==lock['evaluation_source_sha256']; assert checkpoints==lock['checkpoints']; assert additional==lock['additional_source_hashes']
    calibration=calibrate_pd(root,cfg)
    for family in cfg['families']:
        for dim in cfg['test_dimensions']:
            n=dim//2 if family=='mechanical' else dim
            p=Problem(family,n); test_case(root,cfg,p,'nominal',calibration)
            shifted=Problem(family,n,coupling=p.gamma*cfg['test_shift_coupling_multiplier'],budget=p.bound*cfg['test_shift_budget_multiplier'])
            test_case(root,cfg,shifted,'shift',calibration)
    print('INDEPENDENT EVALUATION COMPLETE',flush=True)


if __name__=='__main__': main()
