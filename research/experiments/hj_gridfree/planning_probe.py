"""Development-only initialization and analytical-feedback baseline audit."""
import argparse
import gc
import itertools
import json
from pathlib import Path
import time
import torch
from .core import Problem,LQR,sample,project,rollout
from .engine import AdamGraph,write,hashes


def pd_policy(p,kq,kv,kg):
    def policy(x,t):
        if p.mechanical:
            q,v=x[...,:p.n],x[...,p.n:]
            return project(-kq*torch.atan2(q.sin(),q.cos())-kv*v-kg*q.sin(),p)
        return project(-kq*x-kg*x.pow(3),p)
    return policy


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--family',default='mechanical'); ap.add_argument('--n',type=int,default=16)
    args=ap.parse_args(); torch.set_num_threads(1); p=Problem(args.family,args.n)
    base=LQR(p); train=sample(p,256,550801); val=sample(p,128,21801)
    start=time.perf_counter(); records=[]; best=float('inf'); chosen=None
    settings=itertools.product([0.,1.,3.,6.,12.],[.5,2.,5.],[-5.,0.,5.]) if p.mechanical else itertools.product([.5,1.,2.,4.,8.],[0.],[-.4,0.,.4])
    with torch.no_grad():
        for kq,kv,kg in settings:
            policy=pd_policy(p,kq,kv,kg); cost=rollout(train,policy,p).mean().item()
            records.append(dict(kq=kq,kv=kv,kg=kg,train_cost=cost))
            if cost<best: best=cost; chosen=(kq,kv,kg)
        jb,_,ub=rollout(val,base,p,keep=True)
        jp,_,up=rollout(val,pd_policy(p,*chosen),p,keep=True)
    torch.cuda.synchronize(); tune_seconds=time.perf_counter()-start
    result=dict(problem=p.__dict__,source_hashes=hashes(),tuned_pd=chosen,tune_seconds=tune_seconds,lqr=jb.mean().item(),pd=jp.mean().item(),records=records,planning={})
    solver=AdamGraph(val,p,lr=.12*p.bound)
    for name,init in [('lqr',ub),('zero',torch.zeros_like(ub)),('half_lqr',ub*.5),('tuned_pd',up)]:
        start=time.perf_counter(); c,u=solver.solve(val,init,1024); torch.cuda.synchronize()
        result['planning'][name]=dict(mean=c.mean().item(),seconds=time.perf_counter()-start)
    write(f'experiments/results/hj_gridfree_planning_probe/{p.family}_d{p.dim}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__': main()
