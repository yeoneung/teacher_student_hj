import json
import numpy as np
import torch
from .core import Problem,LQR,sample,rollout
from .native import solve
from .engine import write


def main():
    torch.set_num_threads(1); reports={}
    for family in ['mechanical','reaction']:
        p=Problem(family,16 if family=='mechanical' else 32)
        x=sample(p,16,21801,device='cpu').double(); base=LQR(p,device='cpu')
        with torch.no_grad(): j,_,u=rollout(x,base,p,keep=True)
        rows=[]
        for i in range(len(x)):
            a=solve(x[i].numpy(),u[i].numpy(),p,1024)
            b=solve(x[i].numpy(),np.zeros((p.horizon,p.n)),p,1024)
            rows.append(dict(base=j[i].item(),lqr_cost=a['cost'],zero_cost=b['cost'],lqr_seconds=a['seconds'],zero_seconds=b['seconds'],lqr_nit=a['nit'],zero_nit=b['nit']))
        reports[family]=dict(rows=rows,mean_best=float(np.mean([min(r['lqr_cost'],r['zero_cost']) for r in rows])),median_seconds=float(np.median([r['lqr_seconds']+r['zero_seconds'] for r in rows])))
    write('experiments/results/hj_gridfree_planning_probe/native_d32.json',reports)
    print(json.dumps({f:{k:v for k,v in r.items() if k!='rows'} for f,r in reports.items()}),flush=True)


if __name__=='__main__': main()
