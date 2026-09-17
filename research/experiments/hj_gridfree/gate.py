"""Exact-return block gate; guarantees relative cost only under the exact model."""
import torch
from .core import rollout,running,step,terminal


def gated_rollout(x,actor,base,p,length=16):
    total=torch.zeros(x.shape[0],device=x.device,dtype=x.dtype); actions=[]; choices=[]; advantages=[]
    for k in range(0,p.horizon,length):
        m=min(k+length,p.horizon)
        value,_,baseu=rollout(x,base,p,t0=k,keep=True)
        y=x; prefix=torch.zeros_like(total); candidate=[]
        for t in range(k,m):
            u=actor(y,t); candidate.append(u); prefix+=running(y,u,p); y=step(y,u,p)
        q=prefix+rollout(y,base,p,t0=m)
        choose=q<=value; choices.append(choose); advantages.append(q-value)
        candidate=torch.stack(candidate,1)
        chosen=torch.where(choose[:,None,None],candidate,baseu[:,:m-k])
        for t in range(m-k):
            u=chosen[:,t]; total+=running(x,u,p); x=step(x,u,p); actions.append(u)
    return total+terminal(x,p),torch.stack(actions,1),torch.stack(choices,1),torch.stack(advantages,1)


if __name__=='__main__':
    from .core import Problem,LQR,sample
    from .engine import write
    torch.set_num_threads(1); records={}
    for family in ['mechanical','reaction']:
        p=Problem(family,8,horizon=12); base=LQR(p,device='cpu')
        x=sample(p,32,991,device='cpu').double()
        with torch.no_grad():
            jb=rollout(x,base,p)
            j,u,chosen,a=gated_rollout(x,lambda y,t:-base(y,t),base,p,length=5)
        increase=(j-jb).max().item(); assert increase<1e-10,increase
        assert (~chosen).any().item(),'Audit must exercise teacher fallback'
        records[family]=dict(max_cost_increase=increase,teacher_fallback_count=int((~chosen).sum().item()),blocks=chosen.numel())
    write('experiments/results/hj_gridfree_audit_v1/gate.json',records)
    print(records,flush=True)
