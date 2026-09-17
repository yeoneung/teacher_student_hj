import copy
import hashlib
import json
from pathlib import Path
import time
import torch
from .core import project, rollout, exact_q, sequence_cost, prefix_return, truncated_return


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True),encoding='utf-8')


def hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('experiments/hj_gridfree').glob('*.py'))}


def capture(update,warmup=3):
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup): update()
    torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): update()
    return graph


class AdamGraph:
    """Static-buffer Adam, exact reset after graph setup; explicit best incumbent."""
    def __init__(self,x,p,kind='sequence',lr=.1):
        from .core import LQR
        self.p=p; self.kind=kind; self.base=LQR(p,x.device)
        self.x=torch.zeros_like(x); self.t=torch.zeros(x.shape[0],device=x.device,dtype=torch.long)
        shape=(x.shape[0],p.horizon,p.n) if kind=='sequence' else (x.shape[0],p.n)
        self.u=torch.zeros(shape,device=x.device,requires_grad=True)
        self.m=torch.zeros_like(self.u); self.v=torch.zeros_like(self.u)
        self.b1=torch.ones((),device=x.device); self.b2=torch.ones((),device=x.device)
        self.best=torch.full((x.shape[0],),float('inf'),device=x.device)
        self.best_u=torch.zeros_like(self.u)
        self.u.grad=torch.zeros_like(self.u)
        def update():
            self.u.grad.zero_()
            cost=self.cost(); cost.sum().backward()
            with torch.no_grad():
                self.retain(cost)
                self.m.mul_(.9).add_(self.u.grad,alpha=.1)
                self.v.mul_(.999).addcmul_(self.u.grad,self.u.grad,value=.001)
                self.b1.mul_(.9); self.b2.mul_(.999)
                self.u.add_(-lr*(self.m/(1-self.b1))/((self.v/(1-self.b2)).sqrt()+1e-8))
                self.u.copy_(project(self.u,p))
        self.graph=capture(update)

    def cost(self):
        return sequence_cost(self.x,self.u,self.p) if self.kind=='sequence' else exact_q(self.x,self.u,self.t,self.base,self.p)

    def retain(self,cost):
        better=cost<self.best; self.best.copy_(torch.minimum(cost,self.best))
        self.best_u.copy_(torch.where(better.reshape(-1,*([1]*(self.u.ndim-1))),self.u,self.best_u))

    def solve(self,x,u,steps,t=None):
        with torch.no_grad():
            self.x.copy_(x); self.u.copy_(u); self.m.zero_(); self.v.zero_()
            self.b1.fill_(1); self.b2.fill_(1); self.best.fill_(float('inf'))
            if t is not None: self.t.copy_(t)
        for _ in range(steps): self.graph.replay()
        with torch.no_grad(): self.retain(self.cost())
        return self.best.clone(),self.best_u.clone()


class TrainGraph:
    def __init__(self,actor,x,p,mode,lr=.002):
        self.actor=actor; self.p=p; self.mode=mode
        self.x=torch.zeros_like(x); self.t=torch.zeros(x.shape[0],device=x.device,dtype=torch.long)
        self.y=torch.zeros(x.shape[0],p.n,device=x.device)
        params=list(actor.net.parameters()); initial=copy.deepcopy(actor.state_dict())
        self.m=[torch.zeros_like(z) for z in params]; self.v=[torch.zeros_like(z) for z in params]
        self.b1=torch.ones((),device=x.device); self.b2=torch.ones((),device=x.device)
        self.loss=torch.zeros((),device=x.device)
        for z in params: z.grad=torch.zeros_like(z)
        def update():
            for z in params: z.grad.zero_()
            if mode=='mse': loss=(actor(self.x,self.t)-self.y).square().mean()/p.bound**2
            elif mode=='bellman': loss=exact_q(self.x,actor(self.x,self.t),self.t,actor.base,p).mean()
            elif mode=='dpc': loss=rollout(self.x,actor,p,self.t).mean()
            elif mode.startswith('block'): loss=prefix_return(self.x,self.t,actor,actor.base,p,int(mode[5:])).mean()
            elif mode.startswith('truncated'): loss=truncated_return(self.x,self.t,actor,p,int(mode[9:])).mean()
            else: raise ValueError(mode)
            loss.backward()
            with torch.no_grad():
                self.loss.copy_(loss); self.b1.mul_(.9); self.b2.mul_(.999)
                norm=torch.stack([z.grad.square().sum() for z in params]).sum().sqrt()
                scale=(10/norm.clamp_min(1e-12)).clamp_max(1)
                for z,m,v in zip(params,self.m,self.v):
                    g=z.grad*scale; m.mul_(.9).add_(g,alpha=.1); v.mul_(.999).addcmul_(g,g,value=.001)
                    z.add_(-lr*(m/(1-self.b1))/((v/(1-self.b2)).sqrt()+1e-8))
        self.graph=capture(update)
        actor.load_state_dict(initial)
        for z in self.m+self.v: z.zero_()
        self.b1.fill_(1); self.b2.fill_(1)

    def update(self,x,t,y=None):
        self.x.copy_(x); self.t.copy_(t)
        if y is not None: self.y.copy_(y)
        self.graph.replay()


class EvalGraph:
    def __init__(self,policy,p,batch,dtype=torch.float32):
        self.x=torch.zeros(batch,p.dim,device='cuda',dtype=dtype)
        self.cost=torch.zeros(batch,device='cuda',dtype=dtype)
        def update():
            with torch.no_grad(): self.cost.copy_(rollout(self.x,policy,p))
        self.graph=capture(update)
    def __call__(self,x):
        self.x.copy_(x); self.graph.replay(); return self.cost.clone()


def cem(x,incumbent,p,samples=256,iterations=6,seed=0,knots=10):
    """CEM over linearly interpolated action residuals, with incumbent retention.

    This is a specified smooth-noise CEM baseline, not a reproduction of iCEM.
    """
    gen=torch.Generator(device=x.device).manual_seed(seed)
    b=x.shape[0]; elite=max(4,samples//10)
    mean=torch.zeros(b,knots,p.n,device=x.device); std=torch.ones_like(mean)*p.bound
    best_u=incumbent.clone(); best=sequence_cost(x,best_u,p)
    with torch.no_grad():
        for _ in range(iterations):
            z=mean[:,None]+std[:,None]*torch.randn(b,samples,knots,p.n,device=x.device,generator=gen)
            z[:,0]=mean
            delta=torch.nn.functional.interpolate(z.flatten(0,1).transpose(1,2),size=p.horizon,mode='linear',align_corners=True).transpose(1,2)
            u=project(incumbent[:,None]+delta.reshape(b,samples,p.horizon,p.n),p)
            cost=sequence_cost(x[:,None].expand(-1,samples,-1).reshape(-1,p.dim),u.flatten(0,1),p).reshape(b,samples)
            vals,idx=cost.topk(elite,largest=False)
            winners=z.gather(1,idx[:,:,None,None].expand(-1,-1,knots,p.n))
            mean=.2*mean+.8*winners.mean(1); std=(.2*std+.8*winners.std(1,unbiased=False)).clamp_min(.03*p.bound)
            j=idx[:,0]; candidate=u[torch.arange(b,device=x.device),j]
            better=vals[:,0]<best; best=torch.minimum(best,vals[:,0]); best_u=torch.where(better[:,None,None],candidate,best_u)
    return best,best_u
