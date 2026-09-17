"""Scalar-feedback interfaces for exact quadratic teacher Bellman models.

Receiver inputs: states, feasible target, known isotropic curvature, constraint
geometry and a fixed orthonormal student output subspace. Only Oracle methods
access the hidden coefficient. This is an explicitly restricted feedback
interface, not a computational lower bound for a learner given the model.
"""
import hashlib,json
from pathlib import Path
import torch

A=.8;B=.2;CURV=.09;H=12;GAIN=A*B/CURV
METHODS=['point','gain','rank','random','face','full']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def project(u,radius,shape):
    if shape=='box':return u.clamp(-radius,radius)
    return u*(radius/u.norm(dim=-1,keepdim=True).clamp_min(1e-30)).clamp_max(1.)

def initials(n,d,seed):
    gen=torch.Generator().manual_seed(seed);rare=torch.rand(n,generator=gen)<.1
    x=2*torch.rand(n,d,generator=gen)-1
    x[rare]*=.1;x[~rare,0]*=.1
    sign=torch.where(torch.rand(n,generator=gen)<.5,-1.,1.)
    mag=6+4*torch.rand(n,generator=gen);x[rare,0]=(sign*mag)[rare]
    return x.cuda().double(),rare.cuda()

def states(x):
    decay=torch.tensor([A**t for t in range(H)],device=x.device,dtype=x.dtype)
    return (x[:,None,:]*decay[None,:,None]).reshape(-1,x.shape[-1])

def embedding(d,m,seed):
    gen=torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d,m,generator=gen,dtype=torch.float64),mode='reduced').Q.cuda()

class Oracle:
    def __init__(self,x,radius,shape):
        self._b=2*A*B*x;self.radius=radius;self.shape=shape
        self.target=project(-self._b/(2*CURV),radius,shape)
        self.calls=torch.zeros(len(x),device=x.device,dtype=torch.int64)
    def cost_difference(self,u,index):
        # One response is Q_x(u)-Q_x(0), from one feasible counterfactual.
        # The common baseline is already supplied by teacher evaluation.
        bound=u.abs().amax(-1) if self.shape=='box' else u.norm(dim=-1)
        assert float(bound.max())<=self.radius+1e-9 if len(u) else True
        self.calls.index_add_(0,index,torch.ones_like(index))
        return CURV*u.square().sum(-1)+(self._b[index]*u).sum(-1)
    def normal_reference(self):
        n=2*CURV*self.target+self._b
        if self.shape=='box':active=self.target.abs()>=self.radius*(1-1e-10)
        else:active=(self.target.norm(dim=-1)>=self.radius*(1-1e-10))[:,None]
        return torch.where(active,n,torch.zeros_like(n))
    def value_reference(self,u):return CURV*u.square().sum(-1)+(self._b*u).sum(-1)

def geometry(v,P,radius,shape):
    # Columns of U span the student-visible normal directions in latent space.
    n,d=v.shape;m=P.shape[1]
    if shape=='ball':
        active=v.norm(dim=-1)>=radius*(1-1e-10)
        latent=v@P;norm=latent.norm(dim=-1);rank=(active&(norm>1e-10)).long()
        U=torch.zeros(n,m,m,device=v.device,dtype=v.dtype)
        U[:,:,0]=latent/norm[:,None].clamp_min(1e-30)
        return dict(active=active,face_dim=active.long(),rank=rank,U=U)
    active=v.abs()>=radius*(1-1e-10)
    L=P.T[None,:,:]*active[:,None,:]
    gram=L@L.transpose(1,2)
    val,U=torch.linalg.eigh(gram);val=val.flip(-1);U=U.flip(-1)
    rank=(val>1e-10).sum(-1)
    return dict(active=active,face_dim=active.sum(-1),rank=rank,U=U)

def feedback(oracle,u,index):
    ans=oracle.cost_difference(u,index)
    # Remove the known quadratic and known target terms, leaving n^T u.
    return ans-CURV*u.square().sum(-1)+2*CURV*(oracle.target[index]*u).sum(-1)

@torch.no_grad()
def transmit(oracle,P,method,seed,geo=None):
    v=oracle.target;n,d=v.shape;m=P.shape[1];radius=oracle.radius;shape=oracle.shape
    geo=geometry(v,P,radius,shape) if geo is None else geo
    before=oracle.calls.clone();beta=torch.zeros(n,m,device=v.device,dtype=v.dtype)
    if method=='full':beta=oracle.normal_reference()@P
    elif method=='gain':
        mask=geo['face_dim']>0;ix=mask.nonzero().flatten()
        if len(ix):
            nv=feedback(oracle,v[ix],ix)
            w=v[ix] if shape=='ball' else v[ix]*geo['active'][ix]
            nhat=w*(nv/w.square().sum(-1).clamp_min(1e-30))[:,None]
            beta[ix]=nhat@P
    elif method=='rank':
        U=geo['U'];valid=torch.arange(m,device=v.device)[None,:]<geo['rank'][:,None]
        ix,j=valid.nonzero(as_tuple=True)
        if len(ix):
            latent=U[ix,:,j];u=(radius/2)*(latent@P.T)
            response=feedback(oracle,u,ix)/(radius/2)
            beta.index_add_(0,ix,latent*response[:,None])
    elif method=='face':
        if shape=='ball':
            ix=(geo['face_dim']>0).nonzero().flatten()
            if len(ix):
                direction=v[ix]/radius;u=(radius/2)*direction
                response=feedback(oracle,u,ix)/(radius/2)
                beta[ix]=(direction*response[:,None])@P
        else:
            ix,j=geo['active'].nonzero(as_tuple=True)
            if len(ix):
                u=torch.zeros(len(ix),d,device=v.device,dtype=v.dtype);u[torch.arange(len(ix),device=v.device),j]=radius/2
                response=feedback(oracle,u,ix)/(radius/2)
                beta.index_add_(0,ix,P[j]*response[:,None])
    elif method=='random':
        gen=torch.Generator(device=v.device).manual_seed(seed)
        directions=torch.randn(n,m,d,device=v.device,dtype=v.dtype,generator=gen)
        if shape=='box':directions*=geo['active'][:,None,:]
        else:directions=(directions*v[:,None,:]).sum(-1,keepdim=True)*v[:,None,:]
        directions/=directions.norm(dim=-1,keepdim=True).clamp_min(1e-30)
        valid=torch.arange(m,device=v.device)[None,:]<geo['rank'][:,None]
        directions*=valid[:,:,None]
        ix,j=valid.nonzero(as_tuple=True)
        y=torch.zeros(n,m,device=v.device,dtype=v.dtype)
        if len(ix):y[ix,j]=feedback(oracle,(radius/2)*directions[ix,j],ix)/(radius/2)
        nhat=(torch.linalg.pinv(directions,rtol=1e-12)@y[:,:,None]).squeeze(-1)
        beta=nhat@P
    elif method!='point':raise ValueError(method)
    return dict(coefficient=-2*CURV*(v@P)+beta,normal_latent=beta,
                queries=oracle.calls-before,rank=geo['rank'],face_dim=geo['face_dim'])

@torch.no_grad()
def checks():
    maximum=0.;cases=0
    for shape in ['ball','box']:
        for d,m in [(8,2),(32,4),(128,8)]:
            x,_=initials(96,d,917+d);P=embedding(d,m,1717+d)
            oracle=Oracle(x,.5,shape);g=geometry(oracle.target,P,.5,shape)
            exact=transmit(oracle,P,'full',99,g)
            for method in ['rank','face']:
                z=transmit(oracle,P,method,99,g)
                e=float((z['coefficient']-exact['coefficient']).abs().max());maximum=max(maximum,e)
                assert e<1e-9
                assert torch.equal(z['queries'],g['rank' if method=='rank' else 'face_dim'])
            gain=transmit(oracle,P,'gain',99,g)
            if shape=='ball':assert float((gain['coefficient']-exact['coefficient']).abs().max())<1e-9
            random=transmit(oracle,P,'random',99,g);assert torch.equal(random['queries'],g['rank'])
            cases+=len(x)
    # A normal can be nonzero and yet irrelevant to every student action.
    x=torch.tensor([[0.,0.,-10.,0.]],device='cuda',dtype=torch.float64);P=torch.eye(4,device='cuda',dtype=torch.float64)[:,:2]
    oracle=Oracle(x,.5,'ball');z=transmit(oracle,P,'rank',91)
    assert z['queries'].item()==0 and z['normal_latent'].abs().max().item()==0
    # Same target and same gain, opposite student action ranking in a box.
    v=torch.ones(2,dtype=torch.float64);aa=torch.tensor([1.,0.],dtype=torch.float64);bb=aa.flip(0)
    witnesses=[]
    for b in [torch.tensor([-3.,-5.],dtype=torch.float64),torch.tensor([-5.,-3.],dtype=torch.float64)]:
        q=lambda u:float(u.square().sum()+b@u)
        assert torch.equal((-b/2).clamp(-1,1),v)
        witnesses.append(dict(gain=-q(v),cost_A=q(aa),cost_B=q(bb)))
    assert witnesses[0]['gain']==witnesses[1]['gain']==6
    assert witnesses[0]['cost_A']>witnesses[0]['cost_B'] and witnesses[1]['cost_A']<witnesses[1]['cost_B']
    return dict(passed=True,query_recovery_cases=cases,maximum_coefficient_error=maximum,
                irrelevant_normal_zero_queries=True,box_gain_witness=witnesses,cuda_device=torch.cuda.get_device_name(0))
