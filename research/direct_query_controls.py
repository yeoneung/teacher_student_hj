"""Direct student-space and hybrid controls; no hidden coefficient access."""
import torch
import minimal_teaching as mt

METHODS=['student_basis','student_orthogonal','hybrid','pruned_hybrid']

@torch.no_grad()
def transmit(oracle,P,method,seed):
    if method not in METHODS:return mt.transmit(oracle,P,method,seed)
    v=oracle.target;N,d=v.shape;m=P.shape[1];alpha=oracle.radius/2
    before=oracle.calls.clone();beta=torch.zeros(N,m,device=v.device,dtype=v.dtype)
    if oracle.shape=='box':
        active=v.abs()>=oracle.radius*(1-1e-10);k=active.sum(-1)
        normal_visible=active&(P.norm(dim=1)>1e-12)[None,:]
        student_visible=(active.double()@P.square())>1e-24
    else:
        active=v.norm(dim=-1)>=oracle.radius*(1-1e-10);k=active.long()
        student_visible=((v@P).abs()>1e-12)&active[:,None]
        normal_visible=active&student_visible.any(-1)
    if method=='student_orthogonal':
        gen=torch.Generator(device=v.device).manual_seed(seed)
        H=torch.linalg.qr(torch.randn(m,m,device=v.device,dtype=v.dtype,generator=gen)).Q
    else:H=torch.eye(m,device=v.device,dtype=v.dtype)
    use_face=torch.zeros(N,device=v.device,dtype=torch.bool)
    visible=torch.ones(N,m,device=v.device,dtype=torch.bool)&(k>0)[:,None]
    if method=='hybrid':use_face=(k<m)&(k>0)
    if method=='pruned_hybrid':
        visible=student_visible
        kv=normal_visible.sum(-1) if oracle.shape=='box' else normal_visible.long()
        use_face=(kv<visible.sum(-1))&(kv>0)
    ix,j=(visible&~use_face[:,None]).nonzero(as_tuple=True)
    if len(ix):
        latent=H[:,j].T
        response=mt.feedback(oracle,alpha*(latent@P.T),ix)/alpha
        beta.index_add_(0,ix,latent*response[:,None])
    if oracle.shape=='box':
        mask=normal_visible if method=='pruned_hybrid' else active
        ix,j=(mask&use_face[:,None]).nonzero(as_tuple=True)
        if len(ix):
            u=torch.zeros(len(ix),d,device=v.device,dtype=v.dtype)
            u[torch.arange(len(ix),device=v.device),j]=alpha
            response=mt.feedback(oracle,u,ix)/alpha
            beta.index_add_(0,ix,P[j]*response[:,None])
    else:
        ix=use_face.nonzero().flatten()
        if len(ix):
            direction=v[ix]/oracle.radius
            response=mt.feedback(oracle,alpha*direction,ix)/alpha
            beta[ix]=(direction*response[:,None])@P
    return dict(coefficient=-2*mt.CURV*(v@P)+beta,normal_latent=beta,queries=oracle.calls-before)

def structured_initials(n,d,seed):
    gen=torch.Generator().manual_seed(seed)
    x=.2*torch.rand(n,d,generator=gen)-.1
    sign=torch.where(torch.rand(n,3,generator=gen)<.5,-1.,1.)
    x[:,:3]=sign*(6+4*torch.rand(n,3,generator=gen))
    return x.cuda().double(),torch.ones(n,device='cuda',dtype=torch.bool)

def structured_embedding(seed):
    P=torch.zeros(8,4,dtype=torch.float64);P[:3,0]=3**-.5
    P[3,1]=P[4,2]=P[5,3]=1.
    gen=torch.Generator().manual_seed(seed)
    H=torch.linalg.qr(torch.randn(4,4,generator=gen,dtype=torch.float64)).Q
    return (P@H).cuda()

def checks():
    maximum=0.;cases=0
    for shape in ['box','ball']:
        for d,m in [(8,2),(32,4)]:
            x,_=mt.initials(96,d,511+d);P=mt.embedding(d,m,191+d)
            o=mt.Oracle(x,.5,shape);g=mt.geometry(o.target,P,.5,shape)
            for method in METHODS:
                out=transmit(o,P,method,771)
                err=float((out['coefficient']-2*mt.A*mt.B*(x@P)).abs().max())
                maximum=max(maximum,err);assert err<1e-9
                expected=torch.minimum(g['face_dim'],torch.full_like(g['face_dim'],m)) if 'hybrid' in method else (g['face_dim']>0)*m
                assert torch.equal(expected,out['queries']);cases+=len(x)
    x,_=structured_initials(96,8,555);P=structured_embedding(888);o=mt.Oracle(mt.states(x),.5,'box')
    g=mt.geometry(o.target,P,.5,'box');assert (g['rank']==1).all() and (g['face_dim']==3).all()
    counts={}
    for method in METHODS+['rank','face']:
        out=transmit(o,P,method,777)
        err=float((out['coefficient']-2*mt.A*mt.B*(mt.states(x)@P)).abs().max())
        assert err<1e-9;maximum=max(maximum,err)
        counts[method]=int(out['queries'][0]);assert (out['queries']==counts[method]).all()
    assert counts==dict(student_basis=4,student_orthogonal=4,hybrid=3,pruned_hybrid=3,rank=1,face=3)
    return dict(passed=True,recovery_cases=cases,maximum_error=maximum,structured_counts=counts)
