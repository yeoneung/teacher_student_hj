"""Prospective CUDA intervention on cost information in neural supervision.

Stable, completely specified linear dynamics with an exact fixed-teacher value.
The nonlinear student is trained by ordinary minibatch gradient learning.
No continuation-model error, learned critic, cost gate, or test selection.
Existing numerical sources/results are read-only and are not overwritten.
"""
import copy,gc,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.hj_gridfree.engine import capture

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/neural_information_transfer'
D=8;H=12;A=.8;B=.2;R=.05;Q=1-A*A;CURV=R+B*B;GAIN=A*B/CURV
METHODS=['point','restored','weight','shuffled']
CFG=dict(dim=D,horizon=H,dynamics=dict(A=A,B=B,state_cost=Q,control_cost=R,terminal_cost=1.),
    teacher='Identically zero; V_teacher(t,x)=||x||^2 exactly for every decision time.',
    student='One hidden SiLU layer without biases, linear output, radial projection. Width is an output-subspace bottleneck; width16 can represent the exact greedy map using SiLU(x)-SiLU(-x)=x.',
    widths=[2,4,16],control_radii=[.5,2.,8.,32.],seeds=[12101,12102,12103,12104,12105],methods=METHODS,
    updates=4096,batch=512,learning_rate=.003,train_initials=1024,validation_initials=512,test_initials=4096,
    data='Fixed teacher trajectories, all12 decision times. 10% rare large first-axis excursions; other cases concentrate on remaining axes. x1 rare has random sign and magnitude6..10. Common coordinates2..8 are uniform[-1,1]; rare other coordinates are uniform[-.1,.1]. Common x1 is uniform[-.1,.1].',
    pairing='Same initial network parameters, teacher states, minibatch generator and update count within each width/radius/seed. Same initial deployed zero policy at all widths/radii. Randomized sequential run order; one timed GPU process.',
    primary='Held-out exact local Bellman regret, divided by mean teacher-to-greedy local improvement, at width2/radius2. Compare restored minus point across five seeds with unadjusted paired95%t interval.',
    secondary='Constraint-radius and width response; full12-step student cost, rare/common strata, teacher-advantage telescoping, normal-term contribution, rare-axis projection into output span, restoration versus weight-only/shuffled normal controls.',
    predictions=['At radius32 no teacher target is constrained: normal=0 and all methods have identical losses and training trajectories.',
                 'At limited width and active constraints, correct normal information reduces Bellman regret relative to point labels; the effect may depend on radius and need not be monotone in raw cost.',
                 'Increasing representational capacity should reduce conflict when optimization succeeds; width16 has a known exact greedy representation.',
                 'Improvement from correct information should not be reproduced reliably by fixed random permutation of normal vectors. Weight-only loss tests whether scalar importance recovers much of the effect.'],
    selection='Final4096-update policy is primary. No hyperparameter, seed or checkpoint selection on test costs; no rejection/rollback. Validation curves at1024,2048,4096 are descriptive only.',
    optimizer='Identical captured Adam (.9,.999), epsilon1e-8, gradient norm clipping10. Loss normalized by CURV and dimension, not by input radius.',
    loss='point=||u-target||^2; restored=point+normal/CURV dot(u-target); weight=(1+||normal||/(2*CURV*radius))*point; shuffled=point+permuted(normal)/CURV dot(u-target). Mean over samples and action dimensions.',
    reporting='All240 runs retained. Five seed means are statistical units; states/times are not independent replicates. Constructed mechanism benchmark, not evidence of general real-world superiority or causes of archived nonlinear failures.',
    budget='Matched updates. Record preparation and complete fitting time separately from diagnostic evaluation. No equal-wall-budget or end-to-end speedup claim.')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def project(u,radius):
    return u*(radius/u.norm(dim=-1,keepdim=True).clamp_min(1e-20)).clamp_max(1.)


class Student(nn.Module):
    def __init__(self,width,radius):
        super().__init__();self.radius=radius
        self.input=nn.Linear(D,width,bias=False);self.output=nn.Linear(width,D,bias=False)
        nn.init.zeros_(self.output.weight)
    def forward(self,x):
        dtype=x.dtype
        return project(self.output(torch.nn.functional.silu(self.input(x.float()))),self.radius).to(dtype)


def initials(n,seed):
    gen=torch.Generator(device='cpu').manual_seed(seed)
    rare=torch.rand(n,generator=gen)<.1
    x=2*torch.rand(n,D,generator=gen)-1
    x[rare]*=.1;x[~rare,0]*=.1
    sign=torch.where(torch.rand(n,generator=gen)<.5,-1.,1.)
    mag=6+4*torch.rand(n,generator=gen)
    x[rare,0]=(sign*mag)[rare]
    return x.cuda().double(),rare.cuda()


def teacher_states(x):
    decay=torch.tensor([A**t for t in range(H)],dtype=x.dtype,device=x.device)
    return (x[:,None,:]*decay[None,:,None]).reshape(-1,D)


def teaching_data(x,radius,seed):
    target=project(-GAIN*x,radius)
    normal=2*CURV*target+2*A*B*x
    # Exact zero on interior teacher targets prevents cancellation artifacts in
    # the declared inactive-constraint negative control.
    active=(-GAIN*x).norm(dim=-1)>radius
    normal=torch.where(active[:,None],normal,torch.zeros_like(normal))
    weight=1+normal.norm(dim=-1)/(2*CURV*radius)
    gen=torch.Generator(device='cuda').manual_seed(seed+701)
    permutation=torch.randperm(len(x),device='cuda',generator=gen)
    return dict(x=x.float(),target=target.float(),normal=(normal/CURV).float(),
                shuffled=(normal[permutation]/CURV).float(),weight=weight.float(),active=active)


class Trainer:
    def __init__(self,model,method):
        self.model=model;self.method=method
        self.x=torch.zeros(CFG['batch'],D,device='cuda');self.target=torch.zeros_like(self.x)
        self.normal=torch.zeros_like(self.x);self.weight=torch.ones(CFG['batch'],device='cuda')
        self.loss=torch.zeros((),device='cuda');self.nonfinite=torch.zeros_like(self.loss)
        self.clipped=torch.zeros_like(self.loss)
        params=list(model.parameters());initial=copy.deepcopy(model.state_dict())
        self.m=[torch.zeros_like(p) for p in params];self.v=[torch.zeros_like(p) for p in params]
        self.b1=torch.ones((),device='cuda');self.b2=torch.ones_like(self.b1)
        for p in params:p.grad=torch.zeros_like(p)
        def update():
            for p in params:p.grad.zero_()
            delta=model(self.x)-self.target
            per=delta.square().sum(-1)
            if method in ['restored','shuffled']:per=per+(self.normal*delta).sum(-1)
            elif method=='weight':per=per*self.weight
            loss=per.mean()/D;loss.backward()
            with torch.no_grad():
                self.loss.copy_(loss)
                norm=torch.stack([p.grad.double().square().sum() for p in params]).sum().sqrt()
                good=torch.isfinite(norm)&torch.isfinite(loss)
                self.nonfinite.add_((~good).float());self.clipped.add_((good&(norm>10)).float())
                b1=torch.where(good,.9,1.);b2=torch.where(good,.999,1.)
                self.b1.mul_(b1);self.b2.mul_(b2)
                scale=(10/norm.clamp_min(1e-12)).clamp_max(1)
                for p,m,v in zip(params,self.m,self.v):
                    g=torch.where(good,p.grad*scale,torch.zeros_like(p))
                    m.mul_(b1).add_(g,alpha=.1);v.mul_(b2).addcmul_(g,g,value=.001)
                    delta=-CFG['learning_rate']*(m/(1-self.b1).clamp_min(1e-12))/((v/(1-self.b2).clamp_min(1e-12)).sqrt()+1e-8)
                    p.add_(torch.where(good,delta,torch.zeros_like(p)))
        self.graph=capture(update);model.load_state_dict(initial)
        for z in self.m+self.v:z.zero_()
        self.b1.fill_(1);self.b2.fill_(1);self.nonfinite.zero_();self.clipped.zero_()

    def update(self,data,idx):
        self.x.copy_(data['x'][idx]);self.target.copy_(data['target'][idx])
        self.normal.copy_(data['shuffled' if self.method=='shuffled' else 'normal'][idx])
        self.weight.copy_(data['weight'][idx]);self.graph.replay()


@torch.no_grad()
def local_metrics(model,x,radius,rare=None):
    u=model(x);target=project(-GAIN*x,radius)
    normal=2*CURV*target+2*A*B*x
    normal=torch.where(((-GAIN*x).norm(dim=-1)>radius)[:,None],normal,torch.zeros_like(normal))
    q=lambda a:CURV*a.square().sum(-1)+2*A*B*(x*a).sum(-1)
    regret=q(u)-q(target);opportunity=-q(target)
    square=CURV*(u-target).square().sum(-1);term=(normal*(u-target)).sum(-1)
    err=float((regret-square-term).abs().max())
    assert err<1e-8 and float(regret.min())>-1e-8
    out=dict(normalized_regret=float(regret.mean()/opportunity.mean()),regret=float(regret.mean()),
        opportunity=float(opportunity.mean()),target_mse=float((u-target).square().mean()),
        normal_contribution=float(term.mean()),normal_contribution_fraction=float(term.mean()/regret.mean().clamp_min(1e-30)),
        active_target_fraction=float((target.norm(dim=-1)>=radius*(1-1e-10)).double().mean()),
        student_boundary_fraction=float((u.norm(dim=-1)>=radius*(1-1e-6)).double().mean()),identity_error=err)
    if rare is not None:
        for label,mask in [('rare',rare),('common',~rare)]:out[label+'_regret']=float(regret[mask].mean())
    return out,dict(action=u,target=target,normal=normal,regret=regret,opportunity=opportunity)


@torch.no_grad()
def trajectory(model,x,radius):
    initial=x.clone();cost=torch.zeros(len(x),device=x.device,dtype=x.dtype);adv=cost.clone()
    for _ in range(H):
        u=project(-GAIN*x,radius) if model is None else model(x)
        cost+=Q*x.square().sum(-1)+R*u.square().sum(-1)
        adv+=CURV*u.square().sum(-1)+2*A*B*(x*u).sum(-1)
        x=A*x+B*u
    cost+=x.square().sum(-1);teacher=initial.square().sum(-1)
    assert float((cost-teacher-adv).abs().max())<1e-9
    return cost,teacher,adv


def checks():
    torch.manual_seed(91);x,_=initials(128,991)
    net=Student(16,2.).cuda()
    with torch.no_grad():
        net.input.weight.copy_(torch.cat([torch.eye(D,device='cuda'),-torch.eye(D,device='cuda')]))
        net.output.weight.copy_(torch.cat([-GAIN*torch.eye(D,device='cuda'),GAIN*torch.eye(D,device='cuda')],dim=1))
    assert float((net(x)-project(-GAIN*x,2.)).abs().max())<1e-6
    # Check restored action gradients against the direct exact Bellman model.
    z=x[:32].float();u=torch.randn_like(z,requires_grad=True)
    target=project(-GAIN*z,2.);normal=2*CURV*target+2*A*B*z
    l1=((u-target).square()+(normal/CURV)*(u-target)).mean()
    l2=(u.square()+(2*A*B/CURV)*z*u).mean()
    assert torch.allclose(torch.autograd.grad(l1,u)[0],torch.autograd.grad(l2,u)[0],atol=1e-7,rtol=1e-6)
    data=teaching_data(teacher_states(x),32.,91);assert not bool(data['active'].any()) and float(data['normal'].abs().max())==0
    cost,teacher,adv=trajectory(None,x,2.)
    assert bool((cost<teacher).all())
    return dict(passed=True,greedy_representation_error=float((net(x)-project(-GAIN*x,2.)).abs().max()),
        teacher_telescoping_error=float((cost-teacher-adv).abs().max()),cuda_device=torch.cuda.get_device_name(0))


def run():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'protocol_lock.json';protocol=dict(config=CFG,source_sha256=digest(Path(__file__)),checks=checks())
    if lock.exists():assert json.loads(lock.read_text())==protocol
    else:write(lock,protocol)
    rows=[];total=len(CFG['seeds'])*len(CFG['widths'])*len(CFG['control_radii'])*len(METHODS)
    for seed in CFG['seeds']:
        setup=time.perf_counter();trainx,trainrare=initials(CFG['train_initials'],51000000+seed)
        valx,valrare=initials(CFG['validation_initials'],52000000+seed)
        pool=teacher_states(trainx);valpool=teacher_states(valx)
        torch.cuda.synchronize();pool_seconds=time.perf_counter()-setup
        for radius in CFG['control_radii']:
            setup=time.perf_counter();data=teaching_data(pool,radius,seed)
            torch.cuda.synchronize();common_seconds=pool_seconds+time.perf_counter()-setup
            for width in CFG['widths']:
                torch.manual_seed(seed+width*107);initial=Student(width,radius).cuda()
                order=METHODS.copy();random.Random(seed+width*107+int(radius*2)).shuffle(order)
                for method in order:
                    key=f's{seed}_r{radius:g}_w{width}_{method}';path=OUT/(key+'.json')
                    if path.exists():rows.append(json.loads(path.read_text()));continue
                    write(OUT/'status.json',dict(status='training',case=key,completed=len(rows),total=total))
                    model=copy.deepcopy(initial);torch.cuda.synchronize();start=time.perf_counter()
                    trainer=Trainer(model,method);gen=torch.Generator(device='cuda').manual_seed(seed+177)
                    curves=[]
                    for update in range(1,CFG['updates']+1):
                        ix=torch.randint(len(pool),(CFG['batch'],),device='cuda',generator=gen)
                        trainer.update(data,ix)
                        if update in [1024,2048,4096]:
                            metric,_=local_metrics(model,valpool,radius)
                            curves.append(dict(update=update,validation_normalized_regret=metric['normalized_regret'],training_last_loss=float(trainer.loss)))
                    torch.cuda.synchronize();fit_seconds=time.perf_counter()-start
                    checkpoint=OUT/(key+'.pt');torch.save(model.state_dict(),checkpoint)
                    # Primary final policy is saved before any test generation.
                    testx,testrare=initials(CFG['test_initials'],53000000+seed);testpool=teacher_states(testx)
                    metrics,raw=local_metrics(model,testpool,radius,testrare.repeat_interleave(H))
                    costs,teacher,adv=trajectory(model,testx,radius);greedy,_,_=trajectory(None,testx,radius)
                    with torch.no_grad():
                        uu,ss,_=torch.linalg.svd(model.output.weight.double(),full_matrices=False)
                        keep=ss>1e-8;rare_axis=float(uu[0,keep].square().sum())
                    result=dict(seed=seed,width=width,radius=radius,method=method,updates=CFG['updates'],
                        metrics=metrics,test_cost=float(costs.mean()),teacher_cost=float(teacher.mean()),greedy_cost=float(greedy.mean()),
                        rare_cost=float(costs[testrare].mean()),common_cost=float(costs[~testrare].mean()),rare_axis_span=rare_axis,
                        charged_seconds=common_seconds+fit_seconds,common_seconds=common_seconds,fit_seconds=fit_seconds,
                        nonfinite_updates=int(trainer.nonfinite),clipped_updates=int(trainer.clipped),curves=curves,
                        checkpoint_sha256=digest(checkpoint),test_rare_count=int(testrare.sum()),
                        maximum_telescoping_error=float((costs-teacher-adv).abs().max()))
                    # Save per-initial-state local averages to preserve pairing and
                    # avoid treating 12 time points as independent observations.
                    torch.save(dict(cost=costs,teacher_cost=teacher,greedy_cost=greedy,advantage_sum=adv,rare=testrare,
                        local_regret=raw['regret'].reshape(-1,H).mean(-1),local_opportunity=raw['opportunity'].reshape(-1,H).mean(-1)),OUT/(key+'_evaluation.pt'))
                    write(path,result);rows.append(result)
                    print(key,'regret',round(metrics['normalized_regret'],5),'cost',round(result['test_cost'],4),'seconds',round(fit_seconds,2),flush=True)
                    del trainer,model,raw;gc.collect();torch.cuda.empty_cache()
                del initial
    write(OUT/'report.json',dict(config=CFG,rows=rows))
    write(OUT/'status.json',dict(status='complete',completed=len(rows),total=total))


if __name__=='__main__':run()
