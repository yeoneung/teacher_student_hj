"""Frozen neural test of student-relevant scalar teaching information.

The primary is recovery of the privileged full-information learning signal
with the predicted number of scalar value-difference queries. Comparisons of
final policy cost are secondary; no gain over full information is expected.
"""
import copy,gc,random,time
from pathlib import Path
import torch
from torch import nn
from experiments.hj_gridfree.engine import capture
import minimal_teaching as mt

HERE=Path(__file__).resolve().parent;OUT=HERE/'results/minimal_teaching_study'
CFG=dict(dimensions=[8,32],student_subspace_dimensions=[2,4],radii=[.5,2.],shapes=['ball','box'],
    seeds=list(range(13101,13106)),methods=mt.METHODS,updates=4096,batch=512,learning_rate=.003,
    hidden_width=32,train_initials=1024,validation_initials=512,test_initials=4096,
    dynamics='x_next=.8*x+.2*u, horizon12; stage=.36||x||^2+.05||u||^2, terminal=||x||^2. Exact zero teacher has V=||x||^2, exact advantage=.09||u||^2+.32*x dot u.',
    data='Same anisotropic initial-state law as the separate neural information study, extended to d dimensions. All12 teacher states are used. New seeds, independently generated train/validation/test initials.',
    student='Nonlinear bias-free d->32 SiLU->m network with fixed random orthonormal d-by-m embedding P. Uniform radial rescaling preserves range(P) and enforces the ball or box constraint. Box output uses uniform scaling, not coordinate clipping.',
    interface='Receiver has x, target v, known curvature, constraint and P. The linear Bellman coefficient is accessible only inside a scalar cost-difference oracle; full is a privileged derivative reference. Restriction is an experimental information interface, not a lower bound for agents given the analytic plant/value formula.',
    feedback='One scalar query returns Q_x(u)-Q_x(0) for a feasible u. Common baseline teacher evaluation and target generation are excluded shared information. All value queries are counted per cached state. Full derivative responses are reported separately and are not charged as scalar queries.',
    methods_description=dict(point='Targets only, missing normal set to zero.',gain='One target-improvement scalar per active target; minimum-norm normal consistent with that scalar and known face.',rank='Queries feasible directions in the student-visible normal span, exactly r=rank(P^T Gamma) queries.',random='Same r scalar queries in random ambient normal-space directions; minimum-norm normal decoder.',face='Queries all k normal-space basis directions, reconstructing the complete normal.',full='Privileged exact normal-vector reference; no scalar-query budget equivalence to gradient calls is claimed.'),
    primary='Across all cache states, rank and face reproduce the full projected quadratic coefficient to relative RMS error1e-9 or less, and count exactly r and k scalar queries respectively. This is a deterministic recovery/implementation check, not a statistical performance-superiority test.',
    secondary='Paired final neural normalized local regret and12-step cost across five seeds. Rank versus full fidelity tolerance1e-3 normalized regret, reported per pair without a population equivalence claim. Gain and random-budget failures, query savings, finite-update differences and all adverse results retained. Unadjusted paired95%t intervals; no multiple-comparison significance claim.',
    predictions=['For a ball, gain/rank/face recover full information when the normal is relevant; rank uses zero queries for an orthogonal normal.',
        'For a box, r may be smaller than k. Rank recovers the student cost model, whereas a gain scalar or r generic ambient queries need not do so.',
        'Matching the recovered projected coefficient should reproduce full-information neural learning up to numerical differences; no claim of better optimization than full information.',
        'Matched-update cost differences versus point/gain/random can vary by constraint and student space. Recovery itself does not guarantee closed-loop improvement.'],
    selection='All480 fits retained. Same initial parameters and minibatches within a seed/dimension/subspace/radius/shape. Final4096-update checkpoint saved before test evaluation; no early stopping, cost gate, rollback or test selection.',
    optimizer='Captured Adam(.9,.999), epsilon1e-8, clip10; loss=[a||z||^2+b_latent dot z]/(a*d), where feasible action=Pz. Float32 parameters with float64 action geometry, loss coefficients and diagnostics.',
    budget='Matched4096 updates, one timed CUDA process. Query construction, fitting and diagnostic times recorded separately. No end-to-end acceleration claim. Generic random and rank use equal scalar query counts; face generally uses more; full uses a different privileged interface.')

class Student(nn.Module):
    def __init__(self,P,radius,shape):
        super().__init__();self.radius=radius;self.shape=shape
        self.input=nn.Linear(P.shape[0],CFG['hidden_width'],bias=False)
        self.output=nn.Linear(CFG['hidden_width'],P.shape[1],bias=False)
        self.register_buffer('P',P.clone());nn.init.zeros_(self.output.weight)
    def latent(self,x):
        z=self.output(torch.nn.functional.silu(self.input(x.float()))).double()
        norm=z.norm(dim=-1,keepdim=True) if self.shape=='ball' else (z@self.P.T).abs().amax(-1,keepdim=True)
        return z*(self.radius/norm.clamp_min(1e-30)).clamp_max(1.)
    def forward(self,x):return self.latent(x)@self.P.T

class Trainer:
    def __init__(self,model):
        self.model=model;d,m=model.P.shape;dev=model.P.device
        self.x=torch.zeros(CFG['batch'],d,device=dev);self.b=torch.zeros(CFG['batch'],m,device=dev,dtype=torch.float64)
        self.loss=torch.zeros((),device=dev,dtype=torch.float64);self.bad=torch.zeros((),device=dev);self.clips=self.bad.clone()
        params=list(model.parameters());initial=copy.deepcopy(model.state_dict())
        self.mm=[torch.zeros_like(p) for p in params];self.vv=[torch.zeros_like(p) for p in params]
        self.b1=torch.ones((),device=dev);self.b2=self.b1.clone()
        for p in params:p.grad=torch.zeros_like(p)
        def update():
            for p in params:p.grad.zero_()
            z=model.latent(self.x);loss=(mt.CURV*z.square().sum(-1)+(self.b*z).sum(-1)).mean()/(mt.CURV*d)
            loss.backward()
            with torch.no_grad():
                self.loss.copy_(loss);norm=torch.stack([p.grad.double().square().sum() for p in params]).sum().sqrt()
                good=torch.isfinite(norm)&torch.isfinite(loss);self.bad.add_((~good).float());self.clips.add_((good&(norm>10)).float())
                b1=torch.where(good,.9,1.);b2=torch.where(good,.999,1.);self.b1.mul_(b1);self.b2.mul_(b2)
                scale=(10/norm.clamp_min(1e-12)).clamp_max(1)
                for p,m,v in zip(params,self.mm,self.vv):
                    g=torch.where(good,p.grad*scale,torch.zeros_like(p));m.mul_(b1).add_(g,alpha=.1);v.mul_(b2).addcmul_(g,g,value=.001)
                    step=-CFG['learning_rate']*(m/(1-self.b1).clamp_min(1e-12))/((v/(1-self.b2).clamp_min(1e-12)).sqrt()+1e-8)
                    p.add_(torch.where(good,step,torch.zeros_like(p)))
        self.graph=capture(update);model.load_state_dict(initial)
        for z in self.mm+self.vv:z.zero_()
        self.b1.fill_(1);self.b2.fill_(1);self.bad.zero_();self.clips.zero_()
    def update(self,x,b,index):
        self.x.copy_(x[index]);self.b.copy_(b[index]);self.graph.replay()

@torch.no_grad()
def evaluate(model,x,radius,shape):
    pool=mt.states(x);oracle=mt.Oracle(pool,radius,shape);u=model(pool)
    actual=oracle.value_reference(u);optimal=oracle.value_reference(oracle.target)
    regret=actual-optimal;opportunity=-optimal
    z=model.latent(pool);b=2*mt.A*mt.B*(pool@model.P)
    coefficient_error=float((actual-(mt.CURV*z.square().sum(-1)+(b*z).sum(-1))).abs().max())
    assert coefficient_error<1e-9 and float(regret.min())>-1e-8
    xx=x.clone();cost=torch.zeros(len(x),device=x.device,dtype=x.dtype);adv=cost.clone()
    for _ in range(mt.H):
        uu=model(xx);cost+=(1-mt.A**2)*xx.square().sum(-1)+.05*uu.square().sum(-1)
        adv+=mt.CURV*uu.square().sum(-1)+2*mt.A*mt.B*(xx*uu).sum(-1);xx=mt.A*xx+mt.B*uu
    cost+=xx.square().sum(-1);teacher=x.square().sum(-1);err=float((cost-teacher-adv).abs().max());assert err<1e-9
    metrics=dict(normalized_regret=float(regret.mean()/opportunity.mean()),regret=float(regret.mean()),test_cost=float(cost.mean()),teacher_cost=float(teacher.mean()),
        target_mse=float((u-oracle.target).square().mean()),maximum_telescoping_error=err,maximum_subspace_cost_error=coefficient_error)
    raw=dict(cost=cost,teacher_cost=teacher,advantage_sum=adv,local_regret=regret.reshape(-1,mt.H).mean(-1),local_opportunity=opportunity.reshape(-1,mt.H).mean(-1))
    return metrics,raw

def run():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    import json
    protocol=dict(config=CFG,source_sha256=mt.digest(__file__),interface_sha256=mt.digest(mt.__file__),checks=mt.checks())
    lock=OUT/'protocol_lock.json'
    if lock.exists():assert json.loads(lock.read_text())==protocol
    else:mt.write(lock,protocol)
    total=480;rows=[]
    for seed in CFG['seeds']:
        for d in CFG['dimensions']:
            trainx,_=mt.initials(CFG['train_initials'],d,61000000+seed);valx,_=mt.initials(CFG['validation_initials'],d,62000000+seed)
            pool=mt.states(trainx);training_x=pool.float()
            for m in CFG['student_subspace_dimensions']:
                P=mt.embedding(d,m,seed+311)
                for shape in CFG['shapes']:
                    for radius in CFG['radii']:
                        prefix=f's{seed}_d{d}_m{m}_{shape}_r{radius:g}'
                        common_start=time.perf_counter();oracle=mt.Oracle(pool,radius,shape);geo=mt.geometry(oracle.target,P,radius,shape)
                        torch.cuda.synchronize();common_seconds=time.perf_counter()-common_start
                        reference=mt.transmit(oracle,P,'full',seed,geo);data={};records={}
                        for method in mt.METHODS:
                            start=time.perf_counter();data[method]=mt.transmit(oracle,P,method,seed+933,geo)
                            torch.cuda.synchronize();elapsed=time.perf_counter()-start
                            delta=data[method]['coefficient']-reference['coefficient']
                            relative=float(delta.square().mean().sqrt()/reference['coefficient'].square().mean().sqrt().clamp_min(1e-30))
                            if method in ['rank','face']:assert relative<1e-9
                            if method in ['rank','random']:assert torch.equal(data[method]['queries'],geo['rank'])
                            if method=='face':assert torch.equal(data[method]['queries'],geo['face_dim'])
                            records[method]=dict(query_seconds=elapsed,queries=int(data[method]['queries'].sum()),queries_per_state=float(data[method]['queries'].double().mean()),
                                mean_relevant_rank=float(geo['rank'].double().mean()),mean_normal_dimension=float(geo['face_dim'].double().mean()),
                                relative_coefficient_error=relative,maximum_coefficient_error=float(delta.abs().max()),privileged_normal_vectors=len(pool) if method=='full' else 0)
                        cache=OUT/(prefix+'_cache.pt')
                        if not cache.exists():torch.save(dict(P=P,target=oracle.target,x=pool,data=data),cache)
                        torch.manual_seed(seed+107*m+d);initial=Student(P,radius,shape).cuda()
                        order=mt.METHODS.copy();random.Random(seed+d+m+int(radius*2)+(1 if shape=='box' else 0)).shuffle(order)
                        for method in order:
                            key=prefix+'_'+method;path=OUT/(key+'.json')
                            if path.exists():rows.append(json.loads(path.read_text()));continue
                            mt.write(OUT/'status.json',dict(status='training',case=key,completed=len(rows),total=total))
                            model=copy.deepcopy(initial);torch.cuda.synchronize();start=time.perf_counter();trainer=Trainer(model)
                            gen=torch.Generator(device='cuda').manual_seed(seed+177);curves=[]
                            for update in range(1,CFG['updates']+1):
                                ix=torch.randint(len(pool),(CFG['batch'],),device='cuda',generator=gen);trainer.update(training_x,data[method]['coefficient'],ix)
                                if update in [1024,2048,4096]:
                                    vv,_=evaluate(model,valx,radius,shape);curves.append(dict(update=update,validation_normalized_regret=vv['normalized_regret']))
                            torch.cuda.synchronize();fit_seconds=time.perf_counter()-start
                            cp=OUT/(key+'.pt');torch.save(model.state_dict(),cp)
                            testx,rare=mt.initials(CFG['test_initials'],d,63000000+seed)
                            metrics,raw=evaluate(model,testx,radius,shape);raw['rare']=rare
                            torch.save(raw,OUT/(key+'_evaluation.pt'))
                            row=dict(seed=seed,dimension=d,student_dimension=m,shape=shape,radius=radius,method=method,metrics=metrics,
                                feedback=records[method],common_seconds=common_seconds,fit_seconds=fit_seconds,nonfinite_updates=int(trainer.bad),clipped_updates=int(trainer.clips),
                                checkpoint_sha256=mt.digest(cp),cache_sha256=mt.digest(cache),curves=curves)
                            assert row['nonfinite_updates']==0;mt.write(path,row);rows.append(row)
                            print(key,'regret',round(metrics['normalized_regret'],6),'cost',round(metrics['test_cost'],5),'queries',records[method]['queries'],'sec',round(fit_seconds,2),flush=True)
                            del trainer,model,raw;gc.collect();torch.cuda.empty_cache()
                        del data,initial,oracle
    assert len(rows)==total
    mt.write(OUT/'report.json',dict(config=CFG,rows=rows));mt.write(OUT/'status.json',dict(status='complete',completed=len(rows),total=total))

if __name__=='__main__':run()
