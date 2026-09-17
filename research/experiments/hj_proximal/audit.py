"""Independent analytic and captured-update checks before development outcomes."""
import copy
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem, teacher, actor, sample, running, step
from .targets import coefficients, feasible, target_action, model_difference, expose, reconstruct, RegressionGraph


def main():
    torch.set_num_threads(1);torch.manual_seed(819031)
    rows=[]
    for family in ['mechanical','reaction','building']:
        p=problem(family,4,16)
        x=sample(p,32,819032).double();t=torch.arange(32,device='cuda')%p.horizon
        old=feasible(torch.randn(32,p.n,device='cuda',dtype=torch.float64),p)
        cov=torch.randn_like(x)
        r,b=coefficients(p,t,cov)
        u=old.clone().requires_grad_(True)
        exact=running(x,u,p,t)+(cov*step(x,u,p,t)).sum(-1)
        grad=torch.autograd.grad(exact.sum(),u)[0]
        grad_error=float((grad-(2*r*u+b)).abs().max());assert grad_error<1e-12
        for reg in [.25,16.,4096.]:
            target=target_action(old,t,cov,p,reg);rho=reg*2*r
            g=2*r*target+b+rho*(target-old)
            # Projection fixed point is the constrained first-order condition.
            kkt=float((feasible(target-g/(2*r+rho),p)-target).abs().max())
            delta=model_difference(target,old,t,cov,p)+rho/2*(target-old).square().sum(-1)
            assert kkt<1e-9 and delta.max()<=1e-9
            rows.append(dict(family=family,regularization=reg,kkt_error=kkt,max_proximal_change=float(delta.max()),gradient_error=grad_error))
        a=actor(p,teacher(p),16).cuda()
        with torch.no_grad():a.net[-1].weight.normal_(std=.03)
        f,base,deployed=expose(a,x.float(),t)
        with torch.no_grad():reconstructed=reconstruct(a.net,f,base,p)
        error=float((deployed-reconstructed).abs().max());assert error==0
        for kind in ['action','latent']:
            reference=copy.deepcopy(a)
            graph=RegressionGraph(a,p,32,.002,kind)
            target=torch.randn_like(base)*.1 if kind=='latent' else target_action(old,t,cov,p,16).float()
            opt=torch.optim.Adam(reference.net.parameters(),lr=.002)
            for _ in range(4):
                graph.update(f,base,target)
                opt.zero_grad()
                if kind=='latent':loss=(reference.net(f).squeeze(-1)-target).square().mean()
                else:loss=((reconstruct(reference.net,f,base,p)-target)/p.bound).square().mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(reference.net.parameters(),10.);opt.step()
            torch.cuda.synchronize()
            error=max(float((v-reference.state_dict()[k]).abs().max()) for k,v in a.state_dict().items())
            assert error<3e-6,(family,kind,error)
            rows.append(dict(family=family,kind=kind,captured_adam_max_error=error,exposed_policy_error=0.))
            del graph
    write('build/hj_v8_preflight_audit.json',dict(passed=True,rows=rows))
    print('PASS: target KKT, surrogate descent, independent gradient, identical deployed actor, and captured Adam.')


if __name__=='__main__':main()
