"""Curvature identity and finite-difference audit; run after timed training."""
import torch
from scipy.linalg import hadamard
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem,teacher,actor,sample
from experiments.hj_adaptive.trajectory import trajectory_covectors
from .targets import coefficients, feasible
from .curvature import perturbed_trajectory, weighted_target, CurvatureGraph


def main():
    torch.set_num_threads(1);torch.manual_seed(850123);rows=[]
    for family in ['mechanical','reaction','building']:
        p=problem(family,2,4);a=actor(p,teacher(p),8).cuda()
        for z in a.parameters():z.requires_grad_(False)
        x=sample(p,2,850124)
        # A complete Hadamard set cancels off-diagonal terms exactly.
        delta=torch.zeros(2,p.horizon,p.n,device='cuda',requires_grad=True)
        cost,states,controls=perturbed_trajectory(x,delta,a,p)
        g=torch.autograd.grad(cost.sum(),delta,create_graph=True)[0]
        signs=torch.tensor(hadamard(p.horizon*p.n),device='cuda',dtype=torch.float32).reshape(-1,p.horizon,p.n)
        h=torch.zeros_like(g)
        for sign in signs:
            h=h+sign*torch.autograd.grad(g,delta,grad_outputs=sign.expand_as(g),retain_graph=True)[0]/len(signs)
        xx=x.clone().requires_grad_(True)
        _,ss,uu,pp=trajectory_covectors(xx,a,p)
        t=torch.arange(p.horizon,device='cuda').repeat(len(x))
        r,b=coefficients(p,t,pp.flatten(0,1));expected=(2*r*uu.flatten(0,1)+b).reshape_as(g)
        gradient_error=float((expected-g).abs().max());assert gradient_error<2e-5
        # Independently perturb each control coordinate and differentiate the
        # corresponding objective once; finite-difference the first gradient.
        fd=torch.zeros_like(g);eps=1e-3
        for j in range(p.horizon*p.n):
            values=[]
            for direction in [-1.,1.]:
                dd=torch.zeros_like(delta)
                dd.reshape(len(x),-1)[:,j]=direction*eps;dd.requires_grad_(True)
                value=perturbed_trajectory(x,dd,a,p)[0]
                values.append(torch.autograd.grad(value.sum(),dd)[0].reshape(len(x),-1)[:,j])
            fd.reshape(len(x),-1)[:,j]=(values[1]-values[0])/(2*eps)
        curvature_error=float(((fd-h).abs()/(1+h.abs())).max());assert curvature_error<2e-3,(family,curvature_error)
        old=controls.detach().flatten(0,1);gradient=g.detach().flatten(0,1);diag=h.detach().flatten(0,1)
        target,positive=weighted_target(old,gradient,diag,p,16.)
        r=p.dt*({'mechanical':.04,'reaction':.1,'building':.025}[family])/p.n
        change=target-old;model=(gradient*change+.5*(positive+16*2*r)*change.square()).sum(-1)
        assert model.max()<=1e-5
        if family=='building':assert target.min()>=0 and target.max()<=p.bound
        else:assert target.square().mean(-1).sqrt().max()<=p.bound+1e-6
        graph=CurvatureGraph(a,p,2,4);gen=torch.Generator(device='cuda').manual_seed(850125)
        got=graph(x,gen);torch.cuda.synchronize()
        replay_error=float((got[3]-g).abs().max());assert replay_error<2e-5
        assert all(torch.isfinite(z).all() for z in got)
        rows.append(dict(family=family,gradient_identity_error=gradient_error,finite_difference_curvature_error=curvature_error,
            captured_gradient_error=replay_error,max_target_model_change=float(model.max())))
        del graph
    write('build/hj_v8_curvature_preflight.json',dict(passed=True,rows=rows))
    print('PASS: Bellman action-gradient identity, curvature finite differences, constrained target, and CUDA replay.')


if __name__=='__main__':main()
