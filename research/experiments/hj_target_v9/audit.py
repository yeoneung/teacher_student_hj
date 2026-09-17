"""Independent quadratic identities, arithmetic inference, and GPU controls."""
import argparse
import copy
import math
import numpy as np
import torch
from scipy.stats import t as student_t
from experiments.hj_gridfree.engine import write
from .statistics import paired,family_decision


def cpu_audit():
    rng=np.random.default_rng(991731);records=[]
    for constraint in ['ball','box']:
        for rho in [0.,.1,16.,4096.]:
            n=7;count=400;r=.07;bound=1.3
            def projection(u):
                if constraint=='box':return u.clip(0,bound)
                return u/np.maximum(1,np.linalg.norm(u,axis=-1,keepdims=True)/bound)
            old=projection(rng.normal(size=(count,n)));b=rng.normal(size=(count,n))*5
            target=projection((rho*old-b)/(2*r+rho));student=projection(rng.normal(size=(count,n))*2)
            delta=target-old;e=student-target
            q=lambda u:r*(u*u).sum(-1)+(b*u).sum(-1)
            descent=q(target)-q(old)+(r+rho)*(delta*delta).sum(-1)
            normal=(2*r+rho)*target+b-rho*old
            qr=lambda u:q(u)+rho/2*((u-old)**2).sum(-1)
            error=np.max(np.abs(qr(student)-qr(target)-(r+rho/2)*(e*e).sum(-1)-(normal*e).sum(-1)))
            assert descent.max()<1e-9 and error<1e-8 and (normal*e).sum(-1).min()>-1e-8
            records.append(dict(constraint=constraint,rho=rho,max_descent_residual=float(descent.max()),normal_identity_error=float(error)))
    # On a sphere the normal term exactly equals lambda/2 times squared error.
    target=rng.normal(size=(100,5));target/=np.linalg.norm(target,axis=-1,keepdims=True)
    student=rng.normal(size=(100,5));student/=np.linalg.norm(student,axis=-1,keepdims=True)
    multiplier=rng.uniform(0,100,(100,1));normal=-multiplier*target;e=student-target
    sphere=float(np.max(np.abs((normal*e).sum(-1)-(multiplier/2*(e*e)).sum(-1))))
    assert sphere<1e-10
    b=np.linspace(1.,2.,10);a=b+np.array([-.04,-.03,-.02,-.05,-.04,-.03,-.02,-.05,-.04,-.03])
    got=paired(a,b);d=a-b
    independent=float(d.mean()+student_t.isf(.05/3,9)*np.sqrt(np.sum((d-d.mean())**2)/(9*10)))
    assert abs(got['upper_difference']-independent)<1e-12
    assert family_decision(dict(hjb_target=a,short=b,dpc=b*1.1,warm=b))['all_reference_superiority_with_observed_1pct']
    assert not family_decision(dict(hjb_target=b*1.02,short=b,dpc=b,warm=b))['all_reference_1pct_noninferiority']
    assert paired(b,b)['upper_difference']==0.
    result=dict(passed=True,quadratic_rows=records,sphere_identity_error=sphere,paired_statistics_error=abs(got['upper_difference']-independent))
    write('build/hj_v9_cpu_audit.json',result);print('PASS: local descent, normal-cone identity, sphere weighting and arithmetic inference.')


def cuda_audit():
    from experiments.hj_cotangent.systems import problem,teacher,actor,sample
    from experiments.hj_adaptive.trajectory import TrajectoryGraph
    from experiments.hj_proximal.targets import expose,RegressionGraph
    from .controls import CachedHamiltonianGraph,collect_control
    torch.manual_seed(991741);rows=[]
    for family in ['mechanical','reaction','building']:
        p=problem(family,4,8);a=actor(p,teacher(p),16).cuda()
        with torch.no_grad():a.net[-1].weight.normal_(std=.02)
        reference=copy.deepcopy(a);x=sample(p,32,991742);t=torch.arange(32,device='cuda')%p.horizon
        features,base,_=expose(a,x,t)
        target=torch.zeros_like(base)+(p.bound/2 if family=='building' else 0.)
        direct=CachedHamiltonianGraph(a,p,32,.002,'hamiltonian');regression=RegressionGraph(reference,p,32,.002,'action')
        for _ in range(8):direct.update(features,base,-2*target);regression.update(features,base,target)
        torch.cuda.synchronize()
        error=max(float((v-reference.state_dict()[name]).abs().max()) for name,v in a.state_dict().items())
        assert error<3e-6,(family,error)
        teaching=copy.deepcopy(a);jet=TrajectoryGraph(teaching,p,32)
        initial=sample(p,64,991743)
        full=collect_control(jet,initial,p,32,False);zero=collect_control(jet,initial,p,32,True)
        assert torch.equal(full[0],zero[0])
        for u,v in zip(full[1][:4],zero[1][:4]):assert torch.equal(u,v)
        assert torch.count_nonzero(zero[1][4])==0
        rows.append(dict(family=family,cached_quadratic_vs_interior_target_update_error=error,zero_signal_preserves_actual_cost=True))
        del a,reference,direct,regression,teaching,jet
    write('build/hj_v9_controls_audit.json',dict(passed=True,rows=rows));print('PASS: cached Hamiltonian versus interior target CUDA updates, and isolated covector removal.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);cpu_audit()
    if args.cuda:cuda_audit()


if __name__=='__main__':main()
