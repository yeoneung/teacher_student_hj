"""Independent exact-quadratic audit of the cached block performance bound."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def audit(n,horizon=80,length=16):
    identity=np.eye(n)
    lap=np.roll(identity,1,axis=1)+np.roll(identity,-1,axis=1)-2*identity
    a=identity+.05*(.1*lap+.4*identity)
    b=.05*identity
    teacher_gain=.9
    closed=a-teacher_gain*b
    q=.05*(1+.1*teacher_gain**2)*identity/n
    values=[None]*(horizon+1)
    values[-1]=2*identity/n
    for k in range(horizon-1,-1,-1):
        values[k]=q+closed.T@values[k+1]@closed
        values[k]=(values[k]+values[k].T)/2
    def quadratic(x,p):return np.einsum('bi,ij,bj->b',x,p,x)
    def block(x,gain,steps):
        x=x.copy();cost=np.zeros(len(x))
        for _ in range(steps):
            u=-gain*x
            cost+=.05*np.mean(x*x+.1*u*u,axis=1)
            x=x@a.T+u@b.T
        return cost,x
    gen=np.random.default_rng(691000+n)
    initial=gen.normal(size=(32,n))
    cb,xb=block(initial,teacher_gain,horizon)
    teacher_cost=cb+2*np.mean(xb*xb,axis=1)
    value_error=float(np.max(np.abs(teacher_cost-quadratic(initial,values[0]))))
    assert value_error<1e-11
    rows=[]
    for gain in [.3,.9,1.5,3.]:
        cs,xs=block(initial,gain,horizon)
        student_cost=cs+2*np.mean(xs*xs,axis=1)
        for radius in [0.,.05,.25,.5]:
            current=initial.copy()
            estimated=np.zeros(len(initial));error=np.zeros(len(initial));bound=np.zeros(len(initial))
            exact_advantages=np.zeros(len(initial))
            max_block_bound_violation=0.
            for start in range(0,horizon,length):
                end=min(start+length,horizon)
                cs,zs=block(current,gain,end-start)
                cb,zb=block(current,teacher_gain,end-start)
                pv=values[end]
                exact=cs-cb+quadratic(zs,pv)-quadratic(zb,pv)
                exact_advantages+=exact
                if end==horizon:
                    approximation=exact
                    remainder=np.zeros(len(initial));budget=np.zeros(len(initial))
                else:
                    anchor=zb+radius*gen.normal(size=zb.shape)
                    covector=2*anchor@pv
                    approximation=cs-cb+np.sum(covector*(zs-zb),axis=1)
                    remainder=exact-approximation
                    m=2*np.linalg.eigvalsh(pv)[-1]
                    budget=.5*m*(np.sum((zs-anchor)**2,axis=1)+np.sum((zb-anchor)**2,axis=1))
                    max_block_bound_violation=max(max_block_bound_violation,float(np.max(np.abs(remainder)-budget)))
                estimated+=approximation;error+=remainder;bound+=budget
                current=zs
            truth=student_cost-teacher_cost
            residual=float(np.max(np.abs(truth-estimated-error)))
            telescoping=float(np.max(np.abs(truth-exact_advantages)))
            violation=float(np.max(np.abs(truth-estimated)-bound))
            assert residual<1e-10 and telescoping<1e-10 and violation<1e-10 and max_block_bound_violation<1e-10
            certified=estimated+bound<0
            assert np.all(truth[certified]<0)
            rows.append(dict(student_gain=gain,anchor_coordinate_noise=radius,
                actual_mean_cost_difference=float(truth.mean()),estimated_mean_advantage=float(estimated.mean()),
                mean_remainder=float(error.mean()),mean_error_bound=float(bound.mean()),
                max_identity_error=residual,max_telescoping_error=telescoping,
                max_trajectory_bound_violation=violation,max_block_bound_violation=max_block_bound_violation,
                certified_improvement_count=int(certified.sum()),states=len(initial)))
    return dict(dimension=n,horizon=horizon,length=length,teacher_gain=teacher_gain,
                independently_evaluated_teacher_value_error=value_error,rows=rows)


def main():
    with threadpool_limits(limits=1):records=[audit(n) for n in [32,128,256]]
    target=Path('experiments/results/hj_cotangent_dev_v6c/cached_performance_identity.json')
    target.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=sum(len(r['rows']) for r in records),
        max_identity_error=max(row['max_identity_error'] for r in records for row in r['rows']),
        max_bound_violation=max(row['max_trajectory_bound_violation'] for r in records for row in r['rows']),
        output=str(target)),indent=2))


if __name__=='__main__':main()
