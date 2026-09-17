"""Exact quadratic teacher-value check of local covector error bounds."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    records=[]
    with threadpool_limits(limits=1):
        for n in [32,128,256]:
            eye=np.eye(n)
            lap=np.roll(eye,1,axis=1)+np.roll(eye,-1,axis=1)-2*eye
            a=eye+.05*(.1*lap+.4*eye)
            b=.05*eye
            gain=.9*eye
            closed=a-b@gain
            q=.05*(eye+.1*gain.T@gain)/n
            p=2*eye/n
            for _ in range(80):p=q+closed.T@p@closed
            p=(p+p.T)/2
            m=2*np.linalg.eigvalsh(p)[-1]
            gen=np.random.default_rng(68400+n)
            anchors=gen.normal(size=(512,n))
            directions=gen.normal(size=(512,n))
            directions/=np.linalg.norm(directions,axis=1,keepdims=True)
            value=np.einsum('bi,ij,bj->b',anchors,p,anchors)
            covector=2*anchors@p
            rows=[]
            for radius in [.001,.01,.1,1.]:
                dz=radius*directions
                z=anchors+dz
                actual=np.einsum('bi,ij,bj->b',z,p,z)
                linear=value+np.sum(covector*dz,axis=1)
                remainder=actual-linear
                bound=.5*m*np.sum(dz*dz,axis=1)
                gradient_error=np.linalg.norm(2*dz@p,axis=1)
                slack=bound-remainder
                assert slack.min()>-1e-12
                assert np.max(gradient_error-m*np.linalg.norm(dz,axis=1))<1e-12
                exact_quadratic=linear+np.einsum('bi,ij,bj->b',dz,p,dz)
                identity_error=np.max(np.abs(actual-exact_quadratic))
                assert identity_error<1e-12
                rows.append(dict(radius=radius,mean_remainder=float(remainder.mean()),
                                 max_remainder_to_bound=float(np.max(remainder/bound)),
                                 min_majorization_slack=float(slack.min()),
                                 exact_quadratic_identity_error=float(identity_error)))
            records.append(dict(dimension=n,horizon=80,teacher_gain=.9,
                                closed_loop_spectral_radius=float(np.max(np.abs(np.linalg.eigvalsh(closed)))),
                                gradient_lipschitz_constant=float(m),
                                corresponding_mean_squared_penalty_rho=float(m*n),cases=rows))
    path=Path('experiments/results/hj_cotangent_dev_v6c/quadratic_jet_audit.json')
    path.write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
