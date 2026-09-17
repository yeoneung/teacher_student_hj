"""Constructed spectral-truncation and response-noise checks, no training."""
from pathlib import Path
import json,hashlib
import numpy as np

class mt:
    @staticmethod
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    @staticmethod
    def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n',encoding='utf-8')

HERE=Path(__file__).resolve().parent;OUT=HERE/'results/approximate_feedback'
CFG=dict(angles=[0.,1e-6,1e-4,.01,.1,.5],queries=[1,3],response_error_norms=[0.,1e-6,1e-3],
    normals=64,seed=13301,dimension=8,student_dimension=4,normal_dimension=3,radius=.5,normal_bound=1.,
    scope='Constructed fixed normal cone, exact SVD and known norm bound. Near alignment changes exact rank; this is a local approximation upper bound, not noisy neural training or an approximate-query minimax theorem.')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(config=CFG,source_sha256=mt.digest(__file__))
    lock=OUT/'protocol_lock.json'
    if lock.exists():assert json.loads(lock.read_text())==protocol
    else:mt.write(lock,protocol)
    rng=np.random.default_rng(CFG['seed']);basis=np.eye(8)
    g=(basis[0]+basis[1]+basis[2])/np.sqrt(3)
    h1=(basis[0]-basis[1])/np.sqrt(2);h2=(basis[0]+basis[1]-2*basis[2])/np.sqrt(6)
    H=np.linalg.qr(rng.normal(size=(4,4)))[0];Gamma=basis[:3].T;v=np.r_[np.full(3,.5),np.zeros(5)]
    rows=[];raw=[];alpha=.25
    for theta in CFG['angles']:
        P=np.stack([g,np.cos(theta)*basis[3]+np.sin(theta)*h1,np.cos(theta)*basis[4]+np.sin(theta)*h2,basis[5]],1)@H
        U,s,_=np.linalg.svd(P.T@Gamma,full_matrices=False)
        assert np.allclose(s,[1,np.sin(theta),np.sin(theta)],atol=1e-14)
        exact_rank=1 if theta==0 else 3
        for q in CFG['queries']:
            E=U[:,:q];tail=float(s[q]) if q<len(s) else 0.
            for noise in CFG['response_error_norms']:
                errors=[];bounds=[];cost_errors=[]
                for sample in range(64):
                    lam=rng.uniform(.1,1,3);lam/=np.linalg.norm(lam);n=-Gamma@lam
                    beta=P.T@n;b=n-2*.09*v
                    actions=alpha*P@E
                    Q=lambda u:.09*np.sum(u*u,axis=0)+b@u
                    y=Q(actions)-.09*np.sum(actions*actions,axis=0)+2*.09*v@actions
                    e=rng.normal(size=q);e*=noise/np.linalg.norm(e)
                    estimate=E@(y+e)/alpha
                    bound=float(np.sqrt(tail**2+(noise/alpha)**2));err=float(np.linalg.norm(estimate-beta))
                    assert err<=bound+2e-14 and np.max(np.abs(actions))<=.5
                    z=rng.normal(size=(2,4));z*=.25/np.linalg.norm(z,axis=1)[:,None]
                    gap=float(abs((estimate-beta)@(z[0]-z[1])));cost_bound=bound*np.linalg.norm(z[0]-z[1])
                    assert gap<=cost_bound+2e-14
                    errors.append(err);bounds.append(bound);cost_errors.append(gap)
                    raw.append([theta,q,noise,sample,err,bound,gap,cost_bound])
                rows.append(dict(angle=theta,queries=q,noise=noise,exact_rank=exact_rank,singular_values=s.tolist(),
                    maximum_coefficient_error=max(errors),coefficient_bound=max(bounds),maximum_cost_difference_error=max(cost_errors)))
    np.savez_compressed(OUT/'raw.npz',values=np.array(raw),columns=np.array(['angle','q','noise','sample','coefficient_error','coefficient_bound','cost_error','cost_bound']))
    report=dict(passed=True,cases=len(raw),rows=rows,protocol_sha256=mt.digest(lock),raw_sha256=mt.digest(OUT/'raw.npz'),source_sha256=mt.digest(__file__))
    mt.write(OUT/'report.json',report);print(json.dumps(dict(passed=True,cases=len(raw))))

if __name__=='__main__':main()
