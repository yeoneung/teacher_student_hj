"""CPU witnesses for the query lower bound and exact noisy reconstruction.

No learned policies, fitted hyperparameters or outcome-based case selection.
Each witness uses two valid box normals with identical r-1 query responses
and different student cost coefficients. Files are separate from neural data.
"""
import json,hashlib
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def main():
    rows=[];raw={}
    for d in [32,128]:
        for m in [2,4,8]:
            for k in [4,8,16,32]:
                for seed in range(5):
                    rng=np.random.default_rng(771000+d*1000+m*100+k*5+seed)
                    P=np.linalg.qr(rng.normal(size=(d,m)),mode='reduced')[0]
                    Gamma=np.eye(d)[:,:k];L=P.T@Gamma
                    E,sv,_=np.linalg.svd(L,full_matrices=False);r=int((sv>1e-12).sum());E=E[:,:r]
                    alpha=.25;W=alpha*(P@E).T
                    eta=-(1+rng.random(k));beta=L@eta;y=W@Gamma@eta
                    recovered=E@y/alpha;recovery=np.linalg.norm(recovered-beta)
                    # Select an invisible normal perturbation maximizing visibility to S.
                    _,_,vh=np.linalg.svd(W[:-1]@Gamma,full_matrices=True)
                    null=vh[r-1:].T
                    _,_,rv=np.linalg.svd(L@null,full_matrices=False)
                    h=null@rv[0];h/=np.linalg.norm(h)
                    eta0=-np.ones(k);plus=eta0+.25*h;minus=eta0-.25*h
                    observation_gap=np.linalg.norm(W[:-1]@Gamma@(plus-minus))
                    student_gap=np.linalg.norm(L@(plus-minus))
                    assert np.all(plus<0) and np.all(minus<0)
                    assert recovery<1e-12 and observation_gap<1e-12 and student_gap>1e-6
                    noises=[]
                    for sigma in [0.,1e-8,1e-5,1e-3]:
                        e=rng.normal(size=r);e*=sigma/max(np.linalg.norm(e),1e-30)
                        err=np.linalg.norm(E@(y+e)/alpha-beta);bound=np.linalg.norm(e)/alpha
                        assert err<=bound+1e-12
                        noises.append(dict(response_l2=sigma,coefficient_error=err,bound=bound))
                    key=f'd{d}_m{m}_k{k}_s{seed}'
                    raw[key+'_P']=P;raw[key+'_W']=W;raw[key+'_normals']=np.stack([Gamma@plus,Gamma@minus])
                    rows.append(dict(d=d,m=m,k=k,seed=seed,r=r,queries_below_bound=r-1,
                        reconstruction_error=recovery,indistinguishable_response_gap=observation_gap,
                        distinguishable_student_coefficient_gap=student_gap,noise=noises))
    assert len(rows)==120
    out=HERE/'results/minimal_information_exact';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'witnesses.npz',**raw)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(passed=True,cases=len(rows),training_runs=0,source_sha256=digest(__file__),
        raw_sha256=digest(out/'witnesses.npz'),rows=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,cases=len(rows),maximum_reconstruction_error=max(z['reconstruction_error'] for z in rows),
        maximum_indistinguishable_response_gap=max(z['indistinguishable_response_gap'] for z in rows),
        minimum_student_gap=min(z['distinguishable_student_coefficient_gap'] for z in rows))))
if __name__=='__main__':main()
