"""CPU quadratic check: teacher feedback contraction and value curvature."""
import datetime
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits


def audit(dimension,gain,horizon,initial):
    eye=np.eye(dimension)
    lap=np.roll(eye,1,axis=1)+np.roll(eye,-1,axis=1)-2*eye
    dynamics=eye+.05*(.1*lap+.4*eye)
    closed=dynamics-.05*gain*eye
    q=.05*(1+.1*gain**2)*eye/dimension
    terminal=2*eye/dimension
    alpha=float(np.linalg.norm(closed,2))
    stage_curvature=float(np.linalg.norm(2*q,2))
    terminal_curvature=4/dimension
    matrix=terminal.copy();bound=terminal_curvature
    for _ in range(horizon):
        matrix=q+closed.T@matrix@closed
        matrix=.5*(matrix+matrix.T)
        bound=stage_curvature+alpha**2*bound
    actual=float(np.linalg.eigvalsh(2*matrix)[-1])
    x=initial.copy();cost=np.zeros(len(x))
    for _ in range(horizon):
        u=-gain*x
        cost+=.05*np.mean(x*x+.1*u*u,axis=1)
        x=x@dynamics.T+.05*u
    cost+=2*np.mean(x*x,axis=1)
    quadratic=np.einsum('bi,ij,bj->b',initial,matrix,initial)
    replay_error=float(np.max(np.abs(cost-quadratic)/(1+np.abs(cost))))
    relative_curvature_error=abs(actual-bound)/(1+abs(actual))
    assert replay_error<1e-11 and relative_curvature_error<1e-10
    assert actual<=bound+1e-10*(1+actual)
    cap=max(terminal_curvature,stage_curvature/(1-alpha**2)) if alpha<1 else None
    if cap is not None:assert actual<=cap+1e-10
    return dict(dimension=dimension,teacher_gain=gain,horizon=horizon,
        feedback_operator_norm=alpha,exact_value_gradient_lipschitz_constant=actual,
        mean_squared_penalty_curvature=dimension*actual,recursive_curvature_bound=bound,
        contractive_uniform_bound=cap,independent_rollout_scaled_cost_error=replay_error,
        recursive_bound_scaled_error=relative_curvature_error,initial_states=len(initial))


def main():
    started=time.perf_counter();records=[]
    with threadpool_limits(limits=1):
        for dimension in [32,128,256]:
            seed=693100+dimension
            initial=np.random.default_rng(seed).normal(size=(8,dimension))
            for gain in [.2,.9,3.]:
                for horizon in [16,40,80,160,320]:
                    record=audit(dimension,gain,horizon,initial)
                    record['initial_seed']=seed;records.append(record)
            print(json.dumps(dict(curvature_audit_dimension=dimension,completed=len(records))),flush=True)
        from .systems import problem
        from .building import coefficients
        building=[]
        for dimension in [32,128,256]:
            for shifted in [False,True]:
                kwargs=dict(weather_shift=-3.,budget=4.,topology_seed=62001) if shifted else {}
                p=problem('building',dimension,192,**kwargs)
                c=coefficients(p);a=c['a'];smallest=a-np.diag(1.2*c['b'])
                assert np.min(smallest)>=0
                # For every secant saturation slope D in [0,I], the teacher
                # secant matrix satisfies 0 <= A-1.2 diag(B)D <= A entrywise.
                singular=float(np.linalg.norm(a,2))
                _,vectors=np.linalg.eigh(a.T@a)
                positive=np.abs(vectors[:,-1])+1e-10
                perron_ratio=float(np.max((a.T@(a@positive))/positive))
                eps=np.finfo(np.float64).eps
                gamma=(2*dimension*eps)/(1-2*dimension*eps)
                padded_bound=float(np.sqrt(perron_ratio/(1-gamma)**2)*(1+1e-10))
                assert singular<=padded_bound and padded_bound<.99
                building.append(dict(dimension=dimension,model_shift=shifted,
                    min_secant_jacobian_entry=float(np.min(smallest)),matrix_operator_norm=singular,
                    positive_vector_spectral_upper_bound=padded_bound,conservative_lipschitz_bound=.99,
                    regional_mean_scaled_stage_curvature=2*p.dt*(1+.025*1.2**2),
                    regional_mean_scaled_uniform_value_curvature=2*p.dt*(1+.025*1.2**2)/(1-.99**2),
                    scope='Coefficient-only structural check; no primary initial states sampled. Global feedback contraction and curvature within a fixed thermostat-activation sequence; no bound on gradient jumps between pieces.'))
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-started,records=records,configurations=len(records),
        distinct_initial_vectors=24,building_structural_checks=building,
        scope='Analytic coupled linear-quadratic examples; same eight vectors per dimension reused across gains and horizons. These are not primary nonlinear tests or a certificate of curvature for the nonlinear teachers.')
    path=Path('experiments/results/hj_cotangent_dev_v6c/teacher_curvature_horizon.json')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(path),seconds=result['wall_seconds'],
        max_scaled_cost_error=max(r['independent_rollout_scaled_cost_error'] for r in records),
        max_scaled_curvature_error=max(r['recursive_bound_scaled_error'] for r in records))),flush=True)


if __name__=='__main__':main()
