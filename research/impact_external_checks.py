"""Common-state failure factorial and independent building quality/latency check."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from impact_core import HERE,PROJECT,write,digest
from experiments.hj_cotangent.systems import problem,teacher,actor,sample,rollout
from experiments.hj_cotangent.evaluation import PlanGraph
from experiments.hj_cotangent.building import solve_qp


def factorial():
    out=HERE/'results/factorial';out.mkdir(parents=True,exist_ok=True)
    cfg=dict(states=256,state_seed=491001,dimension=256,horizon=320,
        models=['nominal','changed_coefficients'],initials=['nominal','shifted'],
        actor_seeds={'transfer':[9501,9502,9503],'direct':[9701,9702,9703]},
        methods=['hjb_target','warm','dpc'],scope='Post-review diagnostic on common new states; no retuning or causal attribution to reduced training batch.')
    write(out/'plan_lock.json',dict(config=cfg,source_sha256=digest(__file__)))
    nominal=problem('reaction',256,320)
    initial=sample(nominal,256,cfg['state_seed']).double();rows=[]
    for changed in [False,True]:
        p=problem('reaction',256,320,**(dict(coupling=nominal.gamma*.8,budget=nominal.bound*.9) if changed else {}))
        net=actor(p,teacher(p),64).cuda();planner=PlanGraph(net,p,64)
        zero=copy.deepcopy(net.net.state_dict())
        nominal_base=teacher(nominal).cuda().state_dict()
        for shifted in [False,True]:
            x=initial*1.25 if shifted else initial
            for base_kind in (['adapted','nominal_gains'] if changed else ['adapted']):
                net.base.load_state_dict(nominal_base if base_kind=='nominal_gains' else teacher(p).state_dict())
                net.net.load_state_dict(zero)
                values=[]
                for start in range(0,len(x),64):values.append(planner(x[start:start+64])[0])
                base_cost=torch.cat(values)
                rows.append(dict(model='changed' if changed else 'nominal',initial='shifted' if shifted else 'nominal',
                    base=base_kind,training='base_only',method='base',seed=None,cost=float(base_cost.mean())))
                for training,seeds in cfg['actor_seeds'].items():
                    for method in cfg['methods']:
                        for seed in seeds:
                            root='hj_target_primary_v9' if training=='transfer' else 'hj_target_scaling_v9'
                            dim=32 if training=='transfer' else 256;budget='30.0' if training=='transfer' else '60.0'
                            path=PROJECT/f'experiments/results/{root}/training/reaction_d{dim}_h320_s{seed}/{method}.pt'
                            state=torch.load(path,map_location='cuda',weights_only=False)['budgets'][budget]['state_dict']
                            net.net.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('net.')})
                            values=[]
                            for start in range(0,len(x),64):values.append(planner(x[start:start+64])[0])
                            cost=torch.cat(values);high=x.mean(-1).abs()>=.8
                            rows.append(dict(model='changed' if changed else 'nominal',initial='shifted' if shifted else 'nominal',
                                base=base_kind,training=training,method=method,seed=seed,cost=float(cost.mean()),
                                high_coherence_cost=float(cost[high].mean()),other_cost=float(cost[~high].mean()),
                                high_coherence_fraction=float(high.double().mean()),source_sha256=digest(path)))
                print(json.dumps(dict(model=changed,shifted=shifted,base=base_kind,rows=len(rows))),flush=True)
        del planner,net;torch.cuda.empty_cache()
    torch.save(initial.cpu(),out/'common_initials.pt');write(out/'report.json',dict(config=cfg,rows=rows,completed=True))


def building():
    out=HERE/'results/building';out.mkdir(parents=True,exist_ok=True)
    cfg=dict(states=64,state_seed=501001,dimension=32,horizon=192,training_seeds=list(range(9501,9511)),
        scope='Independent nominal states; approximate convex QP optimum, deterministic full-horizon planning reference. CPU solve versus GPU resident-input policy latency reported separately, not matched learning acceleration.')
    write(out/'plan_lock.json',dict(config=cfg,source_sha256=digest(__file__)))
    p=problem('building',32,192);x=sample(p,64,cfg['state_seed']).double();net=actor(p,teacher(p),64).cuda();planner=PlanGraph(net,p,64)
    rows=[]
    for method in ['hjb_target','warm','dpc','short']:
        for seed in cfg['training_seeds']:
            path=PROJECT/f'experiments/results/hj_target_primary_v9/training/building_d32_h192_s{seed}/{method}.pt'
            state=torch.load(path,map_location='cuda',weights_only=False)['budgets']['30.0']['state_dict'];net.load_state_dict(state)
            costs=planner(x)[0]
            with torch.no_grad():
                _,xs,us=rollout(x,net,p,keep=True)
                tracking=float(xs.square().mean().sqrt());heating=float(p.dt*us.mean((0,2)).sum())
            timings=[]
            with torch.no_grad():
                for _ in range(10):net(x[:1],0)
                torch.cuda.synchronize()
                for _ in range(100):
                    start=time.perf_counter();net(x[:1],0);torch.cuda.synchronize();timings.append(time.perf_counter()-start)
            rows.append(dict(method=method,seed=seed,cost=float(costs.mean()),costs=costs.cpu().tolist(),
                tracking_rms_temperature_deviation=tracking,mean_integrated_heating_control=heating,
                gpu_policy_median_seconds=float(np.median(timings)),gpu_policy_p95_seconds=float(np.quantile(timings,.95))))
    qp=[]
    with threadpool_limits(limits=1):
        for i,initial in enumerate(x.cpu().numpy()):
            start=time.perf_counter();result=solve_qp(initial,p,eps=1e-7);elapsed=time.perf_counter()-start
            assert result['status']=='solved' and result['objective_replay_error']<1e-4
            qp.append(dict(index=i,cost=result['cost'],cpu_complete_call_seconds=elapsed,
                tracking_rms_temperature_deviation=float(np.sqrt(np.mean(result['x'][:-1]**2))),
                mean_integrated_heating_control=float(p.dt*np.mean(result['u'],axis=-1).sum()),
                **{k:v for k,v in result.items() if k not in ['x','u','cost']}))
    reference=np.array([r['cost'] for r in qp])
    for r in rows:
        assert np.min(np.asarray(r['costs'])-reference)>-1e-4
        r['qp_gap_percent']=float(100*(r['cost']/reference.mean()-1))
    write(out/'report.json',dict(config=cfg,rows=rows,qp=qp,reference_cost=float(reference.mean()),completed=True))
    print(json.dumps(dict(building_reference_cost=float(reference.mean()),states=64)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);factorial();building()
