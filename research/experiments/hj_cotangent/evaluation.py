"""Frozen fresh tests with common states, transferred weights and NumPy replay."""
import argparse
from dataclasses import asdict
import datetime
import gc
import json
from pathlib import Path
import torch
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.hj_gridfree.engine import capture, write
from experiments.hj_gridfree.evaluate import independent_cost as ring_cost
from .systems import problem, teacher, actor, sample, rollout, step
from .building_audit import numpy_cost as building_cost
from .conditioning import local_trajectory_audit, independent_feedback_cost
from .study import verify, digest


class PlanGraph:
    def __init__(self, policy, p, batch):
        self.x = torch.zeros(batch,p.dim,device='cuda',dtype=torch.float64)
        self.cost = torch.zeros(batch,device='cuda',dtype=torch.float64)
        self.u = torch.zeros(batch,p.horizon,p.n,device='cuda',dtype=torch.float64)
        self.states = torch.zeros(batch,p.horizon+1,p.dim,device='cuda',dtype=torch.float64)
        def evaluate():
            with torch.no_grad():
                cost,xs,u = rollout(self.x,policy,p,keep=True)
                self.cost.copy_(cost)
                self.u.copy_(u)
                self.states[:,:-1].copy_(xs)
                self.states[:,-1].copy_(step(xs[:,-1],u[:,-1],p,p.horizon-1))
        self.graph = capture(evaluate)

    def __call__(self,x):
        self.x.copy_(x)
        self.graph.replay()
        return self.cost.clone(),self.u.clone()


def replay(x,u,p):
    if p.family!='building':
        # A fixed per-state shape makes the canonical score independent of
        # outer batching, which matters for a long unstable open-loop replay.
        xx,uu=x.cpu(),u.cpu()
        return torch.cat([ring_cost(xx[i:i+1],uu[i:i+1],p) for i in range(len(xx))])
    xn,un = x.cpu().numpy(),u.cpu().numpy()
    with threadpool_limits(limits=1):
        return torch.tensor([building_cost(a,b,p) for a,b in zip(xn,un)],dtype=torch.float64)


def teacher_evaluation(x,p,ncheck,base=None):
    """Audit the analytic teacher on its original full evaluation batch."""
    base=teacher(p) if base is None else base
    with torch.no_grad():
        cost,xs,u=rollout(x,base,p,keep=True)
        final=step(xs[:ncheck,-1],u[:ncheck,-1],p,p.horizon-1)
        states=torch.cat((xs[:ncheck],final[:,None]),dim=1)
    if p.family=='building':
        ratio=u.max().item()/p.bound
        assert u.min().item()>=-1e-7
    else:ratio=u.square().mean(-1).sqrt().max().item()/p.bound
    assert torch.isfinite(cost).all() and ratio<1.00001
    local=local_trajectory_audit(states,u[:ncheck],cost[:ncheck],p)
    independent=independent_feedback_cost(x,base,p)
    assert torch.isfinite(independent).all()
    original=cost.cpu()
    meta=dict(**local,count=len(x),audit_trajectories=ncheck,max_budget_ratio=ratio,
        independent_feedback_scaled_cost_difference=((independent-original).abs()/(1+independent.abs())).max().item(),
        independent_feedback_relative_mean_difference=abs(independent.mean().item()/original.mean().item()-1),
        scope='Analytic teacher with the primary float64 plant/state arithmetic and its specified stored coefficients; original full evaluation batch preserved for both plants. This is separate from the float32 neural actor.')
    raw=dict(states=states.cpu(),controls=u[:ncheck].cpu(),feedback_costs=independent)
    return original,raw,meta


def train_dir(root, task, seed):
    return root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'


def lock_evaluation(root,cfg):
    checkpoints = []
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            folder = train_dir(root,task,seed)
            assert (folder/'complete.json').exists(), folder
            for spec in task['methods']:
                checkpoints.append(folder/(spec['name']+'.pt'))
    sources = [Path('experiments/hj_cotangent')/p for p in
               ['evaluation.py','building_audit.py','systems.py','building.py','core.py',
                'classical.py','convex_certificate.py','isolation.py','report.py','conditioning.py',
                'evaluation_preflight.py','feedback_native.py','feedback_study.py']]
    sources += [Path('experiments/hj_gridfree')/name for name in
                ['evaluate.py','native.py','planning_probe.py','engine.py','core.py','report.py']]
    sources.append(Path('experiments/results/hj_gridfree_primary_v1/pd_calibration.json'))
    sources.append(root/'evaluation_addendum.md')
    sources.append(root/'feedback_protocol_lock.json')
    sources += [Path('experiments/hj_cotangent')/name for name in
                ['cached_identity.py','jet_lq.py','feedback_native_audit.py','curvature.py','teacher_audit_pretest.py']]
    sources += [Path('experiments/results/hj_cotangent_dev_v6c')/name for name in
                ['final_numerical_audit.json','quadratic_jet_audit.json','cached_performance_identity.json',
                 'cpu_conditioning_audit.json','feedback_native_audit.json','teacher_curvature_horizon.json',
                 'teacher_evaluation_cpu_pretest.json']]
    target = root/'evaluation_lock.json'
    if target.exists():
        saved = json.loads(target.read_text())
        for path,expected in saved['sources'].items():
            assert digest(path)==expected,path
        for path,expected in saved['checkpoints'].items():
            assert digest(path)==expected,path
    else:
        write(target,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          sources={str(p):digest(p) for p in sources},
                          checkpoints={str(p):digest(p) for p in checkpoints}))


def cases(cfg):
    for ti,task in enumerate(cfg['tasks']):
        family,horizon = task['family'],task['horizon']
        for dim in cfg['test_dimensions']:
            n = dim//2 if family=='mechanical' else dim
            for ci,condition in enumerate(cfg['test_conditions']):
                kwargs = {}
                if condition=='model_shift':
                    if family=='building':
                        kwargs = dict(weather_shift=-3.,budget=4.,topology_seed=62001)
                    else:
                        nominal = problem(family,n,horizon)
                        kwargs = dict(coupling=nominal.gamma*1.5,budget=nominal.bound*.8)
                p = problem(family,n,horizon,**kwargs)
                # Common initial states across physical horizons.
                family_index={'mechanical':0,'reaction':1,'building':2}[family]
                seed = cfg['test_seed_base']+family_index*100000+dim*10+ci
                count = cfg['test_nominal_states'] if condition=='nominal' else cfg['test_shift_states']
                yield task,p,condition,seed,count


def evaluate_case(root,cfg,task,p,condition,seed,count):
    key = f'{p.family}_h{p.horizon}_d{p.dim}_{condition}'
    out = root/'evaluation'
    out.mkdir(exist_ok=True)
    if (out/(key+'.json')).exists():
        return
    x = sample(p,count,seed,shift=condition=='initial').double()
    policy = actor(p,teacher(p),cfg['width']).cuda()
    policy.eval()
    planner = PlanGraph(policy,p,cfg['evaluation_batch'])
    base_cost,teacher_raw,teacher_meta=teacher_evaluation(x,p,cfg['audit_states'])
    raw = dict(x=x.cpu(),teacher=base_cost,teacher_audit=teacher_raw,
               methods={},audit_controls={},audit_states={},feedback_replay_costs={})
    meta = dict(problem=asdict(p),condition=condition,test_seed=seed,count=count,
                methods={},teacher_mean=base_cost.mean().item(),teacher_audit=teacher_meta,audits=[])
    for spec in task['methods']:
        name = spec['name']
        rows,control_rows,state_rows,feedback_rows = [],[],[],[]
        for training_seed in cfg['seeds']:
            source = train_dir(root,task,training_seed)/(name+'.pt')
            saved = torch.load(source,map_location='cuda',weights_only=False)['state_dict']
            policy.net.load_state_dict({k[4:]:v for k,v in saved.items() if k.startswith('net.')})
            costs = []
            max_error = max_budget = feedback_error = feedback_mean_error = 0.
            local_audit={}
            for offset in range(0,count,cfg['evaluation_batch']):
                batch = x[offset:offset+cfg['evaluation_batch']]
                j,u = planner(batch)
                costs.append(j.cpu())
                if p.family=='building':
                    ratio = u.max().item()/p.bound
                    assert u.min().item()>=-1e-7
                else:
                    ratio = u.square().mean(-1).sqrt().max().item()/p.bound
                max_budget = max(max_budget,ratio)
                assert torch.isfinite(j).all().item() and ratio<1.00001,(key,name,ratio)
                if offset==0:
                    ncheck = cfg['audit_states']
                    jj = replay(batch[:ncheck],u[:ncheck],p)
                    error = ((jj-j[:ncheck].cpu()).abs()/(1+jj.abs())).max().item()
                    max_error = max(max_error,error)
                    control_rows.append(u[:ncheck].cpu())
                    states=planner.states[:ncheck].clone()
                    state_rows.append(states.cpu())
                    local_audit=local_trajectory_audit(states,u[:ncheck],j[:ncheck],p)
                    independent=independent_feedback_cost(batch,policy,p)
                    assert torch.isfinite(independent).all(),(key,name,'NumPy feedback')
                    feedback_error=((independent-j.cpu()).abs()/(1+independent.abs())).max().item()
                    feedback_mean_error=abs(independent.mean().item()/j.mean().item()-1)
                    feedback_rows.append(independent)
            row = torch.cat(costs)
            rows.append(row)
            meta['audits'].append(dict(method=name,training_seed=training_seed,
                independent_scaled_cost_error=local_audit['independent_scaled_trajectory_cost_error'],
                max_scaled_single_step_residual=local_audit['max_scaled_single_step_residual'],
                fixed_control_open_loop_scaled_difference=max_error,
                independent_feedback_scaled_cost_difference=feedback_error,
                independent_feedback_relative_mean_difference=feedback_mean_error,
                feedback_audit_count=cfg['evaluation_batch'],max_budget_ratio=max_budget))
        values = torch.stack(rows)
        raw['methods'][name] = values
        raw['audit_controls'][name] = torch.stack(control_rows)
        raw['audit_states'][name] = torch.stack(state_rows)
        raw['feedback_replay_costs'][name] = torch.stack(feedback_rows)
        meta['methods'][name] = dict(mean=values.mean().item(),seed_means=values.mean(1).tolist(),
                                     p95=float(np.quantile(values.numpy(),.95)),max=values.max().item())
    torch.save(raw,out/(key+'.pt'))
    write(out/(key+'.json'),meta)
    print(json.dumps(dict(case=key,count=count,means={k:v['mean'] for k,v in meta['methods'].items()})),flush=True)
    del x,policy,planner,raw
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='experiments/results/hj_cotangent_primary_v6')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root)
    cfg = verify(root)
    from .evaluation_preflight import run as preflight
    preflight(root,PlanGraph,replay,teacher_evaluation)
    lock_evaluation(root,cfg)
    for task,p,condition,seed,count in cases(cfg):
        evaluate_case(root,cfg,task,p,condition,seed,count)
    print('V6 FRESH EVALUATION COMPLETE',flush=True)


if __name__=='__main__':
    main()
