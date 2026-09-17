"""Fresh-process memory and synchronized inference timing, after quality runs."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import torch
from experiments.hj_gridfree.engine import capture,write
from .systems import problem,teacher,actor,sample,rollout
from .core import LocalGraph,JetGraph
from .study import verify
from .evaluation import train_dir


def time_replays(graph,repeats=200):
    for _ in range(20): graph.replay()
    values=[]
    for _ in range(7):
        torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(repeats): graph.replay()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)/repeats)
    return dict(median_seconds=statistics.median(values),batch_means_seconds=values,
                repetitions_per_batch=repeats,warmup_replays=20)


def memory(root,cfg,task,spec):
    p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
    torch.manual_seed(cfg['seeds'][0])
    policy=actor(p,teacher(p),cfg['width']).cuda()
    base=teacher(p)
    if spec.get('teacher_parent'):
        parent=train_dir(root,task,cfg['seeds'][0])/(spec['teacher_parent']+'.pt')
        saved=torch.load(parent,map_location='cuda',weights_only=False)['state_dict']
        policy.load_state_dict(saved)
        base=actor(p,teacher(p),cfg['width']).cuda()
        base.load_state_dict(saved)
        base.eval()
        for parameter in base.parameters():parameter.requires_grad_(False)
    data=sample(p,cfg['dataset'],68301)
    times=torch.arange(cfg['dataset'],device='cuda')%p.horizon
    torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph=LocalGraph(policy,p,cfg['batch'],spec.get('length',cfg['length']),
                     rho=spec.get('rho',0.),lr=spec.get('lr',cfg['lr']),mode=spec['mode'],teacher=base)
    cache=None
    jet=None
    if spec['mode']=='cache':
        cache=(torch.zeros_like(data),torch.zeros(len(data),device='cuda'),torch.zeros_like(data))
        jet=JetGraph(policy,base,p,cfg['batch'],spec.get('length',cfg['length']))
        for offset in range(0,len(data),cfg['batch']):
            labels=jet(data[offset:offset+cfg['batch']],times[offset:offset+cfg['batch']])
            for dst,src in zip(cache,labels): dst[offset:offset+cfg['batch']].copy_(src)
    for _ in range(5):
        labels=tuple(v[:cfg['batch']] for v in cache) if cache else ()
        graph.update(data[:cfg['batch']],times[:cfg['batch']],*labels)
    torch.cuda.synchronize()
    result=dict(problem=p.__dict__,method=spec,baseline_allocated_bytes=baseline,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                additional_peak_bytes=torch.cuda.max_memory_allocated()-baseline,
                final_reserved_bytes=torch.cuda.memory_reserved(),
                scope='Fresh process: actor, teacher, training data, local cache where applicable, optimizer and captured graphs. CUDA driver/context and external desktop GPU allocations excluded. No validation graph.')
    out=root/'isolated_memory'/f'{p.family}_h{p.horizon}_{spec["name"]}.json'
    write(out,result)


def timing(root,cfg,task,dim):
    preparation_started=time.perf_counter()
    n=dim//2 if task['family']=='mechanical' else dim
    p=problem(task['family'],n,task['horizon'])
    policy=actor(p,teacher(p),cfg['width']).cuda()
    checkpoint=train_dir(root,task,cfg['seeds'][0])/'fresh.pt'
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False)['state_dict']
    policy.net.load_state_dict({k[4:]:v for k,v in saved.items() if k.startswith('net.')})
    x=sample(p,1,68303)
    action=torch.zeros(1,p.n,device='cuda')
    action_time=torch.zeros(1,device='cuda',dtype=torch.long)
    def call_action():
        with torch.no_grad(): action.copy_(policy(x,action_time))
    action_graph=capture(call_action)
    costs=torch.zeros(1,device='cuda')
    def call_plan():
        with torch.no_grad(): costs.copy_(rollout(x,policy,p))
    plan_graph=capture(call_plan)
    x64=x.double();costs64=torch.zeros(1,device='cuda',dtype=torch.float64)
    def call_plan64():
        with torch.no_grad():costs64.copy_(rollout(x64,policy,p))
    plan64_graph=capture(call_plan64)
    torch.cuda.synchronize()
    preparation_seconds=time.perf_counter()-preparation_started
    result=dict(problem=p.__dict__,action=time_replays(action_graph,500),plan=time_replays(plan_graph,20),
                plan_float64=time_replays(plan64_graph,20),gpu_preparation_seconds=preparation_seconds,
                scope='Synchronized CUDA graph replay, batch one, input already resident on GPU; action time is a mutable device input, measured at time zero. Actor arithmetic is float32. Plan uses float32 state/cost; plan_float64 uses the primary float64 state/cost definition. Both include full-horizon cost calculation. GPU preparation includes policy loading and graph capture, separately from replay.')
    with torch.no_grad():
        eager32=rollout(x,policy,p)
        eager64=rollout(x64,policy,p)
    graph_errors={label:((captured-eager).abs()/(1+eager.abs())).max().item()
                  for label,captured,eager in [('float32',costs,eager32),('float64',costs64,eager64)]}
    assert all(value<1e-6 for value in graph_errors.values()),(p,graph_errors)
    result['batch_one_graph_vs_eager']=graph_errors
    states=sample(p,128,68304)
    with torch.no_grad():
        f32=rollout(states,policy,p).double()
        f64=rollout(states.double(),policy,p)
    result['precision']=dict(mean_float32=f32.mean().item(),mean_float64=f64.mean().item(),
        max_scaled_difference=((f32-f64).abs()/(1+f64.abs())).max().item(),
        mean_relative_difference=(f32.mean()/f64.mean()-1).item())
    with torch.no_grad():
        single=torch.cat([rollout(states[i:i+1].double(),policy,p).cpu() for i in range(8)])
    batched=f64[:8].cpu()
    assert torch.isfinite(single).all() and torch.isfinite(batched).all()
    result['batch_sensitivity']=dict(source_seed=68304,count=8,source_batch_size=128,
        single_costs=single.tolist(),batched_costs=batched.tolist(),
        max_scaled_difference=((single-batched).abs()/(1+batched.abs())).max().item(),
        relative_mean_difference=(single.mean()/batched.mean()-1).item(),
        scope='Same first eight diagnostic states, first-seed Fresh policy, and primary float64 plant; compare separate batch-one rollouts with their outputs in the original batch of 128. This is a numerical deployment check, not a new quality benchmark.')
    if p.family=='building':
        from threadpoolctl import threadpool_limits
        from .building import solve_qp
        from .building_audit import numpy_cost
        import numpy as np
        initial_states=sample(p,8,68305,device='cpu').double().numpy()
        with threadpool_limits(limits=1):
            prepared=solve_qp(initial_states[0],p,return_workspace=True)
            workspace=prepared.pop('_workspace')
            solver=workspace['solver']
            lower,upper=workspace['lower'],workspace['upper']
            times,qp_costs,replay_errors=[],[],[]
            for xx in initial_states:
                started=time.perf_counter()
                lower[:p.n]=-xx
                upper[:p.n]=-xx
                solver.update(l=lower,u=upper)
                solved=solver.solve()
                controls=solved.x[(p.horizon+1)*p.n:].reshape(p.horizon,p.n).clip(0,p.bound)
                times.append(time.perf_counter()-started)
                assert solved.info.status_val==1
                actual=numpy_cost(xx,controls,p)
                qp_costs.append(actual)
                replay_errors.append(abs(actual-solved.info.obj_val))
            result['reused_qp']=dict(median_seconds=statistics.median(times),sample_seconds=times,
                preparation_seconds=prepared['setup_seconds'],mean_cost=float(np.mean(qp_costs)),
                max_objective_replay_error=max(replay_errors),
                scope='CPU, one native thread, reusable OSQP matrix factorization and warm starts. Timed bound update, solve and control extraction for eight distinct initial states. Model setup and independent cost replay excluded from steady-state solve time and reported separately.')
    else:
        from threadpoolctl import threadpool_limits
        prepare_started=time.perf_counter()
        from .feedback_native import solve as feedback_solve
        initial_states=sample(p,8,68307,device='cpu').double().numpy()
        warm_state=sample(p,1,68306,device='cpu').double().numpy()[0]
        with threadpool_limits(limits=1):
            feedback_solve(warm_state,p,maxiter=cfg['native_iterations'])
            cpu_preparation=time.perf_counter()-prepare_started
            records=[]
            for state in initial_states:
                started=time.perf_counter()
                solved=feedback_solve(state,p,maxiter=cfg['native_iterations'])
                records.append(dict(seconds=time.perf_counter()-started,cost=solved['cost'],
                    nit=solved['nit'],nfev=solved['nfev'],success=solved['success']))
        with torch.no_grad():neural_cost=rollout(torch.from_numpy(initial_states).cuda(),policy,p).cpu()
        result['single_start_feedback']=dict(median_seconds=statistics.median(r['seconds'] for r in records),
            mean_cost=statistics.mean(r['cost'] for r in records),matched_neural_mean_cost=neural_cost.mean().item(),
            preparation_seconds=cpu_preparation,records=records,
            scope='CPU, one thread, zero-reference LQR tracking feedback with at most 2048 L-BFGS-B iterations on eight distinct initial states. Gain setup and one warmup solve are excluded from steady-state timing and reported separately. This is a single-start solver; it is not the full multistart/hybrid quality reference. Matched neural costs use these same eight states and the first seed fresh policy.')
    write(root/'timing'/f'{p.family}_h{p.horizon}_d{dim}.json',result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='experiments/results/hj_cotangent_primary_v6')
    parser.add_argument('--task',type=int)
    parser.add_argument('--method')
    parser.add_argument('--dimension',type=int)
    args=parser.parse_args()
    torch.set_num_threads(1)
    root=Path(args.root)
    cfg=verify(root)
    if args.task is not None:
        task=cfg['tasks'][args.task]
        if args.method:
            spec=next(s for s in task['methods'] if s['name']==args.method)
            memory(root,cfg,task,spec)
        else:
            timing(root,cfg,task,args.dimension)
        return
    # Added before fresh test access after the long-horizon conditioning audit.
    # Complete this CPU reference before any isolated deployment timing starts.
    if not (root/'feedback_complete.json').exists():
        subprocess.run([sys.executable,'-B','-m','experiments.hj_cotangent.feedback_study',
                        '--root',str(root),'--workers','4'],check=True)
    for index,task in enumerate(cfg['tasks']):
        for spec in task['methods']:
            if spec.get('warm_start') and not spec.get('teacher_parent') or spec.get('covector_signal'):
                continue
            target=root/'isolated_memory'/f'{task["family"]}_h{task["horizon"]}_{spec["name"]}.json'
            if not target.exists():
                subprocess.run([sys.executable,'-B','-m','experiments.hj_cotangent.isolation','--root',str(root),
                                '--task',str(index),'--method',spec['name']],check=True)
        for dim in [32,256]:
            target=root/'timing'/f'{task["family"]}_h{task["horizon"]}_d{dim}.json'
            if not target.exists():
                subprocess.run([sys.executable,'-B','-m','experiments.hj_cotangent.isolation','--root',str(root),
                                '--task',str(index),'--dimension',str(dim)],check=True)


if __name__=='__main__':
    main()
