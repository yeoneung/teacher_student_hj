"""Resumable training with complete refresh costs and validation-only selection."""
import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from experiments.hj_gridfree.engine import write
from .systems import problem, teacher as make_teacher, actor as make_actor, sample, rollout
from .core import JetGraph, LocalGraph, EvalGraph


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    cfg = json.loads((root/'config.json').read_text(encoding='utf-8'))
    lockpath = root/'training_lock.json'
    if lockpath.exists():
        lock = json.loads(lockpath.read_text(encoding='utf-8'))
        assert digest(root/'config.json') == lock['config_sha256']
        assert digest(root/'protocol.md') == lock['protocol_sha256']
        for path, expected in lock['source_hashes'].items():
            assert digest(path) == expected, path
    else:
        assert cfg.get('development'), 'Primary training requires a source lock'
    return cfg


def make_data(p, cfg, seed, out):
    if (out/'data.pt').exists():
        saved = torch.load(out/'data.pt', map_location='cuda', weights_only=False)
        return saved
    torch.cuda.synchronize()
    start = time.perf_counter()
    count = cfg['dataset']
    gen = torch.Generator(device='cuda').manual_seed(310000+seed)
    initial = sample(p, cfg['training_initial_states'], 110000+seed)
    with torch.no_grad():
        _, xs, _ = rollout(initial, make_teacher(p), p, keep=True)
        take = torch.randperm(xs.shape[0]*p.horizon, device='cuda', generator=gen)[:count//2]
        broad = sample(p, count//2, 120000+seed)
        x = torch.cat((xs.flatten(0, 1)[take], broad))
        t = torch.cat((take % p.horizon, torch.randint(p.horizon, (count//2,), device='cuda', generator=gen)))
        valx = sample(p, cfg['validation_states'], cfg.get('validation_seed_base',210000)+(0 if p.mechanical else 10000)+p.horizon)
    torch.cuda.synchronize()
    data = dict(x=x, t=t, valx=valx, data_seconds=time.perf_counter()-start)
    torch.save({k:v.cpu() if isinstance(v, torch.Tensor) else v for k,v in data.items()}, out/'data.pt')
    return data


def train_method(p, cfg, seed, spec, data, out):
    name = spec['name']
    target = out/name
    if target.with_suffix('.json').exists():
        return json.loads(target.with_suffix('.json').read_text())
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    actor = make_actor(p, make_teacher(p), cfg['width']).cuda()
    teacher = make_teacher(p)
    parent_seconds = 0.
    if spec.get('warm_start'):
        parent = out/spec['warm_start']
        parent_record = json.loads(parent.with_suffix('.json').read_text())
        parent_seconds = parent_record['compute_seconds']-data['data_seconds']
        actor.load_state_dict(torch.load(parent.with_suffix('.pt'),map_location='cuda',weights_only=False)['state_dict'])
    if spec.get('teacher_parent'):
        teacher=make_actor(p,make_teacher(p),cfg['width']).cuda()
        teacher.load_state_dict(torch.load((out/spec['teacher_parent']).with_suffix('.pt'),map_location='cuda',weights_only=False)['state_dict'])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad=None
    dx, dt, valx = data['x'], data['t'], data['valx']
    length = spec.get('length', cfg['length'])
    mode = spec['mode']
    cache = None
    jet = None
    signs = None
    signal = spec.get('covector_signal','normal')
    if signal=='sign':
        sign_gen=torch.Generator(device='cuda').manual_seed(510000+seed)
        signs=2*torch.randint(2,dx.shape,device='cuda',generator=sign_gen)-1
    lr = spec.get('lr',cfg['lr'])
    if mode == 'cache':
        trainer = LocalGraph(actor, p, cfg['batch'], length, spec.get('rho', 0.), lr,signal=signal)
        jet = JetGraph(actor, teacher, p, cfg['batch'], length)
        cache = (torch.zeros_like(dx), torch.zeros(len(dx), device='cuda'), torch.zeros_like(dx))
    else:
        trainer = LocalGraph(actor,p,cfg['batch'],length,lr=lr,mode=mode,teacher=teacher)
    # Long chaotic trajectories require the same float64 state/cost arithmetic
    # used for final evaluation. Actor layers remain float32 by design.
    evaluator = EvalGraph(actor, p, len(valx), dtype=torch.float64)
    torch.cuda.synchronize()
    setup = time.perf_counter()-started
    with torch.no_grad():
        initial = evaluator(valx).mean().item()
    best = initial
    state = copy.deepcopy(actor.state_dict())
    best_step = 0
    fit = refresh = validation = checkpoint_io = 0.
    refresh_count = 0
    jet_diagnostics = []
    curvature = torch.ones(len(dx),device='cuda')
    gen = torch.Generator(device='cuda').manual_seed(410000+seed)
    history = []
    checkpoint_dir = out/(name+'_history')
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(dict(state_dict=state, problem=asdict(p), best_step=0), checkpoint_dir/'step_000000.pt')
    history.append(dict(step=0, mean=initial, best=best, best_step=0, compute_seconds=data['data_seconds']+parent_seconds+setup,
                        fit_seconds=0., refresh_seconds=0., validation_seconds=0.))
    start = time.perf_counter()
    for k in range(1, cfg['steps']+1):
        if mode == 'cache' and (k == 1 or (k-1) % spec['refresh_every'] == 0):
            torch.cuda.synchronize()
            fit += time.perf_counter()-start
            refresh_start = time.perf_counter()
            for offset in range(0, len(dx), cfg['batch']):
                result = jet(dx[offset:offset+cfg['batch']], dt[offset:offset+cfg['batch']])
                if refresh_count:
                    aa,vv,pp = result
                    olda,oldv,oldp = [v[offset:offset+cfg['batch']] for v in cache]
                    dz,dp = aa-olda,pp-oldp
                    curvature[offset:offset+cfg['batch']] = (p.dim*dp.norm(dim=-1)/dz.norm(dim=-1).clamp_min(1e-6)).clamp(1.,1000.)
                    if offset in [0,len(dx)//2]:
                        error = vv-oldv-(oldp*dz).sum(-1)
                        norms = pp.norm(dim=-1)*oldp.norm(dim=-1)
                        valid = norms>1e-10
                        cosine = (pp*oldp).sum(-1)/norms.clamp_min(1e-10)
                        jet_diagnostics.append(dict(step=k,pool='teacher_states' if offset==0 else 'broad_states',
                            mean_boundary_rms=dz.square().mean(-1).sqrt().mean().item(),
                            meaningful_gradient_count=valid.sum().item(),
                            mean_gradient_cosine=cosine[valid].mean().item() if valid.any().item() else None,
                            mean_abs_value_error=error.abs().mean().item(),
                            mean_secant_curvature=curvature[offset:offset+cfg['batch']].mean().item()))
                for saved, value in zip(cache, result):
                    saved[offset:offset+cfg['batch']].copy_(value)
            torch.cuda.synchronize()
            refresh += time.perf_counter()-refresh_start
            refresh_count += 1
            start = time.perf_counter()
        idx = torch.randint(len(dx), (cfg['batch'],), device='cuda', generator=gen)
        if mode == 'cache':
            labels=[v[idx] for v in cache]
            trainer.update(dx[idx], dt[idx], *labels,
                           curvature=curvature[idx] if spec.get('adaptive_curvature') else None,
                           signal_covector=labels[2]*signs[idx] if signal=='sign' else None)
        else:
            trainer.update(dx[idx], dt[idx])
        if k % cfg['validation_every'] == 0 or k == cfg['steps']:
            torch.cuda.synchronize()
            fit += time.perf_counter()-start
            validation_start = time.perf_counter()
            cost = evaluator(valx)
            torch.cuda.synchronize()
            finite = torch.isfinite(cost).all().item()
            mean = cost.mean().item() if finite else None
            if finite and mean < best:
                best, best_step = mean, k
                state = copy.deepcopy(actor.state_dict())
            validation += time.perf_counter()-validation_start
            elapsed = data['data_seconds']+parent_seconds+setup+fit+refresh+validation
            grad_norm=trainer.grad_norm.item()
            row = dict(step=k, mean=mean, finite=finite, best=best, best_step=best_step,
                       compute_seconds=elapsed, fit_seconds=fit, refresh_seconds=refresh,
                       validation_seconds=validation,raw_gradient_norm=grad_norm if math.isfinite(grad_norm) else None,
                       clipped_update_fraction=trainer.clipped_updates.item()/k,
                       nonfinite_updates_skipped=int(trainer.nonfinite_updates.item()))
            history.append(row)
            io_start = time.perf_counter()
            torch.save(dict(state_dict=state, problem=asdict(p), best_step=best_step),
                       checkpoint_dir/f'step_{k:06d}.pt')
            checkpoint_io += time.perf_counter()-io_start
            print(json.dumps(dict(run=str(target), **row)), flush=True)
            if not finite or elapsed >= cfg.get('max_compute_seconds', float('inf')):
                break
            start = time.perf_counter()
    actor.load_state_dict(state)
    with torch.no_grad():
        final = rollout(valx.double(), actor, p)
    torch.cuda.synchronize()
    result = dict(problem=asdict(p), seed=seed, method=spec, initial_mean=initial,
                  teacher_type='learned_actor' if spec.get('teacher_parent') else 'analytic_controller',
                  mean=final.mean().item(), best_step=best_step, steps_completed=k,
                  setup_seconds=setup, fit_seconds=fit, refresh_seconds=refresh,
                  validation_seconds=validation, data_seconds=data['data_seconds'],
                  compute_seconds=data['data_seconds']+parent_seconds+setup+fit+refresh+validation,
                  parent_seconds=parent_seconds,
                  stopped_on_nonfinite_validation=not history[-1]['finite'],
                  nonfinite_updates_skipped=history[-1]['nonfinite_updates_skipped'],
                  checkpoint_io_seconds=checkpoint_io, refresh_count=refresh_count,
                  jet_diagnostics=jet_diagnostics,
                  peak_process_allocated_bytes=torch.cuda.max_memory_allocated(), history=history,
                  student_transition_evaluations=k*cfg['batch']*(p.horizon if mode in ['dpc','tbptt'] else 0 if mode=='bc' else length)+refresh_count*len(dx)*length,
                  teacher_transition_evaluations=(refresh_count*len(dx)*p.horizon if mode=='cache' else k*cfg['batch']*(p.horizon-length) if mode=='exact' else 0),
                  scope='Compute includes common data generation, graph setup, every cache refresh, fit and validation; excludes artifact I/O and final independent precision replay. Counts include masked fixed-graph slots.')
    torch.save(dict(state_dict=state, problem=asdict(p), best_step=best_step), target.with_suffix('.pt'))
    if spec.get('teacher_parent'):
        reference=torch.load((out/spec['teacher_parent']).with_suffix('.pt'),map_location='cuda',weights_only=False)['state_dict']
        assert all(torch.equal(value,reference[key]) for key,value in teacher.state_dict().items())
        assert all(parameter.grad is None for parameter in teacher.parameters())
    torch.save(final.cpu(), out/(name+'_validation.pt'))
    write(target.with_suffix('.json'), result)
    del actor, teacher, trainer, evaluator, jet, cache, state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--family')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--horizon', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root)
    cfg = verify(root)
    tasks = cfg.get('tasks') or [dict(family=f,horizon=h,methods=cfg['methods']) for h in cfg['horizons'] for f in cfg['families']]
    for task in tasks:
        family,horizon = task['family'],task['horizon']
        if args.horizon and args.horizon!=horizon or args.family and args.family!=family:
            continue
        for seed in [args.seed] if args.seed else cfg['seeds']:
                out = root/'training'/f'{family}_h{horizon}_s{seed}'
                if (out/'complete.json').exists():
                    continue
                out.mkdir(parents=True, exist_ok=True)
                p = problem(family, cfg['training_nodes'][family], horizon=horizon)
                data = make_data(p, cfg, seed, out)
                # Rotate execution order across seeds to avoid a fixed time-order confound.
                shift = seed % len(task['methods'])
                specs = task['methods'][shift:] + task['methods'][:shift]
                # Warm-start controls are dependent and must follow their parent.
                specs = [s for s in specs if not s.get('warm_start')]+[s for s in specs if s.get('warm_start')]
                for spec in specs:
                    train_method(p, cfg, seed, spec, data, out)
                write(out/'complete.json', dict(methods=[s['name'] for s in specs]))
                del data
                gc.collect()
                torch.cuda.empty_cache()
    print('COTANGENT TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
