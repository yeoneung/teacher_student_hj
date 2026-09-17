"""Full-pool gradient checks at the actual mixed-time local training batch."""
import argparse
import datetime
import gc
import json
from pathlib import Path
import torch
from experiments.hj_cotangent.systems import problem,teacher,actor,sample
from experiments.hj_gridfree.engine import capture,write
from .trajectory import TrajectoryGraph,advantage
from .transport import collect
from .study import verify,digest


class LocalGradientGraph:
    def __init__(self,net,p,batch):
        self.net=net
        self.inputs=[torch.zeros(batch,p.dim,device='cuda'),torch.zeros(batch,device='cuda',dtype=torch.long),
            torch.zeros(batch,p.n,device='cuda'),torch.zeros(batch,p.dim,device='cuda'),torch.zeros(batch,p.dim,device='cuda')]
        params=list(net.net.parameters())
        for z in params:z.grad=torch.zeros_like(z)
        self.gradient=torch.zeros(sum(z.numel() for z in params),device='cuda')
        self.loss=torch.zeros((),device='cuda')
        def backward():
            for z in params:z.grad.zero_()
            loss=p.horizon*advantage(*self.inputs,net,p).mean();loss.backward()
            with torch.no_grad():
                self.gradient.copy_(torch.cat([z.grad.reshape(-1) for z in params]));self.loss.copy_(loss)
        self.graph=capture(backward)

    def __call__(self,values):
        for target,value in zip(self.inputs,values):target.copy_(value)
        self.graph.replay()


def metrics(g,h):
    finite=bool(torch.isfinite(g).all() and torch.isfinite(h).all())
    if not finite:return dict(finite=False,relative_error=None,scaled_error=None,reference_norm=None,passed=False)
    error=(g-h).norm().item();norm=h.norm().item()
    return dict(finite=True,relative_error=error/max(norm,1e-12),scaled_error=error/(1+norm),
        reference_norm=norm,passed=error/max(norm,1e-12)<2e-4)


def make_lock(root):
    verify(root);target=root/'batch_gradient_audit_lock.json'
    assert not target.exists(),'An additional audit lock already exists'
    assert not (root/'evaluation_lock.json').exists(),'Declare this diagnostic before fresh evaluation'
    plan=Path('docs/hj_v7_batch_gradient_audit_plan.md');source=Path(__file__).resolve().relative_to(Path.cwd())
    write(target,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_training_lock_sha256=digest(root/'training_lock.json'),
        sources={str(source):digest(source),str(plan):digest(plan)},
        local_batches=[128,1024],permutation_seed_base=4820000,relative_tolerance=2e-4,
        policies=60,scope='Additional numerical diagnostic, declared during training before fresh evaluation. No change to primary settings or decisions.'))
    print('Additional batch-gradient diagnostic source and plan locked',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--lock',action='store_true');args=parser.parse_args()
    root=Path(args.root)
    if args.lock:make_lock(root);return
    torch.set_num_threads(1);cfg=verify(root);lock=json.loads((root/'batch_gradient_audit_lock.json').read_text())
    assert digest(root/'training_lock.json')==lock['original_training_lock_sha256']
    for path,sha in lock['sources'].items():assert digest(path)==sha,path
    out=root/'batch_gradient_diagnostics';out.mkdir(exist_ok=True)
    for task in cfg['tasks']:
        if (out/(task['family']+'.json')).exists() and (out/(task['family']+'.pt')).exists():continue
        p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
        base=actor(p,teacher(p),cfg['width']).cuda().eval()
        trajectory=TrajectoryGraph(base,p,cfg['batch'],parameter_gradient=True)
        local={batch:LocalGradientGraph(actor(p,teacher(p),cfg['width']).cuda().eval(),p,batch) for batch in lock['local_batches']}
        rows=[];vectors=[]
        for seed in cfg['seeds']:
            initial=sample(p,cfg['training_initial_states'],110000+seed)
            folder=root/'training'/f'{p.family}_h{p.horizon}_s{seed}'
            for spec in task['methods']:
                if spec['mode']!='transport':continue
                path=folder/(spec['name']+'.pt');sha=digest(path)
                saved=torch.load(path,map_location='cuda',weights_only=False)['budgets']['60']['state_dict']
                base.load_state_dict(saved)
                for graph in local.values():graph.net.load_state_dict(saved)
                costs,pool,anchor=collect(trajectory,initial,p,cfg['batch'])
                generator=torch.Generator(device='cuda').manual_seed(lock['permutation_seed_base']+seed)
                permutation=torch.randperm(len(pool[0]),device='cuda',generator=generator)
                full=anchor.double().cpu();means={};values={};checks={}
                for batch,graph in local.items():
                    assert len(permutation)%batch==0
                    total=torch.zeros_like(anchor,dtype=torch.float64);value=torch.zeros((),device='cuda',dtype=torch.float64)
                    for offset in range(0,len(permutation),batch):
                        index=permutation[offset:offset+batch];graph(tuple(z[index] for z in pool))
                        total.add_(graph.gradient.double());value.add_(graph.loss.double())
                    count=len(permutation)//batch;means[str(batch)]=(total/count).cpu()
                    values[str(batch)]=(value/count).item() if torch.isfinite(value) else None
                    checks[str(batch)]=metrics(means[str(batch)],full)
                    assert all(torch.equal(v,saved[k]) for k,v in graph.net.state_dict().items())
                difference=metrics(means['1024'],means['128'])
                assert digest(path)==sha
                row=dict(family=p.family,seed=seed,method=spec['name'],checkpoint_sha256=sha,
                    initial_states=len(initial),state_time_pairs=len(permutation),
                    training_cost_finite=bool(torch.isfinite(costs).all()),
                    uncorrected_local_gradient_checks=checks,mean_anchor_advantages=values,
                    batch_shape_difference=difference,all_checks_passed=all(r['passed'] for r in checks.values()),
                    scope='Selected policy reevaluated as teacher on original training states; fixed mixed-time permutation; no parameter update or test-state access.')
                rows.append(row);vectors.append(dict(seed=seed,method=spec['name'],full_gradient=full,mean_local_gradients=means))
                print(json.dumps(row),flush=True)
                del saved,costs,pool,anchor,permutation,means,total,value
        write(out/(p.family+'.json'),rows);torch.save(vectors,out/(p.family+'.pt'))
        del base,trajectory,local,rows,vectors;gc.collect();torch.cuda.empty_cache()
    print('ADDITIONAL BATCH-GRADIENT AUDIT COMPLETE; disagreements remain reported',flush=True)


if __name__=='__main__':main()
