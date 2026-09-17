"""Isolate prefix-objective mismatch with identical teacher/student policies."""
import argparse
import datetime
import gc
import json
from pathlib import Path
import torch
from experiments.hj_cotangent.systems import problem,teacher,actor,sample
from experiments.hj_gridfree.engine import write
from .diagnostics import ObjectiveGradient
from .study import verify,digest


def lock(root):
    verify(root);target=root/'prefix_equality_audit_lock.json'
    assert not target.exists()
    assert not (root/'evaluation_lock.json').exists(),'Declare before fresh evaluation'
    source=Path(__file__).resolve().relative_to(Path.cwd())
    plan=Path('docs/hj_v7_prefix_equality_audit_plan.md')
    write(target,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_training_lock_sha256=digest(root/'training_lock.json'),
        sources={str(source):digest(source),str(plan):digest(plan)},
        policies=60,initial_states=512,batch=128,prefix_steps=16,
        value_scaled_tolerance=1e-6,
        scope='Supplementary training-state diagnostic declared during primary training, before fresh evaluation. Identical feedback policies isolate omitted later parameter effects. No optimization, selection or primary-decision change.'))
    print('Prefix-equality diagnostic source and plan locked',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--lock',action='store_true');args=parser.parse_args()
    root=Path(args.root)
    if args.lock:lock(root);return
    torch.set_num_threads(1);cfg=verify(root)
    declared=json.loads((root/'prefix_equality_audit_lock.json').read_text())
    assert digest(root/'training_lock.json')==declared['original_training_lock_sha256']
    for path,sha in declared['sources'].items():assert digest(path)==sha,path
    assert cfg['training_initial_states']==declared['initial_states']
    assert cfg['batch']==declared['batch'] and cfg['length']==declared['prefix_steps']
    out=root/'prefix_equality_diagnostics';out.mkdir(exist_ok=True)
    for task in cfg['tasks']:
        if (out/(task['family']+'.json')).exists() and (out/(task['family']+'.pt')).exists():continue
        p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
        base=actor(p,teacher(p),cfg['width']).cuda().eval()
        net=actor(p,teacher(p),cfg['width']).cuda().eval()
        for parameter in base.parameters():parameter.requires_grad_(False)
        prefix=ObjectiveGradient(net,base,p,cfg['batch'],cfg['length'],'fresh')
        full=ObjectiveGradient(net,base,p,cfg['batch'],cfg['length'],'dpc')
        rows=[];vectors=[]
        for seed in cfg['seeds']:
            initial=sample(p,cfg['training_initial_states'],110000+seed)
            folder=root/'training'/f'{p.family}_h{p.horizon}_s{seed}'
            for spec in task['methods']:
                if spec['mode']!='transport':continue
                path=folder/(spec['name']+'.pt');sha=digest(path)
                saved=torch.load(path,map_location='cuda',weights_only=False)['budgets']['60']['state_dict']
                base.load_state_dict(saved);net.load_state_dict(saved)
                pg=[];fg=[];pv=[];fv=[]
                for offset in range(0,len(initial),cfg['batch']):
                    x=initial[offset:offset+cfg['batch']]
                    time=torch.zeros(len(x),device='cuda',dtype=torch.long)
                    g,j=prefix(x,time);h,k=full(x,time)
                    pg.append(g);fg.append(h);pv.append(j);fv.append(k)
                g=torch.stack(pg).mean(0);h=torch.stack(fg).mean(0)
                pc=torch.tensor(pv,dtype=torch.float64);fc=torch.tensor(fv,dtype=torch.float64)
                finite=bool(torch.isfinite(g).all() and torch.isfinite(h).all())
                finite_values=bool(torch.isfinite(pc).all() and torch.isfinite(fc).all())
                value_error=((pc-fc).abs()/(1+fc.abs())).max().item() if finite_values else None
                row=dict(family=p.family,seed=seed,method=spec['name'],checkpoint_sha256=sha,
                    initial_states=len(initial),prefix_steps=cfg['length'],finite_gradients=finite,
                    finite_values=finite_values,max_scaled_batch_value_difference=value_error,
                    equal_values_within_tolerance=finite_values and value_error<declared['value_scaled_tolerance'],
                    scope='Both feedback networks have the same selected weights. Only the first16 decisions depend directly on student parameters in the prefix objective; the continuation teacher remains frozen. All512 original training initial states, no test states.')
                if finite:
                    gn=g.norm().item();hn=h.norm().item();dot=(g*h).sum().item()
                    row.update(prefix_gradient_norm=gn,full_gradient_norm=hn,
                        gradient_cosine=dot/max(gn*hn,1e-24),
                        relative_to_full_gradient_error=(g-h).norm().item()/max(hn,1e-12),
                        prefix_negative_gradient_is_full_descent=dot>0)
                assert digest(path)==sha
                for policy in [base,net]:assert all(torch.equal(v,saved[k]) for k,v in policy.state_dict().items())
                rows.append(row);vectors.append(dict(seed=seed,method=spec['name'],initial=initial.cpu(),
                    prefix_gradient=g,full_gradient=h,prefix_batch_values=pc,full_batch_values=fc))
                print(json.dumps(row),flush=True)
                del saved,pg,fg,pv,fv
        write(out/(p.family+'.json'),rows);torch.save(vectors,out/(p.family+'.pt'))
        del base,net,prefix,full,rows,vectors;gc.collect();torch.cuda.empty_cache()
    print('PREFIX-EQUALITY DIAGNOSTIC COMPLETE; all discrepancies retained',flush=True)


if __name__=='__main__':main()
