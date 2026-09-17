"""Reevaluate final accepted teachers and stored global-gradient labels."""
import argparse
import gc
import json
from pathlib import Path
import torch
from experiments.hj_cotangent.systems import problem,actor,teacher,sample
from experiments.hj_gridfree.engine import write
from .trajectory import TrajectoryGraph
from .transport import collect
from .study import verify,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    torch.set_num_threads(1);root=Path(args.root);cfg=verify(root);out=root/'teacher_audit';out.mkdir(exist_ok=True)
    for task in cfg['tasks']:
        if (out/(task['family']+'.json')).exists():continue
        p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
        net=actor(p,teacher(p),cfg['width']).cuda();graph=TrajectoryGraph(net,p,cfg['batch'],parameter_gradient=True)
        rows=[];raw=[]
        for seed in cfg['seeds']:
            folder=root/'training'/f'{p.family}_h{p.horizon}_s{seed}'
            initial=sample(p,cfg['training_initial_states'],110000+seed)
            for spec in task['methods']:
                if spec['mode']!='transport':continue
                path=folder/(spec['name']+'.pt');before=digest(path)
                saved=torch.load(path,map_location='cuda',weights_only=False)
                record=json.loads(path.with_suffix('.json').read_text())
                net.load_state_dict(saved['final_teacher_state']);cost,pool,g0=collect(graph,initial,p,cfg['batch'])
                mean_error=abs(cost.mean().item()-record['incumbent_training_mean'])/(1+abs(record['incumbent_training_mean']))
                assert torch.isfinite(cost).all() and mean_error<1e-6,(path,mean_error)
                grad_error=None
                if spec.get('variance_reduction'):
                    reference=saved['final_teacher_gradient']
                    grad_error=((g0-reference).double().norm()/(1+reference.double().norm())).item()
                    assert torch.isfinite(g0).all() and torch.isfinite(reference).all() and grad_error<1e-5,(path,grad_error)
                assert digest(path)==before
                rows.append(dict(family=p.family,seed=seed,method=spec['name'],training_mean=cost.mean().item(),
                    scaled_mean_replay_error=mean_error,scaled_global_gradient_replay_error=grad_error,passed=True))
                raw.append(dict(seed=seed,method=spec['name'],costs=cost.cpu(),global_gradient=g0.cpu(),
                    saved_global_gradient=saved['final_teacher_gradient'].cpu() if spec.get('variance_reduction') else None))
                del saved,pool,cost,g0
        write(out/(p.family+'.json'),rows);torch.save(raw,out/(p.family+'.pt'))
        print(json.dumps(dict(family=p.family,teachers=len(rows),passed=True)),flush=True)
        del net,graph,raw;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
