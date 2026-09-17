"""Separate stale information from mismatch between two learning objectives."""
import argparse
import gc
import json
import math
from pathlib import Path
import torch
from experiments.hj_cotangent.systems import problem,teacher,actor,sample,rollout
from experiments.hj_cotangent.core import continuation
from experiments.hj_gridfree.engine import capture,write
from .study import verify


class ObjectiveGradient:
    def __init__(self,net,base,p,batch,length,mode):
        self.x=torch.zeros(batch,p.dim,device='cuda');self.t=torch.zeros(batch,device='cuda',dtype=torch.long)
        parameters=list(net.net.parameters())
        for v in parameters:
            if v.grad is None:v.grad=torch.zeros_like(v)
        self.gradient_buffers=[v.grad for v in parameters]
        self.gradient=torch.zeros(sum(v.numel() for v in parameters),device='cuda',dtype=torch.float64)
        self.value=torch.zeros((),device='cuda',dtype=torch.float64)
        def evaluate():
            for v in parameters:v.grad.zero_()
            value=(continuation(self.x,self.t,net,base,p,length) if mode=='fresh' else rollout(self.x,net,p,self.t)).mean()
            value.backward()
            with torch.no_grad():
                self.gradient.copy_(torch.cat([v.grad.reshape(-1) for v in parameters]));self.value.copy_(value)
        self.graph=capture(evaluate)

    def __call__(self,x,t):
        self.x.copy_(x);self.t.copy_(t);self.graph.replay()
        return self.gradient.detach().cpu().clone(),self.value.item()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    torch.set_num_threads(1);root=Path(args.root);cfg=verify(root)
    out=root/'objective_diagnostics';out.mkdir(exist_ok=True)
    for task in cfg['tasks']:
        if (out/(task['family']+'.json')).exists():continue
        p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
        base=(actor(p,teacher(p),cfg['width']).cuda() if task.get('teacher_parent_name') else teacher(p))
        net=actor(p,teacher(p),cfg['width']).cuda()
        for v in base.parameters():v.requires_grad_(False)
        fg=ObjectiveGradient(net,base,p,cfg['batch'],cfg['length'],'fresh')
        dg=ObjectiveGradient(net,base,p,cfg['batch'],cfg['length'],'dpc')
        rows=[];vectors=[]
        for seed in cfg['seeds']:
            folder=root/'training'/f'{p.family}_h{p.horizon}_s{seed}'
            if task.get('teacher_parent_name'):
                saved=torch.load(folder/(task['teacher_parent_name']+'.pt'),map_location='cuda',weights_only=False)
                base.load_state_dict(saved['budgets'][str(cfg['parent_budget_seconds'])]['state_dict'])
            data=torch.load(folder/'data.pt',map_location='cuda',weights_only=False)
            generator=torch.Generator(device='cuda').manual_seed(881000+seed)
            idx=torch.randperm(len(data['x']),device='cuda',generator=generator)[:cfg['batch']]
            batches=[('training_pairs',data['x'][idx],data['t'][idx]),
                ('training_initial_states',sample(p,cfg['training_initial_states'],110000+seed)[:cfg['batch']],torch.zeros(cfg['batch'],device='cuda',dtype=torch.long))]
            for spec in task['methods']:
                if spec.get('parent_only'):continue
                bundle=torch.load(folder/(spec['name']+'.pt'),map_location='cuda',weights_only=False)['budgets']['60']
                net.load_state_dict(bundle['state_dict'])
                for label,x,t in batches:
                    g,fvalue=fg(x,t);h,dvalue=dg(x,t)
                    finite=bool(torch.isfinite(g).all() and torch.isfinite(h).all())
                    row=dict(family=p.family,seed=seed,method=spec['name'],batch=label,finite_gradients=finite,
                        teacher_type='frozen_parent' if task.get('teacher_parent_name') else 'analytic',
                        teacher_prefix_value=fvalue if math.isfinite(fvalue) else None,
                        full_student_value=dvalue if math.isfinite(dvalue) else None,
                        scope='Same fixed batch and current parameters; objective discrepancy includes a fixed teacher continuation and omitted later student-parameter dependence. This is not solely teacher suboptimality.')
                    if finite:
                        gn=g.norm().item();hn=h.norm().item();dot=(g*h).sum().item()
                        row.update(fresh_gradient_norm=gn,full_gradient_norm=hn,
                            gradient_cosine=dot/max(gn*hn,1e-24),relative_to_full_gradient_error=(g-h).norm().item()/max(hn,1e-12),
                            fresh_negative_gradient_is_full_descent=dot>0)
                    rows.append(row);vectors.append(dict(seed=seed,method=spec['name'],batch=label,
                        x=x.cpu(),t=t.cpu(),fresh_gradient=g,full_gradient=h))
            del data
        torch.save(vectors,out/(p.family+'.pt'));write(out/(p.family+'.json'),rows)
        print(json.dumps(dict(family=p.family,objective_gradient_checks=len(rows))),flush=True)
        del base,net,fg,dg,vectors;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
