"""Prespecified descriptive strata and policy-conditioned target geometry."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem,teacher,actor
from experiments.hj_adaptive.trajectory import TrajectoryGraph
from experiments.hj_adaptive.transport import collect
from experiments.hj_proximal.targets import coefficients
from experiments.hj_proximal.study import sha,utc
from .study import folder,verify


def strata(roots):
    rows=[]
    for root in roots:
        cfg=json.loads((root/'config.json').read_text());final=str(max(cfg['budgets_seconds']))
        if cfg['stage']=='controls':continue
        for path in (root/'evaluation').glob('reaction*.pt'):
            raw=torch.load(path,map_location='cpu',weights_only=False);meta=json.loads(path.with_suffix('.json').read_text())
            high=raw['x'].mean(-1).abs()>=.8
            groups={'high_coherence':high,'other':~high}
            methods={}
            for name,budgets in raw['methods'].items():
                key='7.5' if name=='parent' else final;cost=budgets[key];parts={}
                for group,mask in groups.items():
                    count=int(mask.sum());fraction=count/len(mask)
                    parts[group]=dict(count=count,fraction=fraction,conditional_mean=float(cost[:,mask].mean()) if count else None,
                        seed_means=cost[:,mask].mean(1).tolist() if count else None,
                        contribution_to_mean=float(cost[:,mask].sum()/cost.numel()))
                reconstructed=sum(row['contribution_to_mean'] for row in parts.values())
                assert abs(reconstructed-float(cost.mean()))<1e-10
                methods[name]=parts
            rows.append(dict(root=str(root),case=meta['key'],methods=methods,
                scope='Input-defined |spatial mean initial state| >= .8. Exploratory conditional means, not a reachability or causal certificate.'))
    return rows


def geometry(root):
    cfg=verify(root);rows=[]
    for task in cfg['tasks']:
        p=problem(task['family'],task['nodes'],task['horizon'])
        source=root/'evaluation'/f'{p.family}_train32_h{p.horizon}_eval32_nominal.pt'
        initial=torch.load(source,map_location='cpu',weights_only=False)['x'][:128].cuda().float()
        policy=actor(p,teacher(p),cfg['width']).cuda();graph=TrajectoryGraph(policy,p,64)
        for spec in task['methods']:
            name=spec['name'];budget='7.5' if spec.get('parent_only') else '30.0'
            for seed in cfg['seeds']:
                bundle=torch.load(folder(root,task,seed)/(name+'.pt'),map_location='cuda',weights_only=False)['budgets'][budget]
                policy.load_state_dict(bundle['state_dict'])
                costs,cache=collect(graph,initial,p,64);old=cache[2].double();r,b=coefficients(p,cache[1],cache[4].double())
                rho=16*2*r;unconstrained=(rho*old-b)/(2*r+rho)
                if p.family=='building':
                    active=(unconstrained<0)|(unconstrained>p.bound)
                    info=dict(coordinate_bound_activity=float(active.double().mean()))
                else:
                    weights=(unconstrained.square().mean(-1).sqrt()/p.bound).clamp_min(1.)
                    info=dict(active_target_fraction=float((weights>1+1e-10).double().mean()),
                        multiplier_weight_quantiles=torch.quantile(weights,torch.tensor([.5,.95,.99,1.],device='cuda',dtype=torch.float64)).tolist())
                rows.append(dict(family=p.family,seed=seed,method=name,diagnostic_regularization=16.,
                    teacher_precision='float32 trajectory differentiation, float64 local quadratic arithmetic',**info))
        del policy,graph
    return rows


def main():
    torch.set_num_threads(1)
    roots=[Path('experiments/results/hj_target_'+name+'_v9') for name in ['primary','controls','scaling']]
    for root in roots:
        assert (root/'report.json').exists();verify(root)
    rows=strata(roots);geo=geometry(roots[0])
    # Analytic homogeneous scalar reference, not the general ring's reachability.
    roots_changed=np.roots([.4,0.,-1.2,.72]).real.tolist()
    result=dict(completed_utc=utc(),plan_sha256=sha('docs/hj_v9_reaction_diagnostic_plan.md'),
        source_sha256=sha(__file__),strata=rows,geometry=geo,
        homogeneous_reference=dict(maximum_positive_reaction_drift=.8,location=1.,changed_uniform_equilibria=sorted(roots_changed)),
        scope='Secondary descriptive diagnostics. No primary policy was selected or retrained using these results.')
    write('build/hj_v9_diagnostics.json',result)
    print(json.dumps(dict(strata_cases=len(rows),geometry_policies=len(geo))),flush=True)


if __name__=='__main__':main()
