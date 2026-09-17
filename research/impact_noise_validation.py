"""Independent five-teacher validation of conditional local margin predictions."""
import copy,gc,json
import torch
from impact_core import HERE,load_policy,write,digest,ValueGraph,exact_probe
from impact_learning import student_states
from experiments.hj_cotangent.systems import problem,sample


def main():
    torch.set_num_threads(1);out=HERE/'results/noise_validation';out.mkdir(parents=True,exist_ok=True)
    calibration=json.loads((HERE/'results/exact_diagnostic/report.json').read_text())['rows']
    kappas={f:1.5*next(r for r in calibration if r['family']==f and r['mode']=='fresh')['kappa_quantiles'][2] for f in ['mechanical','reaction']}
    cfg=dict(teacher_seeds=[9503,9504,9505,9506,9507],qualities=[7.5,15.,30.],points=32,
        perturbations=[['none',0.],['bias',.5],['gaussian',.5],['bias',1.5],['gaussian',1.5]],
        eta=16.,kappa_calibration=kappas,kappa_rule='1.5 times 95th percentile from earlier two-trajectory weak-teacher development diagnostic',
        scope='Independent local-action diagnostic; no student retraining, certified curvature or population guarantee. Five teacher/data replications.')
    write(out/'plan_lock.json',dict(config=cfg,source_sha256=digest(__file__),calibration_sha256=digest(HERE/'results/exact_diagnostic/report.json')))
    rows=[]
    for fi,family in enumerate(['mechanical','reaction']):
        p=problem(family,16 if family=='mechanical' else 32,160 if family=='mechanical' else 320)
        for seed in cfg['teacher_seeds']:
            weak,_=load_policy(p,seed,'parent',7.5)
            teacher,_=load_policy(p,seed,'short' if family=='mechanical' else 'warm',7.5)
            jet=ValueGraph(teacher,p,32,True);value=ValueGraph(teacher,p,32,False)
            initial=sample(p,4,51100000+seed+fi*100000).double()
            gen=torch.Generator(device='cuda').manual_seed(seed+71)
            x,t=student_states(weak,p,initial,32,gen)
            for quality in cfg['qualities']:
                source,provenance=load_policy(p,seed,'short' if family=='mechanical' else 'warm',quality)
                teacher.load_state_dict(source.state_dict());del source
                for kind,magnitude in cfg['perturbations']:
                    raw=exact_probe(teacher,weak,p,x,t,jet,value,16.,magnitude,kind,seed+17)
                    r=p.dt*(.04 if p.mechanical else .1)/p.n;rho=32*r
                    bound=-(r+rho-.5*kappas[family])*raw['dnorm'].square()+raw['sensitivity_error']*raw['dnorm']
                    naive=raw['target_model'] < -1e-8;guard=bound < -1e-8;bad=raw['actual_target']>1e-8
                    row=dict(family=family,teacher_seed=seed,teacher_quality_budget=quality,noise=kind,magnitude=magnitude,
                        points=32,naive_predictions=int(naive.sum()),naive_false_positives=int((naive&bad).sum()),
                        guarded_predictions=int(guard.sum()),guarded_false_positives=int((guard&bad).sum()),
                        actual_improvements=int((raw['actual_target'] < -1e-8).sum()),
                        mean_actual_target_advantage=float(raw['actual_target'].mean()),
                        bound_violations=int((raw['actual_target']>bound+1e-7).sum()),
                        source_sha256=provenance['sha256'])
                    key=f'{family}_s{seed}_q{quality:g}_{kind}{magnitude:g}'
                    torch.save(dict(raw={k:v.cpu() for k,v in raw.items()},empirical_bound=bound.cpu()),out/(key+'.pt'))
                    rows.append(row)
            print(json.dumps(dict(family=family,seed=seed,cases=len(rows))),flush=True)
            del jet,value,weak,teacher;gc.collect();torch.cuda.empty_cache()
    write(out/'report.json',dict(config=cfg,rows=rows,completed=True))


if __name__=='__main__':main()
