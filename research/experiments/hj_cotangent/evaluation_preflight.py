"""GPU evaluation checks using development checkpoints before fresh test access."""
import gc
import json
from pathlib import Path

import torch

from experiments.hj_gridfree.engine import write
from .systems import problem,teacher,actor,sample,rollout
from .conditioning import local_trajectory_audit,independent_feedback_cost
from .study import digest


def run(root,planner_class,replay,teacher_evaluation):
    target=root/'pretest_evaluation_audit.json'
    if target.exists():
        previous=json.loads(target.read_text())
        for path,expected in previous['source_hashes'].items():assert digest(path)==expected,path
        return
    sources=[Path('experiments/hj_cotangent')/name for name in
             ['evaluation.py','evaluation_preflight.py','conditioning.py']]
    settings=[('mechanical',320,32,'exact16'),('mechanical',320,256,'exact16'),
              ('reaction',320,32,'cache512'),('building',192,32,'exact16')]
    records=[]
    for family,h,dim,name in settings:
        p=problem(family,dim//2 if family=='mechanical' else dim,h)
        source=Path('experiments/results/hj_cotangent_dev_v6c/training')/f'{family}_h{h}_s67041'/(name+'.pt')
        policy=actor(p,teacher(p),64).cuda()
        saved=torch.load(source,map_location='cuda',weights_only=False)['state_dict']
        policy.net.load_state_dict({k[4:]:v for k,v in saved.items() if k.startswith('net.')})
        policy.eval()
        x=sample(p,128,692301).double()
        planner=planner_class(policy,p,128)
        costs,controls=planner(x)
        local=local_trajectory_audit(planner.states[:8],controls[:8],costs[:8],p)
        independent=independent_feedback_cost(x,policy,p)
        feedback=((independent-costs.cpu()).abs()/(1+independent.abs())).max().item()
        with torch.no_grad():direct=rollout(x,policy,p)
        direct_error=((direct-costs).abs()/(1+costs.abs())).max().item()
        open_loop=replay(x[:8],controls[:8],p)
        open_error=((open_loop-costs[:8].cpu()).abs()/(1+open_loop.abs())).max().item()
        assert feedback<.003 and direct_error<1e-6,(family,dim,feedback,direct_error)
        teacher_cost,teacher_raw,teacher_meta=teacher_evaluation(x,p,8)
        with torch.no_grad():original_teacher=rollout(x,teacher(p),p).cpu()
        assert torch.equal(teacher_cost,original_teacher),'Teacher audit changed GPU baseline costs.'
        records.append(dict(family=family,horizon=h,dimension=dim,development_checkpoint=str(source),
            checkpoint_sha256=digest(source),seed=692301,count=128,local=local,
            independent_feedback_scaled_difference=feedback,graph_vs_eager_scaled_difference=direct_error,
            fixed_control_open_loop_scaled_difference=open_error,teacher_audit=teacher_meta,
            teacher_mean=teacher_cost.mean().item(),teacher_original_cost_bitwise_preserved=True))
        print(json.dumps(dict(pretest_evaluation_audit=records[-1])),flush=True)
        del policy,planner,x,costs,controls,direct,teacher_cost,teacher_raw
        gc.collect();torch.cuda.empty_cache()
    write(target,dict(source_hashes={str(p):digest(p) for p in sources},records=records,
                      scope='Development checkpoints and a separate diagnostic seed only; completed before primary test access.'))
