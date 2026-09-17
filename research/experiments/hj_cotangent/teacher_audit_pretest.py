"""CPU development check of analytic-teacher evaluation before primary tests."""
import datetime
import json
from pathlib import Path

import torch

from experiments.hj_gridfree.core import LQR
from .building import Thermostat
from .systems import problem,sample,rollout
from .evaluation import teacher_evaluation
from .study import digest


def main():
    torch.set_num_threads(1)
    root=Path('experiments/results/hj_cotangent_primary_v6')
    assert not (root/'evaluation_lock.json').exists(),'Do not rewrite a pretest audit after test locking.'
    records=[]
    for family,horizon,dimension in [('mechanical',320,32),('mechanical',320,256),
                                     ('reaction',320,32),('building',192,32)]:
        p=problem(family,dimension//2 if family=='mechanical' else dimension,horizon)
        base=Thermostat(p,'cpu') if family=='building' else LQR(p,'cpu')
        x=sample(p,128,694001,device='cpu').double()
        cost,raw,meta=teacher_evaluation(x,p,8,base=base)
        with torch.no_grad():original=rollout(x,base,p)
        assert torch.equal(cost,original),'Keeping audit trajectories changed the original teacher costs.'
        assert raw['states'].shape==(8,horizon+1,dimension)
        assert torch.equal(raw['states'][:,0],x[:8])
        assert raw['controls'].shape==(8,horizon,p.n)
        assert raw['feedback_costs'].shape==(128,)
        record=dict(family=family,horizon=horizon,dimension=dimension,seed=694001,
                    mean_cost=cost.mean().item(),original_cost_bitwise_preserved=True,**meta)
        records.append(record);print(json.dumps(record),flush=True)
    sources=[Path('experiments/hj_cotangent')/name for name in
             ['teacher_audit_pretest.py','evaluation.py','conditioning.py','systems.py','building.py']]
    target=Path('experiments/results/hj_cotangent_dev_v6c/teacher_evaluation_cpu_pretest.json')
    target.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes={str(p):digest(p) for p in sources},records=records,
        scope='CPU development check; no primary initial states or learned-policy checkpoints accessed. GPU preflight separately follows primary training.'),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
