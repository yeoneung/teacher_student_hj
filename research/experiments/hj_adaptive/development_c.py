"""Third development: all-time teacher covectors and matched initial objectives."""
import copy
import datetime
import json
from pathlib import Path
import shutil
from experiments.hj_gridfree.engine import write
from .prepare import A,complete
from .study import digest


def main():
    root=Path('experiments/results/hj_adaptive_dev_v7c');cfg=complete(A)
    assert not (root/'config.json').exists(),'Do not silently replace started development'
    for name in ['transport_cuda_expanded_audit.json','transport_update_audit.json']:
        assert json.loads((root/name).read_text())['passed'],name
    methods=[]
    for label,lr in [('002',.002),('0005',.0005)]:
        methods += [dict(name='dpc_initial_'+label,mode='dpc',lr=lr,initial_objective=True),
            dict(name='warm_initial_'+label,mode='dpc',lr=lr,initial_objective=True,parent='short_002')]
        for period in [32,256]:methods.append(dict(name=f'transport_fixed_{period}_{label}',mode='transport',lr=lr,student_batch=1024,refresh_every=period))
        for label2,tol in [('005',.05),('015',.15)]:methods.append(dict(name=f'transport_adaptive_{label2}_{label}',mode='transport',lr=lr,student_batch=1024,adaptive=True,probe_every=32,action_tolerance=tol,max_age=512))
    for task in cfg['tasks']:task['methods']=copy.deepcopy(methods)
    write(root/'config.json',cfg);shutil.copyfile('docs/hj_v7_transport_plan.md',root/'protocol.md')
    copied={}
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            name=f'{task["family"]}_h{task["horizon"]}_s{seed}';dest=root/'training'/name;dest.mkdir(parents=True,exist_ok=True)
            for file in ['data.pt','short_002.pt','short_002.json']:
                source=A/'training'/name/file;target=dest/file
                if target.exists():assert digest(target)==digest(source)
                else:shutil.copyfile(source,target)
                copied[str(target)]=dict(source=str(source),sha256=digest(source))
    snap=root/'source_at_development_start';snap.mkdir(exist_ok=True)
    for p in Path('experiments/hj_adaptive').glob('*.py'):
        target=snap/p.name;assert not target.exists();shutil.copyfile(p,target)
    write(root/'development_start.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        copied_parent_evidence=copied,sources={str(p):digest(p) for p in snap.glob('*.py')},
        note='Third development uses original development states and charged existing parents. Primary training/test have not started.'))
    print('THIRD DEVELOPMENT PREPARED: 72 NEW PHASES')


if __name__=='__main__':main()
