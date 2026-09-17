"""Additional strong full-set controls; no new teacher-student configuration."""
import copy
import datetime
import json
from pathlib import Path
import shutil
from experiments.hj_gridfree.engine import write
from .prepare import A,complete
from .study import digest


def main():
    cfg=complete(A);complete(Path('experiments/results/hj_adaptive_dev_v7e'))
    root=Path('experiments/results/hj_adaptive_dev_v7f_controls');root.mkdir(parents=True,exist_ok=True)
    assert json.loads((root/'full_batch_audit.json').read_text())['passed']
    assert not (root/'config.json').exists()
    methods=[]
    for warm in [False,True]:
        for label,lr in [('002',.002),('0005',.0005)]:
            spec=dict(name=f'{"warm" if warm else "dpc"}_full_{label}',mode='dpc',lr=lr,
                initial_objective=True,full_batch=True,update_block=1)
            if warm:spec['parent']='short_002'
            methods.append(spec)
    for task in cfg['tasks']:task['methods']=copy.deepcopy(methods)
    write(root/'config.json',cfg);shutil.copyfile('docs/hj_v7_full_batch_controls.md',root/'protocol.md')
    copied={}
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            name=f'{task["family"]}_h{task["horizon"]}_s{seed}';dest=root/'training'/name;dest.mkdir(parents=True,exist_ok=True)
            for file in ['data.pt','short_002.pt','short_002.json']:
                source=A/'training'/name/file;target=dest/file;assert not target.exists();shutil.copyfile(source,target)
                copied[str(target)]=dict(source=str(source),sha256=digest(source))
    snap=root/'source_at_development_start';snap.mkdir(exist_ok=True)
    for p in Path('experiments/hj_adaptive').glob('*.py'):
        target=snap/p.name;assert not target.exists();shutil.copyfile(p,target)
    write(root/'development_start.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        copied_parent_evidence=copied,sources={str(p):digest(p) for p in snap.glob('*.py')},
        note='24 strong full-initial-set DPC controls. No additional teacher-student candidate settings; every new baseline becomes a primary reference.'))
    print('FULL-SET CONTROL DEVELOPMENT PREPARED: 24 NEW PHASES')


if __name__=='__main__':main()
