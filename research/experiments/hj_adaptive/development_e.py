"""Final bounded development: eight teacher-anchor control-variate variants."""
import copy
import datetime
import json
from pathlib import Path
import shutil
from experiments.hj_gridfree.engine import write
from .prepare import A,complete
from .study import digest


def main():
    cfg=complete(A);complete(Path('experiments/results/hj_adaptive_dev_v7d'))
    root=Path('experiments/results/hj_adaptive_dev_v7e');root.mkdir(parents=True,exist_ok=True)
    assert json.loads((root/'variance_cuda_audit.json').read_text())['passed']
    assert not (root/'config.json').exists()
    methods=[]
    for warm in [False,True]:
        for label,lr in [('002',.002),('0005',.0005)]:
            for period in [32,256]:
                spec=dict(name=f'cv_{"warm" if warm else "cold"}_{period}_{label}',mode='transport',lr=lr,
                    student_batch=1024,refresh_every=period,variance_reduction=True)
                if warm:spec['parent']='short_002'
                methods.append(spec)
    for task in cfg['tasks']:task['methods']=copy.deepcopy(methods)
    write(root/'config.json',cfg);shutil.copyfile('docs/hj_v7_variance_plan.md',root/'protocol.md')
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
        note='Final48 control-variate development phases. Same states, parents, rates and fixed intervals as C/D; no new primary tests.'))
    print('FIFTH DEVELOPMENT PREPARED: 48 NEW PHASES')


if __name__=='__main__':main()
