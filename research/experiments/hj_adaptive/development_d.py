"""Bounded warm-start replication of all eight transport configurations."""
import copy
import datetime
import json
from pathlib import Path
import shutil
from experiments.hj_gridfree.engine import write
from .prepare import A,complete
from .study import digest


def main():
    previous=Path('experiments/results/hj_adaptive_dev_v7c');cfg=complete(previous)
    root=Path('experiments/results/hj_adaptive_dev_v7d');root.mkdir(parents=True,exist_ok=True)
    assert not (root/'config.json').exists()
    for task in cfg['tasks']:
        methods=[]
        for source in task['methods']:
            if source['mode']!='transport':continue
            spec=copy.deepcopy(source);spec['name']='warm_'+spec['name'];spec['parent']='short_002';methods.append(spec)
        task['methods']=methods
    write(root/'config.json',cfg);shutil.copyfile('docs/hj_v7_transport_plan.md',root/'protocol.md')
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
        note='48 warm transport child phases. Same charged parents as Warm DPC. No new parameter settings or fresh tests.'))
    print('FOURTH DEVELOPMENT PREPARED: 48 NEW PHASES')


if __name__=='__main__':main()
