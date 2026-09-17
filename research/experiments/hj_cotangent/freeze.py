"""Freeze prospective v6 primary training after completed development audits."""
import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import torch
from experiments.hj_gridfree.engine import write
from .study import digest


def main():
    root=Path('experiments/results/hj_cotangent_primary_v6')
    root.mkdir(exist_ok=True)
    assert not (root/'training_lock.json').exists(),'An immutable primary lock already exists'
    config=Path('experiments/configs/hj_cotangent_primary_v6.json')
    cfg=json.loads(config.read_text())
    assert not cfg.get('development')
    development=Path('experiments/results/hj_cotangent_dev_v6c')
    state=json.loads((development/'pipeline_status.json').read_text())
    assert state['stage']=='preprimary_development_complete'
    audit=development/'final_numerical_audit.json'
    assert len(json.loads(audit.read_text()))==3
    promoted=Path('experiments/results/hj_promoted_cache_dev_v6/training/mechanical_h160_s67051/complete.json')
    assert promoted.exists(),'Complete the matched-parent development check first'
    sources=[Path('experiments/hj_cotangent')/name for name in
             ['__init__.py','core.py','systems.py','building.py','study.py','freeze.py','final_audit.py']]
    sources += [Path('experiments/hj_gridfree/core.py'),Path('experiments/hj_gridfree/engine.py')]
    audited_hashes=json.loads((development/'audit_source_hashes.json').read_text())
    for path in sources:
        if str(path) in audited_hashes:assert digest(path)==audited_hashes[str(path)],path
    for path in sources:
        target=root/'source'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    shutil.copy2(config,root/'config.json')
    shutil.copy2('docs/hj_v6_protocol.md',root/'protocol.md')
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                     torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
                     device_memory=torch.cuda.get_device_properties(0).total_memory,
                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                     float32_matmul_precision=torch.get_float32_matmul_precision(),
                     packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),
                     nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
    write(root/'environment.json',environment)
    old=[Path('build/hj_gridfree_release_v4/release_manifest.json'),Path('build/hj_gridfree_release_v5/release_manifest.json'),
         Path('experiments/results/hj_extension_v4/report/summary.json')]
    lock=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              config_sha256=digest(root/'config.json'),protocol_sha256=digest(root/'protocol.md'),
              source_hashes={str(p):digest(p) for p in sources},development_audit_sha256=digest(audit),
              environment_sha256=digest(root/'environment.json'),prior_records={str(p):digest(p) for p in old},
              planned_fits=sum(len(t['methods']) for t in cfg['tasks'])*len(cfg['seeds']),
              planned_updates_per_phase=cfg['steps'],fresh_test_accessed=False)
    write(root/'training_lock.json',lock)
    print(json.dumps(lock,indent=2),flush=True)


if __name__=='__main__':main()
