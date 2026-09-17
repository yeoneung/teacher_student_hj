"""Serial final baseline audit/development after E; then pre-primary replay checks."""
import json
from pathlib import Path
import time
from .development_pipeline import run


def main():
    root=Path('experiments/results/hj_adaptive_dev_v7e');log=Path('build/hj_variance_development_e.log');start=time.monotonic()
    while not log.exists() or 'V7 TRAINING COMPLETE' not in log.read_text(errors='replace'):
        assert time.monotonic()-start<5400
        if log.exists():assert 'Traceback (most recent call last)' not in log.read_text(errors='replace')
        time.sleep(15)
    target=Path('experiments/results/hj_adaptive_dev_v7f_controls')
    run('hj_full_batch_audit','audit_full_batch','--output',str(target/'full_batch_audit.json'))
    if not (target/'config.json').exists():run('hj_development_f_prepare','development_f')
    run('hj_full_batch_development_f','study','--root',str(target))
    for folder in [root,target]:
        run('hj_'+folder.name+'_validation_replay','validation_audit','--root',str(folder))
    run('hj_variance_teacher_replay','teacher_audit','--root',str(root))
    print('ALL DEVELOPMENT AND PRE-PRIMARY REPLAY CHECKS COMPLETE; PRIMARY NOT STARTED',flush=True)


if __name__=='__main__':main()
