"""Finish the predeclared development work serially after round C."""
import json
from pathlib import Path
import subprocess
import sys
import time


def run(stage,module,*args):
    target=Path('build')/(stage+'.log');print('START '+stage,flush=True)
    with target.open('a',encoding='utf-8') as log:
        code=subprocess.run([sys.executable,'-B','-m','experiments.hj_adaptive.'+module,*args],stdout=log,stderr=subprocess.STDOUT).returncode
    if code:raise RuntimeError(stage+' failed; retained log '+str(target))
    print('COMPLETE '+stage,flush=True)


def main():
    log=Path('build/hj_transport_development_c.log');start=time.monotonic()
    def contents():
        data=log.read_bytes()
        return data.decode('utf-16' if data.startswith(b'\xff\xfe') else 'utf-8',errors='replace')
    while 'V7 TRAINING COMPLETE' not in contents():
        assert time.monotonic()-start<5400,'Round C did not complete within90 minutes'
        assert 'Traceback (most recent call last)' not in contents(),'Round C failed'
        time.sleep(15)
    run('hj_variance_cuda_audit','audit_variance','--device','cuda','--output','experiments/results/hj_adaptive_dev_v7e/variance_cuda_audit.json')
    rootd=Path('experiments/results/hj_adaptive_dev_v7d')
    if not (rootd/'config.json').exists():run('hj_development_d_prepare','development_d')
    run('hj_transport_development_d','study','--root',str(rootd))
    roote=Path('experiments/results/hj_adaptive_dev_v7e')
    if not (roote/'config.json').exists():run('hj_development_e_prepare','development_e')
    run('hj_variance_development_e','study','--root',str(roote))
    print('ALL DECLARED DEVELOPMENT COMPLETE; PRIMARY NOT STARTED',flush=True)


if __name__=='__main__':main()
