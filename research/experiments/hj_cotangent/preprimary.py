"""Final numerical audit and selected development settings, serialized on GPU."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('experiments/results/hj_cotangent_dev_v6c')
    logs=root/'stage_logs'
    logs.mkdir(exist_ok=True)
    def status(stage,**kwargs):
        item=dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
        (root/'pipeline_status.json').write_text(json.dumps(item,indent=2),encoding='utf-8')
        print(json.dumps(item),flush=True)
    status('waiting_for_baseline_tuning')
    previous=Path('experiments/results/hj_cotangent_dev_v6b/pipeline_status.json')
    while True:
        state=json.loads(previous.read_text())['stage']
        if state=='development_complete':
            break
        if state=='failed':
            raise RuntimeError('Previous development pipeline failed')
        time.sleep(5)
    time.sleep(3)
    for stage,module,args in [('audit','experiments.hj_cotangent.final_audit',[]),
                              ('selected_development','experiments.hj_cotangent.study',['--root',str(root)])]:
        status(stage)
        started=time.perf_counter()
        with (logs/(stage+'.log')).open('w',encoding='utf-8') as output:
            code=subprocess.run([sys.executable,'-B','-m',module,*args],stdout=output,stderr=subprocess.STDOUT).returncode
        if code:
            status('failed',job=stage,returncode=code)
            raise RuntimeError(f'Inspect {logs/(stage+".log")}')
        (logs/(stage+'.complete.json')).write_text(json.dumps(dict(seconds=time.perf_counter()-started)),encoding='utf-8')
    status('preprimary_development_complete')


if __name__=='__main__':
    main()
