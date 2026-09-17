"""Serial GPU development jobs with durable status and separate logs."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('experiments/results/hj_cotangent_dev_v6b')
    logs = root/'stage_logs'
    logs.mkdir(parents=True,exist_ok=True)
    def status(stage, **kwargs):
        value = dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
        (root/'pipeline_status.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
        print(json.dumps(value),flush=True)
    markers = [Path('experiments/results/hj_cotangent_dev_v6/training')/f'{f}_h160_s67011'/'complete.json'
               for f in ['mechanical','reaction']]
    status('waiting_for_initial_development')
    while not all(p.exists() for p in markers):
        time.sleep(5)
    time.sleep(3)
    jobs = [('audit','experiments.hj_cotangent.audit',[]),
            ('building_development','experiments.hj_cotangent.study',['--root','experiments/results/hj_building_dev_v6']),
            ('long_horizon_tuning','experiments.hj_cotangent.study',['--root','experiments/results/hj_cotangent_dev_v6b'])]
    for name,module,arguments in jobs:
        if (logs/(name+'.complete.json')).exists():
            continue
        status(name)
        started = time.perf_counter()
        with (logs/(name+'.log')).open('w',encoding='utf-8') as output:
            result = subprocess.run([sys.executable,'-B','-m',module,*arguments],stdout=output,stderr=subprocess.STDOUT)
        record = dict(returncode=result.returncode,seconds=time.perf_counter()-started)
        if result.returncode:
            status('failed',job=name,**record)
            raise RuntimeError(f'Inspect {logs/(name+".log")}')
        (logs/(name+'.complete.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    status('development_complete')


if __name__ == '__main__':
    main()
