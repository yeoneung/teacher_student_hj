"""Complete locked GPU training, evaluation, references and reporting."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('experiments/results/hj_cotangent_primary_v6')
    logs=root/'stage_logs'
    logs.mkdir(parents=True,exist_ok=True)
    ledger={}
    def status(stage,**extra):
        record=dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        (root/'pipeline_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)
    def run(stage,module,*arguments):
        status(stage)
        started=time.perf_counter()
        # Append on resume, preserving all prior stage output.
        with (logs/(stage+'.log')).open('a',encoding='utf-8') as output:
            code=subprocess.run([sys.executable,'-B','-m',module,*arguments],stdout=output,stderr=subprocess.STDOUT).returncode
        ledger[stage]=dict(seconds=time.perf_counter()-started,returncode=code)
        (root/'stage_times.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
        if code:
            raise RuntimeError(f'{stage} failed; inspect {logs/(stage+".log")}')
    try:
        run('training','experiments.hj_cotangent.study','--root',str(root))
        run('evaluation','experiments.hj_cotangent.evaluation','--root',str(root))
        run('gpu_classical','experiments.hj_cotangent.classical','--root',str(root),'--stage','gpu')
        run('native_classical','experiments.hj_cotangent.classical','--root',str(root),'--stage','native','--workers','4')
        run('isolated_memory_and_timing','experiments.hj_cotangent.isolation','--root',str(root))
        run('report','experiments.hj_cotangent.report','--root',str(root))
    except BaseException as error:
        status('failed',error=str(error))
        raise
    status('experiments_complete',next='Independent completion audit, manuscript integration, compile/visual review and archival release')


if __name__=='__main__':main()
