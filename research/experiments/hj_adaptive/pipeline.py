"""Resumable serial pipeline for the locked v7 primary experiment."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);logs=root/'stage_logs';logs.mkdir(parents=True,exist_ok=True)
    ledgerpath=root/'stage_times.json';ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else {}
    def status(stage,**extra):
        record=dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
        (root/'pipeline_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    try:
        for stage in ['study','validation_audit','teacher_audit','evaluation','diagnostics','transport_diagnostics','qp_reference','report']:
            if ledger.get(stage,{}).get('returncode')==0:continue
            status(stage);started=time.perf_counter()
            with (logs/(stage+'.log')).open('a',encoding='utf-8') as log:
                code=subprocess.run([sys.executable,'-B','-m','experiments.hj_adaptive.'+stage,'--root',str(root)],stdout=log,stderr=subprocess.STDOUT).returncode
            ledger[stage]=dict(seconds=time.perf_counter()-started,returncode=code)
            ledgerpath.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
            if code:raise RuntimeError(stage+' failed; inspect retained log')
    except BaseException as error:
        status('failed',error=str(error));raise
    status('numerical_pipeline_complete',remaining='Independent completion audit, manuscript, visual review and archive')


if __name__=='__main__':main()
