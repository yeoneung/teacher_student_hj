"""Record bounded progress observations without changing numerical inputs."""
import argparse
import datetime
import json
from pathlib import Path
import time
import subprocess
import sys


def observe(root):
    cfg=json.loads((root/'config.json').read_text())
    expected=sum(len(t['methods'])*len(cfg['seeds']) for t in cfg['tasks'])
    records=[];families={}
    for task in cfg['tasks']:
        count=0
        for seed in cfg['seeds']:
            folder=root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'
            for spec in task['methods']:
                path=folder/(spec['name']+'.json')
                if path.exists():records.append(path);count+=1
        families[task['family']]=count
    status_path=root/'pipeline_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    latest=max(records,key=lambda p:p.stat().st_mtime) if records else None
    telemetry=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
        '--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    sample=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        nvidia_smi=telemetry.stdout.strip(),returncode=telemetry.returncode,
        completed_phases=len(records),stage=status.get('stage'))
    with (root/'host_telemetry.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(sample)+'\n')
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completed_training_phases=len(records),expected_training_phases=expected,
        families=families,pipeline=status,
        latest_phase=str(latest) if latest else None,
        latest_phase_age_seconds=time.time()-latest.stat().st_mtime if latest else None,
        sampled_gpu_telemetry=sample['nvidia_smi'])


def prepare_after_primary(root):
    """Wait for the numerical pipeline, then prepare audited presentation data."""
    status_path=root/'pipeline_status.json'
    print('Waiting for the locked numerical pipeline; no extra GPU job is started',flush=True)
    while True:
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('stage')=='failed':raise RuntimeError('Primary numerical pipeline failed; retain logs and inspect the cause')
        if status.get('stage')=='numerical_pipeline_complete':break
        time.sleep(5)
    commands=[('batch_gradient_audit',[sys.executable,'-B','-m','experiments.hj_adaptive.batch_gradient_audit','--root',str(root)]),
        ('prefix_equality_audit',[sys.executable,'-B','-m','experiments.hj_adaptive.prefix_equality_audit','--root',str(root)]),
        ('completion',[sys.executable,'-B','-m','experiments.hj_adaptive.completion','--root',str(root)]+
        (['--verify-only'] if (root/'completion_audit.json').exists() else [])),
        ('writeup_tables',[sys.executable,'-B','-m','experiments.hj_adaptive.writeup','--tables','--root',str(root)]),
        ('writeup_figures',[sys.executable,'-B','-m','experiments.hj_adaptive.figures','--outcomes','--root',str(root)])]
    for name,command in commands:
        print('START '+name,flush=True)
        with (root/'stage_logs'/(name+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError(name+' failed; inspect retained stage log')
        print('COMPLETE '+name,flush=True)
    print('AUDIT AND PRESENTATION DATA COMPLETE. Empirical manuscript text, final PDF review and release remain.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/results/hj_transport_primary_v7')
    parser.add_argument('--wait',type=float,default=0)
    parser.add_argument('--prepare-after-primary',action='store_true');args=parser.parse_args()
    if args.prepare_after_primary:prepare_after_primary(Path(args.root));return
    assert 0<=args.wait<=55
    if args.wait:time.sleep(args.wait)
    print(json.dumps(observe(Path(args.root)),indent=2),flush=True)


if __name__=='__main__':main()
