"""Observe a serial experiment without touching its numerical outputs."""
import argparse
import datetime
import json
from pathlib import Path
import time


def read_json(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError):return {}


def observe(root):
    cfg=read_json(root/'config.json')
    phases=[]
    for folder in (root/'training').glob('*'):
        if not folder.is_dir():continue
        for path in folder.glob('*.json'):
            value=read_json(path)
            if 'steps_completed' in value:phases.append(value)
    latest={}
    try:
        with (root/'stage_logs/training.log').open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-16000))
            lines=stream.read().decode('utf-8',errors='replace').splitlines()
        for line in reversed(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if 'run' in row and 'step' in row:
                latest={key:row[key] for key in ['run','step','best']};break
    except OSError:pass
    return dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pipeline=read_json(root/'pipeline_status.json'),postprocess=read_json(root/'postprocess_status.json'),
        completed_phase_endpoints=len(phases),planned_phase_endpoints=sum(len(t['methods']) for t in cfg['tasks'])*len(cfg['seeds']),
        completed_evaluation_cases=len(list((root/'evaluation').glob('*.json'))),
        completed_gpu_reference_cases=len(list((root/'classical_gpu').glob('*.json'))),
        completed_native_reference_cases=len(list((root/'classical_native').glob('*.json'))),
        completed_feedback_reference_cases=len(list((root/'feedback_classical').glob('*.json'))),
        completed_memory_probes=len(list((root/'isolated_memory').glob('*.json'))),
        completed_timing_cases=len(list((root/'timing').glob('*.json'))),
        development_gpu_preflight_complete=(root/'pretest_evaluation_audit.json').exists(),
        fresh_evaluation_locked=(root/'evaluation_lock.json').exists(),
        latest_training=latest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='experiments/results/hj_cotangent_primary_v6')
    parser.add_argument('--minutes',type=float,default=180)
    parser.add_argument('--interval',type=float,default=60)
    args=parser.parse_args();root=Path(args.root)
    deadline=time.monotonic()+60*args.minutes
    while True:
        record=observe(root)
        (root/'live_progress.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(record),flush=True)
        if record['pipeline'].get('stage')=='failed' or record['postprocess'].get('stage') in ['failed','awaiting_manuscript_review']:
            return
        remaining=deadline-time.monotonic()
        if remaining<=0:return
        time.sleep(min(args.interval,remaining))


if __name__=='__main__':main()
