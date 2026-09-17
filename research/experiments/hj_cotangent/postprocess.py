"""Finish evidence auditing and draft generation when the running pipeline ends."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def compile_manuscript(source):
    source=Path(source)
    target=Path('build')/(source.stem+'_compile.log')
    latex=['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=../build',source.name]
    commands=[latex,['bibtex','../build/'+source.stem],latex,latex]
    with target.open('w',encoding='utf-8') as log:
        for command in commands:
            result=subprocess.run(command,cwd=source.parent,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Manuscript build failed: inspect {target}')
    text=(Path('build')/(source.stem+'.log')).read_text(encoding='utf-8',errors='replace')
    return dict(pdf=str(Path('build')/(source.stem+'.pdf')),
        overfull_warnings=text.count('Overfull'),undefined_warnings=text.count('undefined'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='experiments/results/hj_cotangent_primary_v6')
    parser.add_argument('--compile')
    args=parser.parse_args()
    if args.compile:
        print(json.dumps(compile_manuscript(args.compile),indent=2));return
    root=Path(args.root)
    def status(stage,**details):
        row=dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**details)
        (root/'postprocess_status.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(row),flush=True)
    observed=None
    try:
        while True:
            try:current=json.loads((root/'pipeline_status.json').read_text())['stage']
            except (OSError,json.JSONDecodeError):
                time.sleep(10);continue
            if current=='experiments_complete':break
            if current=='failed':raise RuntimeError('Primary pipeline failed; repair and resume it before postprocessing.')
            if current!=observed:
                status('waiting_for_primary_pipeline',observed=current);observed=current
            time.sleep(10)
        for stage,module,parameters in [
            ('completion_audit','experiments.hj_cotangent.completion_audit',['--root',str(root)]),
            ('numerical_writeup','experiments.hj_cotangent.writeup',['--root',str(root)]),
            ('concept_figure','experiments.hj_cotangent.figures',[])]:
            status(stage)
            with (root/'stage_logs'/(stage+'.log')).open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-B','-m',module,*parameters],stdout=log,stderr=subprocess.STDOUT,check=True)
        status('draft_build')
        result=compile_manuscript('paper/hj_cotangent_draft.tex')
        status('awaiting_manuscript_review',**result,
            remaining='Interpret complete evidence, write discussion and final abstract, review all pages, update canonical manuscript, create and verify v6 archives.')
    except BaseException as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
