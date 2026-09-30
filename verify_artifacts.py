"""Verify code, complete archived evidence and all reported policy checkpoints.

CPU and Python standard library only; no training or checkpoint execution.
Use --archive-dir DIR for the three v1.0.0 ZIPs, or verify extracted results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
STUDIES={
    'minimal_teaching_study':480,
    'direct_query_study':120,
    'neural_information_transfer':240,
    'neural_information_confirmation':360,
    'replication_v3':50,
    'improving_sets_pilot':24,
    'improving_sets_warm':12,
    'improving_sets_replication':20,
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def policy_key(stage,row):
    if stage=='direct_query_study':return row['key']
    if stage=='minimal_teaching_study':
        return f"s{row['seed']}_d{row['dimension']}_m{row['student_dimension']}_{row['shape']}_r{row['radius']:g}_{row['method']}"
    if stage.startswith('neural_information_'):
        return f"s{row['seed']}_r{row['radius']:g}_w{row['width']}_{row['method']}"
    if stage.startswith('improving_sets_'):
        return f"{row['family']}_w{row['width']}_s{row['seed']}_{row['method']}"
    return f"{row['family']}_s{row['seed']}_{row['comparison_label']}"


def check_policies(read,hashes):
    counts={};checkpoints=set();evaluations=set();locks=0
    for stage,expected in STUDIES.items():
        prefix='research/results/'+stage+'/'
        report=json.loads(read(prefix+'report.json'))
        assert len(report['rows'])==expected,(stage,'row count')
        lock=json.loads(read(prefix+'protocol_lock.json'));locks+=1
        for field in ['sources','dependencies']:
            for name,sha in lock.get(field,{}).items():
                path=ROOT/'research'/name.replace('\\','/')
                assert path.is_file() and digest(path)==sha,('locked dependency',stage,name)
        for row in report['rows']:
            stem=prefix+policy_key(stage,row)
            cp=stem+'.pt';ev=stem+'_evaluation.pt'
            expected_hash=row.get('checkpoint_sha256',row.get('policy_sha256'))
            assert hashes.get(cp)==expected_hash,(stage,cp,'checkpoint hash')
            assert ev in hashes,(stage,ev,'missing per-run evaluation')
            if 'evaluation_sha256' in row:assert hashes[ev]==row['evaluation_sha256']
            if 'feedback_sha256' in row:
                cache_prefix=prefix+f"s{row['seed']}_d{row['dimension']}_m{row['student_dimension']}_{row['shape']}_r{row['radius']:g}"
                assert hashes[cache_prefix+'_feedback.json']==row['feedback_sha256']
            if stage=='minimal_teaching_study':
                cache=prefix+f"s{row['seed']}_d{row['dimension']}_m{row['student_dimension']}_{row['shape']}_r{row['radius']:g}_cache.pt"
                assert hashes[cache]==row['cache_sha256']
            assert row.get('nonfinite_updates',0)==0
            checkpoints.add(cp);evaluations.add(ev)
        counts[stage]=expected
    assert len(checkpoints)==1306 and len(evaluations)==1306
    return dict(studies=counts,reported_fitted_policies=len(checkpoints),per_run_evaluations=len(evaluations),protocols=locks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    code=json.loads((ROOT/'code-manifest.json').read_text())
    for name,record in code['files'].items():
        assert digest(ROOT/name)==record['sha256'],('current code',name)
        if record.get('frozen'):assert record['sha256']==record['release_sha256']
    manifest=json.loads((ROOT/'artifact-manifest.json').read_text())
    asset_checks={}
    if args.archive_dir:
        for name,record in manifest['assets'].items():
            p=args.archive_dir/name
            assert p.stat().st_size==record['bytes'] and digest(p)==record['sha256'],('asset',name)
            asset_checks[name]=record['sha256']
        with zipfile.ZipFile(args.archive_dir/'research-artifacts-v1.zip') as z:
            names={i.filename for i in z.infolist() if not i.is_dir()}
            assert names==set(manifest['files']),'Archive member list differs'
            for name in sorted(names):
                with z.open(name) as f:
                    h=hashlib.sha256()
                    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
                assert h.hexdigest()==manifest['files'][name],('archived file',name)
            report=check_policies(z.read,manifest['files'])
            report['verified_archive_files']=len(names)
    else:
        hashes={}
        for name,sha in manifest['files'].items():
            if name.startswith('research/results/'):
                p=ROOT/name
                assert p.is_file() and digest(p)==sha,('result',name)
                hashes[name]=sha
        report=check_policies(lambda n:(ROOT/n).read_bytes(),hashes)
        report['verified_result_files']=len(hashes)
    report.update(passed=True,code_files=len(code['files']),release_assets=asset_checks,
                  scope='Availability, byte integrity, protocols and policy/evaluation correspondence; no training or numerical re-evaluation.')
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
