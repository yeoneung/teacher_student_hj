"""Create a new, verified v6 archive; never replace a preceding release."""
import argparse
import datetime
import json
from pathlib import Path
import re

import fitz

from experiments.hj_gridfree.release import pack
from .study import verify,digest
from .evaluation import lock_evaluation


def manuscript_check():
    source=Path('paper/hj_gridfree_rollout.tex')
    assert 'sections/hj_cotangent_results' in source.read_text(encoding='utf-8')
    pdf=Path('build/hj_gridfree_rollout.pdf')
    log=pdf.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log and 'undefined' not in log
    assert 'Output written on' in log
    with fitz.open(pdf) as doc:
        pages=len(doc)
        text='\n'.join(p.get_text() for p in doc)
    assert 'Primary experiments are running' not in text
    assert 'Hamilton' in text and 'teacher' in text.lower()
    assert '250' in text
    review_path=Path('build/hj_cotangent_visual_review/review.json')
    review=json.loads(review_path.read_text(encoding='utf-8'))
    assert review['pdf_sha256']==digest(pdf)
    assert review['reviewed_pages']==pages and review['status']=='reviewed'
    return dict(pdf=str(pdf),pdf_sha256=digest(pdf),pages=pages,
                source_sha256=digest(source),no_overfull_boxes=True,no_undefined_references=True,
                visual_review_sha256=digest(review_path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    root=Path('experiments/results/hj_cotangent_primary_v6')
    cfg=verify(root);lock_evaluation(root,cfg)
    audit=json.loads((root/'completion_audit.json').read_text())
    assert audit['completed_phase_endpoints']==250 and audit['cases']==72
    assert audit['report_sha256']==digest(root/'report/summary.json')
    assert 'preserved_previous' in audit
    for path,expected in audit['preserved_previous']['archive_hashes'].items():
        assert digest(path)==expected,path
    for path,expected in audit['raw_evidence_hashes'].items():assert digest(path)==expected,path
    for row in audit['phase_records']:assert digest(row['path'])==row['sha256'],row['path']
    paper=manuscript_check()
    if args.verify_only:
        print(json.dumps(paper,indent=2));return
    target=Path('build/hj_gridfree_release_v6')
    if (target/'release_manifest.json').exists():
        raise RuntimeError('The v6 release exists; use a new revision instead of replacing it.')
    target.mkdir(parents=True,exist_ok=True)
    sources=[Path('README.md'),Path('experiments/__init__.py'),Path('paper/elsarticle.cls'),Path('paper/elsarticle-num.bst')]
    for folder in ['hj_gridfree','hj_extension','hj_cotangent']:
        sources.extend(Path('experiments',folder).glob('*.py'))
        sources.extend(Path('experiments',folder).glob('requirements*.txt'))
        sources.extend(Path('paper/sections').glob(folder+'*.tex'))
        sources.extend(Path('paper/figures').glob(folder+'*'))
        sources.extend(Path('docs').glob(folder+'*.md'))
    sources.extend(Path('docs').glob('hj_v6*.md'))
    sources.extend(Path('paper').glob('hj_gridfree*'))
    sources.extend(Path('experiments/configs').glob('hj_cotangent*.json'))
    sources.extend(root.glob('host_load_*.json'))
    sources.extend([Path(paper['pdf']),Path(paper['pdf']).with_suffix('.log'),
                    Path('build/hj_gridfree_rollout_compile.log'),
                    Path('build/hj_cotangent_verify_only.log'),
                    Path('build/hj_cotangent_visual_review/render_manifest.json'),
                    Path('build/hj_cotangent_visual_review/review.json'),
                    root/'training_lock.json',root/'evaluation_lock.json',root/'completion_audit.json',
                    root/'report/summary.json',root/'protocol.md',root/'config.json',root/'environment.json',
                    root/'evaluation_addendum.md',root/'feedback_protocol_lock.json',root/'pretest_evaluation_audit.json',
                    Path('experiments/results/hj_gridfree_primary_v1/pd_calibration.json'),
                    Path('experiments/results/hj_gridfree_framework_v2/checks.json'),
                    Path('experiments/results/hj_gridfree_hjb_v3/checks.json')])
    for version in range(1,6):sources.append(Path(f'build/hj_gridfree_release_v{version}/release_manifest.json'))
    folders=[root]
    for pattern in ['hj_cotangent_dev_v6*','hj_building_dev_v6','hj_promoted_cache_dev_v6']:
        folders.extend(Path('experiments/results').glob(pattern))
    data=[p for folder in folders for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    result=dict(version=6,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description='Prospective long-horizon teacher sensitivity reuse, HJB exposition, controls, numerical checks and manuscript',
        manuscript=paper,completion_audit_sha256=digest(root/'completion_audit.json'),
        training_lock_sha256=digest(root/'training_lock.json'),evaluation_lock_sha256=digest(root/'evaluation_lock.json'),
        report_sha256=digest(root/'report/summary.json'),preserved_previous=audit['preserved_previous'],
        verified_counts={k:audit[k] for k in ['completed_phase_endpoints','cases','distinct_initial_vectors',
            'state_horizon_instances','classical_cases','memory_probes','timing_cases']},
        source=pack(target/'hj_gridfree_manuscript_and_source.zip',sources),
        extension_data=pack(target/'hj_cotangent_data_and_checkpoints.zip',data))
    (target/'release_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
