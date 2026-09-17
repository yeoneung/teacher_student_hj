"""Package the completed v7 study with verified files and explicit review."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import zipfile
import fitz
from .study import verify


ROOT=Path('experiments/results/hj_transport_primary_v7')
TARGET=Path('build/hj_teacher_transport_release_v7')


def stream_digest(stream):
    checksum=hashlib.sha256();size=0
    while True:
        chunk=stream.read(1024*1024)
        if not chunk:break
        checksum.update(chunk);size+=len(chunk)
    return size,checksum.hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:return stream_digest(stream)[1]


def pack(target,paths):
    paths=sorted(set(Path(p) for p in paths))
    assert all(p.is_file() for p in paths),'A requested archive member is missing'
    members={p.as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in paths:archive.write(path,path.as_posix())
        archive.writestr('ARCHIVE_MANIFEST.json',json.dumps(members,indent=2))
    record=dict(path=target.as_posix(),bytes=target.stat().st_size,sha256=digest(target),verified_members=len(members))
    verify_zip(record)
    return record


def manuscript_check():
    pdf=Path('build/hj_teacher_transport_v7.pdf')
    log=pdf.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log and 'undefined' not in log
    assert 'Output written on' in log
    with fitz.open(pdf) as document:
        text='\n'.join(p.get_text() for p in document);pages=len(document)
    normalized=' '.join(text.split())
    for forbidden in ['Primary results are pending','Empirical conclusions are pending',
                      'Primary experiments have not started','Primary experiments are running','will be filled only']:
        assert forbidden not in normalized,forbidden
    assert all(word in text for word in ['Hamilton','teacher','270'])
    assert Path('docs/hj_v7_results_ko.md').is_file()
    korean=Path('docs/hj_v7_results_ko.md').read_text(encoding='utf-8')
    assert '진행 중인 작업본' not in korean and '독립 검증 완료 후 작성한다' not in korean
    for name in ['selection','primary','cost','transfer','work','gradient']:
        assert Path(f'paper/sections/hj_v7_{name}_table.tex').is_file()
    for name in ['budget_ratios','reuse_work']:
        assert Path(f'paper/figures/hj_v7_{name}.pdf').is_file()
    review_path=Path('build/hj_transport_visual_review/review.json')
    review=json.loads(review_path.read_text())
    assert review['status']=='reviewed' and review['pdf_sha256']==digest(pdf)
    assert review['reviewed_pages']==list(range(1,pages+1))
    return dict(path=str(pdf),sha256=digest(pdf),pages=pages,
        visual_review_sha256=digest(review_path),no_overfull_boxes=True,no_undefined_references=True)


def verify_zip(record):
    path=Path(record['path']);assert path.stat().st_size==record['bytes']
    assert digest(path)==record['sha256']
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('ARCHIVE_MANIFEST.json'))
        assert len(manifest)==record['verified_members']
        assert len(archive.namelist())==len(manifest)+1
        for name,meta in manifest.items():
            with archive.open(name) as stream:size,sha=stream_digest(stream)
            assert size==meta['bytes'] and sha==meta['sha256'],name


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    cfg=verify(ROOT);audit=json.loads((ROOT/'completion_audit.json').read_text())
    expected=sum(len(t['methods'])*len(cfg['seeds']) for t in cfg['tasks'])
    assert audit['phases']==expected==270
    for path,sha in audit['evidence_hashes'].items():assert digest(path)==sha,path
    preservation=Path('build/hj_v7_preservation_check.json')
    assert json.loads(preservation.read_text())['passed']
    paper=manuscript_check()
    if args.verify_only:
        record=json.loads((TARGET/'release_manifest.json').read_text())
        assert record['manuscript']==paper
        assert record['completion_audit_sha256']==digest(ROOT/'completion_audit.json')
        assert record['preservation_check_sha256']==digest(preservation)
        for key in ['source','data']:verify_zip(record[key])
        print(json.dumps(dict(verified=True,manuscript=paper),indent=2));return
    assert not (TARGET/'release_manifest.json').exists(),'A release already exists; never replace it'
    TARGET.mkdir(parents=True,exist_ok=True)
    sources=[Path(p) for p in json.loads((ROOT/'training_lock.json').read_text())['sources']]
    sources += [Path('experiments/__init__.py'),Path('paper/elsarticle.cls'),Path('paper/elsarticle-num.bst'),
        Path('paper/hj_teacher_transport_v7.tex'),Path('paper/hj_gridfree_references.bib'),Path('paper/hj_v7_references.bib')]
    for folder in ['hj_gridfree','hj_extension','hj_cotangent','hj_adaptive']:
        sources += list(Path('experiments',folder).glob('*.py'))
        sources += list(Path('experiments',folder).glob('requirements*.txt'))
    sources += list(Path('paper/sections').glob('hj_v7_*.tex'))+list(Path('paper/figures').glob('hj_v7_*'))
    sources += list(Path('docs').glob('hj_v7*.md'))
    sources += list(Path('docs').glob('hj_v7*.html'))
    sources += [Path(paper['path']),Path(paper['path']).with_suffix('.log'),
        Path('build/hj_teacher_transport_v7_compile.log'),
        Path('build/hj_transport_visual_review/render_manifest.json'),Path('build/hj_transport_visual_review/review.json')]
    sources += [Path('build/hj_v7_interactive_review')/name for name in ['browser_checks.json','review.json']]
    sources += [preservation,Path('build/hj_v7_completion_verify.log')]
    sources += list(Path('build/hj_v7_interactive_review').glob('*.png'))
    sources += list((ROOT/'report').glob('*'))+[ROOT/name for name in ['config.json','protocol.md','environment.json',
        'training_lock.json','evaluation_lock.json','development_selection.json','completion_audit.json','stage_times.json',
        'package_versions.json','development_reconstruction_audit.json','development_endpoints.csv','development_configuration_means.csv',
        'batch_gradient_audit_lock.json','prefix_equality_audit_lock.json']]
    # Previous release remains separately immutable; retain its manifest as provenance.
    sources.append(Path('build/hj_gridfree_release_v6/release_manifest.json'))
    devroots=[Path('experiments/results/hj_adaptive_dev_v7'+suffix) for suffix in ['', 'b','c','d','e','f_controls']]
    data=[p for folder in [ROOT]+devroots for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    logs=[]
    for pattern in ['hj_adaptive_development*.log','hj_transport_development*.log','hj_variance*.log',
                    'hj_full_batch*.log','hj_development_continuation*.log','hj_baseline_continuation*.log',
                    'hj_transport_cuda_audit*.log']:
        logs += list(Path('build').glob(pattern))
    data += logs
    record=dict(version=7,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description='All-time teacher value transport: HJB interpretation, locked budget-matched control experiments, limitations and numerical evidence',
        manuscript=paper,completion_audit_sha256=digest(ROOT/'completion_audit.json'),
        preservation_check_sha256=digest(preservation),
        training_lock_sha256=digest(ROOT/'training_lock.json'),evaluation_lock_sha256=digest(ROOT/'evaluation_lock.json'),
        primary_phases=270,new_development_phases=312,source=pack(TARGET/'hj_teacher_transport_manuscript_and_source.zip',sources),
        data=pack(TARGET/'hj_teacher_transport_data_and_checkpoints.zip',data))
    (TARGET/'release_manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
