"""Verify frozen inputs and package the completed, local grid-free study."""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile
from .engine import write
from .train import verify_lock


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(target,paths):
    paths=sorted({Path(p) for p in paths if Path(p).is_file()})
    members={p.as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths: z.write(p,p.as_posix())
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(members,indent=2))
    with zipfile.ZipFile(target) as z:
        assert len(z.namelist())==len(members)+1
        for name,meta in members.items():
            content=z.read(name); assert len(content)==meta['bytes']; assert hashlib.sha256(content).hexdigest()==meta['sha256'],name
    return dict(path=target.as_posix(),bytes=target.stat().st_size,sha256=digest(target),verified_members=len(members))


def main():
    root=Path('experiments/results/hj_gridfree_primary_v1'); cfg=verify_lock(root)
    lock=json.loads((root/'evaluation_lock.json').read_text())
    assert digest('experiments/hj_gridfree/evaluate.py')==lock['evaluation_source_sha256']
    for file,d in lock['additional_source_hashes'].items(): assert digest(file)==d,file
    for file,d in lock['checkpoints'].items(): assert digest(file)==d,file
    assert len(lock['checkpoints'])==50
    assert len(list((root/'evaluation').glob('*.pt')))==16
    assert len(list((root/'native_evaluation').glob('*.pt')))==16
    assert len(list((root/'timing').glob('*.json')))==8
    assert len(list((root/'isolated_memory').glob('*.json')))==10
    assert (root/'report/summary.json').exists() and Path('build/hj_gridfree_rollout.pdf').exists()
    release=Path('build/hj_gridfree_release_v1'); release.mkdir(parents=True,exist_ok=True)
    source=list(Path('experiments/hj_gridfree').glob('*.py'))+[p for p in Path('docs').glob('hj_gridfree*.md') if p.name not in ('hj_gridfree_active_state.md','hj_gridfree_completed_state.md')]
    source+=list(Path('paper').glob('hj_gridfree*'))+list(Path('paper/sections').glob('hj_gridfree*.tex'))+list(Path('paper/figures').glob('hj_gridfree*'))
    source+=[Path('experiments/__init__.py'),Path('experiments/hj_gridfree/requirements.txt'),Path('paper/elsarticle.cls'),Path('paper/elsarticle-num.bst'),Path('experiments/configs/hj_gridfree_primary_v1.json'),Path('build/hj_gridfree_rollout.pdf')]
    data=[p for p in root.rglob('*') if p.is_file()]
    for folder in Path('experiments/results').glob('hj_gridfree_dev_*'): data.extend(p for p in folder.rglob('*') if p.is_file())
    for folder in [Path('experiments/results/hj_gridfree_audit_v1'),Path('experiments/results/hj_gridfree_planning_probe')]: data.extend(p for p in folder.rglob('*') if p.is_file())
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_lock_sha256=digest(root/'lock.json'),evaluation_lock_sha256=digest(root/'evaluation_lock.json'),pdf_sha256=digest('build/hj_gridfree_rollout.pdf'),checkpoint_hashes_verified=50,source=pack(release/'hj_gridfree_manuscript_and_source.zip',source),data=pack(release/'hj_gridfree_data_and_checkpoints.zip',data))
    write(release/'release_manifest.json',result); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
