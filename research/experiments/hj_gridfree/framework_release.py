"""Archive a theory revision, preserving and referencing the frozen v1 data."""
import argparse
import datetime
import json
from pathlib import Path
import zipfile

from .release import digest, pack
from .train import verify_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, choices=[2, 3], default=3)
    version = parser.parse_args().version
    root = Path('experiments/results/hj_gridfree_primary_v1')
    verify_lock(root)
    lock = json.loads((root/'evaluation_lock.json').read_text(encoding='utf-8'))
    assert digest('experiments/hj_gridfree/evaluate.py') == lock['evaluation_source_sha256']
    for file, expected in lock['additional_source_hashes'].items():
        assert digest(file) == expected, file
    for file, expected in lock['checkpoints'].items():
        assert digest(file) == expected, file
    parent_path = Path('build/hj_gridfree_release_v1/release_manifest.json')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    for key in ['source', 'data']:
        assert digest(parent[key]['path']) == parent[key]['sha256'], key
    previous_source = None
    if version == 3:
        previous = json.loads(Path('build/hj_gridfree_release_v2/release_manifest.json').read_text(encoding='utf-8'))
        previous_source = previous['source']
        assert digest(previous_source['path']) == previous_source['sha256']
    # Verify all originally archived data against current disk contents as well.
    with zipfile.ZipFile(parent['data']['path']) as archive:
        old_data = json.loads(archive.read('ARCHIVE_MANIFEST.json'))
    for file, meta in old_data.items():
        assert Path(file).stat().st_size == meta['bytes'], file
        assert digest(file) == meta['sha256'], file
    checks = Path('experiments/results/hj_gridfree_framework_v2/checks.json')
    proof_checks = json.loads(checks.read_text(encoding='utf-8'))
    assert proof_checks['max_continuous_or_quadrature_error'] < 2e-12
    assert proof_checks['constrained_regret_max_error'] < 2e-12
    extra_checks = []
    if version == 3:
        hjb_checks = Path('experiments/results/hj_gridfree_hjb_v3/checks.json')
        hjb = json.loads(hjb_checks.read_text(encoding='utf-8'))
        assert hjb['max_master_identity_error'] < 2e-12
        assert hjb['optimal_policy_verification_error'] < 2e-12
        assert hjb['optimal_field_hjb_residual_max_abs'] < 2e-12
        extra_checks.append(hjb_checks)
    pdf = Path('build/hj_gridfree_rollout.pdf')
    log = Path('build/hj_gridfree_rollout.log').read_text(encoding='utf-8', errors='replace')
    assert 'undefined references' not in log and 'Overfull' not in log
    assert 'Output written on' in log
    release = Path(f'build/hj_gridfree_release_v{version}')
    if (release/'release_manifest.json').exists():
        raise RuntimeError(f'v{version} archive already exists; use a new release version for further changes')
    release.mkdir(parents=True, exist_ok=True)
    source = list(Path('experiments/hj_gridfree').glob('*.py'))
    source += [p for p in Path('docs').glob('hj_gridfree*.md')
               if p.name not in ('hj_gridfree_active_state.md', 'hj_gridfree_completed_state.md')]
    source += list(Path('paper').glob('hj_gridfree*'))
    source += list(Path('paper/sections').glob('hj_gridfree*.tex'))
    source += list(Path('paper/figures').glob('hj_gridfree*'))
    source += [Path('README.md'), Path('experiments/__init__.py'),
               Path('experiments/hj_gridfree/requirements.txt'), Path('paper/elsarticle.cls'),
               Path('paper/elsarticle-num.bst'), Path('experiments/configs/hj_gridfree_primary_v1.json'),
               root/'lock.json', root/'evaluation_lock.json', checks, pdf]
    source += extra_checks
    result = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        revision=('Explicit HJB evaluation, improvement, and verification connection; primary experiments unchanged'
                  if version == 3 else 'Theory-centered HJ teacher-student interpretation; primary experiments unchanged'),
        version=version,
        parent_manifest_sha256=digest(parent_path),
        preserved_v1_source_sha256=parent['source']['sha256'],
        preserved_previous_source=previous_source,
        training_lock_sha256=digest(root/'lock.json'),
        evaluation_lock_sha256=digest(root/'evaluation_lock.json'),
        checkpoint_hashes_verified=len(lock['checkpoints']),
        original_data_files_verified=len(old_data),
        analytic_checks_sha256=digest(checks), pdf_sha256=digest(pdf),
        extra_analytic_checks={p.as_posix(): digest(p) for p in extra_checks},
        source=pack(release/'hj_gridfree_manuscript_and_source.zip', source),
        shared_data=parent['data'])
    (release/'release_manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
