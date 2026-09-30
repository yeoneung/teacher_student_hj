"""Download verified v1.0.0 data and extract results beside the current code."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-extract',action='store_true')
    parser.add_argument('--offline',action='store_true',help='Verify local archives without network access.')
    parser.add_argument('--archive-dir',type=Path,default=ROOT,help='Location of the three ZIPs; defaults to this repository.')
    args=parser.parse_args();folder=args.archive_dir.resolve();folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'artifact-manifest.json').read_text())
    for name,info in manifest['assets'].items():
        target=folder/name
        assert target.parent==folder and info['url'].startswith('https://github.com/yeoneung/teacher_student_hj/releases/download/v1.0.0/')
        if not target.exists():
            if args.offline:raise FileNotFoundError(target)
            temporary=target.with_suffix('.zip.part')
            req=urllib.request.Request(info['url'],headers={'User-Agent':'teacher-student-hj-reproduction'})
            print('Downloading',name,flush=True)
            with urllib.request.urlopen(req,timeout=90) as response,temporary.open('wb') as f:
                for block in iter(lambda:response.read(1024*1024),b''):f.write(block)
            assert temporary.stat().st_size==info['bytes'] and digest(temporary)==info['sha256'],name
            os.replace(temporary,target)
        assert target.stat().st_size==info['bytes'] and digest(target)==info['sha256'],('asset hash',name)
        print('Verified',name,flush=True)
    if args.no_extract:return
    count=0
    with zipfile.ZipFile(folder/'research-artifacts-v1.zip') as z:
        for item in z.infolist():
            # Use the current checked-in code; do not restore old scripts or manuscript assets.
            if item.is_dir() or not item.filename.startswith('research/results/'):continue
            target=(ROOT/item.filename).resolve()
            assert target.is_relative_to((ROOT/'research/results').resolve()),'Invalid archive path'
            expected=manifest['files'][item.filename]
            if target.exists():
                assert digest(target)==expected,('Existing result differs; preserve it before restaging',item.filename)
                count+=1;continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as source,target.open('wb') as dest:
                for block in iter(lambda:source.read(1024*1024),b''):dest.write(block)
            assert digest(target)==expected,('Extracted result',item.filename)
            count+=1
    print(f'Verified and staged {count} numerical result files. Current code was preserved.')
    if folder!=ROOT:
        print('For verify_evidence.py --stage-inputs, place the two reference ZIPs at the repository root.')


if __name__=='__main__':main()
