"""Download the pinned public evidence, verify SHA-256, and safely extract it."""
import argparse,hashlib,json,os,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-extract',action='store_true',help='Download and verify the three ZIP files only.')
    args=parser.parse_args();manifest=json.loads((ROOT/'artifact-manifest.json').read_text())
    for name,info in manifest['assets'].items():
        target=ROOT/name
        assert target.parent==ROOT and info['url'].startswith('https://github.com/yeoneung/teacher_student_hj/releases/download/v1.0.0/')
        if target.exists():
            if digest(target)!=info['sha256']:raise RuntimeError(f'Existing file differs: {name}; move it aside before downloading.')
            print('Verified existing',name);continue
        temporary=target.with_suffix(target.suffix+'.part');h=hashlib.sha256();received=0;next_progress=32*1024*1024
        request=urllib.request.Request(info['url'],headers={'User-Agent':'teacher-student-hj-reproduction'})
        print('Downloading',name,flush=True)
        with urllib.request.urlopen(request,timeout=90) as response,temporary.open('wb') as f:
            while True:
                block=response.read(1024*1024)
                if not block:break
                f.write(block);h.update(block);received+=len(block)
                if received>=next_progress:
                    print(f'  {received/2**20:.0f} / {info["bytes"]/2**20:.0f} MiB',flush=True);next_progress+=32*1024*1024
        if received!=info['bytes'] or h.hexdigest()!=info['sha256']:raise RuntimeError('Downloaded file failed integrity verification: '+name)
        os.replace(temporary,target);print('Verified',name,flush=True)
    if args.no_extract:return
    with zipfile.ZipFile(ROOT/'research-artifacts-v1.zip') as z:
        for item in z.infolist():
            target=(ROOT/item.filename).resolve()
            if not target.is_relative_to(ROOT):raise RuntimeError('Invalid archive path')
            if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            expected=manifest['files'].get(item.filename)
            if expected is None:raise RuntimeError('Unlisted archive member: '+item.filename)
            if target.exists():
                if digest(target)!=expected:raise RuntimeError('Existing research file differs: '+item.filename)
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as source,target.open('wb') as dest:
                for block in iter(lambda:source.read(1024*1024),b''):dest.write(block)
            if digest(target)!=expected:raise RuntimeError('Extracted file failed integrity verification: '+item.filename)
    print('Research artifacts verified and extracted. Reference ZIPs remain at the repository root.')

if __name__=='__main__':main()
