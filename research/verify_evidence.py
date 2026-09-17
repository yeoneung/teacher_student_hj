"""Verify the supplied evidence using reference inputs staged only inside submission.

Does not train policies. Extraction is explicit and path-checked. Numerical
source files are not modified; only the reference root is set in memory.
"""
import argparse,json,shutil,sys,zipfile
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SUB=HERE.parent
REFERENCE=HERE/'reference_inputs'


def stage():
    REFERENCE.mkdir(exist_ok=True)
    for name in ['07_supporting_evidence.zip','08_data_and_checkpoints.zip']:
        with zipfile.ZipFile(SUB/name) as archive:
            for info in archive.infolist():
                target=(REFERENCE/info.filename).resolve()
                if not target.is_relative_to(REFERENCE.resolve()):
                    raise ValueError('Archive path escapes the reference directory')
                if info.is_dir():
                    target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(info) as src,target.open('wb') as dst:
                        shutil.copyfileobj(src,dst)
    target=REFERENCE/'build/hj_v10_final/manifest.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(HERE/'reference_manifest_v10.json',target)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage-inputs',action='store_true')
    args=parser.parse_args()
    if args.stage_inputs:stage()
    if not (REFERENCE/'build/hj_v10_final/manifest.json').exists():
        parser.error('First use --stage-inputs to unpack the supplied reference evidence inside submission.')
    import impact_core
    impact_core.PROJECT=REFERENCE
    import impact_audit
    impact_audit.main()
    print('Reference inputs were read from:',REFERENCE)


if __name__=='__main__':main()
