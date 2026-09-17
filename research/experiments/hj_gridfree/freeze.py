import datetime
import hashlib
import json
from pathlib import Path
import shutil
import torch
from .engine import write,hashes


def main():
    root=Path('experiments/results/hj_gridfree_primary_v1'); root.mkdir(parents=True,exist_ok=True)
    assert not (root/'lock.json').exists(),'Refusing to replace an existing lock'
    shutil.copyfile('experiments/configs/hj_gridfree_primary_v1.json',root/'config.json')
    lock=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=hashes(),config_sha256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        scope='Training source and configuration frozen before primary training and independent tests. Evaluation/reporting sources will have a separate pretest lock. Development tests are separate. No independent test states have been evaluated.')
    write(root/'lock.json',lock)
    print(json.dumps(lock,indent=2),flush=True)


if __name__=='__main__': main()
