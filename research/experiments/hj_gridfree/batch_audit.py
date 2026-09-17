"""Check reported batched quality against batch-one deployment precision."""
import json
from pathlib import Path
import torch
from .core import Problem
from .engine import EvalGraph,write
from .evaluate import actor_from
from .train import verify_lock


def main():
    import gc
    torch.set_num_threads(1); root=Path('experiments/results/hj_gridfree_primary_v1'); cfg=verify_lock(root); result={}
    for file in sorted((root/'evaluation').glob('*.pt')):
        data=torch.load(file,weights_only=False); meta=json.loads(file.with_suffix('.json').read_text())
        p=Problem(**meta['problem']); x=data['x'].cuda(); result[file.stem]={}
        for mode in ['block16','dpc']:
            actor=actor_from(root,p,mode,cfg['seeds'][0],cfg['width'])
            for dtype,label in [(torch.float64,'float64'),(torch.float32,'float32')]:
                evaluator=EvalGraph(actor,p,1,dtype=dtype); rows=[]
                for i in range(len(x)): rows.append(evaluator(x[i:i+1]))
                costs=torch.cat(rows).cpu().double(); ref=data['neural'][mode][0]
                error=((costs-ref).abs()/(1+ref.abs())).max().item()
                result[file.stem][f'{mode}_{label}']=dict(max_scaled_error=error,mean_difference=(costs-ref).mean().item(),case_count=len(x))
                del evaluator; gc.collect(); torch.cuda.empty_cache()
            del actor; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({file.stem:result[file.stem]}),flush=True)
    write(root/'batch_audit.json',result)
    maximum=max(v['max_scaled_error'] for r in result.values() for v in r.values())
    print(json.dumps(dict(max_scaled_error=maximum)),flush=True)
    assert maximum<.003,'Deployment precision/shape materially changes costs; inspect before claims.'


if __name__=='__main__': main()
