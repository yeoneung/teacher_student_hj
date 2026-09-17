"""Pre-primary audit of variable remaining horizons, plans and all learning modes."""
import copy
import gc
import json
import hashlib
import shutil
from pathlib import Path
import torch
from .systems import problem,teacher,actor,sample,rollout,terminal,step,running
from .core import prefix,continuation,truncated_backprop,local_return,LocalGraph,JetGraph,EvalGraph
from .evaluation import PlanGraph,replay


def gradient(y,net):
    return torch.cat([g.flatten() for g in torch.autograd.grad(y.sum(),net.parameters())])


def physical_tbptt(x,t,policy,p,length):
    # An independent reference with no padding, one trajectory at a time.
    values = []
    for j,start in enumerate(t.tolist()):
        z = x[j:j+1]
        value = torch.zeros(1,device=x.device,dtype=x.dtype)
        for k in range(p.horizon-start):
            if k and k%length==0:
                z=z.detach()
            u=policy(z,start+k)
            value=value+running(z,u,p,start+k)
            z=step(z,u,p,start+k)
        values.append(value+terminal(z,p))
    return torch.cat(values)


def main():
    torch.set_num_threads(1)
    report={}
    for family in ['mechanical','reaction','building']:
        p=problem(family,4,12)
        torch.manual_seed(67301)
        policy=actor(p,teacher(p),64).cuda()
        x=sample(p,8,67302)*.2
        t=torch.tensor([0,1,3,4,6,8,10,11],device='cuda')
        with torch.no_grad():
            policy.net[-1].weight.normal_(0,.01)
        full=rollout(x,policy,p,t)
        cut=truncated_backprop(x,t,policy,p,4)
        reference=physical_tbptt(x,t,policy,p,4)
        forward=(full-cut).abs().max().item()
        gt=gradient(cut,policy.net)
        gr=gradient(reference,policy.net)
        error=(gt-gr).abs().max().item()
        assert forward<1e-6 and error<2e-5,(family,forward,error)
        with torch.no_grad():
            _,anchor,end=prefix(x,t,policy,p,4)
        anchor=anchor.detach().requires_grad_(True)
        value=rollout(anchor,teacher(p),p,end)
        covector=torch.autograd.grad(value.sum(),anchor)[0].detach()
        value=value.detach()
        anchor=anchor.detach()
        with torch.no_grad():policy.net[-1].bias.add_(.02)
        normal=local_return(x,t,policy,p,4,anchor,value,covector,rho=1.)
        zero=local_return(x,t,policy,p,4,anchor,value,covector,rho=1.,gradient_covector=torch.zeros_like(covector))
        signal_forward=(normal-zero).abs().max().item()
        gn=gradient(normal,policy.net)
        gz=gradient(zero,policy.net)
        _,boundary,end=prefix(x,t,policy,p,4)
        transferred=gradient((covector*boundary).sum(-1)*(end<p.horizon),policy.net)
        signal_gradient=(gn-gz-transferred).abs().max().item()
        assert signal_forward<1e-6 and signal_gradient<2e-5,(family,signal_forward,signal_gradient)
        saved=copy.deepcopy(policy.state_dict())
        policy=actor(p,teacher(p),64).cuda()
        policy.load_state_dict(saved)
        evaluator=PlanGraph(policy,p,len(x))
        costs,controls=evaluator(x.double())
        independent=replay(x.double(),controls,p)
        cost_error=((costs.cpu()-independent).abs()/(1+independent.abs())).max().item()
        assert cost_error<1e-10,(family,cost_error)
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
        checks={}
        for method in ['cache','cache_zero','cache_sign','exact','dpc','short','tbptt']:
            mode='cache' if method.startswith('cache') else method
            signal='zero' if method=='cache_zero' else 'sign' if method=='cache_sign' else 'normal'
            policy=actor(p,teacher(p),64).cuda()
            policy.load_state_dict(saved)
            initial=copy.deepcopy(policy.state_dict())
            base=teacher(p)
            graph=LocalGraph(policy,p,len(x),4,rho=1.,mode=mode,teacher=base,signal=signal)
            assert all(torch.equal(v,initial[k]) for k,v in policy.state_dict().items())
            if mode=='cache':
                jet=JetGraph(policy,base,p,len(x),4)
                labels=jet(x,t)
            else:
                labels=()
                jet=None
            changed=labels[2]*torch.where(torch.arange(p.dim,device='cuda')%2==0,1.,-1.) if signal=='sign' else None
            for _ in range(3):
                graph.update(x,t,*labels,signal_covector=changed)
            torch.cuda.synchronize()
            assert torch.isfinite(graph.loss).item()
            checks[method]=dict(loss=graph.loss.item(),gradient_norm=graph.grad_norm.item(),clipped=graph.clipped_updates.item())
            before=copy.deepcopy(policy.state_dict())
            first_moments=[m.clone() for m in graph.m]
            bad=x.clone()
            bad[0,0]=float('nan')
            graph.update(bad,t,*labels,signal_covector=changed)
            torch.cuda.synchronize()
            assert graph.nonfinite_updates.item()==1
            assert all(torch.equal(v,before[k]) for k,v in policy.state_dict().items())
            assert all(torch.equal(m,a) for m,a in zip(graph.m,first_moments))
            graph.update(x,t,*labels,signal_covector=changed)
            torch.cuda.synchronize()
            assert torch.isfinite(graph.loss).item()
            checks[method]['nonfinite_update_skip_and_recovery']=True
            del graph,jet,policy,base
            gc.collect()
            torch.cuda.empty_cache()
        report[family]=dict(tbptt_full_forward_error=forward,tbptt_unpadded_gradient_error=error,
                            ablation_forward_error=signal_forward,ablation_gradient_decomposition_error=signal_gradient,
                            float64_plan_replay_error=cost_error,modes=checks)
    out=Path('experiments/results/hj_cotangent_dev_v6c')
    out.mkdir(parents=True,exist_ok=True)
    (out/'final_numerical_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    hashes={}
    for path in Path('experiments/hj_cotangent').glob('*.py'):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        destination=out/'audited_source'/path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path,destination)
    (out/'audit_source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
