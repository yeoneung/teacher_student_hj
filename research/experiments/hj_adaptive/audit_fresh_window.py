"""Check the fallback gradient/update identity on separate development data."""
import argparse
import copy
import json
import torch
from experiments.hj_cotangent.systems import problem,teacher,actor,sample
from experiments.hj_cotangent.core import LocalGraph,JetGraph
from experiments.hj_gridfree.engine import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1);rows=[]
    for family,horizon in [('mechanical',160),('reaction',320),('building',192)]:
        p=problem(family,16 if family=='mechanical' else 32,horizon)
        torch.manual_seed(98701);base=teacher(p);fresh=actor(p,base,64).cuda()
        queried=actor(p,teacher(p),64).cuda();queried.load_state_dict(copy.deepcopy(fresh.state_dict()))
        direct=LocalGraph(fresh,p,128,16,lr=.002,mode='exact',teacher=base)
        cached=LocalGraph(queried,p,128,16,lr=.002,mode='cache',rho=1.)
        jet=JetGraph(queried,teacher(p),p,128,16)
        differences=[]
        for iteration in range(5):
            x=sample(p,128,98801+iteration);t=torch.arange(128,device='cuda')%horizon
            labels=jet(x,t);cached.update(x,t,*labels);direct.update(x,t);torch.cuda.synchronize()
            dif=max((a-b).abs().max().item() for a,b in zip(fresh.net.parameters(),queried.net.parameters()))
            differences.append(dif)
        assert max(differences)<2e-5,(family,differences)
        rows.append(dict(family=family,horizon=horizon,updates=5,parameter_max_abs_differences=differences))
        del base,fresh,queried,direct,cached,jet;torch.cuda.empty_cache()
    result=dict(passed=True,rows=rows,scope='Matched clipped-Adam updates with newly queried boundary sensitivities versus direct Fresh differentiation. Independent development inputs at the actual primary horizons.')
    write(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
