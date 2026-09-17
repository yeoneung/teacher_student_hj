"""CPU-only independent physical-model and convex-reference verification."""
import json
from pathlib import Path
import numpy as np
import torch
from .building import BuildingProblem, BuildingModel, Thermostat, coefficients, solve_qp


def numpy_cost(x0,u,p):
    c = coefficients(p)
    x = np.asarray(x0,dtype=np.float64).copy()
    total = 0.
    for t in range(p.horizon):
        hour = t*p.dt
        weather = -11+4*np.sin(2*np.pi*hour/24-np.pi/2)+p.weather_shift
        price = .12+.38*np.exp(-((hour%24-18)/3)**2)
        comfort = .3+.7/(1+np.exp(-(hour%24-7)*2))/(1+np.exp(-(21-hour%24)*2))
        total += p.dt*(comfort*np.mean(x*x)+price*np.mean(u[t])+.025*np.mean(u[t]**2))
        x = c['a']@x+c['b']*u[t]+c['g']*weather
    return total+2*np.mean(x*x)


def main():
    torch.set_num_threads(1)
    records = []
    for horizon in [12,96,192]:
        p = BuildingProblem(n=8,horizon=horizon)
        model = BuildingModel(p,'cpu')
        gen = np.random.default_rng(67201+horizon)
        x0 = gen.uniform(-4,4,p.n)
        unp = gen.uniform(.2,4.8,(horizon,p.n))
        u = torch.tensor(unp,dtype=torch.float64,requires_grad=True)
        x = torch.tensor(x0[None],dtype=torch.float64)
        total = torch.zeros(1,dtype=torch.float64)
        for t in range(horizon):
            total = total+model.running(x,u[t:t+1],t)
            x = model.step(x,u[t:t+1],t)
        total = total+model.terminal(x)
        gradient = torch.autograd.grad(total.sum(),u)[0].numpy()
        reference = numpy_cost(x0,unp,p)
        forward_error = abs(total.item()-reference)
        direction = gen.normal(size=unp.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-4
        fd = (numpy_cost(x0,unp+eps*direction,p)-numpy_cost(x0,unp-eps*direction,p))/(2*eps)
        derivative_error = abs(fd-np.sum(direction*gradient))
        qp = solve_qp(x0,p)
        qp_replay = abs(qp['cost']-numpy_cost(x0,qp['u'],p))
        assert forward_error<1e-10 and derivative_error<1e-7 and qp_replay<1e-10
        assert qp['objective_replay_error']<1e-7
        assert qp['cost']<=reference+1e-7
        records.append(dict(horizon=horizon,dimension=p.n,forward_error=forward_error,
                            directional_gradient_error=derivative_error,qp_numpy_replay_error=qp_replay,
                            qp={k:v for k,v in qp.items() if k not in ['u','x']}))
    path = Path('experiments/results/hj_building_dev_v6/independent_audit.json')
    path.write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    main()
