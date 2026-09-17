"""Box-constrained convex first-order lower bound for the building reference."""
import numpy as np
from .building import coefficients


def certificate(x0,u,p):
    c = coefficients(p)
    x = np.asarray(x0,dtype=np.float64).copy()
    xs,weights,prices = [],[],[]
    cost = 0.
    for t in range(p.horizon):
        hour = t*p.dt
        weather = -11+4*np.sin(2*np.pi*hour/24-np.pi/2)+p.weather_shift
        price = .12+.38*np.exp(-((hour%24-18)/3)**2)
        weight = .3+.7/(1+np.exp(-(hour%24-7)*2))/(1+np.exp(-(21-hour%24)*2))
        xs.append(x)
        weights.append(weight)
        prices.append(price)
        cost += p.dt*(weight*np.mean(x*x)+price*np.mean(u[t])+.025*np.mean(u[t]**2))
        x = c['a']@x+c['b']*u[t]+c['g']*weather
    cost += 2*np.mean(x*x)
    adjoint = 4*x/p.n
    gradient = np.zeros_like(u)
    for t in range(p.horizon-1,-1,-1):
        gradient[t] = p.dt*(prices[t]+.05*u[t])/p.n+c['b']*adjoint
        adjoint = 2*p.dt*weights[t]*xs[t]/p.n+c['a'].T@adjoint
    # Convexity: f(v) >= f(u)+grad f(u)^T(v-u). Minimize the RHS on the box.
    gap = np.sum(np.where(gradient>=0,gradient*u,gradient*(u-p.bound)))
    return dict(feasible_cost=float(cost),convex_lower_bound=float(cost-gap),
                first_order_gap=float(gap),max_abs_gradient=float(np.max(np.abs(gradient))))
