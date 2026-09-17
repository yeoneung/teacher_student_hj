"""Independent NumPy plant checks distinguish local error from unstable replay."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .building import coefficients


def numpy_step_cost(x,u,p,t=0,building_coefficients=None):
    if p.family=='building':
        c=building_coefficients if building_coefficients is not None else coefficients(p)
        hour=t*p.dt
        weather=-11+4*np.sin(2*np.pi*hour/24-np.pi/2)+p.weather_shift
        price=.12+.38*np.exp(-((hour%24-18)/3)**2)
        weight=.3+.7/(1+np.exp(-(hour%24-7)*2))/(1+np.exp(-(21-hour%24)*2))
        cost=p.dt*(weight*np.mean(x*x,axis=-1)+price*np.mean(u,axis=-1)+.025*np.mean(u*u,axis=-1))
        return x@c['a'].T+c['b']*u+c['g']*weather,cost
    if p.mechanical:
        q=x[...,:p.n];v=x[...,p.n:]
        cost=p.dt*np.mean(2*(1-np.cos(q))+.08*v*v+.04*u*u+.2*(1-np.cos(q-np.roll(q,-1,axis=-1))),axis=-1)
        vn=v+p.dt*(5*np.sin(q)+p.gamma*(np.sin(np.roll(q,1,axis=-1)-q)+np.sin(np.roll(q,-1,axis=-1)-q))+u-.2*v)
        return np.concatenate((q+p.dt*vn,vn),axis=-1),cost
    cost=p.dt*np.mean(x*x+.1*u*u+.1*(x-np.roll(x,-1,axis=-1))**2,axis=-1)
    y=x+p.dt*(p.gamma*(np.roll(x,1,axis=-1)+np.roll(x,-1,axis=-1)-2*x)+1.2*x-.4*x**3+u)
    return y,cost


def numpy_terminal(x,p):
    if p.family=='building':return 2*np.mean(x*x,axis=-1)
    if p.mechanical:return np.mean(8*(1-np.cos(x[...,:p.n]))+.4*x[...,p.n:]**2,axis=-1)
    return 3*np.mean(x*x,axis=-1)


def local_trajectory_audit(states,controls,costs,p):
    """Check every stored transition and cost without accumulating state errors."""
    x=states.detach().cpu().numpy().astype(np.float64)
    u=controls.detach().cpu().numpy().astype(np.float64)
    j=costs.detach().cpu().numpy().astype(np.float64)
    assert x.shape[1]==p.horizon+1 and u.shape[1]==p.horizon
    c=coefficients(p) if p.family=='building' else None
    accumulated=np.zeros(len(x));residual=0.
    with threadpool_limits(limits=1):
        for t in range(p.horizon):
            y,cost=numpy_step_cost(x[:,t],u[:,t],p,t,c)
            residual=max(residual,float(np.max(np.abs(y-x[:,t+1])/(1+np.abs(x[:,t+1])))))
            accumulated+=cost
    accumulated+=numpy_terminal(x[:,-1],p)
    cost_error=float(np.max(np.abs(accumulated-j)/(1+np.abs(j))))
    assert residual<1e-11 and cost_error<1e-11,(residual,cost_error)
    return dict(max_scaled_single_step_residual=residual,independent_scaled_trajectory_cost_error=cost_error)


def independent_feedback_cost(x,policy,p):
    """NumPy float64 plant with the same feedback actor, same batch and device.

    This checks plant implementation, not a second neural inference engine.
    All states in the matching batch are propagated to preserve inference shape.
    """
    device=x.device
    state=x.detach().cpu().numpy().astype(np.float64).copy()
    total=np.zeros(len(state))
    c=coefficients(p) if p.family=='building' else None
    with threadpool_limits(limits=1),torch.no_grad():
        for t in range(p.horizon):
            controls=policy(torch.from_numpy(state).to(device),t).cpu().numpy().astype(np.float64)
            state,cost=numpy_step_cost(state,controls,p,t,c)
            total+=cost
    return torch.tensor(total+numpy_terminal(state,p),dtype=torch.float64)
