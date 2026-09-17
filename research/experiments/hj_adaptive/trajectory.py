"""All teacher state covectors in one reverse sweep; no fitted value network."""
import torch
from experiments.hj_cotangent.systems import running,terminal,step
from experiments.hj_gridfree.engine import capture


def trajectory_covectors(initial,teacher,p,parameter_gradient=False):
    x=initial;states=[x];controls=[];cost=torch.zeros(len(x),device=x.device,dtype=x.dtype)
    for t in range(p.horizon):
        u=teacher(x,t);controls.append(u)
        cost=cost+running(x,u,p,t);x=step(x,u,p,t);states.append(x)
    cost=cost+terminal(x,p)
    parameters=list(teacher.net.parameters()) if parameter_gradient else []
    gradients=torch.autograd.grad(cost.sum(),states[1:]+parameters)
    result=(cost,torch.stack(states,1),torch.stack(controls,1),torch.stack(gradients[:p.horizon],1))
    if parameter_gradient:result=result+(torch.cat([g.reshape(-1) for g in gradients[p.horizon:]])/len(initial),)
    return result


def advantage(x,t,old_u,anchor,covector,student,p):
    u=student(x,t);z=step(x,u,p,t)
    value=(covector*(z-anchor)).sum(-1)
    value=torch.where(t==p.horizon-1,terminal(z,p)-terminal(anchor,p),value)
    return running(x,u,p,t)-running(x,old_u,p,t)+value


class TrajectoryGraph:
    def __init__(self,teacher,p,batch,parameter_gradient=False):
        self.parameter_gradient=parameter_gradient
        teacher.eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        if parameter_gradient:
            for parameter in teacher.net.parameters():parameter.requires_grad_(True)
        self.x=torch.zeros(batch,p.dim,device='cuda',requires_grad=True)
        self.cost=torch.zeros(batch,device='cuda')
        self.states=torch.zeros(batch,p.horizon+1,p.dim,device='cuda')
        self.controls=torch.zeros(batch,p.horizon,p.n,device='cuda')
        self.covectors=torch.zeros(batch,p.horizon,p.dim,device='cuda')
        outputs=[self.cost,self.states,self.controls,self.covectors]
        if parameter_gradient:
            self.full_gradient=torch.zeros(sum(z.numel() for z in teacher.net.parameters()),device='cuda')
            outputs.append(self.full_gradient)
        def evaluate():
            result=trajectory_covectors(self.x,teacher,p,parameter_gradient)
            with torch.no_grad():
                for dest,src in zip(outputs,result):dest.copy_(src)
        self.graph=capture(evaluate)

    def __call__(self,x):
        with torch.no_grad():self.x.copy_(x)
        self.graph.replay()
        result=(self.cost.clone(),self.states.clone(),self.controls.clone(),self.covectors.clone())
        if self.parameter_gradient:result=result+(self.full_gradient.clone(),)
        return result
