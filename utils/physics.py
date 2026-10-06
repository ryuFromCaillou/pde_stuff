"""Physical-coordinate losses shared by exact diagnostic replays."""
import torch
from torch import nn


def phase19b_batch_losses(model, sym, scales, t, x, u, idx):
    """Preserve the archived Phase 19B forward/autodiff operation order exactly."""
    tb=t[idx].detach().clone().requires_grad_(True)
    xb=x[idx].detach().clone().requires_grad_(True)
    up=model(tb,xb)
    ut=torch.autograd.grad(up,tb,torch.ones_like(up),create_graph=True,retain_graph=True)[0]
    ux=torch.autograd.grad(up,xb,torch.ones_like(up),create_graph=True,retain_graph=True)[0]
    uxx=torch.autograd.grad(ux,xb,torch.ones_like(ux),create_graph=True,retain_graph=True)[0]
    rhs=sym(torch.cat([up,ux,uxx],1)/scales.reshape(1,3))
    data=nn.functional.mse_loss(up,u[idx]);pde=nn.functional.mse_loss(ut,rhs)
    return data,pde,data+.5*pde
