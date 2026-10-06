"""Read-only gradient and actual-update diagnostics on archived replay states."""
import numpy as np
import torch
from utils.physics import phase19b_batch_losses
from utils.transition_geometry import cosine


def arr(t):return t.detach().cpu().numpy().astype(np.float64)


def gradients(model,sym,scales,t,x,u,indices):
    ps=list(model.parameters())+list(sym.parameters())
    data,pde,total=phase19b_batch_losses(model,sym,scales,t,x,u,indices)
    gd=torch.autograd.grad(data,ps,retain_graph=True,allow_unused=True)
    gp=torch.autograd.grad(pde,ps,allow_unused=True)
    def cat(gs,params):return np.concatenate([arr(torch.zeros_like(p) if g is None else g).ravel() for g,p in zip(gs,params)])
    n=len(list(model.parameters()))
    return {'data_loss':float(data.detach()),'pde_loss':float(pde.detach()),'total_loss':float(total.detach())}, {
        'surrogate':(cat(gd[:n],ps[:n]),cat(gp[:n],ps[:n])),
        'symnet':(cat(gd[n:],ps[n:]),cat(gp[n:],ps[n:]))}


def update_metrics(model,sym,update,grads):
    row={}
    allparams=list(model.parameters())+list(sym.parameters())
    group=update['optimizer_after']['param_groups'][0]
    ids=group['params'];b1,b2=group['betas'];eps=group['eps'];lr=group['lr']
    offset=0
    for name,params in [('surrogate',list(model.parameters())),('symnet',list(sym.parameters()))]:
        gd,gp=grads[name];gt=gd+.5*gp
        delta=arr(update[name+'_delta']);theta=np.concatenate([arr(p).ravel() for p in params])
        nm=lambda z:float(np.linalg.norm(z))
        row.update({name+'_data_gradient_norm':nm(gd),name+'_pde_gradient_norm':nm(gp),name+'_total_gradient_norm':nm(gt),
                    name+'_update_norm':nm(delta),name+'_relative_update_norm':nm(delta)/nm(theta),
                    name+'_update_cos_negative_pde':cosine(delta,-gp),name+'_update_cos_negative_total':cosine(delta,-gt)})
        if name=='surrogate':
            row.update(surrogate_data_pde_cosine=cosine(gd,gp),surrogate_raw_pde_data_ratio=nm(gp)/nm(gd),
                       surrogate_weighted_pde_data_ratio=.5*nm(gp)/nm(gd),surrogate_update_cos_negative_data=cosine(delta,-gd),
                       surrogate_gradient_cancellation_ratio=nm(gt)/(nm(gd)+.5*nm(gp)))
        mh=[];vh=[]
        for i in ids[offset:offset+len(params)]:
            st=update['optimizer_after']['state'][i];step=float(st['step'])
            mh.append(arr(st['exp_avg']).ravel()/(1-b1**step));vh.append(arr(st['exp_avg_sq']).ravel()/(1-b2**step))
        offset+=len(params);mh=np.concatenate(mh);vh=np.concatenate(vh)
        inv=1/(np.sqrt(vh)+eps);expected=-lr*mh*inv
        row.update({name+'_momentum_norm':nm(mh),name+'_momentum_cos_total_gradient':cosine(mh,gt),
                    name+'_update_cos_negative_momentum':cosine(delta,-mh),
                    name+'_effective_preconditioning_gain':nm(mh*inv)/nm(mh),
                    name+'_adam_formula_update_max_abs_error':float(np.max(abs(delta-expected))),
                    name+'_adam_formula_update_relative_error':nm(delta-expected)/nm(delta)})
        for q in [.1,.5,.9]:row[name+f'_inverse_denominator_p{int(q*100)}']=float(np.quantile(inv,q))
    return row
