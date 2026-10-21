"""Correct-support Burgers recovery; no symbolic search or LS extraction."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from prog.featlib import FeatureTensor
from prog.mlps import SirenMLP
from utils.derivative_utils import (build_burgers_reference_derivative_grids,
    evaluate_diagnostic_fields, evaluate_primitive_feature_metrics, compute_error_metrics)

FEATURES = ('u', 'u_x', 'u_xx')
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ('pauls_tweaks_diag_notebook.ipynb', 'notebook/diagnostics/burgers_minimal_discovery_story.ipynb')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def hashes(paths):
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


class PhysicalHead(torch.nn.Module):
    """Unconstrained physical coefficients, initialized to zero without truth access."""
    def __init__(self):
        super().__init__()
        self.xi = torch.nn.Parameter(torch.zeros(2))

    def forward(self, features):
        return self.xi[0]*features[:, 0:1]*features[:, 1:2] + self.xi[1]*features[:, 2:3]


def initialize(seed):
    # Deliberately independent of the notebook's discarded pretraining RNG history.
    torch.manual_seed(seed)
    model = SirenMLP(hidden_size=64, hidden_layers=3, first_omega_0=1., hidden_omega_0=1.)
    return model, PhysicalHead()


def dataset():
    x, _, _, (t, u) = solve_burgers(seed=0, nu=.02, T=.1, return_history=True)
    ti = np.linspace(0, len(t)-1, 10, dtype=int)
    xi = np.linspace(0, len(x)-1, 64, dtype=int)
    tm, xm = np.meshgrid(t, x, indexing='ij')
    full = tuple(torch.tensor(a.reshape(-1,1), dtype=torch.float32) for a in (tm,xm,u))
    indices = (ti[:,None]*len(x)+xi[None,:]).ravel()
    train = tuple(a[indices] for a in full)
    ref = build_burgers_reference_derivative_grids(u_grid=u, x_grid=x, nu=.02)
    return train, full, ref, dict(t=t, x=x, u=u, indices=indices, time_indices=ti, space_indices=xi)


def objective(model, head, batch):
    t, x, target = batch
    t, x = (a.detach().clone().requires_grad_(True) for a in (t,x))
    u = model(t,x)
    ut = torch.autograd.grad(u,t,torch.ones_like(u),create_graph=True)[0]
    f = FeatureTensor(FEATURES, normalize=False).build(u,t=t,x=x)
    data = torch.nn.functional.mse_loss(u,target)
    pde = torch.nn.functional.mse_loss(ut,head(f.F))
    return dict(data_loss=data, pde_loss=pde, total_loss=data+pde)


def coefficients(head):
    a,b = head.xi.detach().tolist()
    te,de = abs(a+1),abs(b-.02)
    loose,strong = te<.25 and de<.02, te<.10 and de<.01
    return dict(transport=a,diffusion=b,transport_abs_error=te,diffusion_abs_error=de,
                transport_relative_error=te,diffusion_relative_error=de/.02,
                loose=loose,strong=strong)


def evaluate(model, head, train, full, ref, grid):
    t,x,u = (a.numpy().ravel() for a in full)
    def reference_builder(_t,_x,_u,names):
        return dict(feature_names=names, values=np.column_stack([ref[n]['physical'].ravel() for n in names]))
    primitive = evaluate_primitive_feature_metrics(model,t,x,u,FEATURES,reference_feature_builder=reference_builder)
    fields = evaluate_diagnostic_fields(model,t,x,grid['u'].shape,FEATURES,chunk_size=2048)
    a,b = head.xi.detach().double().numpy()
    # Independent NumPy residual from canonical evaluation, bypasses head/objective.
    residual = fields['u_t'].astype(float)-a*fields['u'].astype(float)*fields['u_x'].astype(float)-b*fields['u_xx'].astype(float)
    metrics = dict(primitive['summary_flat'])
    for name in (*FEATURES,'u_t'):
        pred = fields[name].astype(float); truth=ref[name]['physical']
        key=name.replace('_','')
        metrics.update({f'{key}_{k}':v for k,v in compute_error_metrics(pred,truth).items()})
        metrics[f'{key}_cosine']=float(np.vdot(pred,truth)/(np.linalg.norm(pred)*np.linalg.norm(truth)+1e-30))
        metrics[f'{key}_pred_rms']=float(np.sqrt(np.mean(pred**2)))
        metrics[f'{key}_reference_rms']=float(np.sqrt(np.mean(truth**2)))
    with torch.no_grad():
        metrics['observation_mse']=float(torch.mean((model(train[0],train[1])-train[2])**2))
    metrics['independent_pde_mse']=float(np.mean(residual**2))
    metrics['observation_pde_mse']=float(np.mean(residual.ravel()[grid['indices']]**2))
    metrics['observation_total_loss']=metrics['observation_mse']+metrics['observation_pde_mse']
    check=objective(model,head,train)
    discrepancy=abs(float(check['pde_loss'].detach())-metrics['observation_pde_mse'])
    metrics['independent_residual_check_abs_error']=discrepancy
    if not np.isclose(float(check['pde_loss'].detach()),metrics['observation_pde_mse'],rtol=2e-4,atol=2e-7):
        raise RuntimeError('Independent residual check failed')
    fields['residual']=residual
    return metrics,fields


def validate():
    """Analytic derivatives, head gradient, objective gradient and solver accuracy."""
    torch.set_num_threads(1)
    class Analytic(torch.nn.Module):
        def forward(self,t,x): return torch.exp(-t)*torch.sin(x)
    t=torch.linspace(0,.1,41,dtype=torch.float64).reshape(-1,1)
    x=torch.linspace(.1,6,41,dtype=torch.float64).reshape(-1,1)
    h=PhysicalHead().double()
    with torch.no_grad(): h.xi.copy_(torch.tensor([.3,-.4],dtype=torch.float64))
    u=torch.exp(-t)*torch.sin(x); ux=torch.exp(-t)*torch.cos(x)
    residual=-u-h.xi[0]*u*ux+h.xi[1]*u
    expected=(residual**2).mean()
    actual=objective(Analytic(),h,(t,x,u))
    torch.testing.assert_close(actual['pde_loss'],expected,rtol=1e-12,atol=1e-12)
    g=torch.autograd.grad(actual['total_loss'],h.xi)[0]
    want=torch.stack([(-2*residual*u*ux).mean(),(2*residual*u).mean()])
    torch.testing.assert_close(g,want,rtol=1e-12,atol=1e-12)
    model,head=initialize(0)
    train,full,ref,grid=dataset()
    loss=objective(model,head,train)['total_loss']; loss.backward()
    assert head.xi.grad.abs().min()>0
    assert sum(float(p.grad.square().sum()) for p in model.parameters())>0
    before=head.xi.detach().clone()
    opt=torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=5e-4)
    opt.step(); assert torch.all(head.xi.detach()!=before)
    # Central finite differences of one surrogate weight verify coupled PDE gradient.
    model,head=initialize(7); model.double(); head.double()
    with torch.no_grad(): head.xi.copy_(torch.tensor([.3,-.4]))
    batch=tuple(v[:31].double() for v in train)
    p=next(model.parameters()); value=float(p[0,0].detach())
    ag=float(torch.autograd.grad(objective(model,head,batch)['total_loss'],p)[0][0,0])
    vals=[]
    for sign in (1,-1):
        with torch.no_grad(): p[0,0]=value+sign*1e-6
        vals.append(float(objective(model,head,batch)['total_loss'].detach()))
    fd=(vals[0]-vals[1])/2e-6
    np.testing.assert_allclose(ag,fd,rtol=2e-5,atol=1e-7)
    # Same-grid temporal refinement; this does not alter training observations.
    _,_,_,(fine_t,fine_u)=solve_burgers(seed=0,nu=.02,T=.1,dt=.001,return_history=True)
    temporal=float(np.linalg.norm(grid['u']-fine_u[::2])/np.linalg.norm(grid['u']))
    assert temporal<1e-5
    _,_,_,(_,spatial_u)=solve_burgers(seed=0,nu=.02,T=.1,N=512,dt=.0005,return_history=True,history_every=1)
    spatial=float(np.linalg.norm(grid['u']-spatial_u[::4,::2])/np.linalg.norm(grid['u']))
    assert spatial<.01
    return dict(passed=True,analytic_loss_error=float(abs(actual['pde_loss']-expected).detach()),
                head_gradient_check=True,joint_updates=True,surrogate_autograd=ag,surrogate_finite_difference=fd,
                solver_time_refinement_rel_l2=temporal,solver_space_time_refinement_rel_l2=spatial,
                reference='Numerical centered periodic spatial derivatives; u_t is generator RHS, not analytic truth.')


def train_seed(out,seed,steps):
    torch.set_num_threads(1)
    out.mkdir(exist_ok=False)
    model,head=initialize(seed)
    train,full,ref,grid=dataset()
    optimizer=torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=5e-4)
    batch_rng=torch.Generator().manual_seed(10000+seed)
    checkpoints=sorted(set([0,1,10,50,100,200,500,1000,2000,5000]+list(range(10000,steps+1,10000))+[steps]))
    checkpoints=[s for s in checkpoints if s<=steps]
    first=dict(first_loose_step=None,first_strong_step=None)
    start=time.monotonic(); metrics_rows=[]
    with (out/'history.csv').open('w') as f:
        writer=None
        for step in range(steps+1):
            idx=torch.randint(0,640,(640,),generator=batch_rng)
            losses=objective(model,head,tuple(v[idx] for v in train))
            row=dict(step=step,**{k:float(v.detach()) for k,v in losses.items()},**coefficients(head))
            if not all(np.isfinite(v) for v in row.values()): raise RuntimeError('Nonfinite training')
            for level in ('loose','strong'):
                if row[level] and first[f'first_{level}_step'] is None: first[f'first_{level}_step']=step
            if writer is None: writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
            if step in checkpoints:
                metrics,fields=evaluate(model,head,train,full,ref,grid)
                metrics_rows.append(dict(step=step,**metrics))
                write_json(out/'checkpoint_metrics.json',metrics_rows)
                np.savez_compressed(out/f'fields_{step:06d}.npz',**fields)
                torch.save(dict(step=step,surrogate=model.state_dict(),head=head.state_dict(),optimizer=optimizer.state_dict(),
                    batch_rng=batch_rng.get_state(),next_batch_indices=idx),out/f'state_{step:06d}.pt')
            if step%1000==0 or step==steps:
                f.flush()
                write_json(out/'progress.json',dict(seed=seed,**row,**first,elapsed_seconds=time.monotonic()-start))
                print(f'seed={seed} step={step} xi=({row["transport"]:.6g},{row["diffusion"]:.6g}) loss={row["total_loss"]:.5g} seconds={time.monotonic()-start:.1f}',flush=True)
            if step==steps: break
            optimizer.zero_grad(set_to_none=True)
            losses['total_loss'].backward(); optimizer.step()
    summary=dict(seed=seed,**row,**metrics,**first,elapsed_seconds=time.monotonic()-start,
                 status='strong' if row['strong'] else 'loose' if row['loose'] else 'not_recovered')
    write_json(out/'summary.json',summary)
    return summary


def summarize(out,summaries):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.DataFrame(summaries); df.to_csv(out/'seed_summary.csv',index=False)
    reliability=df[df.seed!=0]
    distributions={c:{k:(float(v) if np.isfinite(v) else None) for k,v in reliability[c].describe().items()} for c in ('transport','diffusion')} if len(reliability) else {}
    write_json(out/'seed_distribution.json',dict(repeats=len(reliability),loose_count=int(reliability.loose.sum()),
        strong_count=int(reliability.strong.sum()),distributions=distributions,
        failures=reliability.loc[~reliability.strong,'seed'].tolist()))
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for s in summaries:
        h=pd.read_csv(out/f'seed_{s["seed"]}'/'history.csv').iloc[::100]
        for ax,col in zip(axs.flat,('transport','diffusion','data_loss','pde_loss')):
            ax.plot(h.step,h[col],label=f'seed {s["seed"]}'); ax.set(xlabel='Optimizer steps',ylabel=col)
            if 'loss' in col: ax.set_yscale('log')
    axs[0,0].axhline(-1,color='black',ls='--');axs[0,1].axhline(.02,color='black',ls='--')
    axs[0,0].legend();fig.suptitle('Clean Burgers: physical coefficient and minibatch loss trajectories');fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(out/f'trajectories.{ext}')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for s in summaries:
        m=pd.DataFrame(json.loads((out/f'seed_{s["seed"]}'/'checkpoint_metrics.json').read_text()))
        axs[0].plot(m.step,m.u_rel_l2,label=f'seed {s["seed"]}')
        for col in ('ut_rel_l2','ux_rel_l2','uxx_rel_l2'):
            axs[1].plot(m.step,m[col],label=f'{s["seed"]}: {col}')
    for ax in axs: ax.set(xlabel='Optimizer steps',ylabel='Relative L2 error',yscale='log');ax.legend(fontsize=6)
    fig.suptitle('Clean-grid field and numerical-reference derivative fidelity');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'fidelity.{ext}')
    plt.close(fig)
    lines=['# Physics-informed Burgers clean baseline','',
        'This is correct-support parameter estimation, not unrestricted PDE discovery. No LS or L1 was used.',
        'Seed 0 is the gate; seeds 1–5 are five additional independent optimization replicates on the identical clean dataset.',
        '', '|seed|transport|diffusion|observation MSE|grid MSE|field relative L2|ut relative L2|ux relative L2|uxx relative L2|independent PDE MSE|observation total|first loose|first strong|final|',
        '|---|'+'---|'*13]
    for s in summaries:
        vals=[s['seed']]+[f'{s[k]:.7g}' for k in ('transport','diffusion','observation_mse','u_mse','u_rel_l2','ut_rel_l2','ux_rel_l2','uxx_rel_l2','independent_pde_mse','observation_total_loss')]+[s['first_loose_step'],s['first_strong_step'],s['status']]
        lines.append('|'+ '|'.join(map(str,vals))+'|')
    lines+=['',f'Reliability endpoint recovery: loose {int(reliability.loose.sum())}/{len(reliability)}; strong {int(reliability.strong.sum())}/{len(reliability)}.',
        'Loose absolute-error limits: transport <0.25, diffusion <0.02. Strong: <0.10 and <0.01. Both must hold; no spurious terms exist. These intentionally retain historical loose/strong limits, which allow substantial relative diffusion error. Continuous absolute and relative errors are in seed_summary.csv.',
        'First crossings are checked every optimizer step and need not persist. Coefficient distributions (mean/std/min/quartiles/max) and failed seeds are in seed_distribution.json. All checkpoint metrics and figure data are retained. Finite differences are numerical references; u_t is the clean generator RHS. Low residual does not establish accurate derivatives.',
        'Sampling uses 640 draws with replacement per optimizer step; these are not epochs. Full evaluation uses all 51×256 clean solver points, including the original 640 observations. This is interpolation within one numerical solution, not generalization to unseen initial conditions.',
        'See config.json, validation.json, reproduction.txt, and runs/PHYSICS_INFORMED_BURGERS.md. Noise and sparsity studies remain unexecuted.']
    if len(reliability) and not reliability.strong.all():
        lines+=['','Clean recovery is unreliable under the strong criterion. Inspect failed coefficient trajectories and field/derivative errors before any robustness study; no per-seed tuning is permitted.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
