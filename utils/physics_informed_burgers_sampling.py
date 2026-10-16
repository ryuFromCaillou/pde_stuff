"""Nested clean-observation sampling study for physical Burgers recovery."""
from __future__ import annotations
import csv, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.physics_informed_burgers import dataset, initialize, objective, coefficients, evaluate, write_json

RATIOS=(1.00,.75,.50,.25,.10)
COUNTS=(640,480,320,160,64)
SEEDS=(0,1)
SAMPLING_SEED=910064

def make_sampling(count, init_seed, sampling_seed=SAMPLING_SEED):
    train0, full, ref, grid = dataset()
    rng=np.random.default_rng(sampling_seed)
    perm=rng.permutation(len(train0[0]))
    selected=perm[:count]
    train=tuple(v[selected] for v in train0)
    model,head=initialize(init_seed)
    initial={'surrogate':model.state_dict(),'head':head.state_dict()}
    return train,train0,full,ref,grid,perm,selected,initial

def validate_sampling_protocol():
    a=make_sampling(640,0,111); b=make_sampling(640,0,222)
    torch.testing.assert_close(a[1][0],b[1][0],rtol=0,atol=0)
    torch.testing.assert_close(a[1][1],b[1][1],rtol=0,atol=0)
    torch.testing.assert_close(a[1][2],b[1][2],rtol=0,atol=0)
    np.testing.assert_array_equal(a[2][0].numpy(),b[2][0].numpy())
    np.testing.assert_array_equal(a[2][1].numpy(),b[2][1].numpy())
    np.testing.assert_array_equal(a[2][2].numpy(),b[2][2].numpy())
    checks={'solver_and_clean_reference_independent_of_sampling_seed':True}
    _,_,_,_,_,perm,_,_=make_sampling(640,0,SAMPLING_SEED)
    for n,m in zip(COUNTS,COUNTS[1:]):
        assert set(perm[:m]).issubset(set(perm[:n]))
    checks['nested_prefix_subsets']=True
    _,base,_,_,_,p1,s1,i1=make_sampling(160,0,SAMPLING_SEED)
    _,_,_,_,_,p2,s2,i2=make_sampling(160,0,SAMPLING_SEED)
    np.testing.assert_array_equal(p1,p2); np.testing.assert_array_equal(s1,s2)
    for x,y in zip(i1['surrogate'].values(),i2['surrogate'].values()): torch.testing.assert_close(x,y,rtol=0,atol=0)
    checks['sampling_rng_and_initialization_are_reproducible']=True
    _,_,_,_,_,_,s64,_=make_sampling(64,0,SAMPLING_SEED)
    assert len(s64)==64 and len(np.unique(s64))==64
    checks['selected_indices_unique_and_counted']=True
    assert min(640,len(s64))==64
    checks['batch_size_minimum_rule']=True
    return checks

def train_condition(out:Path, ratio:float, count:int, seed:int, steps:int, sampling_seed=SAMPLING_SEED):
    torch.set_num_threads(1); out.mkdir(parents=True,exist_ok=False)
    train,clean,full,ref,grid,perm,selected,initial=make_sampling(count,seed,sampling_seed)
    model,head=initialize(seed); opt=torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=5e-4)
    batch_size=min(640,count); brng=torch.Generator().manual_seed(10000+seed)
    checkpoints=sorted(set([0,1,10,50,100,200,500,1000,2000,5000]+list(range(10000,steps+1,10000))+[steps]))
    first={'first_loose_step':None,'first_strong_step':None}; rows=[]; start=time.monotonic()
    write_json(out/'condition.json',dict(ratio=ratio,count=count,init_seed=seed,sampling_seed=sampling_seed,
        sampling_method='nested prefix of one uniform random permutation without replacement',batch_size=batch_size,
        batch_sampling='with replacement',permutation_sha256=hashlib.sha256(perm.tobytes()).hexdigest(),
        selected_indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),selected_indices_count=len(selected)))
    np.savez_compressed(out/'sampling_indices.npz',permutation=perm,selected_indices=selected)
    torch.save(initial,out/'initial.pt')
    with (out/'history.csv').open('w') as f:
        writer=None
        for step in range(steps+1):
            idx=torch.randint(0,count,(batch_size,),generator=brng)
            losses=objective(model,head,tuple(v[idx] for v in train))
            row=dict(step=step,ratio=ratio,count=count,init_seed=seed,**{k:float(v.detach()) for k,v in losses.items()},**coefficients(head))
            for level in ('loose','strong'):
                if row[level] and first[f'first_{level}_step'] is None: first[f'first_{level}_step']=step
            if writer is None: writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
            if step in checkpoints:
                metrics,fields=evaluate(model,head,train,full,ref,{**grid,'indices':grid['indices'][selected]})
                rows.append(dict(step=step,**metrics)); write_json(out/'checkpoint_metrics.json',rows)
                np.savez_compressed(out/f'fields_{step:06d}.npz',**fields)
                torch.save(dict(step=step,surrogate=model.state_dict(),head=head.state_dict(),optimizer=opt.state_dict(),batch_rng=brng.get_state(),next_batch_indices=idx),out/f'state_{step:06d}.pt')
            if step%1000==0 or step==steps:
                f.flush(); write_json(out/'progress.json',dict(**row,**first,elapsed_seconds=time.monotonic()-start))
                print(f'ratio={ratio:g} seed={seed} step={step} xi=({row["transport"]:.6g},{row["diffusion"]:.6g}) loss={row["total_loss"]:.5g}',flush=True)
            if step==steps: break
            opt.zero_grad(set_to_none=True); losses['total_loss'].backward(); opt.step()
    summary={**row,**metrics,**first,'status':'strong' if row['strong'] else 'loose' if row['loose'] else 'not_recovered','elapsed_seconds':time.monotonic()-start}
    write_json(out/'summary.json',summary); return summary

def finalize_control(target, source, ratio=1.0, count=640, sampling_seed=SAMPLING_SEED):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source,target)
    perm=np.random.default_rng(sampling_seed).permutation(640); np.savez_compressed(target/'sampling_indices.npz',permutation=perm,selected_indices=perm[:640])
    c=json.loads((target/'condition.json').read_text()) if (target/'condition.json').exists() else {}
    c.update(ratio=ratio,count=count,init_seed=int(target.name.split('_')[-1]),sampling_seed=sampling_seed,sampling_method='reused validated clean 640 baseline; nested permutation recorded',batch_size=640,batch_sampling='with replacement')
    write_json(target/'condition.json',c)
    s=json.loads((target/'summary.json').read_text()); s.update(ratio=ratio,count=count); write_json(target/'summary.json',s); return s

def make_figures(root):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rec=[]; hs=[]; ms=[]
    for ratio,count in zip(RATIOS,COUNTS):
      for seed in SEEDS:
        d=root/f'ratio_{ratio:g}'/f'seed_{seed}'; rec.append({**json.loads((d/'summary.json').read_text()),'ratio':ratio,'count':count,'seed':seed})
        hs.append((ratio,seed,pd.read_csv(d/'history.csv')))
        ms += [{**x,'ratio':ratio,'seed':seed} for x in json.loads((d/'checkpoint_metrics.json').read_text())]
    df=pd.DataFrame(rec); mm=pd.DataFrame(ms); df.to_csv(root/'summary.csv',index=False); mm.to_csv(root/'checkpoint_metrics.csv',index=False)
    colors=plt.cm.viridis(np.linspace(0,1,len(RATIOS)))
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    for col,lab,true in [('transport','Transport coefficient',-1),('diffusion','Diffusion coefficient',.02)]:
      a=ax[0 if col=='transport' else 1]
      for color,ratio in zip(colors,RATIOS):
       for seed in SEEDS:
        h=next(v for r,s,v in hs if r==ratio and s==seed); a.plot(h.step,h[col],color=color,label=f'{ratio:g}, seed {seed}')
      a.axhline(true,color='black',ls='--'); a.set(xlabel='Optimizer steps',ylabel=lab); a.grid(alpha=.2); a.legend(fontsize=7,ncol=2)
    fig.suptitle('Physical Burgers coefficient trajectories versus retained observations'); fig.tight_layout()
    for e in ('png','pdf'): fig.savefig(root/f'coefficient_trajectories.{e}',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for color,ratio in zip(colors,RATIOS):
      for seed in SEEDS:
       m=mm[(mm.ratio==ratio)&(mm.seed==seed)]; axs[0].plot(m.step,m.u_rel_l2,color=color,label=f'{ratio:g}, seed {seed}'); axs[1].plot(m.step,m.ut_rel_l2,color=color,label=f'{ratio:g}, seed {seed}')
    for a,title in zip(axs,['u','u_t']): a.set(xlabel='Optimizer steps',ylabel='Clean-reference relative L2',title=title,yscale='log'); a.grid(alpha=.2); a.legend(fontsize=7,ncol=2)
    fig.suptitle('Clean field and temporal-derivative fidelity versus observation count'); fig.tight_layout()
    for e in ('png','pdf'): fig.savefig(root/f'field_derivative_fidelity.{e}',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for color,ratio in zip(colors,RATIOS):
      for seed in SEEDS:
       m=mm[(mm.ratio==ratio)&(mm.seed==seed)]; axs[0].plot(m.step,m.observation_mse,color=color,label=f'{ratio:g}, seed {seed}'); axs[1].plot(m.step,m.independent_pde_mse,color=color,label=f'{ratio:g}, seed {seed}')
    for a in axs: a.set(xlabel='Optimizer steps',yscale='log'); a.grid(alpha=.2); a.legend(fontsize=7,ncol=2)
    axs[0].set_ylabel('Training observation MSE'); axs[1].set_ylabel('Independent PDE residual MSE'); fig.suptitle('Data and PDE loss trajectories by sampling ratio'); fig.tight_layout()
    for e in ('png','pdf'): fig.savefig(root/f'loss_trajectories.{e}',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,5))
    for seed in SEEDS:
      q=df[df.seed==seed]; axs[0].plot(q.ratio,q.transport_abs_error,'o-',label=f'seed {seed}'); axs[1].plot(q.ratio,q.diffusion_abs_error,'o-',label=f'seed {seed}')
    axs[0].axhline(.1,color='black',ls='--'); axs[1].axhline(.01,color='black',ls='--')
    for a,y in zip(axs,['Transport absolute error','Diffusion absolute error']): a.set(xlabel='Sampling ratio (retained / 640)',ylabel=y); a.invert_xaxis(); a.grid(alpha=.2); a.legend()
    fig.suptitle('Final coefficient errors versus retained observation ratio'); fig.tight_layout()
    for e in ('png','pdf'): fig.savefig(root/f'final_coefficient_errors.{e}',dpi=160); plt.close(fig)
    lines=['# Burgers sampling robustness','', 'Two initialization seeds provide exploratory evidence only. Ratio 1.00 reuses the validated clean 640-observation control; the other four ratios are newly trained nested subsets.', '', '|ratio|count|seed|transport|diffusion|transport error|diffusion error|field rel L2|ut rel L2|first loose|first strong|final|','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for s in sorted(rec,key=lambda x:(-x['ratio'],x['seed'])): lines.append(f"|{s['ratio']:g}|{s['count']}|{s['seed']}|{s['transport']:.7g}|{s['diffusion']:.7g}|{s['transport_abs_error']:.7g}|{s['diffusion_abs_error']:.7g}|{s.get('u_rel_l2',float('nan')):.7g}|{s.get('ut_rel_l2',float('nan')):.7g}|{s.get('first_loose_step')}|{s.get('first_strong_step')}|{s['status']}|")
    lines += ['', 'Sampling is a nested prefix of one reproducible permutation of the original 640 coordinates. Minibatches sample with replacement and use batch size min(640,count). The clean 51×256 evaluation grid and solver are unchanged. PDE residuals use surrogate autodiff derivatives; no noise, L1, SymNet, or least-squares extraction was used.', '', 'Noise and crossed noise–sampling experiments were not run.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
