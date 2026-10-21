"""Full solver-grid observation density study; unchanged physical-head training."""
from __future__ import annotations
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.physics_informed_burgers import dataset, initialize, objective, coefficients, evaluate, write_json, validate
from utils.burgers_six_seed import average_frames, sha, CHECKPOINTS
RATIOS=(1.,.75,.5,.25,.1)
COUNTS=(13056,9792,6528,3264,1306)
SEEDS=tuple(range(6))
SAMPLING_SEED=910064
IDENTITY=('ratio','count','init_seed')

def coverage(selected,grid):
    ti,xi=np.divmod(selected,len(grid['x']))
    tc=np.bincount(ti,minlength=len(grid['t'])); xc=np.bincount(xi,minlength=len(grid['x']))
    return dict(temporal_snapshots=int((tc>0).sum()),spatial_coordinates=int((xc>0).sum()),
        per_time_counts=tc.tolist(),per_space_counts=xc.tolist(),
        t_min=float(grid['t'][ti].min()),t_max=float(grid['t'][ti].max()),
        x_min=float(grid['x'][xi].min()),x_max=float(grid['x'][xi].max()),
        actual_percentage=100*len(selected)/13056)

def make_sampling(count,init_seed,sampling_seed=SAMPLING_SEED):
    historical,full,ref,grid=dataset()
    perm=np.random.default_rng(sampling_seed).permutation(len(full[0]))
    selected=perm[:count]
    train=tuple(v[selected] for v in full)
    model,head=initialize(init_seed)
    initial={'surrogate':model.state_dict(),'head':head.state_dict()}
    return train,full,full,ref,grid,perm,selected,initial

def validate_protocol():
    checks=validate()
    historical,full,ref,grid=dataset()
    assert grid['u'].shape==(51,256)
    # Independent explicit indexing checks every coordinate/target, including float32 conversion.
    ti=np.repeat(np.arange(51),256);xi=np.tile(np.arange(256),51)
    for got,want in zip(full,(grid['t'][ti],grid['x'][xi],grid['u'][ti,xi])):
        np.testing.assert_array_equal(got.numpy().ravel(),want.astype(np.float32))
    assert len(np.unique(np.column_stack([v.numpy().ravel() for v in full[:2]]),axis=0))==13056
    all_coverage=[];previous=set(range(13056))
    for ratio,count in zip(RATIOS,COUNTS):
        a=make_sampling(count,0);b=make_sampling(count,0)
        selected=a[6];assert len(set(selected))==count and set(selected)<=previous
        previous=set(selected);np.testing.assert_array_equal(a[5],b[5])
        for got,source in zip(a[0],full):torch.testing.assert_close(got,source[selected],rtol=0,atol=0)
        for seed in SEEDS:
            m,h=initialize(seed);m2,h2=initialize(seed)
            for x,y in zip(m.parameters(),m2.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
            rng=torch.Generator().manual_seed(10000+seed)
            idx=torch.randint(0,count,(640,),generator=rng)
            assert len(idx)==640 and idx.min()>=0 and idx.max()<count
        all_coverage.append(dict(ratio=ratio,count=count,**coverage(selected,grid)))
    for got,source in zip(historical,full):torch.testing.assert_close(got,source[grid['indices']],rtol=0,atol=0)
    checks.update(full_mapping_verified=True,unique_coordinates=13056,nested=True,fixed_batch_size=640,
        coverage=all_coverage,historical_reference=dict(count=640,percentage=100*640/13056,
        method='structured 10 temporal by 64 spatial Cartesian observations, not random full-grid sampling'))
    return checks

def train_condition(out:Path, ratio:float, count:int, seed:int, steps:int, sampling_seed=SAMPLING_SEED):
    torch.set_num_threads(1); out.mkdir(parents=True,exist_ok=False)
    train,clean,full,ref,grid,perm,selected,initial=make_sampling(count,seed,sampling_seed)
    model,head=initialize(seed); opt=torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=5e-4)
    batch_size=640; brng=torch.Generator().manual_seed(10000+seed)
    checkpoints=sorted(set([0,1,10,50,100,200,500,1000,2000,5000]+list(range(10000,steps+1,10000))+[steps]))
    first={'first_loose_step':None,'first_strong_step':None}; rows=[]; start=time.monotonic()
    write_json(out/'condition.json',dict(ratio=ratio,count=count,init_seed=seed,sampling_seed=sampling_seed,
        population=13056, noise=0., batch_seed=10000+seed, coverage=coverage(selected,grid),
        sampling_method='nested prefix of one uniform random permutation without replacement',batch_size=batch_size,
        batch_sampling='with replacement',permutation_sha256=hashlib.sha256(perm.tobytes()).hexdigest(),
        selected_indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),selected_indices_count=len(selected)))
    np.savez_compressed(out/'sampling_indices.npz',permutation=perm,selected_indices=selected)
    np.savez_compressed(out/'observations.npz', indices=selected, t=train[0].numpy(), x=train[1].numpy(), targets=train[2].numpy())
    torch.save(initial,out/'initial.pt')
    with (out/'history.csv').open('w') as f:
        writer=None
        for step in range(steps+1):
            idx=torch.randint(0,count,(batch_size,),generator=brng)
            losses=objective(model,head,tuple(v[idx] for v in train))
            row=dict(step=step,ratio=ratio,count=count,init_seed=seed,**{k:float(v.detach()) for k,v in losses.items()},**coefficients(head))
            if not all(np.isfinite(v) for v in row.values()): raise RuntimeError('Nonfinite training')
            for level in ('loose','strong'):
                if row[level] and first[f'first_{level}_step'] is None: first[f'first_{level}_step']=step
            if writer is None: writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
            if step in checkpoints:
                metrics,fields=evaluate(model,head,train,full,ref,{**grid,'indices':selected})
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

def validate_condition(folder,ratio,count,seed,steps=100000,deep=True):
    h=pd.read_csv(folder/'history.csv',float_precision='round_trip')
    np.testing.assert_array_equal(h.step,np.arange(steps+1))
    assert np.isfinite(h.to_numpy(dtype=float)).all()
    for key,value in zip(IDENTITY,(ratio,count,seed)):assert (h[key]==value).all()
    for name,target in [('transport',-1.),('diffusion',.02)]:
        np.testing.assert_allclose(h[name+'_abs_error'],abs(h[name]-target),rtol=1e-14,atol=1e-16)
        np.testing.assert_allclose(h[name+'_relative_error'],abs(h[name]-target)/abs(target),rtol=1e-14,atol=1e-16)
    np.testing.assert_allclose(h.total_loss,h.data_loss+h.pde_loss,rtol=2e-7,atol=1e-12)
    assert (h[['data_loss','pde_loss','total_loss']]>=0).all().all()
    summary=json.loads((folder/'summary.json').read_text())
    for key in h.columns:
        assert summary[key]==h[key].iloc[-1],key
    for level,t,d in [('loose',.25,.02),('strong',.10,.01)]:
        np.testing.assert_array_equal(h[level],(h.transport_abs_error<t)&(h.diffusion_abs_error<d))
        crossed=h.loc[h[level],'step']
        assert summary['first_'+level+'_step']==(int(crossed.iloc[0]) if len(crossed) else None)
    expected_steps=sorted(set([s for s in CHECKPOINTS if s<=steps]+[steps]))
    metrics=pd.DataFrame(json.loads((folder/'checkpoint_metrics.json').read_text()))
    np.testing.assert_array_equal(metrics.step,expected_steps)
    assert np.isfinite(metrics.to_numpy(dtype=float)).all()
    for key in metrics.columns:assert summary[key]==metrics[key].iloc[-1],key
    condition=json.loads((folder/'condition.json').read_text())
    for key,value in dict(ratio=ratio,count=count,init_seed=seed,batch_size=640,population=13056,noise=0.,batch_seed=10000+seed,sampling_seed=SAMPLING_SEED).items():assert condition[key]==value
    if deep:
        train,_,full,ref,grid,perm,selected,initial=make_sampling(count,seed)
        with np.load(folder/'sampling_indices.npz') as a:
            np.testing.assert_array_equal(a['permutation'],perm);np.testing.assert_array_equal(a['selected_indices'],selected)
        assert condition['permutation_sha256']==hashlib.sha256(perm.tobytes()).hexdigest()
        assert condition['selected_indices_sha256']==hashlib.sha256(selected.tobytes()).hexdigest()
        assert condition['coverage']==coverage(selected,grid)
        with np.load(folder/'observations.npz') as a:
            for key,want in zip(('indices','t','x','targets'),(selected,*(v.numpy() for v in train))):np.testing.assert_array_equal(a[key],want)
        saved=torch.load(folder/'initial.pt',weights_only=False)
        for group in initial:
            for key in initial[group]:torch.testing.assert_close(saved[group][key],initial[group][key],rtol=0,atol=0)
        rng=torch.Generator().manual_seed(10000+seed)
        for step in range(steps+1):
            idx=torch.randint(0,count,(640,),generator=rng)
            if step not in expected_steps:continue
            state=torch.load(folder/f'state_{step:06d}.pt',weights_only=False)
            assert state['step']==step
            torch.testing.assert_close(state['next_batch_indices'],idx,rtol=0,atol=0)
            torch.testing.assert_close(state['batch_rng'],rng.get_state(),rtol=0,atol=0)
            np.testing.assert_array_equal(state['head']['xi'].numpy(),h.loc[step,['transport','diffusion']].to_numpy(dtype=np.float32))
            model,head=initialize(seed);model.load_state_dict(state['surrogate']);head.load_state_dict(state['head'])
            loss=objective(model,head,tuple(v[idx] for v in train))
            for key in loss:np.testing.assert_allclose(float(loss[key].detach()),h.loc[step,key],rtol=0,atol=0)
            if step==0:
                for key in initial['surrogate']:torch.testing.assert_close(state['surrogate'][key],initial['surrogate'][key],rtol=0,atol=0)
            if step>0:
                for value in state['optimizer']['state'].values():assert int(value['step'])==step
            with np.load(folder/f'fields_{step:06d}.npz') as fields:
                a,b=state['head']['xi'].double().numpy()
                residual=fields['u_t'].astype(float)-a*fields['u'].astype(float)*fields['u_x'].astype(float)-b*fields['u_xx'].astype(float)
                np.testing.assert_array_equal(residual,fields['residual'])
                m=metrics.set_index('step').loc[step]
                np.testing.assert_allclose(m.independent_pde_mse,np.mean(residual**2),rtol=1e-14)
                for name in ('u','u_t','u_x','u_xx'):
                    truth=ref[name]['physical'];pred=fields[name].astype(float)
                    assert pred.shape==(51,256) and np.isfinite(pred).all()
                    np.testing.assert_allclose(m[name.replace('_','')+'_rel_l2'],np.linalg.norm(pred-truth)/np.linalg.norm(truth),rtol=1e-12)
            if step in (0,steps):
                fresh,_=evaluate(model,head,train,full,ref,{**grid,'indices':selected})
                for key in fresh:np.testing.assert_allclose(fresh[key],metrics.set_index('step').loc[step,key],rtol=1e-12,atol=1e-15)
    return h,metrics
