"""Strict archival validation and six-seed characterization of the original noise study."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.physics_informed_burgers import write_json
from utils.physics_informed_burgers_noise import SIGMAS, make_condition

SEEDS = tuple(range(6))
STEPS = 100000
CHECKPOINTS = [0,1,10,50,100,200,500,1000,2000,5000] + list(range(10000,100001,10000))
IDENTITY = ['sigma', 'init_seed', 'noise_seed']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def validate_history(h, sigma, seed):
    if not np.array_equal(h.step.to_numpy(), np.arange(STEPS+1)):
        raise ValueError('History must contain every step 0..100000 exactly once')
    for key, value in zip(IDENTITY, (sigma,seed,700000+seed)):
        if key not in h or not (h[key] == value).all(): raise ValueError(f'Wrong {key}')
    if not np.isfinite(h.to_numpy(dtype=float)).all(): raise ValueError('Nonfinite history')
    for name, target in [('transport',-1.),('diffusion',.02)]:
        np.testing.assert_allclose(h[name+'_abs_error'], abs(h[name]-target), rtol=1e-14, atol=1e-16)
        np.testing.assert_allclose(h[name+'_relative_error'], abs(h[name]-target)/abs(target), rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(h.total_loss,h.data_loss+h.pde_loss,rtol=2e-7,atol=1e-12)
    for level,t,d in [('loose',.25,.02),('strong',.10,.01)]:
        np.testing.assert_array_equal(h[level],(h.transport_abs_error<t)&(h.diffusion_abs_error<d))

def validate_condition(folder, sigma, seed, *, observations=True):
    h = pd.read_csv(folder/'history.csv',float_precision='round_trip')
    validate_history(h,sigma,seed)
    m = pd.DataFrame(json.loads((folder/'checkpoint_metrics.json').read_text()))
    if not np.array_equal(m.step, CHECKPOINTS): raise ValueError(f'Checkpoint coordinates differ: {folder}')
    if not np.isfinite(m.to_numpy(dtype=float)).all(): raise ValueError(f'Nonfinite checkpoint: {folder}')
    condition = json.loads((folder/'condition.json').read_text())
    for key,value in zip(IDENTITY,(sigma,seed,700000+seed)):
        if condition[key]!=value: raise ValueError(f'Condition identity: {folder}')
    if observations:
        expected=make_condition(sigma=sigma,init_seed=seed,noise_seed=700000+seed)
        with np.load(folder/'observations.npz') as obs:
            for key,value in [('indices',expected[4]['indices']),('epsilon',expected[5]),
                              ('clean_targets',expected[1][2].numpy().ravel()),('noisy_targets',expected[0][2].numpy().ravel())]:
                np.testing.assert_array_equal(obs[key],value)
        if condition['epsilon_sha256'] != hashlib.sha256(expected[5].tobytes()).hexdigest(): raise ValueError('Noise hash mismatch')
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
        for group in ('surrogate','head'):
            for name,value in expected[7][group].items():
                torch.testing.assert_close(initial[group][name],value,rtol=0,atol=0)
        for step in CHECKPOINTS:
            state=torch.load(folder/f'state_{step:06d}.pt',map_location='cpu',weights_only=False)
            if state['step'] != step: raise ValueError('State step mismatch')
            np.testing.assert_array_equal(state['head']['xi'].numpy(),h.loc[step,['transport','diffusion']].to_numpy(dtype=np.float32))
    s=json.loads((folder/'summary.json').read_text())
    for name in ('transport','diffusion','transport_abs_error','diffusion_abs_error'):
        if s[name]!=h[name].iloc[-1]: raise ValueError(f'Summary mismatch {name}')
    for level in ('loose','strong'):
        crossed=h.loc[h[level],'step']
        first=int(crossed.iloc[0]) if len(crossed) else None
        if s['first_'+level+'_step']!=first: raise ValueError('First recovery mismatch')
    return h,m

def average_frames(frames, identity=()):
    if len(frames)!=6: raise ValueError('Exactly six histories required')
    columns=[c for c in frames[0] if c not in identity and c!='step']
    for f in frames:
        if list(f.columns)!=list(frames[0].columns): raise ValueError('Metric definitions/columns differ')
        if not np.array_equal(f.step,frames[0].step): raise ValueError('Step coordinates differ; no interpolation allowed')
    values=np.stack([f[columns].to_numpy(dtype=float) for f in frames])
    if not np.isfinite(values).all(): raise ValueError('Nonfinite metrics')
    mean=pd.DataFrame(values.mean(axis=0),columns=columns)
    std=pd.DataFrame(values.std(axis=0,ddof=0),columns=columns)
    for f in (mean,std): f.insert(0,'step',frames[0].step.to_numpy())
    return mean,std

# Original 1x2 panels, labels, scales, and target/threshold conventions.
PANELS = {
 'coefficient_trajectories': ('history', ['transport','diffusion'], ['Transport coefficient','Diffusion coefficient'], False,
    'Physical Burgers coefficient trajectories', [[(-1,'black','--')],[(.02,'black','--')]]),
 'coefficient_errors': ('history', ['transport_abs_error','diffusion_abs_error'], ['Absolute transport error','Absolute diffusion error'], True,
    'Physical coefficient errors (mean of per-seed absolute errors)', [[(.10,'black','--'),(.25,'gray',':')],[(.01,'black','--'),(.02,'gray',':')]]),
 'field_derivative_fidelity': ('checkpoints', ['u_rel_l2','ut_rel_l2'], ['Clean-reference relative L2','Clean-reference relative L2'], True,
    'Clean-reference field and temporal-derivative fidelity', [[],[]]),
 'loss_trajectories': ('checkpoints', ['observation_mse','independent_pde_mse'], ['Noisy-training observation MSE','Independent clean-grid PDE residual MSE'], True,
    'Training observation fit and clean-reference PDE residual', [[],[]]),
 'training_losses': ('history', ['total_loss','data_loss','pde_loss'], ['Total training loss','Data loss','PDE loss'], True,
    'Minibatch training losses', [[],[],[]]),
 'trajectories': ('history', ['transport','diffusion','data_loss','pde_loss'], ['transport','diffusion','data_loss','pde_loss'], False,
    'Physical coefficients and minibatch losses', [[(-1,'black','--')],[(.02,'black','--')],[],[]]),
}

def _plot_panels(destination, datasets, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True,exist_ok=True)
    for name,(kind,columns,labels,log,title,targets) in PANELS.items():
        layout=(2,2) if name=='trajectories' else (1,len(columns))
        fig,axes=plt.subplots(*layout,figsize=(12,8) if layout==(2,2) else (6.5*len(columns),5),squeeze=False)
        for i,(column,ylabel,lines) in enumerate(zip(columns,labels,targets)):
            ax=axes.flat[i];use_log=log or (name=='trajectories' and 'loss' in column)
            for sigma,group in datasets.items():
                mean,std=group[kind]
                x=mean.step.to_numpy(); y=mean[column].to_numpy(); d=std[column].to_numpy()
                color=plt.cm.viridis(np.linspace(0,1,len(SIGMAS)))[list(SIGMAS).index(sigma)]
                curve,=ax.plot(x,y,color=color,label=f'σ={sigma:g}, mean of 6')
                if len(datasets)==1:
                    lower=y-d
                    if use_log: lower=np.where(lower>0,lower,np.nan)
                    ax.fill_between(x,lower,y+d,color=curve.get_color(),alpha=.16,linewidth=0)
            for value,color,style in lines:ax.axhline(value,color=color,ls=style)
            ax.set(xlabel='Optimizer steps',ylabel=ylabel)
            if use_log:ax.set_yscale('log')
            if name=='field_derivative_fidelity':ax.set_title(['u','u_t, canonical autodiff'][i])
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7,ncol=2 if len(datasets)>1 else 1)
        fig.suptitle(f'{title} — {label}')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(destination/f'{name}.{ext}',dpi=160)
        plt.close(fig)


def _plot_all_checkpoints(destination,mean):
    import matplotlib.pyplot as plt
    columns=[c for c in mean if c!='step']
    for page,start in enumerate(range(0,len(columns),6),1):
        fig,axes=plt.subplots(2,3,figsize=(16,8))
        for ax,column in zip(axes.flat,columns[start:start+6]):
            ax.plot(mean.step,mean[column]);ax.set(xlabel='Optimizer steps',ylabel=column);ax.grid(alpha=.2)
            if (mean[column]>0).all() and 'cosine' not in column:ax.set_yscale('log')
        for ax in list(axes.flat)[len(columns[start:start+6]):]:ax.set_visible(False)
        fig.suptitle(f'All archived checkpoint diagnostics — {destination.name}, arithmetic mean of six seeds')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(destination/f'checkpoint_diagnostics_{page}.{ext}',dpi=160)
        plt.close(fig)


def publish(root,require_all=True):
    """Publish complete noise levels only; preserve raw histories and reject incompatible input."""
    all_groups={};records=[];coverage=[];missing=[]
    for sigma in SIGMAS:
        folders=[root/'raw'/f'sigma_{sigma:g}'/f'seed_{seed}' for seed in SEEDS]
        absent=[seed for seed,folder in zip(SEEDS,folders) if not (folder/'summary.json').exists()]
        if absent:
            missing.append({'sigma':sigma,'missing_or_incomplete_seeds':absent});continue
        histories=[];checkpoints=[]
        for seed,folder in zip(SEEDS,folders):
            h,m=validate_condition(folder,sigma,seed,observations=False)
            histories.append(h);checkpoints.append(m)
            endpoint=h.iloc[-1]
            record=dict(sigma=sigma,seed=seed)
            for col in ['transport','diffusion','transport_abs_error','diffusion_abs_error',
                        'transport_relative_error','diffusion_relative_error','loose','strong']:
                value=endpoint[col];record['endpoint_'+col]=value.item() if hasattr(value,'item') else value
            for level in ('loose','strong'):
                crossed=h.loc[h[level],'step']
                record['first_'+level+'_step']=int(crossed.iloc[0]) if len(crossed) else None
                record['ever_'+level]=len(crossed)>0
            records.append(record)
        means,stds=average_frames(histories,IDENTITY)
        checkpoint_mean,checkpoint_std=average_frames(checkpoints)
        group={'history':(means,stds),'checkpoints':(checkpoint_mean,checkpoint_std)}
        all_groups[sigma]=group
        folder=root/'averaged'/f'sigma_{sigma:g}';folder.mkdir(parents=True,exist_ok=True)
        for kind,(mean,std) in group.items():
            mean.to_csv(folder/f'{kind}_mean.csv',index=False)
            std.to_csv(folder/f'{kind}_std.csv',index=False)
        write_json(folder/'validation.json',dict(passed=True,n_seeds=6,seeds=list(SEEDS),
            identical_step_coordinates=True,identical_metric_columns=True,finite=True,
            history_rows=len(means),checkpoint_rows=len(checkpoint_mean),
            signed_coefficients_averaged_directly=True,errors_averaged_separately=True,
            uncertainty='Population std, ddof=0; log shading omitted where mean-std <=0'))
        _plot_panels(folder,{sigma:group},f'σ={sigma:g}; six-seed mean ± population SD')
        _plot_all_checkpoints(folder,checkpoint_mean)
        coverage.append({'sigma':sigma,'n_seeds':6,'history_metrics':[c for c in means if c!='step'],
                         'checkpoint_metrics':[c for c in checkpoint_mean if c!='step']})
    write_json(root/'aggregation_validation.json',dict(passed=not missing,complete_noise_levels=list(all_groups),
        expected_conditions=36,validated_conditions=6*len(all_groups),missing_or_incomplete=missing,
        interpolation=False,truncation=False,coverage=coverage))
    if not all_groups:
        if require_all:raise ValueError(f'No complete six-seed noise levels: {missing}')
        return
    _plot_panels(root/'overview',all_groups,'six-seed means across observation noise levels')
    per_seed=pd.DataFrame(records);per_seed.to_csv(root/'per_seed_recovery.csv',index=False)
    summaries=[]
    for sigma,group in all_groups.items():
        subset=per_seed[per_seed.sigma==sigma];mean=group['history'][0]
        r=dict(sigma=sigma,n_seeds=6)
        for level,t,d in [('loose',.25,.02),('strong',.10,.01)]:
            r['ever_'+level+'_count']=int(subset['ever_'+level].sum())
            r['endpoint_'+level+'_count']=int(subset['endpoint_'+level].sum())
            times=subset['first_'+level+'_step'].dropna()
            for stat in ('min','median','max'):
                r['first_'+level+'_step_'+stat]=float(getattr(times,stat)()) if len(times) else None
            hit=(abs(mean.transport+1)<t)&(abs(mean.diffusion-.02)<d)
            r['mean_curve_first_'+level+'_step']=int(mean.loc[hit,'step'].iloc[0]) if hit.any() else None
            r['mean_curve_endpoint_'+level]=bool(hit.iloc[-1])
        for c in ('transport','diffusion','transport_abs_error','diffusion_abs_error','transport_relative_error','diffusion_relative_error'):
            r['endpoint_mean_'+c]=float(subset['endpoint_'+c].mean())
            r['endpoint_std_'+c]=float(subset['endpoint_'+c].std(ddof=0))
        r['endpoint_error_of_mean_transport']=float(abs(mean.transport.iloc[-1]+1))
        r['endpoint_error_of_mean_diffusion']=float(abs(mean.diffusion.iloc[-1]-.02))
        summaries.append(r)
    pd.DataFrame(summaries).to_csv(root/'recovery_summary.csv',index=False)
    write_json(root/'recovery_summary.json',summaries)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,column,target in zip(axes,['transport','diffusion'],[-1,.02]):
        ax.errorbar([s['sigma'] for s in summaries],[s['endpoint_mean_'+column] for s in summaries],
            yerr=[s['endpoint_std_'+column] for s in summaries],fmt='o-',label='Mean ± population SD, n=6')
        ax.axhline(target,color='black',ls='--');ax.set(xlabel='Noise level σ',ylabel=f'Final {column} coefficient');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Final physical coefficient accuracy versus observation noise — six-seed means');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(root/'overview'/f'final_coefficients.{ext}',dpi=160)
    plt.close(fig)
    lines=['# Burgers six-seed averaged convergence','',
        f'Complete noise levels: {len(all_groups)}/6. Validated conditions included in means: {6*len(all_groups)}/36.',
        'Missing or incomplete levels: '+json.dumps(missing), '',
        'Arithmetic means use six histories at identical coordinates, without interpolation, truncation, smoothing, or metric substitution. '
        'The axis is optimizer steps (0–100000), not dataset epochs. Dense minibatch losses and coefficients are recorded each step; '
        'fixed-grid diagnostics are recorded at the original 20 checkpoints only. Population standard deviation uses ddof=0.', '',
        'Original SIREN, two-coefficient physical head, losses, Adam, training budget, data locations, and clean references are unchanged. '
        'Noise exception: the original design varies epsilon with initialization seed (700000+seed), pairing it across noise levels. '
        'This extension preserves that design; outcomes combine initialization, batch-sequence, and noise-realization variation. '
        'No claim of isolating initialization variability is made. Original artifacts remain unchanged.', '',
        'Loose recovery: |transport+1|<0.25 AND |diffusion−0.02|<0.02. Strong: <0.10 AND <0.01. '
        'First crossings are not sustained recovery. A null time means no crossing within 100000 updates. '
        'Recovery of the averaged signed coefficients is separate from per-seed recovery; errors are calculated per seed before averaging.', '',
        '|Noise|Ever loose|Endpoint loose|Ever strong|Endpoint strong|Loose first-step median|Strong first-step median|Mean final transport abs error|Mean final diffusion abs error|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        vals=[s[k] for k in ['sigma','ever_loose_count','endpoint_loose_count','ever_strong_count','endpoint_strong_count','first_loose_step_median','first_strong_step_median','endpoint_mean_transport_abs_error','endpoint_mean_diffusion_abs_error']]
        lines.append('|'+ '|'.join(str(v) if v is not None else 'non-recovery' for v in vals)+'|')
    lines+=['','Per-seed first crossings, endpoints, continuous absolute/relative errors, and explicit non-recoveries: `per_seed_recovery.csv`. '
        'Counts, successful-crossing time ranges/medians, signed endpoint means/SDs, and separate mean-curve crossings: `recovery_summary.csv`.', '',
        'Original four epoch-dependent noise plots are reproduced at each noise level and in `overview/`: '
        '`coefficient_trajectories`, `coefficient_errors`, `field_derivative_fidelity`, `loss_trajectories`. '
        'The additional `training_losses` includes every original objective component and total. '
        '`trajectories` preserves the clean baseline 2×2 layout. All other recorded checkpoint metrics appear in `checkpoint_diagnostics_*`. '
        'The non-temporal endpoint plot is reproduced as `overview/final_coefficients`.', '',
        'Per-seed raw histories, observations, checkpoint metrics/states and fields: `raw/sigma_*/seed_*`. '
        'Averaged numerical histories and population SDs: `averaged/sigma_*/*_mean.csv` and `*_std.csv`. '
        'Recovery booleans averaged in history CSVs are fractions of recovering seeds, not recovery decisions for the mean curve.', '',
        'Limitations: full-grid diagnostic values between the original checkpoints were never recorded and are not reconstructed. '
        'Clean baseline seed 2–5 raw histories were unavailable, so those runs were rerun. Known-support parameter estimation is not unrestricted symbolic discovery.', '',
        'Reproduce figures from archived histories: see `reproduction.txt`. Source code snapshots and original implementation hashes are archived in `code/` and `config.json`.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    if require_all and missing:raise ValueError(f'Incomplete six-seed study: {missing}')
    if not missing:
        manifest={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in root.rglob('*')
                  if p.is_file() and p.name not in ('manifest.json','execution.log','execution.pid','finalization.log','finalization.pid')}
        write_json(root/'manifest.json',manifest)
