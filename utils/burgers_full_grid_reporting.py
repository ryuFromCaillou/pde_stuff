"""Strict six-seed means and auditable full-grid sampling figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from utils.burgers_full_grid import RATIOS,COUNTS,SEEDS,IDENTITY,validate_condition,average_frames,sha,write_json

PANELS={
 'coefficient_trajectories':('history',['transport','diffusion'],False),
 'coefficient_absolute_errors':('history',['transport_abs_error','diffusion_abs_error'],True),
 'coefficient_relative_errors':('history',['transport_relative_error','diffusion_relative_error'],True),
 'training_losses':('history',['total_loss','data_loss','pde_loss'],True),
 'field_derivative_fidelity':('checkpoints',['u_rel_l2','ut_rel_l2','ux_rel_l2','uxx_rel_l2'],True),
 'independent_losses':('checkpoints',['observation_mse','independent_pde_mse'],True),
}

def plots(destination,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True,exist_ok=True)
    for name,(kind,cols,log) in PANELS.items():
        fig,axes=plt.subplots(1,len(cols),figsize=(6*len(cols),5),squeeze=False)
        for ax,col in zip(axes.flat,cols):
            for ratio,g in groups.items():
                mean,std=g[kind];x=mean.step.to_numpy();y=mean[col].to_numpy();d=std[col].to_numpy()
                color=plt.cm.viridis(list(RATIOS).index(ratio)/4)
                ax.plot(x,y,color=color,label=f'{100*ratio:g}% (n=6)')
                lo=y-d
                if log:lo=np.where(lo>0,lo,np.nan)
                ax.fill_between(x,lo,y+d,color=color,alpha=.12,linewidth=0)
            if col in ('transport','diffusion'):ax.axhline(-1 if col=='transport' else .02,color='black',ls='--')
            ax.set(xlabel='Optimizer steps',ylabel=col.replace('_',' '));ax.grid(alpha=.2)
            if log:ax.set_yscale('log')
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(name.replace('_',' ').capitalize()+' — full-grid sampling; mean ± population SD')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(destination/f'{name}.{ext}',dpi=160)
        plt.close(fig)

def publish(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch=__import__('torch');torch.set_num_threads(1)
    config=json.loads((root/'config.json').read_text())
    repo=Path(__file__).resolve().parents[1]
    for name,digest in config['source_hashes'].items():
        assert sha(repo/name)==digest,name
        assert sha(root/'code'/name)==digest,name
    groups={};records=[];audits=[];summaries=[];cover=[]
    for ratio,count in zip(RATIOS,COUNTS):
        hs=[];ms=[]
        for seed in SEEDS:
            folder=root/'raw'/f'ratio_{ratio:g}'/f'seed_{seed}'
            h,m=validate_condition(folder,ratio,count,seed)
            hs.append(h);ms.append(m)
            s=json.loads((folder/'summary.json').read_text())
            records.append(s)
            audits.append(dict(ratio=ratio,seed=seed,passed=True,history_rows=len(h),checkpoint_rows=len(m)))
            if seed==0:cover.append(dict(ratio=ratio,count=count,**json.loads((folder/'condition.json').read_text())['coverage']))
        g={'history':average_frames(hs,IDENTITY),'checkpoints':average_frames(ms)};groups[ratio]=g
        dest=root/'averaged'/f'ratio_{ratio:g}';dest.mkdir(parents=True,exist_ok=True)
        for kind,(mean,std) in g.items():
            mean.to_csv(dest/f'{kind}_mean.csv',index=False);std.to_csv(dest/f'{kind}_std.csv',index=False)
            # Independent round-trip arithmetic checks of every saved mean and SD column.
            frames=hs if kind=='history' else ms
            saved_mean=pd.read_csv(dest/f'{kind}_mean.csv',float_precision='round_trip')
            saved_std=pd.read_csv(dest/f'{kind}_std.csv',float_precision='round_trip')
            for col in mean:
                if col=='step':np.testing.assert_array_equal(saved_mean[col],frames[0][col]);continue
                values=np.array([f[col].to_numpy(dtype=float) for f in frames])
                expected=np.add.reduce(values)/6
                expected_sd=np.sqrt(np.add.reduce((values-expected)**2)/6)
                np.testing.assert_array_equal(saved_mean[col],expected)
                np.testing.assert_array_equal(saved_std[col],expected_sd)
        plots(dest,{ratio:g})
        s=dict(ratio=ratio,nominal_percentage=ratio*100,count=count,n_seeds=6)
        subset=[r for r in records if r['ratio']==ratio]
        for level in ('loose','strong'):
            times=[r['first_'+level+'_step'] for r in subset if r['first_'+level+'_step'] is not None]
            s['endpoint_'+level+'_count']=sum(r[level] for r in subset)
            s['ever_'+level+'_count']=len(times);s['censored_'+level+'_count']=6-len(times)
            for stat,func in [('min',np.min),('median',np.median),('max',np.max),('mean',np.mean),('std',np.std)]:
                s['first_'+level+'_step_'+stat]=float(func(times)) if times else None
        # Include all numeric endpoint diagnostics, not only coefficient metrics.
        for col,value in subset[0].items():
            if col in (*IDENTITY,'step','elapsed_seconds') or col.startswith('first_') or not isinstance(value,(int,float)) or isinstance(value,bool):continue
            values=[r[col] for r in subset];s[col+'_mean']=float(np.mean(values));s[col+'_std']=float(np.std(values))
        summaries.append(s)
    assert len(audits)==30
    pd.DataFrame(records).to_csv(root/'per_seed_summary.csv',index=False)
    recovery_cols=['ratio','count','init_seed','loose','strong','first_loose_step','first_strong_step','transport','diffusion','transport_abs_error','diffusion_abs_error','transport_relative_error','diffusion_relative_error']
    pd.DataFrame(records)[recovery_cols].to_csv(root/'per_seed_recovery.csv',index=False)
    pd.DataFrame(summaries).to_csv(root/'summary_agg.csv',index=False)
    write_json(root/'recovery_summary.json',summaries)
    write_json(root/'coverage.json',cover)
    pd.DataFrame([{k:v for k,v in c.items() if not isinstance(v,list)} for c in cover]).to_csv(root/'coverage.csv',index=False)
    plots(root/'overview',groups)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for ax,col in zip(axes.flat,['transport_abs_error','diffusion_abs_error','transport_relative_error','diffusion_relative_error']):
        ax.errorbar([100*s['ratio'] for s in summaries],[s[col+'_mean'] for s in summaries],yerr=[s[col+'_std'] for s in summaries],fmt='o-',label='Full-grid random subsets: mean ± population SD')
        historical=json.loads((repo/'run_results/physics_informed_burgers/noise_six_seed/recovery_summary.json').read_text())[0]
        ax.errorbar([100*640/13056],[historical['endpoint_mean_'+col]],yerr=[historical['endpoint_std_'+col]],fmt='s',color='gray',label='Historical structured 640 (4.9%), separate reference')
        ax.set(xlabel='Nominal retained full-grid percentage',ylabel=col.replace('_',' '));ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7);fig.suptitle('Endpoint coefficient accuracy versus full-grid sampling density');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(root/'overview'/f'final_coefficient_accuracy.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for row,level in enumerate(['loose','strong']):
        matrix=np.array([[int(r[level]) for r in records if r['ratio']==ratio] for ratio in RATIOS])
        axes[row,0].imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        axes[row,0].set(xticks=range(6),yticks=range(5),yticklabels=[f'{100*r:g}%' for r in RATIOS],xlabel='Initialization seed',ylabel='Full-grid sampling',title=f'Endpoint {level} recovery (1=yes)')
        for i in range(5):
            for j in range(6):axes[row,0].text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j] else 'black')
        for seed in SEEDS:
            rs=[r for r in records if r['init_seed']==seed]
            for r in rs:
                t=r['first_'+level+'_step'];axes[row,1].scatter(100*r['ratio']+(seed-2.5)*.6,t if t is not None else 105000,marker='o' if t is not None else 'x',color=plt.cm.tab10(seed))
        axes[row,1].set(xlabel='Nominal full-grid percentage (seed offsets for visibility)',ylabel='First crossing step',title=f'{level.capitalize()}: circles=crossings; ×=no crossing by 100000')
        axes[row,1].axhline(100000,ls=':',color='gray');axes[row,1].grid(alpha=.2)
    fig.suptitle('Per-seed endpoint outcomes and first recovery times');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(root/'overview'/f'per_seed_recovery.{ext}',dpi=160)
    plt.close(fig)
    protected=json.loads((root/'protected_artifacts.json').read_text())
    for name,digest in protected.items():assert sha(repo/name)==digest,name
    write_json(root/'final_audit.json',dict(passed=True,conditions=audits,validated_conditions=30,protected_artifacts=len(protected),
        original_artifacts_unchanged=True,source_hashes_verified=True,independent_saved_means_and_population_sd=True,
        all_saved_checkpoint_losses_recomputed=True,initial_and_final_full_grid_diagnostics_recomputed=True,
        all_saved_fields_reference_errors_and_residuals_verified=True,batch_sequences_verified=True,interpolation=False))
    lines=['# Six-seed full-grid sampling convergence','',
        'All 30 independently trained conditions passed the artifact audit. Each received 100000 Adam steps with fixed 640-point replacement minibatches, lr 0.0005, unit data/PDE weights, zero noise and the original physical-head SIREN. No model modifications, L1, SymNet or LS extraction.',
        '', 'Arithmetic means and population SD (ddof=0) use exactly six matching histories. Signed coefficients and continuous per-seed errors are averaged separately. Histories contain every state 0–100000; clean diagnostics use only the 20 recorded checkpoints. No interpolation. Log bands omit portions where mean minus SD is nonpositive.',
        '', '|Full-grid %|Count|Endpoint loose|Endpoint strong|Ever loose|Ever strong|Transport absolute error mean ± SD|Diffusion absolute error mean ± SD|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"|{s['nominal_percentage']:g}|{s['count']}|{s['endpoint_loose_count']}/6|{s['endpoint_strong_count']}/6|{s['ever_loose_count']}/6|{s['ever_strong_count']}/6|{s['transport_abs_error_mean']:.7g} ± {s['transport_abs_error_std']:.7g}|{s['diffusion_abs_error_mean']:.7g} ± {s['diffusion_abs_error_std']:.7g}|")
    lines+=['','Loose recovery requires absolute transport/diffusion errors <0.25/<0.02; strong requires <0.10/<0.01. These are per-seed strict inequalities. First crossings need not persist. Noncrossers are censored at 100000; first-time min/median/max/mean/SD summarize crossers only, with denominators and censor counts retained. Six seeds are descriptive optimization replicates, not population recovery probabilities.',
        '', '## Coverage and interpretation',
        'One permutation (NumPy default_rng seed 910064) defines nested subsets shared across all initialization seeds. Time-major flat index i maps to time i//256 and space i%256. All coordinate/target mappings and uniqueness were checked before training. Coverage JSON retains per-time and per-space counts; CSV reports distinct coverage and ranges. Nominal 10% is 1306/13056 = 10.0030637%.',
        'The historical 640 observations form a structured 10×64 Cartesian selection, 4.9019608% of the solver grid. They are a separately labeled reference, never a full-grid random sample.',
        'Observation density and PDE collocation coverage change simultaneously because the unchanged objective evaluates both losses at the sampled observation coordinates. Differences cannot be attributed exclusively to data density. The common 51×256 evaluation grid includes training points (all points at 100%); it is not a held-out or new-initial-condition generalization test. Derivative references use numerical periodic differences; reference u_t is the generator Burgers RHS, not analytic truth. This is known-support coefficient estimation.',
        '', '## Artifacts',
        '`overview/` and `averaged/ratio_*/` contain PNG/PDF figures. Dense means/SD CSVs accompany every level. `per_seed_summary.csv` contains endpoint field/derivative diagnostics, coefficients and recovery. `summary_agg.csv` includes all numeric endpoint mean/SD metrics and crossing statistics. `per_seed_recovery.csv`, `recovery_summary.json`, `coverage.csv/json`, source snapshots, configuration and hashes, validation, smoke evidence, gate review, and final audit support independent review. Raw histories, states, optimizer/batch RNG and field caches remain under `raw/`.',
        '', 'The fixed budget measures finite-horizon behavior; no per-seed tuning or early stopping was used. Endpoint errors and first crossings describe different aspects of convergence; a mean coefficient curve is not a per-seed recovery test.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'manifest.json',{str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in root.rglob('*') if p.is_file() and p.name not in ('manifest.json','execution.log','execution.pid','gate.log')})
    print('All 30 conditions validated; means, figures, report and audit published.',flush=True)
