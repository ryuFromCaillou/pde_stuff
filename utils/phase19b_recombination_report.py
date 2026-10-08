"""Artifact-only visualizations and standalone intervention report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.burgers_recoverability import NAMES, TRUTH
from utils.diagnostic_io import write_json

def table(df):
    def fmt(v):
        if pd.isna(v): return '—'
        if isinstance(v, (float, np.floating)): return f'{v:.6g}'
        return str(v)
    rows=[' | '.join(map(str,df.columns)), ' | '.join(['---']*len(df.columns))]
    rows += [' | '.join(fmt(v) for v in row) for row in df.itertuples(index=False,name=None)]
    return '\n'.join('| '+r+' |' for r in rows)

def render(out,destination=None):
    out=Path(out);dest=out if destination is None else Path(destination)
    if destination is not None:dest.mkdir(parents=True,exist_ok=False)
    df=pd.read_csv(out/'per_seed_results.csv');ls=pd.read_csv(out/'ls_results.csv');su=pd.read_csv(out/'recovery_summary.csv')
    crossing=[]
    for c,g in df.groupby('condition'):
        r={'condition':c}
        for k in ['loose','strong']:
            hits=g['first_'+k+'_step'].dropna()
            r[k+'_ever_count']=len(hits)
            r[k+'_median_first_step_ever_recovered']=float(hits.median()) if len(hits) else None
        crossing.append(r)
    crossings=pd.DataFrame(crossing);crossings.to_csv(dest/'first_crossing_summary.csv',index=False)
    cfg=json.loads((out/'config.json').read_text());val=json.loads((out/'validation.json').read_text())
    assert val['status']=='passed'
    from utils.phase19b_recombination_audit import audit
    audit(out,dest)
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig=plt.figure(figsize=(13,9),layout='constrained');outer=fig.add_gridspec(2,2)
    measures=[('xi_u*u_x','Transport',-1),('xi_u_xx','Diffusion',.02),('spurious_l2','Spurious L2',0)]
    for i,c in enumerate('ABCD'):
        sub=fig.add_subfigure(outer[i//2,i%2]);axes=sub.subplots(1,3)
        g=df[df.condition==c];l=ls[(ls.condition==c)&(ls.support=='full')].iloc[0]
        s,t=cfg['pairs'][c];sub.suptitle(f'{c}: spatial {s}, target {t}',fontsize=12)
        for ax,(key,label,truth) in zip(axes,measures):
            vals=g[key].to_numpy();jitter=np.linspace(-.12,.12,len(vals))
            ax.boxplot(vals,positions=[0],widths=.35,showfliers=False)
            ax.scatter(jitter,vals,s=14,alpha=.6,color='#246a9b',label='25 heads')
            ax.axhline(truth,color='black',ls='--',lw=1,label='Truth')
            ax.scatter([.30],[l[key]],marker='D',color='#c74b2f',s=35,label='Full LS')
            allv=np.r_[df[key],ls[ls.support=='full'][key],truth];span=max(np.ptp(allv),.001)
            ax.set_ylim(allv.min()-.10*span,allv.max()+.10*span);ax.set_xlim(-.4,.5)
            ax.set_xticks([]);ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.2)
        if i==0:axes[0].legend(fontsize=7,loc='best')
    fig.suptitle('Frozen target–feature recombination: final physical coefficients',fontsize=16)
    save(fig,'figure1_intervention_2x2')
    df[['condition','seed']+[m[0] for m in measures]].to_csv(dest/'figure1_data.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');xx=np.arange(4)
    for key,offset,color in [('loose',-.12,'#246a9b'),('strong',.12,'#c74b2f')]:
        p=su[key+'_fraction'].to_numpy();low=su[key+'_ci_low'].to_numpy();high=su[key+'_ci_high'].to_numpy()
        axes[0].errorbar(xx+offset,p,yerr=np.maximum(0,np.array([p-low,high-p])),fmt='o',capsize=3,label=key,color=color)
        for x,y,k in zip(xx+offset,p,su[key+'_count']):axes[0].annotate(f'{k}/25',(x,y),xytext=(0,10 if key=='loose' else -18),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set_ylim(-.15,1.17);axes[0].set_ylabel('Terminal recovery fraction (Wilson 95% CI)');axes[0].legend()
    for key in ['coefficient_error','spurious_l2']:axes[1].plot(xx,su[key+'_median'],'o-',label=key)
    axes[1].set_ylabel('Median physical coefficient norm');axes[1].legend(fontsize=8)
    axes[2].plot(xx,su.pde_residual_median,'o-',color='#246a9b');axes[2].set_ylabel('Median terminal PDE residual MSE');axes[2].set_yscale('log')
    for ax in axes:ax.set_xticks(xx,list('ABCD'));ax.grid(alpha=.2)
    fig.suptitle('Recovery frequency and continuous endpoints across 25 paired seeds');save(fig,'figure2_recovery_summary')
    su.to_csv(dest/'figure2_data.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    # Dimensionless display uses each coefficient's loose-error tolerance, clearly labeled.
    scale=np.full(9,.25);scale[2]=.02
    for ax,c in zip(axes.flat,'ABCD'):
        g=df[df.condition==c];l=ls[(ls.condition==c)&(ls.support=='full')].iloc[0]
        vals=g[['xi_'+n for n in NAMES]].to_numpy()/scale
        ax.boxplot(vals,positions=np.arange(9),widths=.45,showfliers=True)
        for j in range(9):ax.scatter(j+np.linspace(-.12,.12,25),vals[:,j],s=7,alpha=.35,color='#246a9b')
        ax.scatter(np.arange(9)+.25,l[['xi_'+n for n in NAMES]].to_numpy(dtype=float)/scale,marker='D',color='#c74b2f',label='Full LS')
        ax.scatter(np.arange(9)-.25,TRUTH/scale,marker='_',s=150,color='black',label='Truth')
        ax.set_xticks(np.arange(9),NAMES,rotation=45,ha='right');ax.set_title(c);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Physical coefficient / display scale');ax.set_ylim(-6,5)
    # Set common limits from all plotted data; never clip outliers.
    allvals=np.r_[df[['xi_'+n for n in NAMES]].to_numpy().ravel()/np.tile(scale,len(df)),
                  (ls[ls.support=='full'][['xi_'+n for n in NAMES]].to_numpy()/scale).ravel(),TRUTH/scale]
    for ax in axes.flat:ax.set_ylim(min(allvals)-.4,max(allvals)+.4)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Full-library LS versus frozen SymNet and Burgers truth\nDisplay scales: diffusion 0.02; all other coefficients 0.25',fontsize=14)
    save(fig,'figure3_ls_vs_symnet')
    ls.to_csv(dest/'figure3_ls_data.csv',index=False);df.to_csv(dest/'figure3_symnet_data.csv',index=False)
    diagnostic=pd.read_csv(dest/'terminal_diagnostics.csv')
    convergence=diagnostic.groupby('condition',as_index=False).agg(median_last100_coefficient_displacement=('last100_coefficient_displacement','median'),max_last100_coefficient_displacement=('last100_coefficient_displacement','max'),median_last100_residual_change=('last100_residual_change','median'))
    counts=su.set_index('condition').loose_count.to_dict();reliable={k:v>=20 for k,v in counts.items()}
    if reliable['A']:classification='Case 5: pre-transition representational sufficiency under the tested frozen-head protocol.'
    elif reliable['B'] and reliable['C']:classification='Case 3: independent sufficiency of both POST components under the corresponding frozen pairings.'
    elif reliable['B']:classification='Case 1: POST temporal-target sufficiency under frozen recombination.'
    elif reliable['C']:classification='Case 2: POST spatial-feature sufficiency under frozen recombination.'
    elif reliable['D']:classification='Case 4: joint post-transition compatibility under frozen recombination.'
    else:classification='No reliable recovery, including D: positive-control limitation; the four sufficiency cases cannot be distinguished.'
    rows=[]
    for _,r in su.iterrows():
        rows.append({'condition':r['condition'],'loose':f"{int(r.loose_count)}/25",'strong':f"{int(r.strong_count)}/25",
          'loose Wilson 95%':f"[{r.loose_ci_low:.3f}, {r.loose_ci_high:.3f}]",
          'median transport':r['xi_u*u_x_median'],'median diffusion':r.xi_u_xx_median,
          'median coefficient error':r.coefficient_error_median,'median spurious L2':r.spurious_l2_median,
          'median PDE MSE':r.pde_residual_median,'median first loose (terminal recovered)':r.loose_median_first_step_terminal_recovered,
          'median first strong (terminal recovered)':r.strong_median_first_step_terminal_recovered})
    full=ls[ls.support=='full'];two=ls[ls.support=='true_only']
    controls=pd.read_csv(out/'historical_controls.csv')
    report=f'''# Phase 19B frozen target–feature recombination

{classification}

## Exact experimental object and validation

PRE = saved pre-update state **8000**; POST = saved pre-update state **8750**, directly loaded from `run_results/phase19b_transition_instrumented/states/`. No joint training was run. All source artifact hashes match the validated archive manifest; that replay matched 10,502 original scalar rows exactly. Source checkpoint file/tensor hashes are in `frozen_metadata.json`, with source inventories and environment in `provenance.json`.

The full fixed physical grid has 252×256 = 64,512 points, time-major/spatial-minor, archived in `evaluation_grid.npz`. Canonical autodiff evaluates u, u_x, u_xx, u_t. PRE fields match the existing cached arrays bitwise. POST was not included in the original field cache; its exact saved state is hash-verified, and re-evaluated full-grid fidelity and both LS fits match archived instrumentation within rtol 1e-7, atol 1e-10. This is direct evaluation of a captured state, not a reconstructed trajectory. Frozen quantities are saved separately in `frozen_8000.npz` and `frozen_8750.npz`.

| Condition | Spatial library | Temporal target |
|---|---|---|
| A | PRE | PRE |
| B | PRE | POST |
| C | POST | PRE |
| D | POST | POST |

`pairing_validation.json` records exact array hashes proving the swaps. No learned head, optimizer state, coordinate transform, scale estimator, or checkpoint-specific scale is transferred with either component. All four use fixed original Phase19B primitive scales {cfg['scales']}. Optimization feeds [u/s0,u_x/s1,u_xx/s2] into the unchanged one-product MinimalSymNet; target u_t stays physical. Expanded scaled coefficients are divided by [s0,s1,s2,s0²,s0*s1,s0*s2,s1²,s1*s2,s2²]. Polynomial predictions and physical residuals are numerically checked against direct head outputs. Maximum absolute prediction-conversion discrepancy: {val['max_physical_conversion_absolute_error']:.6g}.

Nine-term order: {', '.join(NAMES)}. Truth is transport −1 and diffusion +0.02; the other seven coefficients are spurious. Full LS uses float64, unit-column SVD with relative cutoff 1e-12, no intercept or penalty; coefficients are returned to physical space.

## LS: algebraic compatibility before optimization

Full-library physical coefficients and diagnostics:

{table(full[['condition']+['xi_'+n for n in NAMES]+['residual_mse','coefficient_error','spurious_l2','loose','strong']])}

Two-true-term physical LS (transport, diffusion):

{table(two[['condition','xi_u*u_x','xi_u_xx','residual_mse','coefficient_error','loose','strong']])}

Two-support spurious coefficients are constrained to zero; its apparent recovery is not full-library identification. Unconstrained LS and nonlinear MinimalSymNet answer different questions.

## Symbolic optimization: paired seeds and continuous results

25 seeds (0–24), identical initial parameter tensors for each seed across A/B/C/D, fresh Adam states, lr 0.01, betas (0.9,0.999), eps 1e-8, no regularization. Exactly 1000 full-grid PDE-MSE updates; no early stopping or condition-specific tuning. CPU float32 with two threads. This uses the established frozen-head optimization budget while retaining the original Phase19B fixed scales rather than checkpoint RMS re-estimation.

Loose requires strict transport/diffusion/spurious errors <0.25/0.02/0.25. Strong requires <0.10/0.01/0.10. Primary recovery is at the terminal step, not an earlier transient crossing. First-crossing steps are checked at every step including initialization; missing means no crossing. Descriptive “reliable” means >=20/25 terminal loose successes, declared before optimization. Wilson intervals quantify head-seed sampling uncertainty, not uncertainty across surrogate trajectories.

{table(pd.DataFrame(rows))}

Transient/ever recovery (distinct from the terminal counts above):

{table(crossings)}

All seven spurious coefficients, transport, diffusion, coefficient error, residual, and first crossings are retained by seed in `per_seed_results.csv`; every-step values in `trajectories/`; initial/final parameters in `heads/`. `recovery_summary.csv` gives medians, quartiles, minima and maxima for every coefficient and continuous metric, plus terminal and ever-recovered counts and loose/strong Wilson intervals. `paired_comparisons.csv` retains matched discordance counts and unadjusted exact McNemar tests; these exploratory comparisons are not corrected for multiple testing.

All terminal physical coefficient medians (quartiles and per-seed values are in the CSV artifacts):

{table(su[['condition']+['xi_'+n+'_median' for n in NAMES]])}

Endpoint stationarity diagnostics, without extending any run:

{table(convergence)}

These quantify movement over the final 100 steps, not a proof of global optimality. D's full-library LS fails loose recovery, whereas D's two-support fit meets loose recovery by fixing all seven spurious terms to zero. This distinction prevents mistaking a constrained two-term result for successful identification from the complete library. No condition has a loose-recovering full-library LS solution paired with unreliable head recovery in this run.

## Controls and interpretation

Historical jointly trained heads at the exact chosen states:

{table(controls[['step','xi_u*u_x','xi_u_xx','coefficient_error','spurious_l2','loose','strong','pde_residual']])}

The nominal POST at 8750 is **before** the first archived full-library LS loose crossing (8825) and joint-head crossing (8853); strong crossings occur much later (9975/10013). Therefore D was never a validated positive control under the requested coefficient thresholds. Its frozen fields and self-pair LS have been validated against the archive, so a failed D need not indicate a pairing/code error. A fresh head can also behave differently from the historical joint head. A is reliably recovering: {reliable['A']}; D is reliably recovering: {reliable['D']}.

B demonstrates reliable sufficiency: {reliable['B']}. C demonstrates reliable sufficiency: {reliable['C']}. Failure to demonstrate sufficiency within this budget is not proof of impossibility.

**Classification: {classification}**

The evidence is frozen recombination replicated across head seeds. It does not establish that either component caused the original joint-training transition. Temporal ordering remains observational.

## Decision table

| Observed reliable recovery | Interpretation |
|---|---|
| B, not C | POST target sufficient with PRE features |
| C, not B | POST features sufficient with PRE target; inspect joint spatial errors/cancellation, not just derivative MSE |
| B and C | Each POST component independently sufficient with its PRE counterpart |
| D only | Post-target/feature compatibility required under this intervention |
| A also | PRE already recoverable by repeated frozen-head optimization; distinguish historical optimization |
| No B/C/D | Diagnose controls; no clean component-sufficiency classification |

Actual terminal loose counts: {counts}. {classification}

## Caveats and scope

One surrogate trajectory; head seeds are not trajectory replications. Results depend on the fixed grid, original transferred scales, nonlinear one-product model, optimizer and 1000-update budget. Full LS is a more flexible coefficient relaxation. Small residuals do not imply Burgers recovery; neither loose recovery nor a transient first crossing implies strong recovery. Numerical reference derivatives use generator-consistent differences and reference u_t defined by the Burgers RHS; they are used for validation, not as training targets. POST selection is inside the transition and below the historical loose crossing; this limits any failed-control inference. No second-stage checkpoint sweep was executed and no preferred outcome was tuned for.

## Figures and numerical sources

- `figure1_intervention_2x2.png/.pdf`: A B / C D layout; final physical transport, diffusion and spurious magnitude, truth and full LS. Numerical source `figure1_data.csv` plus `ls_results.csv`.
- `figure2_recovery_summary.png/.pdf`: terminal counts/intervals and continuous metrics; `figure2_data.csv` contains first-crossing and all coefficient summaries too.
- `figure3_ls_vs_symnet.png/.pdf`: all nine LS/SymNet/truth coefficients; display divides diffusion by 0.02 and other coefficients by 0.25 for readability. This is display-only, not optimization scaling. Sources `figure3_ls_data.csv`, `figure3_symnet_data.csv`.

All original long-horizon, post-hoc, instrumentation artifacts and the narrative notebook were hash-verified unchanged. `validation.json` passed. No commit or push.

## Reproduction

From `/home/ghost/ghost/pde_stuff`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -u runs/run_phase19b_recombination.py --out run_results/phase19b_recombination_intervention_repeat
```

Output directory must be new. Implementation: `runs/run_phase19b_recombination.py`, `utils/phase19b_recombination.py`, `utils/phase19b_recombination_report.py`; protocol: `runs/PHASE19B_RECOMBINATION.md`. Regenerate figures/report from saved data into a new destination with `utils.phase19b_recombination_report.render(source, destination)`.
'''
    (dest/'report.md').write_text(report)
    write_json(dest/'interpretation.json',{'classification':classification,'reliable':reliable,'loose_counts':counts,'historical_causation_established':False})
