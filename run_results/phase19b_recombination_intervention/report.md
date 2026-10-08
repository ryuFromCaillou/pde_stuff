# Phase 19B frozen target–feature recombination

No reliable recovery, including D: positive-control limitation; the four sufficiency cases cannot be distinguished.

## Exact experimental object and validation

PRE = saved pre-update state **8000**; POST = saved pre-update state **8750**, directly loaded from `run_results/phase19b_transition_instrumented/states/`. No joint training was run. All source artifact hashes match the validated archive manifest; that replay matched 10,502 original scalar rows exactly. Source checkpoint file/tensor hashes are in `frozen_metadata.json`, with source inventories and environment in `provenance.json`.

The full fixed physical grid has 252×256 = 64,512 points, time-major/spatial-minor, archived in `evaluation_grid.npz`. Canonical autodiff evaluates u, u_x, u_xx, u_t. PRE fields match the existing cached arrays bitwise. POST was not included in the original field cache; its exact saved state is hash-verified, and re-evaluated full-grid fidelity and both LS fits match archived instrumentation within rtol 1e-7, atol 1e-10. This is direct evaluation of a captured state, not a reconstructed trajectory. Frozen quantities are saved separately in `frozen_8000.npz` and `frozen_8750.npz`.

| Condition | Spatial library | Temporal target |
|---|---|---|
| A | PRE | PRE |
| B | PRE | POST |
| C | POST | PRE |
| D | POST | POST |

`pairing_validation.json` records exact array hashes proving the swaps. No learned head, optimizer state, coordinate transform, scale estimator, or checkpoint-specific scale is transferred with either component. All four use fixed original Phase19B primitive scales [0.8564476370811462, 2.418001890182495, 49.38141632080078]. Optimization feeds [u/s0,u_x/s1,u_xx/s2] into the unchanged one-product MinimalSymNet; target u_t stays physical. Expanded scaled coefficients are divided by [s0,s1,s2,s0²,s0*s1,s0*s2,s1²,s1*s2,s2²]. Polynomial predictions and physical residuals are numerically checked against direct head outputs. Maximum absolute prediction-conversion discrepancy: 4.5678e-06.

Nine-term order: u, u_x, u_xx, u^2, u*u_x, u*u_xx, u_x^2, u_x*u_xx, u_xx^2. Truth is transport −1 and diffusion +0.02; the other seven coefficients are spurious. Full LS uses float64, unit-column SVD with relative cutoff 1e-12, no intercept or penalty; coefficients are returned to physical space.

## LS: algebraic compatibility before optimization

Full-library physical coefficients and diagnostics:

| condition | xi_u | xi_u_x | xi_u_xx | xi_u^2 | xi_u*u_x | xi_u*u_xx | xi_u_x^2 | xi_u_x*u_xx | xi_u_xx^2 | residual_mse | coefficient_error | spurious_l2 | loose | strong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | -0.146355 | 0.0433735 | 0.00465605 | 0.000473015 | -0.265147 | -0.000946477 | 0.0117632 | -0.000356155 | 1.02247e-05 | 0.0047338 | 0.750789 | 0.153103 | False | False |
| B | -0.0947912 | 0.235955 | 0.0117772 | -0.0542788 | -0.623409 | -0.00794052 | 0.0247721 | -0.00124422 | -6.11101e-07 | 0.200575 | 0.458446 | 0.261313 | False | False |
| C | -0.148889 | -0.00339827 | 0.00209957 | -0.00929293 | -0.20466 | -0.000683205 | 0.00567709 | -0.000304901 | 6.12857e-06 | 0.0179565 | 0.809435 | 0.149328 | False | False |
| D | -0.0637253 | 0.118871 | 0.0113942 | -0.0381626 | -0.727892 | -0.00384052 | 0.0103424 | -0.00125664 | -4.95359e-06 | 0.00434043 | 0.306411 | 0.140609 | False | False |

Two-true-term physical LS (transport, diffusion):

| condition | xi_u*u_x | xi_u_xx | residual_mse | coefficient_error | loose | strong |
| --- | --- | --- | --- | --- | --- | --- |
| A | -0.375316 | 0.0118625 | 0.0403335 | 0.624737 | False | False |
| B | -0.725866 | 0.0231957 | 0.258744 | 0.274152 | False | False |
| C | -0.268329 | 0.00745569 | 0.0572887 | 0.731778 | False | False |
| D | -0.771555 | 0.0228189 | 0.0399851 | 0.228462 | True | False |

Two-support spurious coefficients are constrained to zero; its apparent recovery is not full-library identification. Unconstrained LS and nonlinear MinimalSymNet answer different questions.

## Symbolic optimization: paired seeds and continuous results

25 seeds (0–24), identical initial parameter tensors for each seed across A/B/C/D, fresh Adam states, lr 0.01, betas (0.9,0.999), eps 1e-8, no regularization. Exactly 1000 full-grid PDE-MSE updates; no early stopping or condition-specific tuning. CPU float32 with two threads. This uses the established frozen-head optimization budget while retaining the original Phase19B fixed scales rather than checkpoint RMS re-estimation.

Loose requires strict transport/diffusion/spurious errors <0.25/0.02/0.25. Strong requires <0.10/0.01/0.10. Primary recovery is at the terminal step, not an earlier transient crossing. First-crossing steps are checked at every step including initialization; missing means no crossing. Descriptive “reliable” means >=20/25 terminal loose successes, declared before optimization. Wilson intervals quantify head-seed sampling uncertainty, not uncertainty across surrogate trajectories.

| condition | loose | strong | loose Wilson 95% | median transport | median diffusion | median coefficient error | median spurious L2 | median PDE MSE | median first loose (terminal recovered) | median first strong (terminal recovered) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0/25 | 0/25 | [0.000, 0.133] | -0.266301 | 0.00548819 | 0.749179 | 0.150812 | 0.00513915 | — | — |
| B | 0/25 | 0/25 | [0.000, 0.133] | -0.630489 | 0.0135901 | 0.447604 | 0.252526 | 0.201838 | — | — |
| C | 0/25 | 0/25 | [0.000, 0.133] | -0.210274 | 0.00274743 | 0.80335 | 0.14631 | 0.0183838 | — | — |
| D | 0/25 | 0/25 | [0.000, 0.133] | -0.729276 | 0.0115274 | 0.304755 | 0.139687 | 0.00436924 | — | — |

Transient/ever recovery (distinct from the terminal counts above):

| condition | loose_ever_count | loose_median_first_step_ever_recovered | strong_ever_count | strong_median_first_step_ever_recovered |
| --- | --- | --- | --- | --- |
| A | 0 | — | 0 | — |
| B | 0 | — | 0 | — |
| C | 0 | — | 0 | — |
| D | 7 | 87 | 0 | — |

All seven spurious coefficients, transport, diffusion, coefficient error, residual, and first crossings are retained by seed in `per_seed_results.csv`; every-step values in `trajectories/`; initial/final parameters in `heads/`. `recovery_summary.csv` gives medians, quartiles, minima and maxima for every coefficient and continuous metric, plus terminal and ever-recovered counts and loose/strong Wilson intervals. `paired_comparisons.csv` retains matched discordance counts and unadjusted exact McNemar tests; these exploratory comparisons are not corrected for multiple testing.

All terminal physical coefficient medians (quartiles and per-seed values are in the CSV artifacts):

| condition | xi_u_median | xi_u_x_median | xi_u_xx_median | xi_u^2_median | xi_u*u_x_median | xi_u*u_xx_median | xi_u_x^2_median | xi_u_x*u_xx_median | xi_u_xx^2_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | -0.144293 | 0.0419932 | 0.00548819 | -0.00227 | -0.266301 | -0.00254309 | 0.0121974 | -0.00028734 | -3.84772e-06 |
| B | -0.0892043 | 0.228109 | 0.0135901 | -0.0552646 | -0.630489 | -0.00982245 | 0.0250251 | -0.00107636 | -2.23701e-05 |
| C | -0.145478 | -0.00741778 | 0.00274743 | -0.0122277 | -0.210274 | -0.00201681 | 0.00582902 | -0.000283018 | -3.20792e-06 |
| D | -0.0629225 | 0.117928 | 0.0115274 | -0.0389735 | -0.729276 | -0.00419861 | 0.0103933 | -0.0012552 | -7.43285e-06 |

Endpoint stationarity diagnostics, without extending any run:

| condition | median_last100_coefficient_displacement | max_last100_coefficient_displacement | median_last100_residual_change |
| --- | --- | --- | --- |
| A | 3.14265e-08 | 0.00376769 | 0 |
| B | 2.80537e-08 | 0.000785395 | 0 |
| C | 2.59185e-08 | 0.00165057 | 0 |
| D | 6.44129e-08 | 0.000631624 | 0 |

These quantify movement over the final 100 steps, not a proof of global optimality. D's full-library LS fails loose recovery, whereas D's two-support fit meets loose recovery by fixing all seven spurious terms to zero. This distinction prevents mistaking a constrained two-term result for successful identification from the complete library. No condition has a loose-recovering full-library LS solution paired with unreliable head recovery in this run.

## Controls and interpretation

Historical jointly trained heads at the exact chosen states:

| step | xi_u*u_x | xi_u_xx | coefficient_error | spurious_l2 | loose | strong | pde_residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8000 | -0.20634 | 0.00411331 | 0.809615 | 0.159144 | False | False | 0.00766946 |
| 8750 | -0.713748 | 0.0101641 | 0.323535 | 0.150458 | False | False | 0.00473484 |

The nominal POST at 8750 is **before** the first archived full-library LS loose crossing (8825) and joint-head crossing (8853); strong crossings occur much later (9975/10013). Therefore D was never a validated positive control under the requested coefficient thresholds. Its frozen fields and self-pair LS have been validated against the archive, so a failed D need not indicate a pairing/code error. A fresh head can also behave differently from the historical joint head. A is reliably recovering: False; D is reliably recovering: False.

B demonstrates reliable sufficiency: False. C demonstrates reliable sufficiency: False. Failure to demonstrate sufficiency within this budget is not proof of impossibility.

**Classification: No reliable recovery, including D: positive-control limitation; the four sufficiency cases cannot be distinguished.**

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

Actual terminal loose counts: {'A': 0, 'B': 0, 'C': 0, 'D': 0}. No reliable recovery, including D: positive-control limitation; the four sufficiency cases cannot be distinguished.

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
