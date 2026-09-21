# Phase 24 — Smooth Burgers Recoverability Control

Run from the repository root:

```sh
.venv/bin/python runs/run_phase24_smooth_burgers_control.py
```

The runner executes the sinusoidal control once. A completed output is checked against its SHA-256 manifest and semantic consistency checks, then reused without writes or training. An incomplete directory is rejected rather than overwritten; individual completed probe histories remain available for diagnosis. `--output-dir` permits a new, separate reproduction directory. No previous phase is modified.

## Scientific contract

The treatment is `u(x,0)=0.5 sin(x)` on the existing physical periodic domain `[0,2*pi)`, with `nu=0.02`, `T=1`, `N=256`, RK4 `dt=0.002`, 252 saved times and 64,512 observations. Only the initial condition changes the data regime. The corresponding inviscid characteristic crossing time is 2, beyond the observation horizon; this motivation is checked quantitatively rather than assumed. Smoothness is measured at all 501 solver states, cross-checked with spectral derivatives and a 512-point / `dt=0.001` verification rollout. The refined grid is not used for learning. Reference discovery derivatives remain the matched centered finite differences and Burgers RHS.

The surrogate trajectory is **joint training**, matching Phase 23, rather than a data-only replacement. The separate scale estimator follows Phase 19B: seed 0, 2,500 Adam steps at `1e-3`, then LBFGS with `max_iter=200`; only full-grid RMS scales transfer. Joint training loads the exact archived surrogate/SymNet tensors and minibatches from Phase 22, uses Adam `5e-4`, data/PDE weights `1/0.5`, 1,000 updates, batch size 4096, and no regularization. Batch 0 is used twice, and epoch 1000 has no update. Checkpoints `0,50,200,500,1000` are saved before the labeled update.

Each frozen state supplies physical `[u,u_x,u_xx]` through the canonical `FeatureTensor` path and autodiff `u_t`. Full-grid RMS scales are recomputed per state. Fresh original `MinimalSymNet` probes use seeds 0–24, Adam `1e-2`, 1,000 full-grid updates, PDE MSE only. The coefficient vector is divided by the primitive-induced monomial scales. Loose thresholds for transport error, diffusion error, and spurious L2 are `(0.25,0.02,0.25)`; strong thresholds are `(0.10,0.01,0.10)`, all strict inequalities. Report final-epoch success, not best-epoch success. Sample SD uses `ddof=1`.

The local PyTorch runtime differs from the historical run. Joint starts use the exact saved tensors. Fresh probe seeds are checked for exact local repeatability and historical initial coefficient agreement at `rtol=2e-5, atol=2e-7`; tiny uniform-initialization rounding differences are recorded. This is a declared numerical matching limit, not a seed or optimizer change.

## Output contract

All files live under `run_results/phase24_smooth_burgers_control/` by default:

- `config.json`: controls, differences, environment, input hashes, and transferred scales.
- `smoothness_over_time.csv`, `smoothness.json`: slope/curvature maxima for both regimes, verification thresholds, spectral/refinement checks and global maxima.
- `reference_fields.npz`: physical coordinates, smooth reference fields and shock reference fields for plotting.
- `reference_identifiability.json`: exact-reference nine-term least-squares recovery, rank and conditioning. This checks numerical feature excitation; it is not a learned-SymNet success result.
- `scale_estimator_history.csv`, `scale_estimator.pt`: scale-estimation history and state; its weights are never transferred to joint training.
- `joint_history.csv`, `checkpoints/theta_epoch_*.pt`: all joint losses/coefficients and five pre-update model states.
- `initial_fields.npz`, `fields_epoch_*.npz`: evaluated physical surrogate fields.
- `frozen_surrogate_metrics.csv`, `derivative_metrics.json`: primitive/time error metrics, RMS scales, deterministic reload checks, and supplementary coefficient-space fits on learned derivatives.
- `probes/checkpoint_*_seed_*.csv` and `.json`: durable per-seed full histories and final results.
- `per_seed_histories.csv`: 125,125 rows (five checkpoints × 25 seeds × 1,001 evaluated epochs).
- `per_seed_results.csv`: 125 outcomes, all nine physical coefficients, loose/strong success, seed/state hashes, coordinate conversion and frozen-state checks.
- `recoverability_by_checkpoint.csv`: five rows with field/derivative MSE, loose count/rate, strong count, median/mean/sample-SD coefficient error, median transport/diffusion coefficients and median spurious L2.
- `phase23_comparison.csv`: identical summary columns for both solution regimes (10 rows).
- `smoothness`, `solution_derivative_slices`, `reference_regime_slices`, `phase23_comparison` (`.pdf` and `.png`): saved comparisons and representative slices.
- `feature_overlays/{u,ux,uxx}_overlay.pdf`: final model/reference overlays through the canonical plotting helper.
- `runtime_compatibility.json`: original-vs-extracted architecture checks for all 25 seeds and one historical terminal-state probe replay.
- `validation.json`, `status.json`, `manifest.json`: semantic checks, completion marker, and hashes of completed artifacts.

The notebook loads these artifacts and presents the experimental logic and results. Execution is implemented in `utils/burgers_recoverability.py`; plot rendering in `utils/burgers_control_plotting.py`; the entrypoint only orchestrates. The dataset solver's optional initial-condition and history-cadence arguments preserve its default behavior.
