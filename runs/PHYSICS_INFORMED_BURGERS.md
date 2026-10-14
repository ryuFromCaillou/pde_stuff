# Physics-informed Burgers coefficient recovery

Scope: correct-support joint estimation of `u_t = xi_transport*u*u_x + xi_diffusion*u_xx`.
The two unconstrained coefficients start at zero, independent of true values −1/+0.02.
No SymNet, LS, L1, pretraining, coordinate normalization, or feature normalization.

Paul's executable settings are authoritative: generator viscosity .02, final time .1,
N=256, dt=.002; ten linspace temporal indices and 64 linspace spatial indices produce
640 observations. The complete clean evaluation grid has 51×256 points. Three hidden
SIREN layers of width 64, frequencies 1/1; Adam .0005, default betas/epsilon, weights
1/1, 640 random draws with replacement per optimizer step, 100000 steps.
Paul's Markdown incorrectly describes warm-starting, a 10×10 grid, and other weights.
His code uses fresh initialization and weights 1/1. Both notebooks remain unchanged.

Declared changes beyond the requested head: initialize a fresh SIREN directly from
`torch.manual_seed(initialization_seed)` rather than replaying RNG draws from discarded
notebook pretraining; a separate batch generator uses seed 10000+initialization_seed.
This is a controlled new experiment, not an exact replay of Paul's trajectory.
CPU float32, one Torch thread. Dataset seed stays 0 across all runs. Seed 0 gates five
additional independent optimization seeds 1–5, all with identical hyperparameters.

Loose recovery: absolute transport error <.25 AND diffusion error <.02. Strong: <.10
AND <.01. These reuse historical thresholds with spurious terms absent by construction;
strong diffusion tolerance is still 50% of truth. Always report continuous absolute
and relative errors. First crossings are checked every optimizer step, not just at
checkpoints; endpoint recovery is separate and crossings need not persist.

Mechanical checks compare the head/residual and coefficient gradients with an analytic
exp(-t)sin(x) field, verify updates of both models, and finite-difference a surrogate
parameter through the PDE objective. Numerical checks compare RK4 time refinement
(relative field discrepancy <1e-5) and joint space/time refinement (<.01).
Each checkpoint independently assembles a float64 NumPy residual from canonical
feature evaluation, bypassing the head and training loss; observation residual MSE
must agree within rtol=2e-4, atol=2e-7. Full-grid primitive metrics use the canonical
helper and generator-consistent reference operators. Reference u_t is the numerical
Burgers RHS, not an analytic derivative. Report spatial and temporal errors separately.

Gate: all checks pass, seed 0 completes 100k steps, endpoint strong recovery and clean
field relative L2 <.05. If gate fails, stop for investigation; no robustness studies.
The horizon is fixed, with no early stopping after recovery and no per-seed tuning.
Checkpoint states/metrics/fields: 0,1,10,50,100,200,500,1000,2000,5000, then every 10000.
State k has completed k optimizer updates; its next sampled batch and RNG are saved.
With-replacement batches do not constitute dataset epochs.

Run from repository root (the available environment in this workspace):

```
/home/ghost/ghost/pde_stuff/.venv/bin/python runs/run_physics_informed_burgers.py --output run_results/physics_informed_burgers/clean_baseline
```

Use a fresh output destination. `--validate-only` runs mechanical/numerical validation;
`--steps 10` runs a smoke check, which cannot open the reliability gate.
Outputs: config/source hashes/environment, validation, exact command, clean reference
and observation indices, per-seed full scalar history, checkpoint fidelity metrics,
field/derivative NPZ arrays, model/optimizer/RNG states, summaries, coefficient
distributions, scientific report, PNG/PDF coefficient/loss and fidelity figures.
CSV/JSON histories are the numerical data underlying figures. Existing artifacts
are never overwritten. No commits, pushes, or notebook promotion are part of this run.

## Prepared noise protocol — do not execute

At full sampling, use sigma = 0,.01,.02,.05,.10,.20, with
`u_obs = u_clean + sigma*std(u_clean)*epsilon`, epsilon iid N(0,1).
Define std on the original 640 clean observations, population convention ddof=0.
Only training targets receive noise; solver states, derivative references, and the
fixed full evaluation grid remain clean. Save the realization and its noise seed
separately from initialization, batch, and dataset seeds. Pair initialization seeds
across levels and use the same standardized epsilon across levels for each noise seed.
Report observation fit against noisy targets AND clean field error, coefficient errors,
recovery times/counts, and derivative fidelity. Review clean reliability first.

## Prepared sampling protocol — do not execute

At zero noise, ratios 1,.75,.50,.25,.10 mean 640,480,320,160,64 observations, always
relative to Paul's original 640. Use nested uniform random subsets without replacement
from a saved permutation of those 640 pairs, with a separate sampling seed. This is
random sampling, not structured or stratified; record retained time/space coverage.
Keep solver N=256, dt=.002, T=.1 and full 51×256 clean evaluation grid unchanged.
Use batch size min(640, retained observation count), sampled with replacement as in
Paul's min-batch rule, and the same 100k-step budget; report this budget in steps.
Collocation follows retained observation coordinates, so reduced data also reduces
PDE constraint locations; this is part of the declared study, not pure target removal.
Pair initialization seeds and subset permutations across levels. Archive selected
indices and separate sampling seeds. No numerical solver coarsening is permitted.

## Deferred combined experiment

Review the individual noise and sampling results before selecting informative levels
for a crossed design. Do not automatically launch the Cartesian product. The runner
intentionally offers no noise or sparsity execution flags for this immediate task.

Read-only final artifact audit (use a new audit output destination):

```
/home/ghost/ghost/pde_stuff/.venv/bin/python runs/run_physics_informed_burgers_audit.py --source run_results/physics_informed_burgers/clean_baseline --output run_results/physics_informed_burgers/clean_audit
```

The audit checks every scalar row, first crossing and endpoint, saved physical
coefficients, initial weights, all cached checkpoint fields/residuals/fidelity metrics,
and source/notebook hashes. CSV parsing uses round-trip float precision so the saved
float32 coefficient values can be compared exactly after textual serialization.
Regression command: `python -m unittest discover -s tests -p test_physics_informed_burgers.py`.
The existing Burgers-control suite also has an archive-dependent check; this checkout
lacks `run_results/phase22_sgd_control/matched_inputs.pt`. That test cannot run here;
the other four existing checks pass. Logs are under `implementation_tests/`.

### Continuation and reporting fix

The original clean run completed seeds 0 and 1, then aggregate reporting rejected
NaN: sample standard deviation is undefined with one independent repeat. The fix
serializes that statistic as JSON null. AST comparison verified every training,
initialization, validation, and evaluation function unchanged. Both completed seeds
passed the read-only artifact audit before continuation; originals remain intact.
Evidence and the original source snapshots are in `reporting_fix_provenance/`.

Continue validated completed seeds into a **new** destination with:

```
/home/ghost/ghost/pde_stuff/.venv/bin/python runs/run_physics_informed_burgers.py --output run_results/physics_informed_burgers/clean_baseline_completed --reuse-completed run_results/physics_informed_burgers/clean_baseline --workers 2
```

`--reuse-completed` requires the seed-0 gate and identical scientific configuration.
Completed seed directories are copied byte-for-byte and hashed in
`reused_artifacts.json`; partial runs are not reused. The original results are not
modified. Two independent CPU processes run remaining seeds, each with one Torch
thread and independent deterministic initialization/batch RNG. This changes scheduling,
not the optimizer trajectories. For a fresh reproduction, omit `--reuse-completed`
and use a new output path; `--workers 2` still waits for seed 0 before repeats.
The consolidated scientific result is `clean_baseline_completed/`; audit that folder
with a new audit destination after all six seed trajectories finish.

### Token-limited follow-up policy

The completed clean baseline retains six seeds for reliability. For the next
exploratory noise and sampling studies, use only initialization seeds `0` and `1`
with the same hyperparameters; do not retune either seed. Treat those results as
two-seed exploratory evidence, not a replacement for the six-seed clean baseline.
Use separate noise/sampling seeds and keep the clean evaluation grid fixed.
