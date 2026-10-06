# Phase 19B transition instrumentation

This is one exact diagnostic reproduction, not a sweep. Preserve archived data, initial tensors, original batch stream/indexing, CPU float32 operations with two Torch threads, Adam 5e-4 and loss weights (1,0.5). Replay steps 0 through 10,501: the final scalar row validates the update out of state 10,500. No original artifact or narrative notebook is changed.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -u runs/run_phase19b_transition_instrumented.py --out run_results/phase19b_transition_instrumented_repeat
```

The output directory must be new. `--stage replay` performs only reproduction/passive capture. `--stage analyze` evaluates a completed validated replay without training; it refuses to overwrite existing analysis tables. The default `--stage all` runs replay, analysis, plots, events and report. Use the same installed environment whose versions are recorded in provenance.json.

## Predeclared schedule and gates

- Full diagnostics at 5000 and every 25 steps from 7000 through 10500, plus 8131, 8631, 8853 and 10013: 146 evaluated states, of which 145 are within the transition.
- Additional saved reproduction anchors: 0, 50, 200, 500, 1000, 2000. Total 152 saved states and corresponding actual updates.
- Geometry at 5000, 7000, 7500, 8000, 8250, 8500, 8750, 9000, 9500, 10000, 10500, plus reference.
- Every scalar row (0–10501) is checked against the original dense history: losses, nine physical coefficients, coefficient error and spurious L2. Gate: absolute error <= 1e-8 + 1e-6*abs(archived value); recovery flags must match exactly. Failures stop mechanistic analysis. Checkpoint parameter tensors are compared bitwise.
- Record hashes before/after for original long-horizon and prior post-hoc artifacts and the notebook.

## Instrumentation isolation

During training, only passive pre-update state/batch/RNG/optimizer capture and post-update actual displacement/combined gradient/Adam-state capture occur. Gradient decomposition and evaluation are performed offline, after reproduction passes, from saved tensors. No diagnostic autograd call is added to training, and save operations are checked not to consume either global or batch-generator RNG. Each saved update at s is the actual update s→s+1.

All field/fidelity/LS evaluations use the identical full 252x256 grid. Gradients use both the exact recorded training minibatch and a fixed systematic 4096-point sample `floor(arange(4096)*64512/4096)`. The latter is deterministic, covers the full flattened grid, and is saved explicitly; it is not claimed to equal full-grid gradients. Actual updates belong to the training minibatch. Comparing these with fixed-sample gradients is explicitly an out-of-batch directional diagnostic. SymNet data gradient is zero by construction.

Reuse canonical primitive evaluation/fidelity, original numerical reference operators, and the previous float64 unit-column SVD least squares. The true-support ordering is transport then diffusion; nine-term ordering is verified from MinimalSymNet. Cancellation remains `||-e_transport+0.02 e_diffusion||² / (||e_transport||² + 0.02²||e_diffusion||²)` with unweighted primitive diffusion error. No intercept or regularization is introduced.

## Timing criteria fixed before replay

- Fractional-progress milestones: first 10/25/50/75/90% of each quantity's net 7000→10500 change, sustained for at least 100 steps, on the regular 25-step grid. These are descriptive relative milestones, not physical onset estimates. A condition already met at 7000 is left-censored. No interpolation pretends to recover sub-25-step timing.
- Rapid transport: forward 100-step slope of `-xi_transport >= 0.0005/step`, sustained across 100 consecutive starting steps, from the every-step history.
- Exact SymNet loose/strong thresholds retain archived definitions. LS uses the same thresholds and 100-step persistence at diagnostic resolution, without equating LS to the nonlinear head.
- Gradient screens: nonnegative data/PDE cosine, weighted PDE/data norm ratio <=1, and actual update aligned with both negative loss gradients, each sustained 100 steps on the fixed sample.
- Geometry screen: adjacent scheduled normalized condition changes >20% or minimum angle changes >5 degrees. This is a descriptive screen, not a significance test.
- Centered 200-step peak progress rates are supplementary descriptions, not onset detectors. Sensitivity across multiple milestones is reported rather than choosing the most striking time.

## Output contract

`config.json`, `provenance.json`, `validation.json`; `optimization_history.csv`, `reproduction_errors.csv`; `states/state_*.pt`, `updates/update_*.pt`; `transition_metrics.csv`, `gradient_update_metrics.csv`, `ls_trajectory.csv`, `geometry_metrics.csv`, `geometry.json`, `fixed_gradient_indices.npy`, `snapshot_fields.npz`; `events.csv`, `events.json`, `descriptive_peak_rates.csv`, `geometry_change_screen.json`; six titled PNG/PDF figures; standalone `report.md`. Config stores exact schedules and definitions. Saved states/updates allow all plotted quantities to be recomputed without another training replay. Artifact-only plot/report regeneration uses `utils.phase19b_instrumented_plotting.render(out)` and `utils.phase19b_instrumented_report.report(out)`.

Raw versus unit-column conditioning, minibatch versus fixed-grid losses, target versus spatial errors, and LS relaxation versus learned SymNet are always distinguished. Temporal precedence and optimizer association are not causality; no bifurcation claim is planned.
