# Six-seed Burgers noise convergence characterization

This extends the completed `noise_robustness_resume2` experiment at authoritative
`sami` commit `bad712797f9b0fb24dd840a356f96ef592d4e97a`. The new branch is
`characterization/burgers-six-seed`. Original artifacts and training code are immutable.

Initialization seeds 0–5 × noise fractions 0,.01,.02,.05,.10,.20; 100000 Adam
updates, fresh 3×64 SIREN with frequencies 1/1 and zero-initialized physical head,
learning rate .0005, unit data/PDE weights, 640 with-replacement draws per step,
CPU float32 and one Torch thread per worker. No tuning, early stopping, normalization,
L1, LS, or architecture changes. Calls the original `train_condition` unchanged.

Noise realization exception: preserve the original seed rule `700000 + init_seed`.
The same epsilon is paired across noise levels for each seed; epsilon differs across
initialization seeds. The clean observations/locations and full clean evaluation grid
are identical. Thus variability includes initialization, batch sequence and noise.
Holding epsilon fixed across all seeds would change the original design and invalidate
reuse of its second seed. At zero noise, targets are identical for all seeds.

The original 12 runs are validated and copied byte-for-byte. Validation checks source
hashes, every step, finite values, loss identity, coefficient error definitions,
recovery flags/times, checkpoint coordinates, original observation/noise realizations,
initial weights and saved checkpoint coefficients. The separate clean six-seed report
has no surviving local histories for seeds 2–5; those four trajectories are rerun.

The current request authorizes this extension, superseding the prior two-seed policy
and historical documentation deferring noise execution.

## Execution

From `/home/ghost/projects/pde_stuff`:

```sh
/home/ghost/ghost/pde_stuff/.venv/bin/python runs/run_burgers_six_seed.py --output run_results/physics_informed_burgers/noise_six_seed --workers 4
```

Use a new output directory for a fresh study. An existing output can reuse complete
validated runs; partial runs are preserved and rejected explicitly. No implicit
continuation or rerun into a partial directory is permitted. Original files are copied
with their hashes in `reused_artifacts.json`. Original training sources and aggregation
sources are snapshotted under `code/`. `config.json` records design/provenance.

As each noise level gains six complete histories, its figures and CSV means are
published. No means with fewer than six seeds are emitted. The final overview requires
all 36. `aggregation_validation.json` explicitly lists missing/incomplete levels.

To reproduce only figures/tables from the archived per-seed data:

```sh
/home/ghost/ghost/pde_stuff/.venv/bin/python runs/run_burgers_six_seed.py --output run_results/physics_informed_burgers/noise_six_seed --plot-only
```

## Aggregation and plot inventory

Arithmetic means at identical optimizer-step coordinates (not dataset epochs).
No interpolation, truncation, smoothing, or downsampling before averaging. Exactly
six finite histories with matching column definitions are required. Population SD
uses ddof=0. Signed coefficient estimates are averaged directly; absolute/relative
errors are averaged separately. Boolean columns become per-step recovery fractions.

|Figure|Source|Metrics|Layout/scale|
|---|---|---|---|
|coefficient_trajectories|history.csv|transport, diffusion|1×2, linear, true −1/+0.02 dashed|
|coefficient_errors|history.csv|transport_abs_error, diffusion_abs_error|1×2, log, original strong dashed/loose dotted thresholds|
|field_derivative_fidelity|checkpoint_metrics.json|u_rel_l2, ut_rel_l2|1×2, log|
|loss_trajectories|checkpoint_metrics.json|observation_mse, independent_pde_mse|1×2, log|
|training_losses (added)|history.csv|total_loss, data_loss, pde_loss|1×3, log|
|trajectories (clean layout)|history.csv|transport, diffusion, data_loss, pde_loss|2×2, original targets and loss scales|
|checkpoint_diagnostics_* (added)|checkpoint_metrics.json|all recorded checkpoint diagnostics|2×3 pages|
|final_coefficients|history.csv endpoint|transport, diffusion vs noise|1×2, linear, true targets|

The first four are all original epoch-dependent noise figures. Every dense history has
100001 rows (states after 0–100000 updates); fixed-grid diagnostics have only the original
20 checkpoints. Unrecorded between-checkpoint metrics cannot be reconstructed and are
not substituted with minibatch losses. Means/SD CSVs retain all recorded numeric fields.

Each noise level has PNG/PDF plots under `averaged/sigma_*`; `overview/` compares
mean curves across noise levels. `raw/sigma_*/seed_*` retains individual histories,
observations, metrics, fields, initial states, model/optimizer/RNG checkpoints and summaries.

## Recovery and validation outputs

Loose: absolute transport error <.25 AND diffusion error <.02. Strong: <.10 AND <.01.
Strict inequalities and first crossings are unchanged; crossings are not necessarily
sustained. `per_seed_recovery.csv` records first crossings, endpoint status, signed
coefficients and absolute/relative errors. Null times mean non-recovery by 100000.
`recovery_summary.csv/json` gives ever/endpoint counts, successful-crossing min/median/max,
endpoint means/population SDs, and separately labeled mean-curve recovery. Mean-curve
recovery does not establish per-seed recovery. No recovery probability is inferred.

`run_status.json` accounts for all 36 conditions. `aggregation_validation.json` and
per-level `validation.json` record strict aggregation gates. Final `manifest.json`
contains SHA-256 and sizes for artifacts, excluding itself and live execution log/PID.
`report.md` provides scientific interpretation and limitations.

Regression check:

```sh
/home/ghost/ghost/pde_stuff/.venv/bin/python -m unittest discover -s tests -p test_burgers_six_seed.py
```

Final independent audit and figure regeneration:

```sh
/home/ghost/ghost/pde_stuff/.venv/bin/python runs/audit_burgers_six_seed.py --output run_results/physics_informed_burgers/noise_six_seed
```

The active execution is followed by an independent audit watcher. Its outcome is
recorded in `finalization_status.json`; failures are retained in `finalization.log`.
