# Smooth sinusoidal Burgers noise × sampling

Implementation: `utils/burgers_smooth_noise_sampling.py`.
Entry point: `runs/run_burgers_smooth_noise_sampling.py`.
Specification: `BURGERS_NOISE_SAMPLING_SPEC.md`.

## Modes

```bash
.venv/bin/python runs/run_burgers_smooth_noise_sampling.py --mode validate  --output DIR
.venv/bin/python runs/run_burgers_smooth_noise_sampling.py --mode prepare   --output DIR
.venv/bin/python runs/run_burgers_smooth_noise_sampling.py --mode smoke     --output DIR --steps 200
.venv/bin/python runs/run_burgers_smooth_noise_sampling.py --mode run-one   --output DIR --ratio 1.0 --sigma 0 --seed 0 --steps 100000
.venv/bin/python runs/run_burgers_smooth_noise_sampling.py --mode shard     --output DIR --shard-index 0 --shards 12 --workers 2
.venv/bin/python runs/run_burgers_smooth_noise_sampling.py --mode benchmark --output DIR --ratio 1.0 --sigma 0 --seed 0 --steps 2000 --device cuda
.venv/bin/python runs/run_burgers_smooth_noise_sampling.py --mode aggregate --output DIR
.venv/bin/python runs/run_burgers_smooth_noise_sampling.py --mode report    --output DIR
.venv/bin/python runs/run_burgers_smooth_noise_sampling.py --mode status    --output DIR
```

`prepare` refuses an existing output directory. `run-one` refuses an existing
partial directory unless `--resume` is passed. `shard` runs a deterministic
interleaved subset of `runs.json`, skips already validated runs, resumes
partial runs, and records per-shard status. `aggregate` and `report` are
incremental and include only runs whose `status.json` is `complete`; every
figure and table reports the completed seed count.

## Output contract

```
DIR/
  config.json                 experiment configuration and source hashes
  runs.json                   180 deterministic run specifications
  protocol.json               counts, nesting, pairing, seed checks
  protected_artifacts.json    hashes of pre-existing research artifacts
  SOURCE_COMMIT.txt           pinned commit/branch/remote
  code/                       source snapshot
  validation/                 numerical validation, identifiability, plots
  raw/ratio_*/sigma_*/seed_*/
  per_seed_recovery.csv
  recovery_summary.csv/json
  completion.json
  report.md
  *.png, *.pdf
```

Per-run files are documented in the specification. `history.csv` has one row
per optimizer step, 0–100000. Checkpoint state `k` is saved after recording the
step-`k` loss and before the step-`k` update, with the sampled batch and the
post-draw batch RNG. This is the established checkpoint convention and makes
exact resume possible.

## Validation

`--mode validate` performs the initial-condition, periodicity, stability,
temporal/spatial refinement, amplitude/derivative magnitude and
identifiability checks and writes `validation/numerical_validation.json`.
`--mode smoke` runs two 200-step engineering conditions (100% clean and 10%
at 10% noise), validates them and labels them as engineering checks, not
scientific evidence. Unit tests are in
`tests/test_burgers_smooth_noise_sampling.py`.

## HPC

Scripts are in `hpc/`. They are written for the verified Moab/Torque
`batch` queue, group `hpc2-cnsm-psun-users`, and the existing
`/home/032480329/mambaforge/envs/pdenet/bin/python`. The runner does not need
pandas. The environment is used read-only; nothing is installed or modified.

`hpc/pilot_cpu.msub` runs 2,000 steps in a separate pilot root. `hpc/shard_cpu.msub`
runs one shard of the 180-run matrix with two single-threaded workers per job.
`hpc/bench_cpu.msub` and `hpc/bench_gpu.msub` measure 2,000-step CPU/GPU
throughput. All jobs write separate stdout/stderr logs and exit with the
payload status.
