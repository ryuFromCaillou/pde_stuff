# HPC2 execution bundle

Target: CSULB HPC2, Moab/Torque `batch` queue, group
`hpc2-cnsm-psun-users`, existing read-only environment
`/home/032480329/mambaforge/envs/pdenet/bin/python`.

The experiment runner does **not** import pandas, so the previously reported
`pdenet` pandas gap does not block this study. No shared environment is
modified. The code uses NumPy, PyTorch and (for aggregation) Matplotlib, all
already present.

## Staging from Ark2

```bash
cd /home/ghost/projects/pde_stuff
hpc/stage.sh /home/032480329/hpc_pde_smooth_noise/20261028
```

`stage.sh` creates a `git archive` of the pinned commit (no `run_results`,
`.venv` or historical artifacts), extracts it to
`$BASE/src`, and writes `BUNDLE.sha256` and `SOURCE_COMMIT.txt`.

## Prepare and validate

```bash
BASE=/home/032480329/hpc_pde_smooth_noise/20261028
ssh ... 032480329@hpc2.its.csulb.edu
cd $BASE/src
/home/032480329/mambaforge/envs/pdenet/bin/python -u \
  runs/run_burgers_smooth_noise_sampling.py --mode prepare --output $BASE/runs --steps 100000
/home/032480329/mambaforge/envs/pdenet/bin/python -u \
  runs/run_burgers_smooth_noise_sampling.py --mode validate --output $BASE/runs
```

`prepare` refuses an existing `$BASE/runs`. Use a new base for a new attempt.

## Pilot

```bash
cd $BASE/src
msub -v HPC_BASE=$BASE,STEPS=2000 hpc/pilot_cpu.msub
```

The pilot writes to `$BASE/pilot` (separate from `$BASE/runs`) so a 2,000-step
engineering run cannot be mistaken for a completed 100,000-step research run.

## CPU / GPU benchmark

```bash
msub -v HPC_BASE=$BASE,STEPS=2000 hpc/bench_cpu.msub
msub -v HPC_BASE=$BASE,STEPS=2000 hpc/bench_gpu.msub
```

`bench_gpu.msub` requests `gres=gpus:1`; the payload records CUDA availability
and does not silently substitute CPU results for GPU measurements.

## Full matrix

```bash
cd $BASE/src
for i in $(seq 0 11); do
  msub -v HPC_BASE=$BASE,SHARD_INDEX=$i,SHARDS=12,WORKERS=2,STEPS=100000 hpc/shard_cpu.msub
done
```

12 jobs × 2 single-threaded workers = 24 concurrent processors, matching the
published guest-group limit. Each shard owns 15 runs (interleaved). Completed
runs are validated and skipped; partial runs resume from the latest checkpoint.
If a job ends before its shard is complete, resubmit the same shard with the
same command; it will continue without duplicating completed work.

## Monitoring and collection

```bash
showq -u 032480329
checkjob -v <Moab.jobid>
```

Per-shard status is written to `$BASE/runs/shard_XX_status.json` and
`shard_XX_complete.json`. Incremental aggregates are available at any time:

```bash
cd $BASE/src
python runs/run_burgers_smooth_noise_sampling.py --mode aggregate --output $BASE/runs
python runs/run_burgers_smooth_noise_sampling.py --mode report --output $BASE/runs
```

Retrieve to Ark2 without deleting local files:

```bash
hpc/collect.sh $BASE /home/ghost/reports/pde/burgers_smooth_noise_20261028
```

## Measured resource envelope

To be filled from the pilot and benchmark job records. Do not treat the
planning estimates below as measurements.

| Quantity | Status |
|---|---|
| Pilot walltime | pending |
| CPU seconds / 1000 steps | pending |
| GPU seconds / 1000 steps | pending |
| Peak memory | pending |
| One-run projection | pending |
| 180-run projection | pending |
