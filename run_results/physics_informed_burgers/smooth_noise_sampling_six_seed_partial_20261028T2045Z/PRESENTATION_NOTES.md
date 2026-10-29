# Smooth Burgers: noise × sampling — partial presentation package

**Status: INCOMPLETE — 57 of 180 runs (31.7%); 8 of 30 conditions have all six seeds.**
Do not present this as the completed 180-run study. Every figure and table below
counts only runs whose `status.json` is `complete`.

Snapshot: 2026-10-28 ~20:45 UTC, from HPC2 base
`/home/032480329/hpc_pde_smooth_noise/20261028`.
Scientific code: commit `19d511a49d53bd1524e50dfd80fda28a2b87d32b`
(plus aggregation-only fixes `77cd0da`, and plotting-only fixes after it).

## 1. What the experiment asks

How does neural recovery of the viscous Burgers coefficients depend jointly on
observation noise and sampling density when the solution is smooth?

```
u_t = -u u_x + nu u_xx,   nu = 0.2
u(x,0) = sin(x),          periodic x in [0, 2*pi),  t in [0, 1]
target coefficients (c1, c2) = (-1, 0.2)
```

Reference: 256 spatial points, `dt = 0.002`, 501 snapshots (128,256 values),
float64 RK4 with periodic centered differences.

## 2. Design

- **Fixed population:** 13,056 points drawn uniformly without replacement from
  the 128,256 numerical points, once per replicate seed.
- **Sampling:** nested prefixes at 100%, 75%, 50%, 25%, 10% →
  13,056 / 9,792 / 6,528 / 3,264 / 1,305 observations.
- **Noise:** `y = u + sigma * std(U) * epsilon`, `std(U) = 0.63703`,
  `sigma in {0, 0.01, 0.02, 0.05, 0.10, 0.20}`. One epsilon per replicate,
  reused across all sigma and sampling ratios (paired design).
- **Seeds:** 6 replicates (0–5), independent streams for population,
  subsampling, noise, initialization and minibatches.
- **Training:** SIREN 3×64 (omega 1/1), zero-initialized two-coefficient head,
  Adam 5e-4, batch 640 with replacement, unit data/PDE weights, 100,000 steps,
  CPU float32, one thread per run.
- **Recovery:** loose `|c1+1|<0.25, |c2-0.2|<0.02`; strong `<0.10, <0.01`;
  sustained = holds for the final 10,000 steps.

## 3. Numerical validation (passed before training)

| Check | Result |
|---|---|
| Initial condition max error | 0.0 |
| Temporal refinement (dt/2) relative L2 | 4.78e-13 |
| Spatial refinement (N=512) relative L2 | 3.31e-5 |
| max abs u_x / u_xx | 1.823 / 2.958 |
| Amplitude decay max|u|(t=1)/max|u|(t=0) | 0.8097 |
| Noise scale std(U) | 0.637027 |

Clean-derivative least squares recovers `(-1, 0.2)` to machine precision on every
sampled domain (raw condition number 3.5–3.7, normalized 1.70–1.75), so the two
coefficients are identifiable; this is a diagnostic, not a neural result.

## 4. Results so far (57 completed runs)

**All 57 completed runs reach strong endpoint and sustained recovery.**
No completed run is a failure.

| Metric | Mean | Median | Max |
|---|---:|---:|---:|
| `|c1 + 1|` | 0.00150 | 0.00113 | 0.00718 |
| `|c2 - 0.2|` | 0.000476 | 0.000272 | 0.00325 |

By noise level (completed runs only):

| sigma | n | strong | mean `|c1+1|` | mean `|c2-0.2|` |
|---:|---:|---:|---:|---:|
| 0.00 | 30 | 30 | 0.00109 | 0.000264 |
| 0.01 | 21 | 21 | 0.00175 | 0.000610 |
| 0.02 | 5 | 5 | 0.00278 | 0.00123 |
| 0.05 | 1 | 1 | 0.00201 | 0.000206 |
| 0.10 | 0 | — | — | — |
| 0.20 | 0 | — | — | — |

By sampling fraction (completed runs only):

| sampling | n | strong |
|---:|---:|---:|
| 100% | 14 | 14 |
| 75% | 14 | 14 |
| 50% | 12 | 12 |
| 25% | 9 | 9 |
| 10% | 8 | 8 |

**Critical caveat:** the run manifest is ordered sigma-major, so the completed
subset is concentrated at low noise. There are **zero** completed runs at
`sigma = 0.10` or `0.20`, and only 8 at 10% sampling. The current data cannot
support any claim about high-noise recovery or about the noise × sampling
interaction. Those are exactly the cells still running.

## 5. Figures in this folder

| File | Content |
|---|---|
| `noise_sampling_recovery_heatmap.png/pdf` | Recovery frequency by noise × sampling, with `n` per cell |
| `coefficient_trajectories_transport.png/pdf` | Mean ± SD coefficient path, 6×5 grid, `n` per panel |
| `coefficient_trajectories_diffusion.png/pdf` | Same for diffusion |
| `final_coefficient_errors.png/pdf` | Final error vs sampling, mean ± SD, one line per sigma |
| `final_coefficient_error_heatmap.png/pdf` | Final absolute error heatmaps |
| `representative_loss_curves.png/pdf` | Data and PDE loss for selected conditions |
| `clean_vs_noisy.png/pdf` | Clean solution and one paired noise realization |
| `report.md` | Auto-generated per-condition table |
| `per_seed_recovery.csv`, `recovery_summary.csv/json`, `completion.json` | Machine-readable results |

## 6. Limitations to state on the slide

- Known-support parameter estimation, not symbolic discovery.
- Noise is added to observations only; coordinates and reference derivatives stay clean.
- The PDE residual is evaluated at the sampled coordinates, so sampling changes
  both data density and collocation coverage — not an isolated target-density effect.
- Partial data are biased toward low noise (manifest ordering); no high-noise claim yet.
- CPU float32, single-threaded per run; one architecture and one learning rate.
- The 10% count is 1,305 (floor rule), not the historical 1,306.

## 7. Reproduction

```bash
BASE=/home/032480329/hpc_pde_smooth_noise/20261028
cd $BASE/src
python runs/run_burgers_smooth_noise_sampling.py --mode status    --output $BASE/runs
python runs/run_burgers_smooth_noise_sampling.py --mode aggregate --output $BASE/runs
python runs/run_burgers_smooth_noise_sampling.py --mode report    --output $BASE/runs
```

Sweep jobs: Torque `158532`–`158543` (Moab `60319`–`60330`), 12 shards × 2 workers.
Expected completion ~23:10 UTC; see `JOB_LEDGER.md`.
