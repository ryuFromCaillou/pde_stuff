# Smooth Burgers noise × sampling specification

Authoritative order: *Ghost Order — Smooth Burgers Noise × Sampling Experiment*
(amended by the supervisory override authorizing HPC execution).
Repository: `https://github.com/ryuFromCaillou/pde_stuff`, branch `sami`.
Implementation commit: recorded in the experiment `config.json` and
`SOURCE_COMMIT.txt`; the pre-existing source was
`c6bfa31827ce0d482453637c012458730fbee2e8`.

## 1. Scientific question

How does neural recovery of the viscous Burgers coefficients depend jointly on
observation noise and sampling density when the solution is smooth and
numerically well resolved? The previous random Gaussian-bump initial condition
produced steep gradients. This experiment replaces it with `u(x,0)=sin(x)` and
uses moderate viscosity `nu=0.2`.

## 2. Governing PDE and numerical reference

```
u_t = -u u_x + nu u_xx,   nu = 0.2
u(x,0) = sin(x)
x in [0, 2*pi),  t in [0,1]
u(0,t) = u(2*pi,t), periodic spatial derivatives
```

Coefficient targets: `(c1, c2) = (-1, 0.2)` in
`u_t = c1 u u_x + c2 u_xx`.

Reference solver: existing `Datasets/data/processed/burg_gen/burg_gen.py:solve_burgers`,
periodic centered differences, classical RK4, float64, `N=256`, `dx=2*pi/256`,
`dt=0.002`, 500 steps, `history_every=1` → **501 × 256 = 128,256** clean
space-time values. `u_t` in the reference is the generator RHS
`-u u_x + 0.2 u_xx`, not analytic truth.

### Measured numerical validation

| Check | Result |
|---|---|
| `u(x,0)=sin(x)` max abs error | `0.0` |
| Temporal refinement (`dt/2`) relative L2 | `4.78e-13` |
| Temporal refinement max abs | `8.42e-13` |
| Spatial refinement (N=512, dt/4) relative L2 | `3.31e-5` |
| Spatial refinement max abs | `1.01e-4` |
| Centered-FD vs spectral `u_x` relative L2 | `3.13e-4` |
| `max abs(u)` at t=1 / t=0 | `0.809663` |
| `max abs(u_x)` | `1.8230` |
| `max abs(u_xx)` | `2.9575` |
| `max abs(u u_x)` | `0.6248` |
| `max abs(u_t)` | `0.6460` |

The explicit RK4 diffusion stability bound at these parameters is
`dt < 2.785 dx^2/(4 nu) ≈ 0.00209`; `dt=0.002` is inside it, and the measured
refinement confirms stability and convergence. No viscosity substitution was
needed.

## 3. Fixed observation population

`N_base = 13,056`, independent of the 128,256-point numerical grid.

For each replicate seed `s`, draw **one** deterministic uniform
without-replacement population of 13,056 points from the 128,256 numerical
points:

```
population(s) = default_rng(SeedSequence([s, 101])).choice(128256, 13056, replace=False)
```

The same population is used by all noise and sampling conditions within
replicate `s`. It is not the complete numerical grid, and 100% does not mean
128,256 observations.

## 4. Sampling grid

```
r in {1.00, 0.75, 0.50, 0.25, 0.10}
N(r) = floor(r * 13056)
```

| r | N(r) |
|---:|---:|
| 1.00 | 13,056 |
| 0.75 | 9,792 |
| 0.50 | 6,528 |
| 0.25 | 3,264 |
| 0.10 | 1,305 |

**Count convention.** The order specifies the floor rule and the table value
1,305. The historical full-grid study used rounded counts and therefore 1,306
at 10%. This experiment follows the order's explicit floor rule (1,305) and
records the deviation. All five counts are fixed before execution.

For each replicate, one randomized permutation of the 13,056 population
positions is drawn:

```
order(s) = default_rng(SeedSequence([s, 102])).permutation(13056)
```

The sample for ratio `r` is the prefix `order(s)[:N(r)]`, so

```
S_10% ⊂ S_25% ⊂ S_50% ⊂ S_75% ⊂ S_100%.
```

## 5. Noise grid

```
sigma in {0, 0.01, 0.02, 0.05, 0.10, 0.20}
s_U = std(U)  (population standard deviation, ddof=0, over all 128,256 clean values)
epsilon(s) = default_rng(SeedSequence([s, 103])).standard_normal(13056)
y_i = u(x_i,t_i) + sigma * s_U * epsilon_i
```

`s_U = 0.6370272738540966` for this reference solution. The scale is computed
once from the full clean solution and held fixed. Noise is added only to
solution values, never to `t` or `x`. One epsilon realization per replicate is
reused across all sampling fractions (nested prefix) and all sigma values
(scaled linearly). This pairs conditions within a replicate.

## 6. Full matrix

6 noise levels × 5 sampling fractions × 6 replicate seeds =
**30 conditions, 180 runs**. Each run has a unique identifier:

```
burgers_nu0.2_r{ratio:.2f}_s{sigma:.2f}_seed{seed}
```

and a unique directory:

```
raw/ratio_{ratio}/sigma_{sigma}/seed_{seed}
```

No run may write into another run's directory.

## 7. Seed derivation

| Stream | Derivation |
|---|---|
| Reference population | `np.random.SeedSequence([s, 101])` |
| Nested subsampling | `np.random.SeedSequence([s, 102])` |
| Observation noise | `np.random.SeedSequence([s, 103])` |
| Neural initialization | `torch.manual_seed(s)` (established `initialize(seed)`) |
| Minibatch selection | `torch.Generator().manual_seed(10000 + s)` |

`s in {0,1,2,3,4,5}`. The four NumPy/Torch streams are independent. The
initialization and minibatch derivations preserve the established research
conventions. Initialization and minibatch sequences are paired across all
conditions within a replicate.

## 8. Neural configuration

| Parameter | Value |
|---|---|
| Surrogate | `SirenMLP(hidden_size=64, hidden_layers=3, first_omega_0=1, hidden_omega_0=1)` |
| Physical head | two unconstrained coefficients, zero initialized |
| Optimizer | Adam, PyTorch defaults |
| Learning rate | `5e-4` |
| Batch size | 640, with replacement |
| Data loss weight | 1 |
| PDE loss weight | 1 |
| Training steps | 100,000 optimizer steps |
| Device / dtype | CPU float32, one Torch thread per run |
| Normalization | none (physical coordinates, unchanged from baseline) |

The PDE residual is evaluated at the sampled observation coordinates, exactly
as in the established implementation. Sampling therefore changes both data
density and PDE collocation coverage; this is not an isolated target-density
intervention.

## 9. Recovery criteria (predeclared)

Targets `(c1, c2) = (-1, 0.2)`.

| Level | Transport | Diffusion |
|---|---|---|
| Loose | `|c1 + 1| < 0.25` | `|c2 - 0.2| < 0.02` |
| Strong | `|c1 + 1| < 0.10` | `|c2 - 0.2| < 0.01` |

These reuse the established absolute limits. Because `nu` increased tenfold,
the diffusion limits correspond to 10% (loose) and 5% (strong) relative error,
which is stricter than the historical 100%/50% relative limits. The criteria
were fixed before any 100,000-step result was produced and were not tuned on
outcomes.

**Sustained recovery window:** a level is sustained if its criterion holds at
every step in the final **10,000** optimizer steps (steps 90,001–100,000).
`first_*_step` is the first crossing and need not persist. The sustained
window and first-crossing definition are both recorded per run.

**Secondary sensitivity only:** `|c2-0.2| < 0.20` (loose) and `< 0.10`
(strong), which preserve the historical relative diffusion tolerance, are
reported as a sensitivity analysis and are not used for primary recovery
counts.

## 10. Identifiability diagnostic

Using clean numerical derivatives at the sampled points, least squares on
`[u u_x, u_xx]` recovers `(-1, 0.2)` to machine precision at every sampling
fraction. Raw condition numbers are 3.51–3.65; column-normalized condition
numbers are 1.70–1.75. Adding the paired noise to the target raises the
least-squares coefficient error only to `3.6e-3` (transport) and `1.4e-3`
(diffusion) at `sigma=0.20`. This is an upper-bound diagnostic, not a neural
recovery result, but it shows the two coefficients are identifiable on all
sampled domains.

## 11. Outputs per run

`condition.json`, `observations.npz`, `initial.pt`, `history.csv` (100,001
rows with data/PDE/total loss, coefficients, absolute/relative errors and
recovery flags), `checkpoint_metrics.json` (20 checkpoints), full-grid field
caches at 0/1k/10k/50k/100k, checkpoint states, `progress.json`,
`summary.json`, `recovery.json`, `status.json`.

Experiment-level outputs: `config.json`, `runs.json`, `protocol.json`,
`protected_artifacts.json`, `validation/`, `per_seed_recovery.csv`,
`recovery_summary.csv/json`, `completion.json`, figures and `report.md`.

## 12. Reproducibility and restart

All 180 run specifications are generated deterministically in `runs.json`.
Each completed run is validated against its identity, counts, nested sampling,
noise scaling, coefficient arithmetic, loss identity, checkpoint schedule and
summary. A run interrupted after a checkpoint is resumed exactly by restoring
the checkpoint, recomputing the pending step's loss, verifying it against the
retained history, applying the update and continuing from the saved batch RNG.
Existing completed runs are validated and skipped; partial directories are
resumed only with `--resume`; no other run's outputs are overwritten.

## 13. Deviations from historical studies

1. The 10% count is 1,305 (floor) rather than the historical 1,306 (round).
2. The reference population is drawn from the 128,256 numerical grid rather
   than using the complete 51×256 grid.
3. `nu=0.2`, `T=1`, `sin(x)` initial condition, and a sustained-recovery
   window are new; the historical studies used `nu=0.02`, `T=0.1`, random
   Gaussian bumps, and first-crossing/endpoint flags only.
4. The physical-head coefficient helper is adapted to the new targets; the
   network, objective, optimizer, loss weights and evaluation helpers are
   unchanged.
5. The new runner avoids pandas so it can use the existing HPC `pdenet`
   environment without modifying it.

## 14. Authorization and execution status

HPC execution was authorized by the supervisory override. The 180-run matrix
is executed through Moab/Torque jobs under group `hpc2-cnsm-psun-users`.
Completed-run counts and the exact job IDs are recorded in the experiment
`completion.json`, `hpc/` logs and the implementation report. Partial results
are always labeled with the number of completed seeds per condition.
