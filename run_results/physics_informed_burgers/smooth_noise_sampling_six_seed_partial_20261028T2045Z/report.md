# Smooth sinusoidal Burgers noise x sampling

Completed runs: **57/180**; complete conditions: **8/30**.

This report is generated from completed runs only. It must not be read as the completed 180-run study while `completion.json` reports missing runs.

## Design

PDE `u_t = -u u_x + 0.2 u_xx`, periodic `[0,2*pi)`, `u(x,0)=sin(x)`, `t in [0,1]`, 256 spatial points, `dt=0.002`, 501 snapshots. A fixed population of 13,056 points is drawn uniformly without replacement from the 128,256 numerical points once per replicate. Nested prefixes give the five sampling fractions. One paired Gaussian realization per replicate is scaled by `sigma * std(U)` and reused across all conditions.

Training reuses the established SIREN (3x64, omega 1/1), zero-initialized two-coefficient head, Adam 5e-4, batch 640 with replacement, unit data/PDE weights, 100,000 steps, CPU float32.

## Recovery criteria (predeclared)

Loose: `|c1+1| < 0.25` and `|c2-0.2| < 0.02`. Strong: `< 0.10` and `< 0.01`. Sustained: the criterion holds for the final 10,000 optimizer steps. First crossings need not persist.

## Per-condition summary

| sigma | sampling | count | n | endpoint loose | endpoint strong | sustained strong | mean c1 | mean c2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100% | 13056 | 6/6 | 6/6 | 6/6 | 6/6 | -0.9987 | 0.1996 |
| 0 | 75% | 9792 | 6/6 | 6/6 | 6/6 | 6/6 | -0.9990 | 0.1998 |
| 0 | 50% | 6528 | 6/6 | 6/6 | 6/6 | 6/6 | -0.9991 | 0.1998 |
| 0 | 25% | 3264 | 6/6 | 6/6 | 6/6 | 6/6 | -0.9988 | 0.1997 |
| 0 | 10% | 1305 | 6/6 | 6/6 | 6/6 | 6/6 | -0.9990 | 0.1998 |
| 0.01 | 100% | 13056 | 6/6 | 6/6 | 6/6 | 6/6 | -0.9986 | 0.1994 |
| 0.01 | 75% | 9792 | 6/6 | 6/6 | 6/6 | 6/6 | -0.9984 | 0.1993 |
| 0.01 | 50% | 6528 | 6/6 | 6/6 | 6/6 | 6/6 | -0.9982 | 0.1994 |
| 0.01 | 25% | 3264 | 1/6 | 1/1 | 1/1 | 1/1 | -0.9974 | 0.1999 |
| 0.01 | 10% | 1305 | 2/6 | 2/2 | 2/2 | 2/2 | -0.9971 | 0.1994 |
| 0.02 | 100% | 13056 | 1/6 | 1/1 | 1/1 | 1/1 | -0.9982 | 0.1989 |
| 0.02 | 75% | 9792 | 2/6 | 2/2 | 2/2 | 2/2 | -0.9991 | 0.1992 |
| 0.02 | 50% | 6528 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.02 | 25% | 3264 | 2/6 | 2/2 | 2/2 | 2/2 | -0.9948 | 0.1985 |
| 0.02 | 10% | 1305 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.05 | 100% | 13056 | 1/6 | 1/1 | 1/1 | 1/1 | -0.9980 | 0.1998 |
| 0.05 | 75% | 9792 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.05 | 50% | 6528 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.05 | 25% | 3264 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.05 | 10% | 1305 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.1 | 100% | 13056 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.1 | 75% | 9792 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.1 | 50% | 6528 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.1 | 25% | 3264 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.1 | 10% | 1305 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.2 | 100% | 13056 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.2 | 75% | 9792 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.2 | 50% | 6528 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.2 | 25% | 3264 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |
| 0.2 | 10% | 1305 | 0/6 | 0/0 | 0/0 | 0/0 | nan | nan |

## Figures

- `noise_sampling_recovery_heatmap.png/pdf`
- `coefficient_trajectories_transport.png/pdf`, `coefficient_trajectories_diffusion.png/pdf`
- `final_coefficient_errors.png/pdf`, `final_coefficient_error_heatmap.png/pdf`
- `representative_loss_curves.png/pdf`
- `clean_vs_noisy.png/pdf`

## Limitations

- Known-support parameter estimation, not symbolic discovery.
- Noise is added to observations only; coordinates and numerical references remain clean.
- The PDE loss is evaluated at the sampled observation coordinates, so sampling changes both data density and PDE collocation coverage.
- Incomplete conditions are reported with their completed seed count and are not extrapolated.
- CPU float32 results; device and thread count are recorded per run.
