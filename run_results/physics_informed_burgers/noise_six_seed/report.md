# Burgers six-seed averaged convergence

Complete noise levels: 6/6. Validated conditions included in means: 36/36.
Missing or incomplete levels: []

Arithmetic means use six histories at identical coordinates, without interpolation, truncation, smoothing, or metric substitution. The axis is optimizer steps (0–100000), not dataset epochs. Dense minibatch losses and coefficients are recorded each step; fixed-grid diagnostics are recorded at the original 20 checkpoints only. Population standard deviation uses ddof=0.

Original SIREN, two-coefficient physical head, losses, Adam, training budget, data locations, and clean references are unchanged. Noise exception: the original design varies epsilon with initialization seed (700000+seed), pairing it across noise levels. This extension preserves that design; outcomes combine initialization, batch-sequence, and noise-realization variation. No claim of isolating initialization variability is made. Original artifacts remain unchanged.

Loose recovery: |transport+1|<0.25 AND |diffusion−0.02|<0.02. Strong: <0.10 AND <0.01. First crossings are not sustained recovery. A null time means no crossing within 100000 updates. Recovery of the averaged signed coefficients is separate from per-seed recovery; errors are calculated per seed before averaging.

|Noise|Ever loose|Endpoint loose|Ever strong|Endpoint strong|Loose first-step median|Strong first-step median|Mean final transport abs error|Mean final diffusion abs error|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0.0|6|6|5|5|44947.5|87997.0|0.09144311149915059|0.0026921900113423664|
|0.01|6|6|4|4|46537.0|92215.5|0.0957583487033844|0.0027993291988968848|
|0.02|6|6|1|1|47818.5|87304.0|0.10394733150800069|0.003676707229266564|
|0.05|6|6|1|1|59693.5|99500.0|0.14532878994941711|0.010374223180115223|
|0.1|2|1|0|0|62098.0|non-recovery|0.20171652237574259|0.019321814632664126|
|0.2|1|0|0|0|54693.0|non-recovery|0.39831385513146716|0.027213040760834698|

Per-seed first crossings, endpoints, continuous absolute/relative errors, and explicit non-recoveries: `per_seed_recovery.csv`. Counts, successful-crossing time ranges/medians, signed endpoint means/SDs, and separate mean-curve crossings: `recovery_summary.csv`.

Original four epoch-dependent noise plots are reproduced at each noise level and in `overview/`: `coefficient_trajectories`, `coefficient_errors`, `field_derivative_fidelity`, `loss_trajectories`. The additional `training_losses` includes every original objective component and total. `trajectories` preserves the clean baseline 2×2 layout. All other recorded checkpoint metrics appear in `checkpoint_diagnostics_*`. The non-temporal endpoint plot is reproduced as `overview/final_coefficients`.

Per-seed raw histories, observations, checkpoint metrics/states and fields: `raw/sigma_*/seed_*`. Averaged numerical histories and population SDs: `averaged/sigma_*/*_mean.csv` and `*_std.csv`. Recovery booleans averaged in history CSVs are fractions of recovering seeds, not recovery decisions for the mean curve.

Limitations: full-grid diagnostic values between the original checkpoints were never recorded and are not reconstructed. Clean baseline seed 2–5 raw histories were unavailable, so those runs were rerun. Known-support parameter estimation is not unrestricted symbolic discovery.

Reproduce figures from archived histories: see `reproduction.txt`. Source code snapshots and original implementation hashes are archived in `code/` and `config.json`.
