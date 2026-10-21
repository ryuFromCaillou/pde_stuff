# Six-seed full-grid sampling convergence

All 30 independently trained conditions passed the artifact audit. Each received 100000 Adam steps with fixed 640-point replacement minibatches, lr 0.0005, unit data/PDE weights, zero noise and the original physical-head SIREN. No model modifications, L1, SymNet or LS extraction.

Arithmetic means and population SD (ddof=0) use exactly six matching histories. Signed coefficients and continuous per-seed errors are averaged separately. Histories contain every state 0–100000; clean diagnostics use only the 20 recorded checkpoints. No interpolation. Log bands omit portions where mean minus SD is nonpositive.

|Full-grid %|Count|Endpoint loose|Endpoint strong|Ever loose|Ever strong|Transport absolute error mean ± SD|Diffusion absolute error mean ± SD|
|---:|---:|---:|---:|---:|---:|---:|---:|
|100|13056|6/6|4/6|6/6|4/6|0.09497203 ± 0.01444702|0.002757077 ± 0.002448363|
|75|9792|6/6|2/6|6/6|2/6|0.1073011 ± 0.0145139|0.003204117 ± 0.002598369|
|50|6528|6/6|4/6|6/6|4/6|0.09014058 ± 0.01653242|0.003857653 ± 0.001062628|
|25|3264|6/6|3/6|6/6|3/6|0.09265677 ± 0.01563065|0.00366166 ± 0.001157882|
|10|1306|6/6|3/6|6/6|3/6|0.1049657 ± 0.02854601|0.003098757 ± 0.001401589|

Loose recovery requires absolute transport/diffusion errors <0.25/<0.02; strong requires <0.10/<0.01. These are per-seed strict inequalities. First crossings need not persist. Noncrossers are censored at 100000; first-time min/median/max/mean/SD summarize crossers only, with denominators and censor counts retained. Six seeds are descriptive optimization replicates, not population recovery probabilities.

## Coverage and interpretation
One permutation (NumPy default_rng seed 910064) defines nested subsets shared across all initialization seeds. Time-major flat index i maps to time i//256 and space i%256. All coordinate/target mappings and uniqueness were checked before training. Coverage JSON retains per-time and per-space counts; CSV reports distinct coverage and ranges. Nominal 10% is 1306/13056 = 10.0030637%.
The historical 640 observations form a structured 10×64 Cartesian selection, 4.9019608% of the solver grid. They are a separately labeled reference, never a full-grid random sample.
Observation density and PDE collocation coverage change simultaneously because the unchanged objective evaluates both losses at the sampled observation coordinates. Differences cannot be attributed exclusively to data density. The common 51×256 evaluation grid includes training points (all points at 100%); it is not a held-out or new-initial-condition generalization test. Derivative references use numerical periodic differences; reference u_t is the generator Burgers RHS, not analytic truth. This is known-support coefficient estimation.

## Artifacts
`overview/` and `averaged/ratio_*/` contain PNG/PDF figures. Dense means/SD CSVs accompany every level. `per_seed_summary.csv` contains endpoint field/derivative diagnostics, coefficients and recovery. `summary_agg.csv` includes all numeric endpoint mean/SD metrics and crossing statistics. `per_seed_recovery.csv`, `recovery_summary.json`, `coverage.csv/json`, source snapshots, configuration and hashes, validation, smoke evidence, gate review, and final audit support independent review. Raw histories, states, optimizer/batch RNG and field caches remain under `raw/`.

The fixed budget measures finite-horizon behavior; no per-seed tuning or early stopping was used. Endpoint errors and first crossings describe different aspects of convergence; a mean coefficient curve is not a per-seed recovery test.
