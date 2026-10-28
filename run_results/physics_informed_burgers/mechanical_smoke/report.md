# Physics-informed Burgers clean baseline

This is correct-support parameter estimation, not unrestricted PDE discovery. No LS or L1 was used.
Seed 0 is the gate; seeds 1–5 are five additional independent optimization replicates on the identical clean dataset.

|seed|transport|diffusion|observation MSE|grid MSE|field relative L2|ut relative L2|ux relative L2|uxx relative L2|independent PDE MSE|observation total|first loose|first strong|final|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|0.003015017|0.004981142|0.7398218|0.7205806|0.9336849|0.9843033|1.019774|1.004304|0.007637393|0.7474949|None|None|not_recovered|

Reliability endpoint recovery: loose 0/0; strong 0/0.
Loose absolute-error limits: transport <0.25, diffusion <0.02. Strong: <0.10 and <0.01. Both must hold; no spurious terms exist. These intentionally retain historical loose/strong limits, which allow substantial relative diffusion error. Continuous absolute and relative errors are in seed_summary.csv.
First crossings are checked every optimizer step and need not persist. Coefficient distributions (mean/std/min/quartiles/max) and failed seeds are in seed_distribution.json. All checkpoint metrics and figure data are retained. Finite differences are numerical references; u_t is the clean generator RHS. Low residual does not establish accurate derivatives.
Sampling uses 640 draws with replacement per optimizer step; these are not epochs. Full evaluation uses all 51×256 clean solver points, including the original 640 observations. This is interpolation within one numerical solution, not generalization to unseen initial conditions.
See config.json, validation.json, reproduction.txt, and runs/PHYSICS_INFORMED_BURGERS.md. Noise and sparsity studies remain unexecuted.
