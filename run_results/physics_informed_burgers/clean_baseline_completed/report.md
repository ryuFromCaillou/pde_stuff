# Physics-informed Burgers clean baseline

This is correct-support parameter estimation, not unrestricted PDE discovery. No LS or L1 was used.
Seed 0 is the gate; seeds 1–5 are five additional independent optimization replicates on the identical clean dataset.

|seed|transport|diffusion|observation MSE|grid MSE|field relative L2|ut relative L2|ux relative L2|uxx relative L2|independent PDE MSE|observation total|first loose|first strong|final|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|-0.914133|0.02261112|2.191119e-05|2.06425e-05|0.004997356|0.1048643|0.0305368|0.1466296|2.736924e-05|5.155817e-05|53773|87997|strong|
|1|-0.9220549|0.02100232|2.300311e-05|2.233231e-05|0.005197876|0.09125031|0.02528102|0.1310363|6.750427e-05|9.269178e-05|33634|79807|strong|
|2|-0.9013773|0.02713715|3.080121e-05|2.848423e-05|0.005870313|0.1303929|0.03071737|0.1421517|9.418063e-05|0.0001290867|36345|98822|strong|
|3|-0.9113712|0.02356916|2.265341e-05|2.105994e-05|0.005047631|0.1070164|0.02630752|0.1276013|5.429535e-05|7.864838e-05|43691|92220|strong|
|4|-0.8834068|0.01925484|2.996613e-05|2.817913e-05|0.00583879|0.1283176|0.03142901|0.1470554|5.66022e-05|9.154538e-05|52301|None|loose|
|5|-0.9189982|0.02108824|2.505835e-05|2.459682e-05|0.005455047|0.09544992|0.02655483|0.1330104|5.234711e-05|7.968501e-05|46204|87985|strong|

Reliability endpoint recovery: loose 5/5; strong 4/5.
Loose absolute-error limits: transport <0.25, diffusion <0.02. Strong: <0.10 and <0.01. Both must hold; no spurious terms exist. These intentionally retain historical loose/strong limits, which allow substantial relative diffusion error. Continuous absolute and relative errors are in seed_summary.csv.
First crossings are checked every optimizer step and need not persist. Coefficient distributions (mean/std/min/quartiles/max) and failed seeds are in seed_distribution.json. All checkpoint metrics and figure data are retained. Finite differences are numerical references; u_t is the clean generator RHS. Low residual does not establish accurate derivatives.
Sampling uses 640 draws with replacement per optimizer step; these are not epochs. Full evaluation uses all 51×256 clean solver points, including the original 640 observations. This is interpolation within one numerical solution, not generalization to unseen initial conditions.
See config.json, validation.json, reproduction.txt, and runs/PHYSICS_INFORMED_BURGERS.md. Noise and sparsity studies remain unexecuted.

Clean recovery is unreliable under the strong criterion. Inspect failed coefficient trajectories and field/derivative errors before any robustness study; no per-seed tuning is permitted.
