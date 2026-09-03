# Research State

## Experiment objective
Evaluate whether low surrogate solution error also yields low derivative error for the current Burgers baseline, using surrogate autodiff derivatives against clean-rollout reference derivatives.

## Exact configuration
```json
{
  "dataset_name": "burg_gen",
  "noise_level": 0.0,
  "seed": 0,
  "stride_t": 2,
  "stride_x": 2,
  "device": "cpu",
  "pretrain_epochs": 150,
  "pretrain_lr": 0.001,
  "epochs": 600,
  "joint_lr": 0.001,
  "batch_size": 1024,
  "weight_decay": 0.0,
  "lam_pde": 0.5,
  "lam_data": 1.0,
  "lam_sparse_eql": 0.0,
  "sparse_eql_s": 0.001,
  "tv_type": "tv_ux",
  "tv_lambda": 0.0,
  "hidden_size": 64,
  "hidden_layers": 3,
  "first_omega_0": 20.0,
  "hidden_omega_0": 1.0,
  "eql_prod_dim": 2,
  "eql_num_layers": 1,
  "feature_terms": [
    "u",
    "u_x",
    "u_xx"
  ],
  "feature_normalize": true,
  "feature_normalize_mode": "global_fixed",
  "max_time_slices": 5,
  "log_every": 50,
  "output_root": null,
  "output_tag": "derivative_diag_20260902T173629Z"
}
```

## Dataset
- dataset: `burg_gen`
- clean solution source: `Datasets/data/processed/burg_gen/burg_gen.py::solve_burgers`
- numerical method: RK4 in time with second-order centered periodic finite differences in space

## Source/method for reference derivatives
- u_t_ref: Clean-rollout Burgers RHS on physical grid, then scaled to normalized t.
- u_x_ref: Generator periodic centered finite differences on physical grid, then scaled to normalized x.
- u_xx_ref: Generator periodic centered second derivative on physical grid, then scaled to normalized x.

## Final surrogate error
- u: MSE=1.178709e-02, RMSE=1.085684e-01, rel_L2=1.267502e-01, max_abs=6.232005e-01

## Derivative errors
- u_t: MSE=1.309988e-01, RMSE=3.619376e-01, rel_L2=7.420186e-01, max_abs=2.681521e+00
- u_x: MSE=9.601796e+00, RMSE=3.098677e+00, rel_L2=4.234345e-01, max_abs=4.331585e+01
- u_xx: MSE=1.273666e+05, RMSE=3.568846e+02, rel_L2=7.180764e-01, max_abs=6.274513e+03

## PDE-feature diagnostics
- u: MSE=1.178709e-02, RMSE=1.085684e-01, rel_L2=1.267502e-01, max_abs=6.232005e-01
- u_x: MSE=9.601796e+00, RMSE=3.098677e+00, rel_L2=4.234345e-01, max_abs=4.331585e+01
- u_xx: MSE=1.273666e+05, RMSE=3.568846e+02, rel_L2=7.180764e-01, max_abs=6.274513e+03
- u*u_x: MSE=7.462580e+00, RMSE=2.731772e+00, rel_L2=6.134777e-01, max_abs=3.836394e+01

## Recovered PDE
`u_t_hat = (-0.0712414095461)*u + (+0.00323493793391)*u_x + (+5.63051937311e-05)*u_xx + (+0.0517073185312)*u*u + (-0.0119660395266)*u*u_x + (-8.20678696186e-06)*u*u_xx + (+0.000691894727604)*u_x*u_x + (+9.43013029704e-07)*u_x*u_xx + (+2.98267840997e-10)*u_xx*u_xx`

## Important visual findings
- Solution and derivative slice plots share the same physical time anchors used in the diagnostic snapshots.
- Absolute derivative-error heatmaps expose where surrogate fit quality and derivative quality diverge across the full rollout.
- The composite feature u*u_x is evaluated independently from the trained PDE readout using the surrogate solution and autodiff u_x.

## Anomalies/failures
- Time-derivative error is substantially larger than solution error.
- Second-derivative fidelity degrades relative to first-derivative fidelity.

## Artifact directory
`run_results/eql_joint_training/burg_gen/seed_000_derivative_diag_20260902T173629Z`

## Next suggested experiment
- Repeat the same diagnostic with the same baseline but compare against a higher-order time-reference estimate from dense saved states to separate generator-discretization error from surrogate derivative error.

## Summary file
- summary: `run_results/eql_joint_training/burg_gen/seed_000_derivative_diag_20260902T173629Z/summary.json`
- metrics: `run_results/eql_joint_training/burg_gen/seed_000_derivative_diag_20260902T173629Z/metrics.json`

## Diagnostic notebooks
- `notebook/diagnostics/burgers_minimal_discovery_story.ipynb`: Linear, bare-bones Burgers discovery walkthrough created on 2026-09-03. The earlier direct least-squares section is not the exact frozen-random-SymNet control because it uses only the 2-term Burgers basis `[u*u_x, u_xx]`. Phase 11 adds the exact coefficient-space control on the same frozen Phase 9 surrogate derivatives and the same 9-term degree-2 library as the minimal SymNet, recovering `u_t_hat = (-0.026607)*u + (+0.031648)*u_x + (+0.009980)*u_xx + (-0.004855)*u^2 + (-0.892455)*u*u_x + (-0.002124)*u*u_xx + (+0.000004)*u_x^2 + (-0.000616)*u_x*u_xx + (+0.000001)*u_xx^2` with PDE MSE `8.851884e-02`. This is close to the hand-initialized frozen-SymNet basin and far better than the random-initialized frozen-SymNet result (`9.651142e-01`), so the evidence currently points primarily to failure in the nonlinear `theta -> xi` SymNet optimization rather than in direct recovery from the frozen derivative field alone.
- Phase 12 extends the same notebook with deterministic local-geometry diagnostics for the frozen SymNet loss `L(theta) = ||Theta xi(theta) - u_t||^2`. It probes 1D parameter slices along `left[u]`, `right[u_x]`, `product_readout`, `linear[u_xx]`, and a product-rescaling symmetry direction, plus a 2D slice in `(product_readout, linear[u_xx])`. The Burgers-like frozen-SymNet point is locally anisotropic rather than uniformly broad: PDE MSE stays within `2x` of the reference out to about `|alpha| = 0.8` along the rescaling direction, about `0.22` to `0.25` along the left/right/product directions, and only about `0.005` along `linear[u_xx]`. The coefficient-map Jacobian `d xi / d theta` has shape `(9, 10)` and numerical rank `8` at both the Burgers-like point and the bad random-final frozen point, with one near-zero singular value in each case consistent with parameter redundancy. Away from that null direction, the Burgers-like point is well conditioned (singular values `[1.624057, 1.016170, 1.015467, 1.000000, 1.000000, 1.000000, 0.748702, 0.745632, 4.27e-17]`, condition number `2.18`), while the bad random-final point is much less so (singular values `[1.000000, 1.000000, 1.000000, 0.121236, 0.044305, 0.035191, 0.025364, 0.019583, 3.85e-18]`, condition number `51.1`). The optional Hessian is positive in all ten reported eigenvalues at the Burgers-like point but has two negative eigenvalues at the bad point, which is consistent with a locally stable anisotropic basin versus a more saddle-like bad solution. Current interpretation: these local slices strengthen the `theta -> xi` optimization-geometry hypothesis without claiming a global map of the loss surface. Next open question: whether reparameterizing the product path to quotient out the scaling redundancy, or initializing directly in coefficient space before lifting to SymNet parameters, improves random-start recovery on the same frozen derivative field.
- Phase 13 adds a controlled reparameterization experiment on the same frozen Phase 9 derivative field. The exact architectural change is local to the notebook: a `NormalizedFactorSymNet` normalizes the left and right product-factor weights before forming the product, so the product readout retains magnitude control while the continuous scaling redundancy `a -> c a`, `b -> b / c` is reduced. The hand-set representability check still recovers Burgers exactly in this minimal setting: `u_t_hat = (+0.020000)*u_xx + (-1.000000)*u*u_x` with the other seven coefficients at zero to numerical precision. But on September 3, 2026 this did not improve frozen random-start optimization. For the matched seed-0 comparison, the original random-init frozen SymNet ended at PDE MSE `9.651142e-01`, `coeff(u*u_x) = -0.005406`, `coeff(u_xx) = -0.001735`, and coefficient-space distance `||xi - xi_LS||_2 = 0.907846`, whereas the normalized-factor run ended much worse at PDE MSE `6.961301e+01`, `coeff(u*u_x) = -0.036762`, `coeff(u_xx) = +0.281238`, and distance `1.154956`. Across the matched 5-seed sweep, the original architecture had lower final PDE MSE in 4 of 5 seeds and a far better median PDE MSE (`1.580443` versus `480.962189`). The normalized-factor variant produced one run closer to the Phase 11 least-squares coefficient vector (`||xi - xi_LS||_2 = 0.628698` at seed 2 versus `0.964913` for the original seed-2 run), but that same run still had very poor PDE MSE (`4.809622e+02`) and did not recover Burgers. At one representative normalized-factor random-final solution, the coefficient-map Jacobian still had shape `(9, 10)`, numerical rank `8`, and one near-null direction, with effective condition number `9.21`: better than the original bad-basin Phase 12 point (`51.1`) but worse than the original Burgers-like basin (`2.18`). Current interpretation: removing the continuous product-factor scale redundancy alone is not a sufficient fix for the frozen random-start failure in this notebook. Next open question: whether a stronger coefficient-aligned reparameterization or an initialization lifted from coefficient space can preserve the mild Jacobian improvement while also improving the actual optimization trajectory.
