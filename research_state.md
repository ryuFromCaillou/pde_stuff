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
