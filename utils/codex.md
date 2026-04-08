The core question is not “does TV lower loss,” but which TV type and strength produce the most stable derivative field and the best PDE recovery?

Since stack already has sweep utilities, plus a trainer that fits data, the clean experiment is a controlled sweep over TV type and TV lambda while holding everything else fixed. The current sweep utilities are already set up for this style of repeated runs and summary reduction   .

## Experiment goal

Determine, for each provided TV regularizer, the lambda regime that best balances:

* data fit
* PDE fit
* derivative smoothness
* coefficient recovery stability

The result should tell you whether TV on (u), TV on (u_x), TV on (u_{xx}), or a spacetime curvature term is the better representative-selector in your surrogate family.

## TV candidates

Assume these four regularizers are available:

* `tv_u`: penalizes roughness in (u) through (u_x)
* `tv_ux`: penalizes roughness in (u_x) through (u_{xx})
* `tv_uxx`: penalizes roughness in (u_{xx}) through (u_{xxx})
* `laplacian_xt_sq`: penalizes spacetime curvature through ((u_{xx}+u_{tt})^2)

The fitting loop already computes (u_t) and uses autograd-built features for PDE fitting, so these regularizers are naturally aligned with the current pipeline  .

## Main sweep design

Use a one-TV-at-a-time sweep first.

For each TV type, sweep over a logarithmic lambda grid:

[
\lambda \in {0,\ 10^{-8},\ 10^{-7},\ 10^{-6},\ 10^{-5},\ 10^{-4},\ 10^{-3},\ 10^{-2}}
]

Run each lambda with multiple seeds, ideally 3 to 5.

So the matrix is:

* TV type in `{tv_u, tv_ux, tv_uxx, laplacian_xt_sq}`
* lambda in `{0, 1e-8, 1e-7, ..., 1e-2}`
* seed in `{0,1,2}` initially

That gives a clean first-stage experiment.

## Fixed controls

Hold constant:

* dataset type, e.g. Burgers first
* architecture
* epochs
* batch size
* learning rate
* noise level
* stride_t, stride_x
* feature library
* `lam_data`
* `lam_pde`

Burgers and Allen–Cahn generators already give you controlled synthetic data with stride and noise knobs  .

Start with Burgers only. It is the sharper test because derivative instability shows up more clearly there.

## Metrics to record

For each run, log:

* final data loss
* final TV loss
* derivative error metrics against a control
* recovered coefficient errors
* seed-to-seed variance

Use your existing derivative comparison utilities as the backbone for derivative metrics, since you already have autograd and finite-difference helpers plus error metrics .

### Required metrics

At minimum:

1. `final_data_loss`
3. `final_tv_loss`
4. `ux_rel_l2`
5. `uxx_rel_l2`
6. `uxxx_rel_l2` if relevant
9. run status

### Strongly recommended summary metrics

For each lambda, aggregate across seeds:

* mean derivative relative L2
* std derivative relative L2

This is what will tell you whether a TV term improves identifiability rather than just making one lucky run look better.

## Recommended run structure

Use a separate sweep root per TV type.

Example directory pattern:

* `runs/tv_u_lambda_sweep`
* `runs/tv_ux_lambda_sweep`
* `runs/tv_uxx_lambda_sweep`
* `runs/laplacian_xt_sq_lambda_sweep`

Each sweep should use `sweep_param = "tv_lambda"` and store `tv_type` in the config. sweep tools already support adding sweep metadata into each run config and summary  .

## Training contract

Have the training function return:

* `history`: rows with `epoch`, `data_loss`, `pde_loss`, `tv_loss`, `total_loss`
* `best_epoch`
* `status`
* `summary_extra` with:

  * `tv_type`
  * `tv_lambda`
  * `ux_rel_l2`
  * `uxx_rel_l2`
  * `uxxx_rel_l2` if used

## Phase 1 experiment spec

Use this as the first task.

### Objective

Evaluate each TV regularizer independently over a log-scale lambda sweep on Burgers data and identify the lambda region that minimizes coefficient recovery error while preserving acceptable data fit.

### Dataset

* Burgers synthetic data
* fixed noise level
* fixed stride
* fixed seed set

### Sweep

For each `tv_type` in:

* `tv_u`
* `tv_ux`
* `tv_uxx`
* `laplacian_xt_sq`

Sweep:

* `tv_lambda ∈ [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]`

Seeds:

* `0,1,2`

### Outputs

For each run:

* history CSV
* summary JSON
* snapshot figure
* derivative error summary

For each sweep:

* sweep results CSV
* plots of final losses vs lambda
* plots of derivative error vs lambda

## Phase 2 experiment spec

After phase 1, select the best two TV types and refine lambda locally.

For example, if `tv_u` and `tv_ux` look best, do a denser search around their best lambdas, e.g.

[
\lambda \in {3e{-6}, 1e{-5}, 3e{-5}, 1e{-4}, 3e{-4}}
]

Then optionally test mixed penalties:

[
\lambda_0 \mathrm{TV}*u + \lambda_1 \mathrm{TV}*{u_x}
]

with a small (3 \times 3) grid around the phase-1 best values.

## What codex should implement

The clean codex scope is:

1. add TV-aware config fields:

   * `tv_type`
   * `tv_lambda`

2. add a TV dispatcher:

   * map `tv_type` string to callable

3. update `fit_data_and_pde(...)` call site to pass:

   * `tv_terms=[(tv_lambda, selected_tv_fn)]` unless `tv_lambda == 0`

4. compute derivative metrics after fitting using your derivative utilities 

6. create one sweep script that accepts:

   * dataset
   * tv_type
   * lambda list
   * seeds

7. emit sweep-level tables and plots via the existing helpers  

## Minimal codex brief

Build a sweep experiment for TV regularization. Sweep one TV type at a time over log-scale lambda values `[0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]`, with seeds `[0,1,2]`, starting on Burgers data. Add config fields `tv_type` and `tv_lambda`, dispatch the selected TV callable, and pass it into `fit_model_to_data` as `tv_terms=[(tv_lambda, fn)]` when lambda is nonzero. For each run, log history rows containing epoch, total_loss, data_loss, pde_loss, and tv_loss. After training, compute derivative error metrics (`ux_rel_l2`, `uxx_rel_l2`, optional `uxxx_rel_l2`) against a finite-difference control and include them in `summary_extra`. Use the existing `train_one`, `run_sweep`, `reduce_sweep`, and `tabulate_sweep` utilities for run management and summary generation. Create one sweep root per TV type and rank best runs primarily by derivative error not by total loss.
