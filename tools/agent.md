# Derivative Accuracy Experiment: SimpleMLP vs SIREN

## Objective

Implement an isolated experiment to compare **derivative accuracy** of two function approximators for PDE solution fitting:

- `SimpleMLP`
- `SirenMLP`

The question is:

> When both models are trained to fit the same PDE solution data, does SIREN yield more accurate spatial derivatives than `SimpleMLP`?

This experiment is **not** a PDE-discovery experiment. Do **not** involve the PDE readout head (`symMLP`, `EQL`, etc.) in training or evaluation. The models are used only to fit `u(t,x)`.

---

## Existing codebase expectations

The codebase already contains:

- `SimpleMLP` in `mlps.py`
- dataset builders for Burgers and Allen–Cahn
- helper plotting utilities in `hlprs.py`

Preserve the existing interfaces where possible.

The model contract should remain:

```python
model(t, x) -> u_pred
````

with `t` and `x` both shaped `(N,1)`.

---

## Required deliverables

Implement the following:

1. A proper `SirenMLP` class in `mlps.py`
2. A standalone experiment driver script
3. Utility functions for:

   * training pure function-fit models
   * autograd derivative extraction up to third order
   * finite-difference control derivatives
   * metric computation
   * plotting derivative comparisons
4. Output directories with plots and logs
5. A summary CSV aggregating all runs

---

## File-by-file instructions

### 1. `mlps.py`

Add a SIREN implementation **without breaking existing code**.

#### Requirements

Implement:

* `SineLayer`
* `SirenMLP`

#### SIREN design

`SineLayer` must:

* apply a linear map
* then apply `sin(omega_0 * linear(x))`

Use SIREN-style initialization:

* first layer bound: `1 / in_features`
* hidden layer bound: `sqrt(6 / in_features) / omega_0`

`SirenMLP` should:

* accept `(t, x)` as separate tensors
* concatenate them into a 2D input
* support configurable:

  * hidden size
  * number of hidden layers
  * `first_omega_0`
  * `hidden_omega_0`
* end with a linear readout to 1 scalar output

#### Constraint

Do **not** replace or modify `SimpleMLP` behavior unless necessary for compatibility.

---

### 2. Create new file: `derivative_utils.py`

Add the following utilities.

#### 2.1 `autograd_spatial_derivatives`

Signature idea:

```python
def autograd_spatial_derivatives(model, t, x, max_order=3):
    ...
```

Input:

* `model`
* `t`, `x` tensors shaped `(N,1)` with `requires_grad=True`

Output:

* dictionary like:

```python
{
    "u": u,
    "ux": ux,
    "uxx": uxx,
    "uxxx": uxxx,
}
```

Use nested `torch.autograd.grad`.

#### 2.2 finite difference utilities

Implement finite-difference controls on a 1D spatial slice.

Suggested functions:

```python
def fd_first_periodic(u_row, dx):
def fd_second_periodic(u_row, dx):
def fd_third_periodic(u_row, dx):
```

for Burgers periodic data.

Also implement non-periodic centered versions for Allen–Cahn, for example:

```python
def fd_first_centered(u_row, dx):
def fd_second_centered(u_row, dx):
def fd_third_centered(u_row, dx):
```

For non-periodic cases:

* use centered stencils in the interior
* return only valid interior points for fairness
* do not fake boundary values unless clearly documented

#### 2.3 `compute_error_metrics`

Signature idea:

```python
def compute_error_metrics(pred, ref, eps=1e-12):
    ...
```

Return:

* relative L2 error
* RMSE
* max absolute error

Format:

```python
{
    "rel_l2": ...,
    "rmse": ...,
    "max_abs": ...,
}
```

#### 2.4 grid reshaping helper

Implement a utility that reconstructs `(Nt, Nx)` arrays from flattened `(t, x, y)` data.

Suggested:

```python
def reconstruct_grid(t_np, x_np, y_np, round_decimals=6):
    ...
```

Return:

* `t_unique`
* `x_unique`
* `Y_grid`

This should be robust to flattened ordering.

---

### 3. Create new file: `fit_utils.py`

This file should handle pure function fitting.

#### 3.1 `fit_model_to_data`

Signature idea:

```python
def fit_model_to_data(
    model,
    t_train,
    x_train,
    y_train,
    *,
    epochs,
    batch_size,
    lr,
    device,
    weight_decay=0.0,
    log_every=100,
):
    ...
```

Requirements:

* standard supervised fit only
* MSE loss between `model(t, x)` and target `y`
* Adam optimizer
* minibatch or full-batch support
* return:

  * trained model
  * per-epoch loss history

Keep this separate from `PDETrainer`.

#### 3.2 `predict_on_grid`

Signature idea:

```python
def predict_on_grid(model, t_np, x_np, device):
    ...
```

Return flattened predictions aligned with inputs.

---

### 4. Create new file: `run_derivative_experiment.py`

This is the main driver.

---

## Experiment design

### Datasets

Use at least:

* Burgers
* Allen–Cahn

Use existing dataset generation code.

For each dataset family:

* define one base config
* sweep over multiple seeds

Suggested default:

* 5 seeds minimum

Keep noise level configurable.

---

## Core loop

For each dataset type:
for each seed:

1. build dataset
2. construct both models:

   * `SimpleMLP`
   * `SirenMLP`
3. train both on the **same** sampled data
4. evaluate both on a dense clean grid
5. compute derivatives by autograd
6. compute control derivatives by finite difference on clean solver data
7. compare errors
8. save plots
9. append one row per model to summary records

---

## Important design constraints

### A. Pure fit experiment only

Do not use:

* `PDETrainer`
* `FeatureTensor`
* `symMLP`
* `EQL`

except possibly borrowing generic code style.

### B. Match model capacity

Choose matched width/depth between `SimpleMLP` and `SirenMLP`.

Example:

* hidden size = 64
* hidden layers = 3 or equivalent depth

### C. Same training conditions

For each pairwise comparison, both models must use:

* same training data
* same epoch count
* same optimizer type
* same learning rate
* same batch size

### D. Dense evaluation

Evaluation should happen on a dense clean grid from the solver, not only sampled training points.

### E. Coordinate consistency

If you standardize `(t, x)` before model input, do it identically for both models.

If derivatives are computed in normalized coordinates, convert them back to physical derivatives before comparison.

Document the conversion clearly.

---

## Recommended coordinate handling

Standardize inputs to `[-1,1]` for both models.

If:

```python
x_phys = a_x * x_norm + b_x
t_phys = a_t * t_norm + b_t
```

then:

* `u_x_phys = (1 / a_x) * u_x_norm`
* `u_xx_phys = (1 / a_x**2) * u_xx_norm`
* `u_xxx_phys = (1 / a_x**3) * u_xxx_norm`

Apply this correction before comparing to FD controls.

This is especially important for SIREN.

---

## Metrics

For each model and run, compute metrics for:

* `u`
* `u_x`
* `u_xx`
* `u_xxx`

Each should include:

* relative L2 error
* RMSE
* max absolute error

Also compute optional restricted-region metrics for Burgers:

* evaluate derivative errors on high-gradient regions, e.g. top 10% of `|u_x|`
* this is optional but recommended

---

## Plotting requirements

### 1. Fit snapshots

Reuse existing plotting helpers if possible.

Save:

* predicted `u(t,x)` heatmap
* several time-slice overlays of predicted vs true

### 2. Derivative overlays

For selected time slices:

* overlay FD control and model derivative for:

  * `u_x`
  * `u_xx`
  * `u_xxx`

Do this for both models.

### 3. Error heatmaps

Save heatmaps over `(t,x)` of:

* `|u_pred - u_true|`
* `|ux_pred - ux_fd|`
* `|uxx_pred - uxx_fd|`
* `|uxxx_pred - uxxx_fd|`

### 4. Loss curves

Save training loss vs epoch for each model.

---

## Output layout

Create run directories like:

```text
runs/
  derivative_compare/
    burgers/
      seed_000/
        simplemlp/
        siren/
      seed_001/
      ...
    allen_cahn/
      seed_000/
      ...
```

Inside each model directory save:

* `loss_curve.pdf`
* `fit_heatmap.pdf`
* `fit_snapshots.pdf`
* `ux_overlay.pdf`
* `uxx_overlay.pdf`
* `uxxx_overlay.pdf`
* `error_heatmaps.pdf`
* `metrics.json`

At dataset level save:

* `summary.csv`

At top level save:

* `all_results.csv`

---

## Summary CSV schema

Each row should represent one `(dataset, seed, model)` run.

Suggested columns:

* `dataset`
* `seed`
* `model`
* `epochs`
* `batch_size`
* `lr`
* `noise_level`
* `u_rel_l2`
* `u_rmse`
* `u_max_abs`
* `ux_rel_l2`
* `ux_rmse`
* `ux_max_abs`
* `uxx_rel_l2`
* `uxx_rmse`
* `uxx_max_abs`
* `uxxx_rel_l2`
* `uxxx_rmse`
* `uxxx_max_abs`
* `final_train_loss`

Optional:

* `ux_rel_l2_highgrad`
* `uxx_rel_l2_highgrad`
* `uxxx_rel_l2_highgrad`

---

## Implementation notes

### Burgers control derivatives

Because Burgers is periodic in the generator, use periodic FD stencils.

### Allen–Cahn control derivatives

Because Allen–Cahn uses Dirichlet boundaries, avoid comparing derivatives at unstable boundary points unless carefully handled.

A clean option:

* restrict derivative comparison to interior spatial indices

### Training targets

Default design:

* train on noisy sampled data
* evaluate against clean dense solver field

This tests whether each architecture recovers clean derivatives from imperfect samples.

Support a switch to train on clean data too if convenient.

---

## Acceptance criteria

The task is complete when:

1. `SirenMLP` exists and runs
2. Both models can be trained on Burgers and Allen–Cahn sampled data
3. Autograd derivatives up to third order are computed
4. FD controls are computed from clean dense solver data
5. Metrics and plots are saved per run
6. Summary CSVs are produced
7. The experiment runs end-to-end without involving the PDE-head models

---

## Nice-to-have extensions

If straightforward, also include:

1. Multiple training restarts per seed
2. Aggregated mean/std summary across seeds
3. A final comparison plot:

   * boxplots or line plots of derivative errors by model and derivative order
4. A simple markdown report summarizing:

   * whether SIREN beats `SimpleMLP`
   * on which datasets
   * at which derivative order
   * whether gains are largest near steep gradients

---

## Final instruction

Optimize for clarity and reproducibility.

Do not over-engineer a framework.

Prefer a clean, readable experiment pipeline that can be inspected and modified easily.

