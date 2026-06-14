## Workspace Identity

This folder contains run experiments for assessing model weaknesses wrt hyperparameter sweeps (usually).Primary outputs are experiment artifacts under `run_results/` (plots, metrics, saved tensors/models) produced by scripts. 

## Task Routing (Read First)

| Task type | Start here |
|---|---|
| Generate Dataset of pde type | `Datasets/data/processed/` | 
| Normalizing data grid to (-1,1) | `utils/data_prep_utils.py` |
| Choose TV type for loss | `tv_utils.py` |

## Expected Outputs
For sweep-style experiments, match the concrete layout used by `run_results/siren_hparam_sweep`:

`Experiment type/`
    `dataset/`
        `sweep_axis_1/`
            `sweep_axis_2/`
                `seed_###/`
                    `config.json`
                    `fit_heatmap.pdf`
                    `fit_snapshots.pdf`
                    `loss_history.csv`
                    `summary.json`
                    `feature_overlays/`
                        `<feature_name>_overlay.pdf`
                    `pde_outputs/`
                        `least_squares/`
                            `pde.json`
                            `pde.txt`
                            `coefficients.csv`
                            `diagnostics.json`
                        `eql/`
                            `pde.json`
                            `pde.txt`
                            `coefficients.csv`
                            `diagnostics.json`
                            `effective_quadratic_matrix.csv`
        `summary.csv`
        `summary_agg.csv`
        `heatmaps/`
            `final_train_loss_heatmap.pdf`
            `min_train_loss_heatmap.pdf`
            `<feature_name>_rel_l2_heatmap.pdf`

### Feature Metric Heatmaps

For SIREN hyperparameter sweeps, training-loss heatmaps are not sufficient. A model may achieve low solution loss while producing poor feature fidelity.

At the dataset sweep level, generate heatmaps for:

- `final_train_loss_mean`
- `min_train_loss_mean`

and for every available feature-fidelity metric:

- `<feature_key>_rel_l2_mean`

Examples:

- `u_rel_l2_mean`
- `ux_rel_l2_mean`
- `uxx_rel_l2_mean`
- `uxxx_rel_l2_mean`
- `uux_rel_l2_mean`

Expected files:

- `final_train_loss_heatmap.pdf`
- `min_train_loss_heatmap.pdf`
- `<feature_key>_rel_l2_heatmap.pdf`

Examples:

- `u_rel_l2_heatmap.pdf`
- `ux_rel_l2_heatmap.pdf`
- `uxx_rel_l2_heatmap.pdf`
- `uxxx_rel_l2_heatmap.pdf`
- `uux_rel_l2_heatmap.pdf`

Feature heatmaps should use the same sweep axes as the training-loss heatmaps.

For `siren_hparam_sweep`, use:

- x-axis: `hidden_omega_0`
- y-axis: `hidden_layers`

The primary diagnostic question is whether low solution error coincides with low feature error. If not, feature fidelity takes priority when evaluating models intended for PDE discovery.

The set of generated feature heatmaps should be determined dynamically from the feature-fidelity metrics present in `summary.csv` or `summary_agg.csv`, rather than from a hardcoded derivative list.

Replace it with this more general version:

### PDE Extraction Outputs

When feature diagnostics are available, each trained model may also be used for post-training PDE extraction.

PDE extraction is method-specific. It should be documented under the method that produced it, not assumed to be least-squares only.

Expected per-run structure:

```text
pde_outputs/
    <pde_method>/
        pde.json
        pde.txt
        coefficients.csv
        diagnostics.json
````

Current methods of `<pde_method>`:

```text
least_squares
eql
```

For least-squares extraction, solve:

[
u_t \approx \Theta(u, u_x, u_{xx}, \ldots)c
]

where:

* `u_t` is computed by autograd from the trained model
* library terms are determined by the active feature library
* coefficients are fit by least squares
* extraction is diagnostic only unless explicitly used in training

`pde_outputs/least_squares/pde.json` should include:

* `method`
* `target`
* `terms`
* `coeffs`
* `residual_rel_l2`
* `residual_rmse`
* `rank`
* `condition_number`
* `true_pde_terms`, when known
* `true_pde_coeffs`, when known
* `coeff_error_l2`, when true coefficients are known

`pde_outputs/least_squares/pde.txt` should contain a human-readable equation.

For EQL/SymNet-style extraction, write analogous outputs under:

```text
pde_outputs/eql/
    pde.json
    pde.txt
    coefficients.csv
    diagnostics.json
    coeff_error_vs_iteration.pdf
    effective_quadratic_matrix.csv
    readout_coefficients.csv
````

For `effective_quadratic_matrix.csv`, do not manually reconstruct the matrix unless necessary. If the active PDE model is the repo’s `EQL` class, use its existing method:

```python
v_model.effective_quadratic_matrix(symmetrize=False)
```

Use `symmetrize=True` only when the analysis explicitly wants symmetric quadratic coefficients.

Only emit `effective_quadratic_matrix.csv` when:

* the PDE model is EQL/SymNet-style,
* the model exposes `effective_quadratic_matrix(...)`, and
* the method is valid for the current EQL configuration.

If the method raises because the EQL configuration is not compatible, record that in `diagnostics.json` instead of inventing a replacement matrix.

`readout_coefficients.csv` should come directly from the EQL readout layer weights.

`product_terms.json` should only be written if product-neuron terms are explicitly extracted from the EQL linear/product layers.

### Per-run summary.json
At minimum, a per-run `summary.json` should track:

- run identity keys: `dataset`, `seed`, `model`
- model identity keys: model
- sweep-coordinate keys for the active experiment
- train config keys: `epochs`, `batch_size`, `lr`, `noise_level`, `stride_t`, `stride_x`
- dataset config keys relevant to the solver used for that run
- optimization outcome keys: `final_train_loss`, `min_train_loss`
- run state keys: `status`, `error`

When feature diagnostics are available, also include learned-vs-reference feature difference norms in the summary keys.

Use the active feature names returned by the feature library as the source of truth.

For each active feature `<feature_name>`, include:

- `<feature_key>_rel_l2`
- `<feature_key>_rmse`
- `<feature_key>_max_abs`

where `<feature_key>` is a filename/column-safe version of the feature name.

Examples:

- `u` → `u_rel_l2`, `u_rmse`, `u_max_abs`
- `u_x` → `ux_rel_l2`, `ux_rmse`, `ux_max_abs`
- `u_xx` → `uxx_rel_l2`, `uxx_rmse`, `uxx_max_abs`
- `u_xxx` → `uxxx_rel_l2`, `uxxx_rmse`, `uxxx_max_abs`
- `uu_x` → `uux_rel_l2`, `uux_rmse`, `uux_max_abs`

These keys should be interpreted as difference norms between:

- feature values computed from the trained model
- reference feature values computed on the clean grid

The trained-model side should use the active feature library implementation as the source of truth. The reference side should use the same feature definitions, using finite differences or analytic/reference operators as needed for clean-grid data.

If a run also performs PDE extraction or regularized fitting, extend summary.json with experiment-specific keys such as:

- tv_type
- tv_lambda
- final_data_loss
- final_tv_loss
- coeff_error_l2
- pde_method
- feature_names
- pde_terms
- pde_coeffs
- true_pde_terms
- true_pde_coeffs

PDE extraction fields are optional and should only be present when the corresponding extraction method was executed.

### Dataset-level summary.csv
`summary.csv` should contain one row per attempted run under a dataset-level sweep folder.

For the active sweep, expected baseline columns are:

- identity: dataset, seed, model
- sweep coordinates for the active experiment
- train configuration
- scalar outcomes
- run state

If feature comparisons are computed for the sweep, include the corresponding feature-fidelity summary keys in `summary.csv` as flat columns.

For each active feature reported by the feature library, include:

- `<feature_key>_rel_l2`
- `<feature_key>_rmse`
- `<feature_key>_max_abs`

Examples:

- `u` → `u_rel_l2`, `u_rmse`, `u_max_abs`
- `u_x` → `ux_rel_l2`, `ux_rmse`, `ux_max_abs`
- `u_xx` → `uxx_rel_l2`, `uxx_rmse`, `uxx_max_abs`
- `u_xxx` → `uxxx_rel_l2`, `uxxx_rmse`, `uxxx_max_abs`
- `uu_x` → `uux_rel_l2`, `uux_rmse`, `uux_max_abs`

This keeps `summary.csv` usable both for heatmaps built from loss values and for downstream analysis of feature fidelity across different PDE libraries and feature sets.

If PDE extraction is computed, include method-specific diagnostic columns.

For least-squares extraction, recommended columns include:

- ls_residual_rel_l2
- ls_residual_rmse
- ls_rank
- ls_condition_number
- ls_coeff_error_l2
- ls_num_active_terms
- ls_active_terms

where `ls_active_terms` is a compact string representation of the active library terms.

Detailed PDE representations, coefficient vectors, and term lists should remain in:

pde_outputs/least_squares/pde.json

rather than being expanded into CSV columns.

### Dataset-level summary_agg.csv

`summary_agg.csv` should contain one row per sweep-coordinate group after aggregating over seeds.

Grouping keys should be the active sweep coordinates for the experiment.

For `siren_hparam_sweep`, expected grouping keys are:

- `hidden_layers`
- `hidden_omega_0`

Expected aggregate columns include:

- `num_seeds`
- `final_train_loss_mean`
- `final_train_loss_std`
- `min_train_loss_mean`
- `min_train_loss_std`

If feature-fidelity metrics are present in `summary.csv`, aggregate them dynamically using the same mean/std pattern:

- `<feature_key>_rel_l2_mean`
- `<feature_key>_rel_l2_std`
- `<feature_key>_rmse_mean`
- `<feature_key>_rmse_std`
- `<feature_key>_max_abs_mean`
- `<feature_key>_max_abs_std`

Examples:

- `ux_rel_l2_mean`
- `ux_rel_l2_std`
- `uxx_rmse_mean`
- `uxx_rmse_std`
- `uux_max_abs_mean`
- `uux_max_abs_std`

Aggregation should be generated from the feature-fidelity columns present in `summary.csv`, not from a hardcoded derivative list.

If least-squares PDE extraction metrics are present in `summary.csv`, aggregate scalar LS diagnostics where mean/std are meaningful, for example:

- `ls_residual_rel_l2_mean`
- `ls_residual_rel_l2_std`
- `ls_residual_rmse_mean`
- `ls_residual_rmse_std`
- `ls_condition_number_mean`
- `ls_condition_number_std`
- `ls_coeff_error_l2_mean`
- `ls_coeff_error_l2_std`

Do not aggregate list/string fields such as `ls_terms`, `ls_coeffs`, or `ls_active_terms` unless a script explicitly defines a stable representation.

### Feature Overlays

Feature overlay requirement:

For each trained SIREN MLP run, generate feature comparison plots for every active feature returned by the feature library.

Compare:

- reference feature values computed from the clean reference solution
- model feature values computed from the trained SIREN prediction

Reference feature values should be computed using the same feature definitions as the active feature library, applied to the clean reference solution. Use finite differences, analytic derivatives, or other documented reference operators as appropriate for the dataset.

The active feature library is the authoritative source of:

- feature names
- feature definitions
- feature ordering

The set of generated overlays should be determined from the active feature names returned by the feature library. Do not maintain a separate hardcoded list of overlay features.

For each active feature:

- use the feature name returned by the feature library as the source of truth
- generate a deterministic feature key used consistently across:
  - overlay filenames
  - summary.json keys
  - summary.csv columns
  - summary_agg.csv columns
  - heatmap filenames

Examples:

- `u → u`
- `u_x → ux`
- `u_xx → uxx`
- `u_xxx → uxxx`
- `uu_x → uux`

When possible, reuse the repository's feature-library implementation for feature naming and feature construction rather than reimplementing feature definitions in plotting code.

Use five evenly spaced time slices over the available time domain, or fewer if the grid has fewer than five time indices.

Each plot should show:

- `x` on the horizontal axis
- feature value on the vertical axis
- clean-grid reference and model prediction overlaid at each selected time slice
- the actual plotted time in each subplot title

For `run_siren_hparam_sweep.py`, save these files directly under the run directory:

- `<feature_key>_overlay.pdf`

Do not document a nested `feature_overlays/` folder for this sweep unless the script is updated to actually emit one.

If a future run script also records feature error summaries, use the same learned-vs-reference naming convention as the summary tables:

- `<feature_key>_rel_l2`
- `<feature_key>_rmse`
- `<feature_key>_max_abs`

If per-slice metrics are written to a separate file in the future, name and document that file explicitly in the corresponding `run_*.py` entrypoint.

## Common Workflows

### Run TV sweep
1. Generate dataset using helpers in `utils/data_prep_utils.py`
2. Configure sweep values
3. Execute run_tv_sweep.py
4. Check run_results/
5. Compare summary.json files

### Sweep SIREN hyperparameters
1. Set `hidden_layers_grid` and `hidden_omega_0s`
2. Execute `run_siren_hparam_sweep.py`
3. Inspect per-run `summary.json` files
4. Compare `summary_agg.csv`
5. Review `final_train_loss_heatmap.pdf`

### Evaluate feature accuracy

1. Open `summary.json`
2. Inspect all available feature-fidelity metrics:
   - `<feature_key>_rel_l2`
   - `<feature_key>_rmse`
   - `<feature_key>_max_abs`
3. Inspect all available feature overlay plots:
   - `<feature_key>_overlay.pdf`
4. Compare dataset-level feature heatmaps when present:
   - `<feature_key>_rel_l2_heatmap.pdf`
   - `<feature_key>_rmse_heatmap.pdf`
   - `<feature_key>_max_abs_heatmap.pdf`
5. Prefer models with stable feature-fidelity metrics, not merely low train loss.

The set of evaluated features is determined by the active feature library used during the run. Do not assume a fixed derivative set (`u`, `u_x`, `u_xx`, `u_xxx`). Examples of valid features include:

- `u`
- `u_x`
- `u_xx`
- `u_xxx`
- `uu_x`

and any future features added to the library.

### Evaluate PDE extraction
1. Open `least_squares_pde.txt`
2. Check whether the recovered active terms match the known PDE
3. Open `least_squares_pde.json`
4. Inspect `residual_rel_l2`, `condition_number`, and `coeff_error_l2`
5. Compare extraction quality against derivative heatmaps

### Conventions

- For sweep scripts, prefer using `utils/data_prep_utils.py` (`PDETrainDataset`, `AffineNormalizer`) for subsampling, noise injection, and coordinate normalization to (-1,1), instead of duplicating `_affine_to_minus1_1`, `_to_norm`, or custom train-sample builders inside new `runs/run_*.py` files.
