## Workspace Identity

This folder contains run experiments for assessing model weaknesses wrt hyperparameter sweeps (usually).Primary outputs are experiment artifacts under `run_results/` (plots, metrics, saved tensors/models) produced by scripts. 

## Architectural Constraints (Strict)

To prevent code bloating and single-file monolithic anti-patterns, all submitted code must strictly adhere to the following modular layout:

1. **Separation of Concerns:** Run scripts (`run_*.py`) are *orchestrators only*. They may not contain localized deep learning training loops, numeric analytical math, data generation algorithms, or plot rendering code.
2. **Module Delegation:**
   - **Data Sourcing:** Must go through a dataset pipeline or a dynamic registry. Direct implementation of physical grid loops inside execution scripts is forbidden.
   - **Core Math/Physics:** Finite difference calculations, autograd chunk loops, and error metric definitions must reside inside `utils/physics.py` or `utils/derivative_utils.py`.
   - **Serialization/IO:** Log tracking, dict payload dumps, and CSV writers must reside in a stateless `utils/io.py` or separate logger module.
   - **Visualization:** Matplotlib logic must be completely isolated from execution runs. Post-processing engines should ingest `.json` or `.npz` artifacts headlessly after execution to generate plots.
3. **No Redundant Imports:** Entrypoint scripts should not import heavy visualization frameworks (`matplotlib`) directly if they are only responsible for executing model optimizations.

## Task Routing (Read First)

|Task / Domain component | System Layer Reference | Permitted Agent Modification|
|---|---|
|Data Generation & Solvers| Datasets/data/processed/ | Consume via registry interfaces only. Do not inline solver math in run files.|
|Grid Processing & Norms|utils/data_prep_utils.py|Mandatory data baseline engine. Do not duplicate arrays/normalization loops.|
|Loss & Regularization|utils/tv_utils.py|Functional registration interface for total variations.|
|Math & Analytical Derivatives|utils/derivative_utils.py|"Add new finite difference operators here only never within the execution routine."|
|Orchestration / Sweeps|runs/run_*.py|"High-level script that loops over configs, triggers imports and dumps variables."|

## Expected Outputs
For sweep-style experiments, match the concrete layout used by `run_results/siren_hparam_sweep`:

`Experiment type/`
    `dataset/`
        `sweep_axis_1/`
            `sweep_axis_2/` (if available)
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

For sweep-style experiments, produce aggregated visualizations of scalar outcomes across the active sweep axes. Whether to emit 2D heatmaps or 1D plots depends on the number of sweep coordinates:

- Two sweep coordinates: emit 2D heatmaps indexed by the two sweep axes (recommended for grid-style parameter sweeps).
- One sweep coordinate: emit 1D summaries (line plots with mean ± std across seeds) rather than a 2D heatmap so results remain interpretable.

Always include these aggregated metrics when available:

- `final_train_loss_mean`
- `min_train_loss_mean`

And aggregate every available feature-fidelity metric over seeds using the same mean/std naming pattern:

- `<feature_key>_rel_l2_mean`

Examples of primitive feature keys and their summary columns:

- `u` -> `u_rel_l2`, `u_rmse`, `u_max_abs`
- `u_x` -> `ux_rel_l2`, `ux_rmse`, `ux_max_abs`
- `u_xx` -> `uxx_rel_l2`, `uxx_rmse`, `uxx_max_abs`

Interpret these keys as difference norms between model-computed features (using the active primitive feature library) and reference features computed on the clean grid (finite differences or analytic operators as appropriate).

File-naming and output conventions (to distinguish 1D vs 2D outputs):

- Two-sweep coords (2D heatmap): `<sweep_x>_vs_<sweep_y>_<metric>_heatmap.pdf` (e.g. `hidden_layers_vs_hidden_omega_0_final_train_loss_heatmap.pdf`).
- One-sweep coord (1D line plot): `<sweep_coord>_<metric>_lineplot.pdf` (e.g. `hidden_layers_final_train_loss_lineplot.pdf`). Optionally append `_1d` to the filename if your tooling requires an explicit suffix.

Scripts should detect the number of active sweep coordinates (from the sweep configuration or from `summary_agg.csv` grouping keys) and choose heatmaps vs lineplots automatically.

If a run also performs PDE extraction or regularized fitting, extend `summary.json` with experiment-specific keys such as:

- tv_type
- tv_lambda
- final_data_loss
- final_tv_loss
- coeff_error_l2
- pde_method
- feature_names
- ls_method
- ls_terms
- ls_coeffs
- pde_terms
- pde_coeffs
- true_pde_terms
- true_pde_coeffs
- ls_true_pde_terms
- ls_true_pde_coeffs
- eql_feature_names
- eql_readout_terms
- eql_readout_coeffs
- eql_readout_dim

PDE extraction fields are optional and should only be present when the corresponding extraction method was executed.

### Dataset-level summary.csv
`summary.csv` should contain one row per attempted run under a dataset-level sweep folder.

For the active sweep, expected baseline columns are:

- identity: dataset, seed, model
- sweep coordinates for the active experiment
- train configuration
- scalar outcomes
- run state

For EQL-style training, the feature library recorded in `feature_names` should be the primitive input set used by the model, not the LS candidate library.

If feature comparisons are computed for the sweep, include the corresponding feature-fidelity summary keys in `summary.csv` as flat columns.

For each active feature reported by the primitive feature library, include:

- `<feature_key>_rel_l2`
- `<feature_key>_rmse`
- `<feature_key>_max_abs`

Examples:

- `u` -> `u_rel_l2`, `u_rmse`, `u_max_abs`
- `u_x` -> `ux_rel_l2`, `ux_rmse`, `ux_max_abs`
- `u_xx` -> `uxx_rel_l2`, `uxx_rmse`, `uxx_max_abs`

This keeps `summary.csv` usable both for heatmaps built from loss values and for downstream analysis of primitive feature fidelity.

If PDE extraction is computed, include method-specific diagnostic columns.

For least-squares extraction, recommended columns include:

- ls_method
- ls_terms
- ls_coeffs
- ls_residual_rel_l2
- ls_residual_rmse
- ls_rank
- ls_condition_number
- ls_coeff_error_l2
- ls_num_active_terms
- ls_active_terms

where `ls_terms` is the LS-specific candidate library and `ls_active_terms` is a compact string representation of the active LS terms.

For EQL outputs, recommended columns include:

- eql_method
- eql_feature_names
- eql_readout_terms
- eql_readout_coeffs
- eql_readout_dim
- eql_effective_quadratic_matrix_available
- eql_effective_quadratic_matrix_error

Detailed PDE representations, coefficient vectors, and term lists should remain in:

- `pde_outputs/least_squares/pde.json`
- `pde_outputs/eql/pde.json`

rather than being expanded into CSV columns.

### Configuration Standards
- Do not add explicit scalar hyperparameter fields for every potential PDE to the core run class configuration.
- Use a polymorphic approach or a generic dictionary block (`dataset_args`) to isolate dataset-specific physics constraints, preventing horizontal configuration explosion.

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

For each trained SIREN MLP run, generate feature comparison plots for every active feature returned by the primitive feature library.

Compare:

- reference feature values computed from the clean reference solution
- model feature values computed from the trained SIREN prediction

Reference feature values should be computed using the same feature definitions as the active feature library, applied to the clean reference solution. Use finite differences, analytic derivatives, or other documented reference operators as appropriate for the dataset.

The active primitive feature library is the authoritative source of:

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

- `u` -> `u`
- `u_x` -> `ux`
- `u_xx` -> `uxx`

When possible, reuse the repository''s feature-library implementation for feature naming and feature construction rather than reimplementing feature definitions in plotting code.

Use five evenly spaced time slices over the available time domain, or fewer if the grid has fewer than five time indices.

Each plot should show:

- `x` on the horizontal axis
- feature value on the vertical axis
- clean-grid reference and model prediction overlaid at each selected time slice
- the actual plotted time in each subplot title

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

The set of evaluated features is determined by the active primitive feature library used during the run. Do not assume a fixed derivative set beyond the primitives supported by the library. Examples of valid features include:

- `u`
- `u_x`
- `u_xx`

and any future primitive features added to the library. Composite terms such as `uu_x` and `u3` belong to LS-specific candidate libraries, not the primitive feature library.

### Evaluate PDE extraction
1. Open `least_squares_pde.txt`
2. Check whether the recovered active terms match the known PDE
3. Open `least_squares_pde.json`
4. Inspect `residual_rel_l2`, `condition_number`, and `coeff_error_l2`
5. Compare extraction quality against derivative heatmaps

### Conventions

- For sweep scripts, prefer using `utils/data_prep_utils.py` (`PDETrainDataset`, `AffineNormalizer`) for subsampling, noise injection, and coordinate normalization to (-1,1), instead of duplicating `_affine_to_minus1_1`, `_to_norm`, or custom train-sample builders inside new `runs/run_*.py` files.

