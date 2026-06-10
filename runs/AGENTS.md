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
                    `ux_overlay.pdf`
                    `uxx_overlay.pdf`
                    `uxxx_overlay.pdf`
        `summary.csv`
        `summary_agg.csv`
        `final_train_loss_heatmap.pdf`
        `min_train_loss_heatmap.pdf`

For the SIREN hyperparameter sweep specifically, this becomes:

`run_results/siren_hparam_sweep/`
    `burgers/`
        `layers_1/`
            `hidden_omega_1/`
                `seed_000/`
                    `config.json`
                    `fit_heatmap.pdf`
                    `fit_snapshots.pdf`
                    `loss_history.csv`
                    `summary.json`
                    `ux_overlay.pdf`
                    `uxx_overlay.pdf`
                    `uxxx_overlay.pdf`
        `summary.csv`
        `summary_agg.csv`
        `final_train_loss_heatmap.pdf`
        `min_train_loss_heatmap.pdf`

### Per-run summary.json
At minimum, a per-run `summary.json` should track:

- run identity keys: `dataset`, `seed`, `model`
- model/sweep keys: `hidden_size`, `hidden_layers`, `first_omega_0`, `hidden_omega_0`
- train config keys: `epochs`, `batch_size`, `lr`, `noise_level`, `stride_t`, `stride_x`
- dataset config keys relevant to the solver used for that run
- optimization outcome keys: `final_train_loss`, `min_train_loss`
- run state keys: `status`, `error`

When derivative diagnostics are available, also include autograd-vs-FD difference norms in the summary keys. Prefer the repo’s existing metric naming pattern:

- field value error: `u_rel_l2`, `u_rmse`, `u_max_abs`
- first derivative error: `ux_rel_l2`, `ux_rmse`, `ux_max_abs`
- second derivative error: `uxx_rel_l2`, `uxx_rmse`, `uxx_max_abs`
- third derivative error: `uxxx_rel_l2`, `uxxx_rmse`, `uxxx_max_abs`

These keys should be interpreted as difference norms between:

- autograd predictions from the trained model
- finite-difference reference derivatives computed on the clean grid

If a run also performs PDE extraction or regularized fitting, extend `summary.json` with experiment-specific keys such as `tv_type`, `tv_lambda`, `final_data_loss`, `final_tv_loss`, `l2_coeff_error`, `pde_names`, `pde_coeffs`, and `true_coeffs`.

### Dataset-level summary.csv
`summary.csv` should contain one row per attempted run under a dataset-level sweep folder.

For a SIREN hyperparameter sweep, expected baseline columns are:

- identity: `dataset`, `seed`, `model`
- sweep coordinates: `hidden_size`, `hidden_layers`, `first_omega_0`, `hidden_omega_0`
- train config: `epochs`, `batch_size`, `lr`, `noise_level`, `stride_t`, `stride_x`
- scalar outcomes: `final_train_loss`, `min_train_loss`
- run state: `status`, `error`

If derivative comparisons are computed for the sweep, include the same autograd-vs-FD summary keys in `summary.csv` as flat columns:

- `u_rel_l2`, `u_rmse`, `u_max_abs`
- `ux_rel_l2`, `ux_rmse`, `ux_max_abs`
- `uxx_rel_l2`, `uxx_rmse`, `uxx_max_abs`
- `uxxx_rel_l2`, `uxxx_rmse`, `uxxx_max_abs`

This keeps `summary.csv` usable both for heatmaps built from loss values and for downstream analysis of derivative fidelity.

### Dataset-level summary_agg.csv
`summary_agg.csv` should contain one row per sweep-coordinate group after aggregating over seeds.

For `siren_hparam_sweep`, expected columns are:

- grouping keys: `hidden_layers`, `hidden_omega_0`
- aggregation count: `num_seeds`
- loss aggregates: `final_train_loss_mean`, `final_train_loss_std`, `min_train_loss_mean`, `min_train_loss_std`

If derivative metrics are present in `summary.csv`, aggregated versions may also be added using the same pattern, for example `ux_rel_l2_mean`, `ux_rel_l2_std`.

### Derivative Overlays
Derivative overlay requirement:

For each trained SIREN MLP run, generate derivative comparison plots for spatial derivatives up to order `n=3`.

Compare:

- finite-difference derivative from the clean reference grid
- autograd derivative from the trained SIREN model

For each derivative order:

- n=1: compare u_x
- n=2: compare u_xx
- n=3: compare u_xxx

Use five evenly spaced time slices over the available time domain, or fewer if the grid has fewer than five time indices.

Each plot should show:

- `x` on the horizontal axis
- derivative value on the vertical axis
- FD reference and model prediction overlaid at each selected time slice
- the actual plotted time in each subplot title

For `run_siren_hparam_sweep.py`, save these files directly under the run directory:

- `ux_overlay.pdf`
- `uxx_overlay.pdf`
- `uxxx_overlay.pdf`

Do not document a nested `derivative_overlays/` folder for this sweep unless the script is updated to actually emit one.

If a future run script also records derivative error summaries, use the same autograd-vs-FD naming convention as the summary tables:

- `ux_rel_l2`, `ux_rmse`, `ux_max_abs`
- `uxx_rel_l2`, `uxx_rmse`, `uxx_max_abs`
- `uxxx_rel_l2`, `uxxx_rmse`, `uxxx_max_abs`

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

### Evaluate derivative accuracy
1. Open summary.json
2. Inspect ux_rel_l2
3. Inspect uxx_rel_l2
4. Inspect derivative_overlays/

## Conventions

- For sweep scripts, prefer using `utils/data_prep_utils.py` (`PDETrainDataset`, `AffineNormalizer`) for subsampling, noise injection, and coordinate normalization to (-1,1), instead of duplicating `_affine_to_minus1_1`, `_to_norm`, or custom train-sample builders inside new `runs/run_*.py` files.
