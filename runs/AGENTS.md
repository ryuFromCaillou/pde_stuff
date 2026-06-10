## Workspace Identity

This folder contains run experiments for assessing model weaknesses wrt hyperparameter sweeps (usually).Primary outputs are experiment artifacts under `run_results/` (plots, metrics, saved tensors/models) produced by scripts. 

## Task Routing (Read First)

| Task type | Start here |
|---|---|
| Generate Dataset of pde type | `Datasets/data/processed/` | 
| Normalizing data grid to (-1,1) | `utils/data_prep_utils.py` |
| Choose TV type for loss | `tv_utils.py` |

## Expected Outputs
For sweep type experiments, file structure must contain: 

Experiment type/
    Dataset/
        Sweep parameter/
            seed/
                experiment_config.json
                    dict_keys(['dataset', 'seed', 'device', 'epochs', 'batch_size', 'lr', 'model', 'hidden_size', 'hidden_layers', 'first_omega_0', 'hidden_omega_0', 'noise_level', 'stride_t', 'stride_x', {dataset params, eg:'burgers_N', 'burgers_L', 'burgers_nu', 'burgers_dt', 'burgers_T', 'allen_N', 'allen_x_min', 'allen_x_max', 'allen_dt', 'allen_T', 'allen_d', 'allen_reaction_scale', 'allen_bc_value'}, 'tv_type', 'tv_lambda', 'eval_chunk_size'])    
                derivative_overlays
                fit_heatmap
                fit_snapshots (using hlprs.snapshot_comp)
                loss_history.csv
                summary.json
                    has keys (['dataset', 'seed', 'model', 'epochs', 'batch_size', 'lr', 'noise_level', 'stride_t', 'stride_x', 'tv_type', 'tv_lambda', 'final_total_loss', 'final_data_loss', 'final_tv_loss', 'ux_rel_l2', 'uxx_rel_l2', 'uxxx_rel_l2', 'l2_coeff_error', 'status', 'error', 'pde_names', 'pde_coeffs', 'true_coeffs'])
            

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
