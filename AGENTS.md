# Layer 1 (ICM): Repository Map

## Project Identity

This repository contains PyTorch-based experiments for PDE-Net style workflows: learning surrogate fields, evaluating derivatives/features, fitting PDE coefficients, and running sweeps/ablations. Primary outputs are experiment artifacts under `run_results/` (plots, metrics, saved tensors/models) produced by scripts in `runs/`.

## Important Regions

- `prog/` -> core training/model code (e.g., trainer, MLPs, feature construction)
- `utils/` -> reusable utilities (derivatives, TV regularization, data prep, PDE extraction via least-squares)
- `Datasets/` -> dataset generation / stored datasets used by experiments
- `runs/` -> runnable experiment entrypoints (single-run experiments, sweeps, ablations)
- `run_results/` -> generated outputs from running experiments (treat as artifacts)
- `notebook/` -> exploratory analysis notebooks
- `setup/` -> environment/dependency setup (`setup/requirements.txt`)
- `.venv/` -> local virtual environment (if present)

## Task Routing (Read First)

| Task type | Start here |
|---|---|
| Run an experiment / reproduce a result | `runs/` (pick the relevant `run_*.py` entrypoint) |
| Add/modify training logic | `prog/trainer.py` and `prog/*.py` |
| Derivative evaluation / feature building | `utils/derivative_utils.py` and `prog/featlib.py` |
| PDE coefficient extraction | `utils/extract_pde_ls.py` and `runs/run_pde_extraction_experiment.py` |
| Fit tests / model fitting utilities | `runs/run_fit_tests.py` and `utils/fit_utils.py` |
| TV regularization behavior | `utils/tv_utils.py` and `runs/run_tv_lambda_sweep.py` |
| Understand outputs | `run_results/` (match folder names to `runs/` scripts) |
| Set up dependencies | `setup/requirements.txt` |


## Global Rules

- Prefer routing new runnable work through `runs/` (keep `prog/` and `utils/` importable and reusable).
- Treat `run_results/` as generated artifacts; don't overwrite or delete prior results unless explicitly requested.
- When changing experiment behavior, update the corresponding `runs/run_*.py` entrypoint so results are reproducible.

## Automated Output Checks

If you want an automated "run -> validate outputs" loop (local or via a remote wrapper), use:

- `runs/run_output_check.py` to validate a specific `run_results/...` folder against a JSON spec
- `runs/run_test_loop.py` to repeatedly run a command and validate the latest output
- Example spec: `runs/regression_specs/tv_sweep_smoke.json`
