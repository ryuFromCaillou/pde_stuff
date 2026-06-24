# Layer 1 (ICM): Repository Map

## Project Identity

This repository contains PyTorch-based experiments for PDE-Net style workflows: learning surrogate fields, evaluating derivatives/features, fitting PDE coefficients, and running sweeps/ablations. Primary outputs are experiment artifacts under `run_results/` (plots, metrics, saved tensors/models) produced by scripts in `runs/`.

## Agent Role

You are maintaining a PDE discovery research codebase.

Your job is to preserve experimental repeatability, keep outputs interpretable, and avoid hardcoding assumptions tied to one PDE such as Burgers.

Prefer abstractions driven by:

- active dataset config
- active feature library
- active PDE extraction method
- declared sweep axes

When adding outputs, update the corresponding run documentation and summary contracts.

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
| Primitive feature fidelity / derivative accuracy metrics | `utils/derivative_utils.py` (`evaluate_primitive_feature_metrics`, `build_reference_primitive_features`) and `prog/featlib.py` |
| Primitive feature overlay plotting | `utils/feature_plotting.py` (`save_primitive_feature_overlays`) with `utils/derivative_utils.py` for reference/model feature evaluation |
| PDE coefficient extraction | `utils/extract_pde_ls.py` and `prog/mlps.py` use mlps.EQL for eql pde coeff extraction |
| Fit tests / model fitting utilities | `runs/run_fit_tests.py` and `utils/fit_utils.py` |
| TV regularization behavior | `utils/tv_utils.py` and `runs/run_tv_lambda_sweep.py` |
| Understand outputs | `run_results/` (match folder names to `runs/` scripts) |
| Set up dependencies | `setup/requirements.txt` |

### Canonical Evaluation Path

For primitive feature/derivative accuracy reporting, follow this path:

1. Treat the active primitive feature library in `prog/featlib.py` as the source of truth for feature names, definitions, and ordering.
2. Evaluate model-side primitive features through `utils/derivative_utils.py:evaluate_model_primitive_features(...)`, which reuses `prog.featlib.FeatureTensor` instead of reimplementing feature construction inside a run script.
3. Build reference primitive features on the clean grid through `utils/derivative_utils.py:build_reference_primitive_features(...)` or a dataset-specific replacement when analytic/reference operators are more appropriate than the default finite-difference path.
4. Compute learned-vs-reference feature metrics through `utils/derivative_utils.py:evaluate_primitive_feature_metrics(...)`.
5. Use `utils/derivative_utils.py:primitive_feature_name_to_key(...)` to derive deterministic summary/plot keys such as `u`, `ux`, and `uxx`.

Do not compute primitive feature metrics from LS candidate libraries, ad hoc derivative code inside `runs/run_*.py`, or a hardcoded derivative list that bypasses the active feature library.

### Canonical Plotting Path

For primitive feature overlays:

1. Keep plotting logic out of `runs/run_*.py`; the run script should call a helper only.
2. Use `utils/feature_plotting.py:save_primitive_feature_overlays(...)` to generate `feature_overlays/<feature_key>_overlay.pdf`.
3. Reuse the canonical model/reference feature evaluation path from `utils/derivative_utils.py` rather than reimplementing feature definitions in plotting code.
4. Use deterministic feature keys from `utils/derivative_utils.py:primitive_feature_name_to_key(...)` so filenames and summary columns stay aligned.


## Global Rules

- Prefer routing new runnable work through `runs/` (keep `prog/` and `utils/` importable and reusable).
- Treat `run_results/` as generated artifacts; don't overwrite or delete prior results unless explicitly requested.
- When changing experiment behavior, update the corresponding `runs/run_*.py` entrypoint so results are reproducible.

## Automated Output Checks

If you want an automated "run -> validate outputs" loop (local or via a remote wrapper), use:

- `runs/run_output_check.py` to validate a specific `run_results/...` folder against a JSON spec
- `runs/run_test_loop.py` to repeatedly run a command and validate the latest output
- Example spec: `runs/regression_specs/tv_sweep_smoke.json`

## Optimization Policy

Helpers may be refactored or optimized when doing so improves clarity, reuse, or consistency with the run contract.

Do not preserve helper behavior merely because it exists. Preserve the output contract unless explicitly changing it.
