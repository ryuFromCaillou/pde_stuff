# Layer 1 (ICM): Repository Map

## Workspace location

On Ark2, use `/home/ghost/projects/pde_stuff` and its `.venv`. This checkout was consolidated on 2026-10-28 onto published `sami` at `3f5c45f`, retaining artifacts from both previous Ark2 checkouts. `/home/ghost/ghost/pde_stuff` has been retired. Historical artifact records may still contain that old absolute prefix; interpret it as historical provenance and map it to the canonical root when locating the same files. Do not rewrite archived scientific evidence merely to update paths. Ark1 remains a separate workspace.

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

For any task involving `notebook/diagnostics/`, a new diagnostic phase, Burgers discovery interpretation, or experiment design based on previous phases, the agent MUST first read:

1. `notebook/diagnostics/RESEARCH_STATE.md`
2. The latest relevant section of `notebook/diagnostics/burgers_minimal_discovery_story.ipynb`.

Reconcile the planned experiment against the current research state before writing or running code.

`notebook/diagnostics/AGENTS.md` governs diagnostic execution and notebook storytelling; consult it for the local code-visibility and figure-title rules.

- Treat `notebook/diagnostics/RESEARCH_STATE.md` as the canonical research-state file for diagnostic work.
- Do not rely solely on older phase descriptions when it contains a newer current-state statement.
- When a diagnostic phase materially changes the scientific interpretation, update `notebook/diagnostics/RESEARCH_STATE.md` in the same change.
- Keep the current-state section concise; append detailed findings to the historical record.

| Task type | Start here |
|---|---|
| Run an experiment / reproduce a result | `runs/` (pick the relevant `run_*.py` entrypoint) |
| Add/modify training logic | `prog/trainer.py` and `prog/*.py` |
| Derivative evaluation / feature building | `utils/derivative_utils.py` and `prog/featlib.py` |
| Primitive feature fidelity / derivative accuracy metrics | `utils/derivative_utils.py` (`evaluate_primitive_feature_metrics`, `build_reference_primitive_features`) and `prog/featlib.py` |
| Primitive feature overlay plotting | `utils/feature_plotting.py` (`save_primitive_feature_overlays`) with `utils/derivative_utils.py` for reference/model feature evaluation |
| PDE coefficient extraction | `utils/extract_pde_ls.py` and `prog/mlps.py` use mlps.EQL for eql pde coeff extraction |
| Model fitting / joint training | `utils/fit_utils.py` and `runs/run_eql_joint_training.py` |
| TV regularization behavior | `utils/tv_utils.py` and `runs/run_eql_joint_training.py` (`--tv-lambda`) |
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

## Diagnostic Validation

- Run lightweight matched-control regression checks with `python -m unittest discover -s tests -v` (`tests/test_burgers_control.py`).
- Run Paul-path mechanical validation with `python runs/run_paul_path_l1.py --validate-only --output <new-audit-directory>`; the destination must not exist. See `runs/PAUL_PATH_L1.md` for the validation and output contract.
- Use each current experiment's protocol in `runs/` for artifact validation; mechanical checks alone do not establish scientific recovery.

## Optimization Policy

Helpers may be refactored or optimized when doing so improves clarity, reuse, or consistency with the run contract.

Do not preserve helper behavior merely because it exists. Preserve the output contract unless explicitly changing it.
