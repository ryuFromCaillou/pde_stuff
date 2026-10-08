# Experiment execution contract

## Workspace identity and routing

`runs/` is the repository's experiment-execution and diagnostic orchestration layer: exact reproductions, controlled experiments, long-horizon runs, diagnostic instrumentation, interventions, ablations, robustness/seed studies, parameter sweeps, artifact-backed post-hoc analyses, and validation/regression checks. `run_results/` contains the resulting scientific artifacts.

Read root `AGENTS.md`. Before diagnostic/research work, read `notebook/diagnostics/RESEARCH_STATE.md`, the latest relevant section of `notebook/diagnostics/burgers_minimal_discovery_story.ipynb`, and `notebook/diagnostics/AGENTS.md`. Reconcile the design with current research state before writing or running code. Inspect the relevant runner, reusable implementation, protocol, and archived config; do not assume a runner's defaults are safe reproduction destinations.

## Sources of truth and historical evidence

For diagnostic/research work, use this hierarchy:

1. `notebook/diagnostics/RESEARCH_STATE.md`: canonical current scientific interpretation.
2. Existing run artifacts: immutable evidence from completed experiments.
3. Run-specific reports/configs: detailed evidence and provenance for individual experiments.
4. `burgers_minimal_discovery_story.ipynb`: curated scientific narrative, not the canonical experiment log.
5. Historical notebook Markdown and old run documentation cannot override a newer validated statement in `RESEARCH_STATE.md`.

After validating a result that materially changes interpretation, update `RESEARCH_STATE.md` in the same change: keep current state concise and append detailed findings to history. Mark earlier conclusions as superseded or horizon-limited in current interpretation; do not rewrite history to imply they were never held. A report of non-recovery by step 1000 remains valid evidence of non-recovery within the first 1000 optimization steps.

## Artifact preservation

Existing `run_results/` directories are scientific evidence. By default, never overwrite, silently regenerate in place, delete, or alter their configs, histories, checkpoints, or reports. A later interpretation does not authorize editing an old artifact.

Write reproductions to new destinations such as `<experiment>_repeat/`, `<experiment>_reproduction/`, or `<experiment>_audit/`, unless the task explicitly specifies another destination. Check actual output/resume behavior before execution. Validate existing evidence by reading, checking, and hashing it; write new audit results separately. Regenerated figures/reports also belong in a new destination unless modification is explicitly requested.

## Exact reproduction and passive diagnostics

“Reproduce” means preserve all scientifically relevant configuration unless explicitly changed: dataset/PDE parameters, physical time horizon, sampling and batch sequence, coordinate representation, architecture, initialization, seeds, feature definitions/order, scaling/normalization, loss weights, optimizer, learning rate/schedule, batch size, regularization, training horizon, and checkpoint semantics (including pre/post-update timing). Record source checkpoints, runtime/device/dtype details, and every deviation.

Never silently substitute a newer “better” pipeline. If historical Markdown disagrees with executable code, report the discrepancy and determine which definition the task requires before proceeding with dependent work. Refactoring permission does not authorize changing a historical trajectory.

Instrumentation is observational unless explicitly designed as an intervention. Preserve RNG state, minibatch sequence, initialization, gradients, optimizer state/Adam moments, parameter updates, and learning-rate schedule. Isolate randomness used by diagnostics. Additional backward passes must not contaminate training gradients or optimizer state; prefer offline analysis of captured states for substantial diagnostics.

When trajectory identity matters, validate overlapping scalar histories and/or checkpoint tensors against the archive before interpreting new diagnostics. Report reproduction as **exact**, **numerically equivalent within stated tolerance**, or **approximate**, with evidence and tolerances. Never label an approximate reconstruction exact.

## Step semantics and evaluation data

One iteration that samples one minibatch, computes one gradient, and calls `optimizer.step()` is an **optimizer step / optimization step**. Use **epoch** only for genuine traversal of a defined training dataset. Preserve historical field names when reading artifacts, but explain their actual semantics.

Distinguish sampled minibatch loss, sampled training-grid metric, fixed evaluation-grid metric, and full-grid metric; they are not interchangeable. Use a fixed evaluation grid or fixed evaluation sample for trajectory comparisons. Training loss may remain minibatch-based; scientific transitions must not be diagnosed solely from changing random minibatches.

Record evaluation grid/indices, sampling method, reference derivative construction, coordinate system, and normalization. Burgers derivatives computed numerically from a generated solution grid are **numerical** or **generator-consistent reference quantities**, not analytic truth. State explicitly when reference `u_t` is defined by the Burgers RHS.

## Scientific layers and metric interpretation

Keep the following layers separate; low error or success in one does not establish success in another:

| Layer | Quantities or question |
|---|---|
| Field fidelity | `u` |
| Derivative/target fidelity | `u_t`, `u_x`, `u_xx`, composite physical terms such as `u*u_x` |
| Feature-library geometry | Correlations, Gram matrices, singular values, rank, condition numbers, true/spurious subspace relationships |
| Symbolic representational sufficiency | Equation supported by a frozen representation; LS as a coefficient-space diagnostic |
| Symbolic optimization | Whether MinimalSymNet actually reaches the equation |
| Joint optimization dynamics | Gradients, update directions, optimizer state, data/PDE loss interactions |
| PDE recovery | Physical coefficient accuracy and spurious terms |

Do not rank derivative quality or feature importance by raw MSE across different units/scales. Where appropriate report MSE, relative L2, cosine similarity, predicted RMS, and reference RMS. Distinguish raw library conditioning from normalized/unit-column conditioning.

Least squares is optional, not a default or required diagnostic. Include it only when the explicitly scoped scientific question calls for it; omit it from L1-only validation and pilots. When used, least squares tests what equation a learned representation supports under more flexible coefficient-space optimization. It does not establish MinimalSymNet recovery or exact representability by its nonlinear parameterization. Report LS and SymNet results separately to distinguish representation from symbolic optimizer behavior.

## Feature scaling and recovery criteria

Always distinguish **frozen-head conditioning effects** from **joint surrogate-training effects**. Phase 18 scaling substantially improved frozen-surrogate symbolic conditioning/optimization in the tested setting. Scaling also changes the magnitude and geometry of PDE gradient pressure on a jointly trained surrogate. Frozen benefits therefore do not imply that the same scaling is optimal or necessary for joint training; avoid generic claims that “scaling helps PDE discovery.”

Record primitive scales, estimation data/procedure, whether scales are fixed or changing/detached, feature ordering, and the exact conversion from scaled to physical coefficients (including product, target, and coordinate scales when applicable).

Predeclare and record recovery criteria, thresholds, coefficient ordering, physical targets, spurious terms, and timing (endpoint, first crossing, or sustained recovery). For the established Burgers nine-term physical library, reuse the current loose/strong criteria from `RESEARCH_STATE.md` where scientifically appropriate, copying their exact definitions into the experiment config. Never silently change thresholds. PDE residual alone is not recovery: evaluate physical coefficients and spurious terms, and report field/derivative fidelity separately.

## Evidence and controlled interventions

Label claims by their evidence type:

1. Observation/correlation.
2. Temporal ordering.
3. Algebraic decomposition or substitution.
4. Frozen recombination / controlled intervention.
5. Replicated intervention across seeds or conditions.

Observational timing alone does not establish causality or a unique trigger, even if one quantity changes first. Do not call a rapid continuous transition a bifurcation without specific mathematical evidence.

For interventions, freeze exactly the claimed component, record checkpoint provenance, use paired seeds where possible, preserve identical head initialization when the paired design requires it, predefine recovery criteria, and include appropriate controls. Distinguish sufficiency from historical causation: POST targets with PRE spatial features recovering under a fresh frozen head establish sufficiency under that intervention, not that target improvement caused the original transition.

## Phase 19B interpretation guardrail

Original Phase 19B was not a terminal failure: it recovers Burgers at a sufficiently long optimization horizon. The canonical long-horizon trajectory first met loose recovery at step **8853** and strong recovery at **10013**. Conclusions from the original 1000-step observation are finite-horizon observations.

Dense instrumentation supports a coupled recovery transition. Target-direction alignment changes early, but no unique causal initiator is established. Library conditioning shows no favorable transition explaining recovery. Endpoint diagnostics support improved target fidelity and spatial-error cancellation while ambiguity remains. The causal mechanism is unresolved without controlled intervention. Consult `RESEARCH_STATE.md` for detailed evidence and subsequent updates.

## Run-script architecture

`runs/run_*.py` should primarily orchestrate experiments. Substantial reusable derivative calculations, numerical physics, metric definitions, plotting, serialization, and training machinery belong in `utils/`, `prog/`, or another appropriate module. Small experiment-specific operations may remain near the runner when extraction would obscure the design. Scientific correctness, provenance, and inspectability take priority over artificial layering.

Use active dataset config, feature library, extraction method, and declared axes rather than Burgers-specific assumptions in general machinery. Keep dataset-specific physics in a suitable config block such as `dataset_args`.

Follow the root canonical primitive evaluation path: `prog/featlib.py` defines names/order; `utils/derivative_utils.py` provides `evaluate_model_primitive_features`, `build_reference_primitive_features` (or documented dataset-specific reference operators), `evaluate_primitive_feature_metrics`, and `primitive_feature_name_to_key`. Do not derive primitive metrics from an LS candidate library or duplicate feature construction in entrypoints. Primitive overlays use `utils/feature_plotting.py:save_primitive_feature_overlays`.

Current routing examples (inspect their protocols, not just defaults):

- Long-horizon reproduction: `run_phase19b_long_horizon.py`, `PHASE19B_LONG_HORIZON.md`.
- Artifact-backed analysis: `run_phase19b_transition_diagnostic.py`, `PHASE19B_TRANSITION.md`.
- Passive replay and offline analysis: `run_phase19b_transition_instrumented.py`, `PHASE19B_TRANSITION_INSTRUMENTED.md`.
- Smooth control and audits: `run_phase24_smooth_burgers_control.py`, `run_phase24_surrogate_fit_audit.py`, `run_phase24_scale_gradient_diagnostic.py`, and their `PHASE24*.md` protocols.

## Artifact contract and figures

Substantial experiments should generally provide `config.json`, metrics/history in CSV/JSON/NPZ as appropriate, `validation.json`, `report.md`, figures, scientifically necessary checkpoints, and a reproduction command. Exact filenames may differ; document equivalents, validation gates, and output contracts in the runner/protocol. Update these when outputs or behavior change.

A reader must be able to determine what ran, from what initial state, with which configuration/data, for how long, what was measured, whether validation passed, and how to reproduce it. Save provenance/hashes, environment details, and initial/optimizer/RNG states as required by the reproduction claim. Historical layouts are evidence, not templates requiring retroactive repair.

Phase 19B instrumentation illustrates saved `states/`, `updates/`, histories, provenance, validation, and offline tables; Phase 24 illustrates checkpoint/probe tables, cached fields, and specialized audits. Do not force `summary.csv`, `summary_agg.csv`, heatmaps, overlays, or seed folders onto every experiment.

Figures are scientific artifacts. Require a descriptive main title (subplot titles are insufficient), labeled axes/quantities, relevant checkpoints/conditions, consistent scales where comparisons require them, and nonmisleading normalization. Save underlying numerical data where practical; important figures should have PNG and vector PDF versions where practical. Normally separate plotting from optimization so figures can be regenerated from saved data without retraining, while preserving existing artifacts.

## Sweep-specific conventions

These conventions apply to genuine sweeps only; the historical `siren_hparam_sweep` layout is an example, not a universal contract.

- Use dataset/active-axis/seed organization where useful. Per-run `config.json` and `summary.json`, dataset-level `summary.csv` (one row per attempted run, including run state), and `summary_agg.csv` (one row per active-coordinate group) support comparison. Record identity, seed/model, active coordinates, training config, and scalar outcomes.
- Aggregate over seeds with `num_seeds` and meaningful `_mean`/`_std` metrics, including final/minimum training loss and available feature metrics. Derive groups from declared axes and metric columns dynamically; never hardcode a derivative list or aggregate list/string coefficients without a defined representation.
- Use 2D heatmaps for two active axes and mean ± std line plots for one. Prefer `<x>_vs_<y>_<metric>_heatmap.pdf` and `<axis>_<metric>_lineplot.pdf`; choose from the active configuration.
- Feature fidelity uses deterministic `<feature_key>_rel_l2`, `_rmse`, `_max_abs` keys across summaries, aggregates, and plots. For SIREN feature-comparison sweeps, produce active-primitive overlays at `feature_overlays/<feature_key>_overlay.pdf`, normally at five evenly spaced available times (fewer for smaller grids), with reference/model curves and actual times. Use the canonical helpers above.
- Record TV type/weight and separate data/TV losses when used. Reuse `utils/tv_utils.py` and appropriate data-preparation helpers; do not impose normalized coordinates on a physical-coordinate reproduction.
- Emit extraction fields only for methods actually run. Distinguish primitive `feature_names` from LS candidate terms. Keep detailed equations/vectors in `pde_outputs/<method>/pde.json` with readable equations and diagnostics; tabulate scalar LS residual/rank/condition/coefficient errors or EQL readout metadata where useful. Preserve useful existing sweep contracts without requiring their structure for diagnostics.

## Validation and notebook promotion

Run appropriate experiment-specific validation before interpreting results; check configuration, data correspondence, reproduction overlap, metric definitions, and preservation of source artifacts as applicable. Verify that any referenced validation runner actually exists in this checkout before invoking it.

Promotion order: run experiment → validate artifacts → interpret evidence → update `RESEARCH_STATE.md` if warranted → decide whether the result materially advances the narrative. Do not automatically add every run to the story notebook. Follow `notebook/diagnostics/AGENTS.md` for code visibility, figure titles, and storytelling; detailed phase histories belong in `RESEARCH_STATE.md`.
