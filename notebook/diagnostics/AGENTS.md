# Diagnostic work instructions

`AGENTS.md` governs how diagnostic work and notebook storytelling are performed. `RESEARCH_STATE.md` is the canonical scientific state file; it records findings, hypotheses, history, and next questions.

- Read `notebook/diagnostics/RESEARCH_STATE.md` before diagnostic work.
- Read the latest relevant phase of `notebook/diagnostics/burgers_minimal_discovery_story.ipynb` before adding a new one; reconcile the plan with the current state before writing or running code.
- Preserve previous experiment artifacts.
- Keep execution machinery outside the notebook when practical, in reusable `runs/`, `prog/`, or `utils/` implementations.
- Keep scientifically important logic visible in the notebook: initial conditions, derivative methods, feature scaling, recovery criteria, coefficient extraction, plots, and interpretation.

## Notebook code visibility

Show code that explains scientific mechanisms and experimental invariants. Notebook code is appropriate when seeing the implementation helps the reader understand an important scientific operation, transformation, assumption, or diagnostic. An interesting function is one whose implementation materially explains how the experiment works or why its evidence should be interpreted a certain way.

### Scientific mechanism code — show

Show the smallest faithful excerpt when its implementation exposes an important operation or invariant, such as:

- computing and detaching feature scales;
- applying fixed scales to `[u, u_x, u_xx]`;
- discarding scale-estimator weights;
- cloning identical SIREN and SymNet states across treatments;
- supplying those states and scales to training;
- derivative construction;
- physical coefficient conversion;
- gradient diagnostics; and
- recovery criteria.

The goal is to expose the scientific dataflow, not the machinery used to execute every treatment. Identify the canonical repository location and function or class whenever one exists.

### Evidence-presentation code — show sparingly

Short code for loading artifacts, selecting meaningful result columns, or constructing the figure or table under discussion is acceptable. Keep it minimal.

### Experiment execution code — normally hide

Do not display code whose primary purpose is epoch or training loops, parameter-sweep loops, optimizer stepping, repeated model construction, batch scheduling, checkpoint orchestration, filesystem setup, serialization, repeated metric collection, or device and boilerplate handling. Keep that code in `runs/`, `utils/`, or other canonical implementation files.

Before showing code, ask: **What scientific fact becomes easier to understand by seeing this implementation?** If the answer is merely “this runs, trains, sweeps, or saves the experiment,” keep it outside the notebook. If the answer is “this shows how scale transfer works,” “this proves weights are discarded,” or “this establishes matched initialization,” show the smallest faithful excerpt.

Two notebook styles are allowed:

1. **Small self-contained diagnostic:** a complete implementation may remain when the code itself is concise and explanatory.
2. **Artifact-backed diagnostic:** substantial experiments run externally; the notebook shows the question, design, relevant mechanism code, implementation location, evidence, and interpretation.

Avoid a hybrid that copies a large external-style runner into a notebook cell.

Examples include concise code for constructing and applying feature scales, obtaining `u_x`, `u_xx`, or `u_t`, forming scientifically meaningful candidate features, extracting or converting physical coefficients, and implementing a diagnostic transformation or comparison that is itself part of the reasoning. Expose the smallest useful piece and prefer calling the canonical repository implementation over duplicating substantial logic.

Do not show code merely because a value is important. Learning rates, epoch counts, batch sizes, loss weights, seeds, checkpoint epochs, and hidden dimensions usually belong in Markdown, compact tables, or concise configuration output. Keep training loops, filesystem handling, serialization, checkpoint I/O, repeated multi-seed execution, generic plotting implementation, and boilerplate tensor/device handling in `runs/`, `prog/`, or `utils/` unless a specific diagnostic requires a small excerpt.

Before exposing implementation code, ask: **Does seeing this code help the reader understand the scientific mechanism, transformation, or evidence being discussed?** If yes, expose the minimal relevant code. If the reader only needs the value, setting, or outcome, explain it in prose, a table, or a figure and keep the implementation outside the notebook.

The notebook should read as a developing scientific argument:

`question → method → scientifically relevant implementation → evidence → interpretation → next question`.

## Diagnostic figures

Every diagnostic figure must have a descriptive main figure title. Subplot titles do not satisfy this requirement. The main title should communicate the scientific quantity or comparison; subplot titles may identify times, checkpoints, features, seeds, or conditions. Keep titles concise and scientific rather than using filenames or internal implementation names. When reusable plotting helpers exist, enforce or support this convention there instead of duplicating plotting code in notebooks.

- Update `RESEARCH_STATE.md` whenever a diagnostic changes the working interpretation. Keep current state concise and append details to its historical record.
- Do not refer to `research_state.md` as the current research-state location unless documenting its historical migration. Existing per-run reports with that filename are historical artifacts, not the canonical state.
