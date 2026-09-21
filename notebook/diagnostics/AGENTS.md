# Diagnostic work instructions

- Read `notebook/diagnostics/agents.md`, the canonical research-state file, before diagnostic work.
- Read the latest relevant phase of `notebook/diagnostics/burgers_minimal_discovery_story.ipynb` before adding a new one; reconcile the plan with the current state before writing or running code.
- Preserve previous experiment artifacts.
- Keep execution machinery outside the notebook when practical, in reusable `runs/`, `prog/`, or `utils/` implementations.
- Keep scientifically important logic visible in the notebook: initial conditions, configuration, derivative methods, scaling, recovery criteria, coefficient extraction, plots, and interpretation.
- Update `notebook/diagnostics/agents.md` whenever a diagnostic changes the working interpretation. Keep current state concise and append details to its historical record.
- Do not refer to `research_state.md` as the current research-state location unless documenting its historical migration. Existing per-run reports with that filename are historical artifacts, not the canonical state.
