# Phase 24 surrogate-fit audit

Run from the repository root:

```sh
.venv/bin/python -m utils.phase24_fit_audit
```

The audit is read-only with respect to Phase 23 and Phase 24 artifacts. It verifies the smooth rollout and flattened correspondence, reproduces all saved Phase 24 field errors, runs the matched data-only SIREN control, replays the joint objective to measure SIREN gradient pressure, and writes outputs under `run_results/phase24_surrogate_fit_audit/`. An existing completed audit is reused.

The data-only control uses the exact archived Phase 24 initial SIREN state and Phase 23/24 batch indices, Adam at `5e-4`, the same architecture, coordinates, batch size, checkpoint convention, and 1,000-update horizon. It removes only the SymNet/PDE contribution. The joint replay is diagnostic and reproduces the Phase 24 objective; it does not modify Phase 24 checkpoints or results.
