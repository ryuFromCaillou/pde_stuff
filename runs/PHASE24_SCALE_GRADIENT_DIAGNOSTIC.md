# Phase 24 scale-gradient diagnostic

This diagnostic holds the archived Phase 24 smooth dataset, SIREN state, MinimalSymNet state, first matched batch, architecture, optimizer, and loss weights fixed while changing only the primitive feature scales supplied to the SymNet. It reports initial gradient pressure, one-at-a-time scale substitutions, and short matched joint trajectories at updates 0, 10, 50, 100, and 200.

The runner is `runs/run_phase24_scale_gradient_diagnostic.py`; reusable implementation is `utils/phase24_scale_gradient_diagnostic.py`. Results are written to `run_results/phase24_scale_gradient_diagnostic/`. A completed directory is reused and is never overwritten automatically.
