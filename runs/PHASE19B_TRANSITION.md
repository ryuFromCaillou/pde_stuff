# Phase 19B post-hoc transition diagnostic

Compare saved 5k/10k joint states and the reference Burgers library, with 20k as supplementary context. No model training or sweep occurs. Main notebook and original long-horizon artifacts are preserved.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python runs/run_phase19b_transition_diagnostic.py --out run_results/phase19b_transition_diagnostic_repeat
```

The output directory must not exist. Data come from archived Phase 22 matched inputs; models come from `run_results/phase19b_long_horizon_control/checkpoint_*.pt`. Autodiff primitives and fidelity use canonical derivative helpers; reference derivatives use the original centered periodic Burgers operator. Computation, validation, plotting and report rendering are separated into `utils/phase19b_transition*.py` and reusable SVD operations in `utils/transition_geometry.py`.

Output contract:

- `config.json`: grid, sources, normalization, precision and SVD tolerance.
- `fields.npz`: common x/t grid; fields named `<reference|5000|10000|20000>__<u|u_x|u_xx|u_t>`.
- `metrics.json`: canonical ordered feature names and true/spurious indices; raw/normalized geometry; field/term/RHS fidelity; full/two-support fits; physical coefficients; error projections, cancellation and coefficient-bias decomposition; trained heads.
- `fidelity.csv`, `alignments.csv`, `projections.csv`: tidy numeric tables.
- `validation.json`: source hashes, archived metric reproduction, feature-order, bias-identity, raw/normalized LS, polynomial expansion and nested-projection checks.
- Eight PNG/PDF pairs: required figures 1–6, supplemental raw Gram, and optional physical snapshots.
- `report.md`: self-contained methods, numerical tables, figure captions, hypothesis assessment and caveats.

Email transmission is not part of the reproducible diagnostic runner. `email_summary.txt` in the reviewed output records the separately sent scientific summary.

The numerical reference is not an analytic continuum solution. Gram cosines are uncentered; Pearson matrices are distinct. Column normalization is unit L2, not the training scales. All LS coefficients are converted to physical units. Least squares relaxes the nonlinear one-product SymNet coefficient constraint; support fit/projection results are observational.
