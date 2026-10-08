# Phase 19B frozen target–feature recombination

Use exact pre-update captured states 8000 and 8750 from the validated instrumented replay. A=(PRE features, PRE target), B=(PRE, POST), C=(POST, PRE), D=(POST, POST). Never replay joint training. Step 8750 precedes the first archived loose LS recovery (8825) and joint-head recovery (8853), so D is a later-state control, not an established positive control.

All 64,512 archived physical-coordinate points are used, in time-major/spatial-minor order. Canonical derivative evaluation produces detached u, u_x, u_xx, u_t. Verify source manifest hashes, PRE bitwise cached fields, both states' archived fidelity and full/two-support LS coefficients. Preserve all source directories and the notebook with before/after SHA256 inventories.

Nine physical terms: u, u_x, u_xx, u², u*u_x, u*u_xx, u_x², u_x*u_xx, u_xx². Retain original fixed Phase19B primitive scales (0.8564476370811462, 2.418001890182495, 49.38141632080078) in every condition; do not recompute checkpoint RMS scales. Feed scaled primitives into the unchanged MinimalSymNet; target remains physical. Expanded coefficients divide by the respective primitive/product scales to recover physical coefficients.

Predeclared protocol: 25 paired seeds 0–24, fresh default MinimalSymNet initialization cloned across A/B/C/D, Adam lr=0.01, betas=(0.9,0.999), eps=1e-8, no weight decay or regularization, 1000 full-grid PDE-MSE updates, CPU float32 with two threads. This borrows the established frozen-probe optimization budget but keeps original Phase19B scales fixed. Optimize only the head. Save all coefficients and residuals at steps 0–1000, first loose/strong crossings, and terminal outcomes. No condition-specific tuning or early stopping.

Loose: abs(transport+1)<0.25, abs(diffusion−0.02)<0.02, spurious L2<0.25. Strong: 0.10/0.01/0.10. All strict. Primary outcome is terminal recovery; ever-recovered counts are secondary. A descriptive reliable-recovery label requires >=20/25 terminal loose successes, accompanied by Wilson 95% intervals. This operational label is not a significance test. Paired discordance counts and unadjusted exact McNemar tests are supplementary. Independent head seeds do not replicate surrogate trajectories.

Unconstrained LS precedes head fitting: float64 unit-column SVD, rcond=1e-12, no intercept/regularization, physical coefficients, full and true-term support. LS compatibility and nonlinear head optimization are separate questions.

```bash
cd /home/ghost/ghost/pde_stuff
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -u runs/run_phase19b_recombination.py --out run_results/phase19b_recombination_intervention_repeat
```

Output must not exist. Artifacts: config, provenance, validation, frozen_metadata, pairing_validation, evaluation_grid, frozen_8000/8750 NPZs, historical_controls, ls_results, per_seed_results, recovery_summary, paired_comparisons, all trajectories and initial/final heads, three PNG/PDF figures with CSV source data, standalone report, artifact manifest. Plot/report regeneration uses `utils.phase19b_recombination_report.render(out, destination)` with a new destination. No notebook edits, second-stage checkpoint experiments, commit, or push. Interpret intervention sufficiency only, never historical causation.

Post-run independent audit: `utils/phase19b_recombination_audit.py` recomputes all 100100 trajectory-row recovery flags and first crossings, coefficient norms, saved initial/final parameter coefficients, pairing, terminal physical residuals, and source preservation. Outputs include `independent_audit.json`, `terminal_diagnostics.csv`, and `first_crossing_summary.csv`.
