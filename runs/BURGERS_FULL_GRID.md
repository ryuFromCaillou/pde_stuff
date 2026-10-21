# Full-grid six-seed observation density study

Separate experiment at `run_results/physics_informed_burgers/sampling_full_grid_six_seed/`.
Authoritative source is current `sami`. Existing studies and narrative notebooks are immutable.

Population: complete clean numerical 51×256 solver grid, time-major coordinates and targets.
Ratios 1,.75,.5,.25,.1; counts 13056,9792,6528,3264,1306. One NumPy default_rng(910064)
permutation, nested prefixes shared across initialization seeds 0–5. Coverage records include
per-time and per-space counts and physical coordinate ranges. Historical structured 640 points
(10×64; 4.9019608%) are a separately labeled reference, not a full-grid random subset.

Training retains original physical SIREN (3 hidden layers, width 64, omega 1/1), zero head,
100000 Adam steps at .0005, unit data/PDE losses, no noise/L1/SymNet/LS, CPU float32,
one Torch thread per worker. Fixed 640-point with-replacement minibatches at every level;
batch RNG 10000+seed and initialization seed unchanged. Objective and evaluation helpers are
reused unchanged. Both observations and PDE collocation coverage vary; this is not an isolated
data-density effect. Full solver/evaluation grid unchanged. Numerical derivative references;
u_t is generator RHS. Known-support parameter estimation, not symbolic support discovery.

Run with `.venv/bin/python runs/run_burgers_full_grid.py --mode MODE` in this checkout.
Modes in order: `prepare`, `smoke`, `gate`, `remaining`, `publish` (the last also rerenders).
Prepare refuses existing output; captures source snapshots, hashes, protected artifacts, mapping
and mechanical validation. Smoke runs 100 steps at 100% and 10%. Gate runs 100% seed 0 to
100000 and audits its trajectory. After inspecting its metrics, write `gate_review.json` with
`approved_for_remaining: true` and the inspection findings. This is a scientific implementation
gate, not a recovery threshold. Remaining requires this review and runs the other 29 conditions.
No tuning based on seed outcomes. Partial run directories are preserved and rejected; complete
runs can be reused only after validation. No silent resume or overwritten histories.

Dense history includes states 0–100000, original checkpoint schedule includes 20 states.
Saved state is after step updates, with the newly drawn batch and its post-draw RNG state;
loss is pre-next-update. All field caches, independent clean diagnostics, optimizer states,
initial models and selected observations are retained. Fixed-grid diagnostics are never interpolated.

Aggregation reuses the noise study's strict six-frame mean/SD implementation. Population SD
is ddof=0; errors are averaged per seed separately from signed coefficients. All saved means/SDs
are independently checked by arithmetic on raw columns. Recovery is computed per seed with
strict absolute error limits .25/.02 (loose), .10/.01 (strong). First crossings need not persist;
noncrossers remain null, with censor counts. First-time summaries describe successful crossers only.

Deliverables: six plot families in PNG/PDF (coefficients, absolute/relative errors, three training
losses, field/all derivatives, endpoint accuracy, per-seed recovery), independent residual curves,
all numeric mean/SD CSVs, per-seed and aggregate endpoints including clean fidelity, coverage,
source/configuration/environment, audit, SHA256 manifest and scientific report.
Audit checks every dense row, endpoint/crossings, original artifact hashes, initial models,
checkpoint coefficients, optimizer step counts, complete batch RNG sequences and saved batches,
recomputed checkpoint losses, all cached field errors/residuals, and freshly evaluated initial/final
full-grid diagnostics. Mechanical validation alone does not establish recovery.
