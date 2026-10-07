# Paul-path parameter L1 validation and pilot

Scope: verify the implemented parameter penalty, then compare seed 0 at lambda
0, 1e-7, 1e-6, 1e-5, 1e-4 for 100000 Adam updates each. No noise, LS, or
multi-seed experiment is part of this plan. Neither source notebook is edited.
Earlier Phase 19B late recovery motivates the horizon but is not a matched control.

The objective is `data_loss + pde_loss + lambda_l1 * symnet_l1`.
`symnet_l1` sums absolute values of all ten internal MinimalSymNet parameters;
this is not L1 on the nine expanded physical coefficients. Product factors make
parameter sparsity and physical coefficient sparsity inequivalent.

Preserve executable Paul settings: generator nu=.02, T=.1, N=256, dt=.002,
10 time and 64 spatial linspace indices, physical coordinates, float32,
SirenMLP 3x64 with frequencies 1/1, fresh joint SIREN and MinimalSymNet,
raw FeatureTensor (u,u_x,u_xx), weights 1/1, Adam lr=.0005, batch 640 sampled
with replacement. Reproduce the notebook RNG consumption preceding the fresh
joint SIREN: construct the discarded data model, draw 2500 batches, construct
the diagnostic SymNet. Data-only optimization consumes no RNG and its weights
are discarded, so it need not run. Reset seed immediately before joint SymNet,
as in the notebook. CPU execution is recorded; no claim of reproducing an
unknown historical CUDA session is made.

Gate pilot execution on exact zero-lambda objective/gradient equality, positive
loss identity, matched SymNet gradient differences equal to lambda*sign(theta),
absent direct SIREN penalty gradients, and matched Adam updates checked against
its explicit first-step formula. Test the pilot lambdas and a larger diagnostic
lambda: first-step Adam nearly cancels gradient magnitude, so a tiny lambda can
produce identical float32 parameters even with a verified gradient contribution.
Use a float64 matched-state check to resolve this rounding effect. Also compare one float32 step from identical Adam moments after ten common unregularized warmup steps, checking the bias-corrected update formula.

One history row is the state after `step` updates and before the next update,
including a newly sampled batch at the terminal state. Record every step: all
nine coefficients, transport/diffusion, spurious L2, recovery flags, five separate
loss components. Full 51x256 generator-grid field MSE and relative L2 are evaluated
every 1000 steps and at endpoints (blank otherwise). First crossings are checked
every step, not inferred from sparse field evaluations. Loose thresholds are
transport error <.25, diffusion error <.02, spurious L2 <.25; strong .10/.01/.10.
Crossings are not sustained recovery. Targets are -1 and .02.

Run: `python3 runs/run_paul_path_l1.py --output run_results/paul_path_l1_pilot_validated --workers 3`.
Use `--steps 10` only for implementation smoke testing; `--validate-only` runs
only mechanical checks. Existing destinations are rejected. Outputs: config,
source hashes, matched_inputs.pt, validation.json, per-lambda history.csv,
field_metrics.csv, initial/final states (including optimizer/RNG), summary.json,
aggregate summary.csv, report.md, and coefficient/loss trajectory PNG/PDF figures.

`--workers 3` executes three independent lambda conditions concurrently, each with
one CPU thread and the same saved initial state/RNG. This does not add seeds or
conditions. Progress JSON is refreshed every 1000 steps; separate state checkpoints
are saved every 10000. Checkpoint state k is pre-update with its already sampled
next batch indices saved. A 10-step smoke test is not scientific recovery evidence.

Mechanical tolerances: loss identity and zero-lambda gradients are bitwise checks;
float32 gradient increment / first-step formula use atol=rtol=2e-6, float64
uses 1e-12. Actual gradient errors are saved, so tiny increments are not inferred
from tolerance alone. The populated-moment float32 formula uses atol=1e-7,
rtol=1e-6 and requires a strictly nonzero matched SymNet update difference.
