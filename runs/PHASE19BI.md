# Phase 19B-i execution

The canonical execution implementation for the transferred-scale PDE loss-weight sweep is `utils/phase19bi_loss_weight_sweep.py`, invoked by `runs/run_phase19bi_loss_weight_sweep.py`. It reuses the archived matched Phase 19B state and batches, clones the same SIREN/SymNet initialization across all six lambda values, and writes the existing artifact contract under `run_results/phase19b_i_pde_loss_weight_sweep/`. Completed artifacts are reused.
