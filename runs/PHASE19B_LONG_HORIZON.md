# Phase 19B long-horizon control

This control extends the original Phase 19B transferred-scale joint SIREN–MinimalSymNet trajectory to 100,000 optimizer steps. It preserves the archived Phase 19B initial states, scales, observations, batch seed and sequence, Adam optimizer, learning rate, loss weights, architecture, and coefficient conversion. The first 1,000 batches are verified against `run_results/phase22_sgd_control/matched_inputs.pt`; subsequent batches continue the same seed-19220 random stream.

Run `runs/run_phase19b_long_horizon.py`. Reusable implementation: `utils/phase19b_long_horizon.py`. Outputs are written to `run_results/phase19b_long_horizon_control/`; completed outputs are reused. Existing Phase 19B and later-phase artifacts are never overwritten.
