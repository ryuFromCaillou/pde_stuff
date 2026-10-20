# Original experiment audit

Authoritative remote sami and working base both resolve to bad712797f9b0fb24dd840a356f96ef592d4e97a on Ark2.
All recorded original source/notebook hashes match the completed noise_robustness_resume2 configuration.
All 12 original conditions have 100001 finite rows and the original 20 checkpoint coordinates.
Read-only validation verified loss totals, coefficient error definitions, recovery flags and first crossings,
endpoint summaries, observation indices/targets/noise, initial model states, and every saved checkpoint coefficient.
Reused directories are copied byte-for-byte with SHA-256 records in reused_artifacts.json.

The other noise_robustness* directories include prior partial/retry work; they are neither overwritten nor mixed into this study.
The earlier six-seed clean summary survives, but seeds 2–5 raw histories are absent from both local PDE checkouts and the tracked clean artifact tree.
Those four conditions are rerun without reconstructing histories from endpoints.

Every time-dependent original noise figure and its source metrics is inventoried in code/runs/BURGERS_SIX_SEED.md.
The original loss_trajectories figure actually displays full-observation MSE and full-grid residual at checkpoints.
The new training_losses figure additionally shows the recorded dense minibatch total, data and PDE losses.
Optimizer steps, not dataset epochs, are the original x-axis.

Noise exception: epsilon varies by initialization seed in the original design, with seed 700000+init_seed.
It is fixed in time and paired across noise levels. That design is retained; clean observations are identical across all seeds.
This is an extension of the original mixed optimization/noise-seed experiment, not an isolation of initialization effects.
