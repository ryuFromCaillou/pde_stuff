# Physics-informed Burgers artifact index

- `clean_baseline_completed/`: consolidated scientific run; includes hash-identical copies of completed seeds 0 and 1 and the continuation of seeds 2–5. Check its validation/seed summary for completion status.
- `clean_baseline/`: preserved original execution. Seeds 0 and 1 completed; aggregate reporting stopped on an undefined one-repeat sample standard deviation. Do not use its partial aggregate report as the final reliability result.
- `reporting_fix_provenance/`: original implementation snapshots, original execution log, AST-equivalence validation for all training/evaluation functions, and reporting-regression evidence. Only summary serialization and orchestration changed.
- `mechanical_smoke/`: ten-step implementation check, not scientific recovery evidence.
- `implementation_tests/`: initial test logs, including the unavailable historical archive fixture.
- `seed0_audit/`, `seed01_audit/`, `continuation_audit/`: read-only audits at intermediate completion stages. Individual completed-seed checks pass; overall completion is false because not all six seeds had finished.
- `smoke_audit/`: first audit attempt exposed CSV-parser round-trip precision loss. `smoke_audit_roundtrip/` verifies the corrected parser on the smoke data. Neither is scientific evidence.

Runner/protocol: `runs/run_physics_informed_burgers.py` and `runs/PHYSICS_INFORMED_BURGERS.md`.
No noise or sampling sweep is included. Numerical references and both notebooks are preserved.
