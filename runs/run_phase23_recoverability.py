#!/usr/bin/env python3
"""Phase 23: frozen-SymNet recoverability along a matched joint trajectory.

Reuses the exact Phase 19B/Phase 21 transferred-scale joint controls and the
Phase 18 frozen/scaled MinimalSymNet protocol. The historical run is validated
before any frozen probes are launched.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "run_results/phase23_joint_trajectory_recoverability"
NOTEBOOK = ROOT / "notebook/diagnostics/burgers_minimal_discovery_story.ipynb"
P22_INPUTS = ROOT / "run_results/phase22_sgd_control/matched_inputs.pt"
P21_TABLE = ROOT / "run_results/phase21_surrogate_update_geometry/checkpoint_table.csv"
P19B_SNAP = ROOT / "run_results/phase19b_scale_transfer/snapshot_metrics.csv"
PHASE18_RUNS = ROOT / "run_results/phase18_feature_scaling_reliability/per_run.csv"
CHECKPOINTS = [0, 50, 200, 500, 1000]
SEEDS = list(range(25))
FEATURE_NAMES = ["u", "u_x", "u_xx", "u^2", "u*u_x", "u*u_xx", "u_x^2", "u_x*u_xx", "u_xx^2"]
PRIMITIVES = ["u", "u_x", "u_xx"]
TRUTH = np.array([0.0, 0.0, 0.02, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
SPURIOUS_IDX = [i for i in range(9) if i not in (2, 4)]
LOOSE = {"transport_abs": 0.25, "diffusion_abs": 0.02, "spurious_l2": 0.25}
STRONG = {"transport_abs": 0.10, "diffusion_abs": 0.01, "spurious_l2": 0.10}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def setup_notebook_context():
    """Reuse the notebook's canonical data/model/derivative definitions."""
    import ast
    import json as _json

    nb = _json.loads(NOTEBOOK.read_text())
    cells = nb["cells"]
    scope = {"__name__": "phase23_reused_notebook_definitions"}
    # Imports, deterministic Burgers dataset, derivative evaluators, and model
    # definitions are sourced from the existing notebook rather than forked.
    exec("".join(cells[1]["source"]), scope)
    exec("".join(cells[2]["source"]), scope)
    exec("".join(cells[4]["source"]), scope)
    cell7 = "".join(cells[7]["source"])
    exec(cell7.split("siren = SirenMLP")[0], scope)
    for node in ast.parse("".join(cells[15]["source"])).body:
        if isinstance(node, ast.ClassDef):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<reused MinimalSymNet>", "exec"), scope)
    from prog.mlps import SirenMLP
    from utils.derivative_utils import build_burgers_reference_derivative_grids
    scope["SirenMLP"] = SirenMLP
    scope["build_burgers_reference_derivative_grids"] = build_burgers_reference_derivative_grids
    scope["device"] = torch.device("cpu")
    torch.set_num_threads(2)
    return scope


def flat_grads(grads, params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for g, p in zip(grads, params)])


def coeff_vector(model, primitive_scales, product_term_fn):
    # Reuse the notebook's established coefficient extraction and ordering.
    beta = np.array(list(product_term_fn(model).values()), dtype=np.float64)
    product_scales = np.array([
        primitive_scales[0], primitive_scales[1], primitive_scales[2],
        primitive_scales[0] ** 2, primitive_scales[0] * primitive_scales[1],
        primitive_scales[0] * primitive_scales[2], primitive_scales[1] ** 2,
        primitive_scales[1] * primitive_scales[2], primitive_scales[2] ** 2,
    ], dtype=np.float64)
    return beta / product_scales


def success(xi, thresholds):
    spurious = float(np.linalg.norm(np.asarray(xi)[SPURIOUS_IDX]))
    return (abs(float(xi[4]) + 1.0) < thresholds["transport_abs"] and
            abs(float(xi[2]) - 0.02) < thresholds["diffusion_abs"] and
            spurious < thresholds["spurious_l2"])


def run():
    resume = OUT.exists()
    if resume:
        # Resume only this script's own Part-A artifact set after a numerical
        # validation gate; never overwrite a completed Phase 23 experiment.
        if (OUT / "status.json").exists() and json.loads((OUT / "status.json").read_text()).get("status") == "complete":
            raise RuntimeError(f"Refusing to overwrite completed Phase 23 output directory: {OUT}")
        if not (OUT / "joint_checkpoint_metrics.csv").is_file():
            raise RuntimeError(f"Incomplete Phase 23 directory is missing joint metrics: {OUT}")
    for required in [P22_INPUTS, P21_TABLE, P19B_SNAP, PHASE18_RUNS, NOTEBOOK]:
        if not required.is_file():
            raise FileNotFoundError(required)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    (OUT / "probe_histories").mkdir(exist_ok=True)
    scope = setup_notebook_context()
    device = torch.device("cpu")
    torch.manual_seed(19219)
    np.random.seed(0)

    # Phase 22 archived the common reconstructed Phase 21 initial states,
    # exact transferred scales, sampled batches, and training tensors.
    matched = torch.load(P22_INPUTS, map_location=device, weights_only=False)
    t_train, x_train, u_train = (matched[k].to(device) for k in ["t", "x", "u"])
    batches = [b.to(device) for b in matched["batches"]]
    scales = matched["scales"].to(device).detach()
    scales_np = scales.cpu().numpy().astype(np.float64)
    assert len(batches) == 1000 and tuple(t_train.shape) == (64512, 1)
    assert len(t_train) == len(x_train) == len(u_train)
    assert np.allclose(scales_np, [0.8564476370811462, 2.418001890182495, 49.38141632080078], rtol=1e-7, atol=1e-9)

    # Confirm archived matched tensors are precisely the deterministic notebook dataset.
    assert torch.equal(t_train, scope["t_train"].cpu())
    assert torch.equal(x_train, scope["x_train"].cpu())
    assert torch.equal(u_train, scope["u_train"].cpu())
    s0, y0 = matched["surrogate"], matched["symnet"]
    from prog.mlps import SirenMLP
    Surrogate = SirenMLP
    SymNet = scope["MinimalSymNet"]
    sm = Surrogate(hidden_size=64, hidden_layers=3, first_omega_0=20.0, hidden_omega_0=1.0).to(device)
    ym = SymNet().to(device)
    sm.load_state_dict(s0); ym.load_state_dict(y0)
    assert tensor_state_hash(sm.state_dict()) == tensor_state_hash(s0)
    assert tensor_state_hash(ym.state_dict()) == tensor_state_hash(y0)

    reference = build_reference = scope["build_burgers_reference_derivative_grids"](
        u_grid=scope["u_grid"], x_grid=scope["x_grid"], nu=0.02)
    historical = pd.read_csv(P21_TABLE).set_index("epoch")
    rows = []
    ckpt_hashes = {}
    if resume:
        trajectory = pd.read_csv(OUT / "joint_checkpoint_metrics.csv")
        for epoch in CHECKPOINTS:
            path = OUT / "checkpoints" / f"theta_epoch_{epoch:04d}.pt"
            if not path.is_file():
                raise RuntimeError(f"Cannot resume: missing saved theta checkpoint {path}")
            ckpt_hashes[str(epoch)] = sha256(path)
    else:
        optimizer = torch.optim.Adam(list(sm.parameters()) + list(ym.parameters()), lr=5e-4)
        sparams = list(sm.parameters())
        for epoch in range(1001):
            ix = batches[epoch - 1] if epoch else batches[0]
            tb = t_train[ix].detach().clone().requires_grad_(True)
            xb = x_train[ix].detach().clone().requires_grad_(True)
            up = sm(tb, xb)
            ut = torch.autograd.grad(up, tb, torch.ones_like(up), create_graph=True, retain_graph=True)[0]
            ux = torch.autograd.grad(up, xb, torch.ones_like(up), create_graph=True, retain_graph=True)[0]
            uxx = torch.autograd.grad(ux, xb, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
            rhs = ym(torch.cat([up, ux, uxx], 1) / scales.reshape(1, 3))
            ld = nn.functional.mse_loss(up, u_train[ix])
            lp = nn.functional.mse_loss(ut, rhs)
            total = ld + 0.5 * lp
            gd = flat_grads(torch.autograd.grad(ld, sparams, retain_graph=True, allow_unused=True), sparams)
            gp = flat_grads(torch.autograd.grad(lp, sparams, retain_graph=True, allow_unused=True), sparams)
            gt = gd + 0.5 * gp
            dot = float(torch.dot(gd, gp))
            xi = coeff_vector(ym, scales_np, scope["product_term_dict"])
            row = dict(epoch=epoch, data_loss=float(ld.detach()), pde_loss=float(lp.detach()), total_loss=float(total.detach()),
                       gradient_cosine=dot / (float(gd.norm()) * float(gp.norm()) + 1e-30),
                       g_data_norm=float(gd.norm()), weighted_g_pde_norm=0.5 * float(gp.norm()),
                       g_total_norm=float(gt.norm()), coeff_error=float(np.linalg.norm(xi - TRUTH)),
                       **{f"xi_{n}": float(v) for n, v in zip(FEATURE_NAMES, xi)})

            if epoch in CHECKPOINTS:
                rows.append(row)
                # theta_k is the exact pre-update parameter state whose metrics are labeled k.
                state = {k: v.detach().cpu().clone() for k, v in sm.state_dict().items()}
                path = OUT / "checkpoints" / f"theta_epoch_{epoch:04d}.pt"
                torch.save({"epoch": epoch, "surrogate": state, "phase19b_scales": scales.cpu(),
                            "source_config": "Phase 19B / Phase 21 transferred-scale joint Adam"}, path)
                ckpt_hashes[str(epoch)] = sha256(path)

            if epoch < 1000:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()

        trajectory = pd.DataFrame(rows)
        trajectory.to_csv(OUT / "joint_checkpoint_metrics.csv", index=False)
    compare_cols = ["data_loss", "pde_loss", "total_loss", "gradient_cosine", "g_data_norm",
                    "weighted_g_pde_norm", "g_total_norm", "coeff_error"] + [f"xi_{n}" for n in FEATURE_NAMES]
    validation_rows = []
    for _, row in trajectory.iterrows():
        epoch = int(row.epoch)
        ref = historical.loc[epoch]
        checks = {}
        for col in compare_cols:
            a, b = float(row[col]), float(ref[col])
            err = abs(a - b)
            # Loss/coefficient state should reproduce tightly. Gradient values
            # involve second-derivative backprop and vary slightly with CPU
            # reduction order, so use a still-small 2e-4 relative envelope.
            if col in {"gradient_cosine", "g_data_norm", "weighted_g_pde_norm", "g_total_norm"}:
                rtol, atol = 2e-4, 3e-5
            else:
                rtol, atol = 3e-5, 3e-6
            checks[col] = {"rerun": a, "historical": b, "abs_error": err,
                           "rtol": rtol, "atol": atol,
                           "pass": bool(np.isclose(a, b, rtol=rtol, atol=atol))}
        validation_rows.append({"epoch": epoch, "pass": all(v["pass"] for v in checks.values()), "metrics": checks})
    validation = {"selected_historical_run": "Phase 21 transferred-scale Adam joint run (Phase 19B controls)",
                  "tolerance": {"losses_and_coefficients": {"rtol": 3e-5, "atol": 3e-6},
                                "gradient_diagnostics": {"rtol": 2e-4, "atol": 3e-5,
                                                         "reason": "float32 second-derivative backward reductions"}}, "checkpoints": validation_rows,
                  "all_pass": all(v["pass"] for v in validation_rows), "checkpoint_sha256": ckpt_hashes}
    (OUT / "joint_reproduction_validation.json").write_text(json.dumps(validation, indent=2))
    if not validation["all_pass"]:
        (OUT / "status.json").write_text(json.dumps({"status": "stopped_validation_mismatch"}, indent=2))
        raise RuntimeError("Joint trajectory did not reproduce Phase 21 metrics; frozen probes were not started")

    # Evaluate frozen surrogate field/derivative errors and run Phase-18-style probes.
    evaluator = scope["evaluate_model_and_derivatives_in_chunks"]
    refs = {"u": scope["u_grid"], "u_x": reference["u_x"]["physical"],
            "u_xx": reference["u_xx"]["physical"], "u_t": reference["u_t"]["physical"]}
    checkpoint_rows, final_rows, history_rows = [], [], []
    phase18_src = "".join(json.loads(NOTEBOOK.read_text())["cells"][71]["source"])
    (OUT / "phase18_protocol_source.py.txt").write_text(phase18_src)
    phase18_scaled = pd.read_csv(PHASE18_RUNS).query("treatment == 'scaled'")
    phase18_control = {"scaled_runs": int(len(phase18_scaled)),
                       "loose_successes": int(phase18_scaled["loose_success"].sum()),
                       "strong_successes": int(phase18_scaled["strong_success"].sum()),
                       "seeds": list(range(25)), "optimizer": "Adam", "learning_rate": 1e-2,
                       "epochs": 1000, "loss": "MSE(SymNet(scaled primitives), frozen surrogate u_t)",
                       "primary_recovery_criterion": LOOSE, "secondary_strong_criterion": STRONG,
                       "convergence_epoch_defined_in_phase18": False,
                       "scaling": "full-grid RMS for [u,u_x,u_xx]; same primitive-induced product coefficient conversion"}
    (OUT / "phase18_protocol.json").write_text(json.dumps(phase18_control, indent=2))
    p19b_snap = pd.read_csv(P19B_SNAP).query("condition == 'transferred'").set_index("epoch")

    # Generate deterministic flattened inputs in the exact Phase-18 row ordering.
    t_np = scope["t_flat"].astype(np.float32)
    x_np = scope["x_flat"].astype(np.float32)
    grid_shape = scope["u_grid"].shape
    prepared = {}
    derivative_gate = []
    # Validate all saved theta states and their Phase 19B field/derivative
    # metrics before allowing any frozen SymNet optimizer to run.
    for epoch in CHECKPOINTS:
        saved = torch.load(OUT / "checkpoints" / f"theta_epoch_{epoch:04d}.pt", map_location=device, weights_only=False)
        frozen = Surrogate(hidden_size=64, hidden_layers=3, first_omega_0=20.0, hidden_omega_0=1.0).to(device)
        frozen.load_state_dict(saved["surrogate"]); frozen.eval()
        for p in frozen.parameters(): p.requires_grad_(False)
        before_hash = tensor_state_hash(frozen.state_dict())
        fields = evaluator(frozen, t_np, x_np, grid_shape, chunk_size=4096, device=device)
        reloaded = Surrogate(hidden_size=64, hidden_layers=3, first_omega_0=20.0, hidden_omega_0=1.0).to(device)
        reloaded.load_state_dict(saved["surrogate"])
        fields_reload = evaluator(reloaded, t_np, x_np, grid_shape, chunk_size=4096, device=device)
        reload_errors = {q: float(np.max(np.abs(fields[q] - fields_reload[q]))) for q in fields}
        assert max(reload_errors.values()) <= 1e-7, f"Reload discrepancy at epoch {epoch}: {reload_errors}"
        scales_k = np.maximum(np.sqrt(np.mean(np.column_stack([fields["u"].ravel(), fields["u_x"].ravel(), fields["u_xx"].ravel()]) ** 2, axis=0)), 1e-12)
        features_np = np.column_stack([fields["u"].ravel(), fields["u_x"].ravel(), fields["u_xx"].ravel()]).astype(np.float32)
        features = torch.tensor(features_np / scales_k.astype(np.float32), dtype=torch.float32, device=device)
        target = torch.tensor(fields["u_t"].reshape(-1, 1), dtype=torch.float32, device=device)
        errors = {q: float(np.mean((fields[q] - refs[q]) ** 2)) for q in ["u", "u_x", "u_xx", "u_t"]}
        checks = {}
        for quantity in ["u", "u_x", "u_xx", "u_t"]:
            actual, expected = errors[quantity], float(p19b_snap.loc[epoch, quantity + "_mse"])
            checks[quantity] = {"rerun": actual, "historical_phase19b": expected,
                                "abs_error": abs(actual - expected),
                                "pass": bool(np.isclose(actual, expected, rtol=2e-5, atol=3e-6))}
        derivative_gate.append({"epoch": epoch, "pass": all(v["pass"] for v in checks.values()), "metrics": checks})
        prepared[epoch] = (fields, scales_k, features, target, errors, reload_errors, before_hash)
    validation["phase19b_derivative_snapshot_comparison"] = {
        "source": str(P19B_SNAP.relative_to(ROOT)),
        "timing": "Phase 19B saved surrogate state before the labeled epoch update",
        "tolerance": {"rtol": 2e-5, "atol": 3e-6}, "checkpoints": derivative_gate,
        "all_pass": all(v["pass"] for v in derivative_gate)}
    validation["all_pass"] = validation["all_pass"] and validation["phase19b_derivative_snapshot_comparison"]["all_pass"]
    (OUT / "joint_reproduction_validation.json").write_text(json.dumps(validation, indent=2))
    if not validation["all_pass"]:
        (OUT / "status.json").write_text(json.dumps({"status": "stopped_derivative_validation_mismatch"}, indent=2))
        raise RuntimeError("Historical joint/derivative metrics did not reproduce; frozen probes were not started")

    for epoch in CHECKPOINTS:
        fields, scales_k, features, target, errors, reload_errors, before_hash = prepared[epoch]
        frozen = Surrogate(hidden_size=64, hidden_layers=3, first_omega_0=20.0, hidden_omega_0=1.0).to(device)
        saved = torch.load(OUT / "checkpoints" / f"theta_epoch_{epoch:04d}.pt", map_location=device, weights_only=False)
        frozen.load_state_dict(saved["surrogate"])
        frozen.eval()
        for p in frozen.parameters(): p.requires_grad_(False)
        checkpoint_rows.append({"checkpoint_epoch": epoch, **errors,
                                **{f"scale_{n}": float(v) for n, v in zip(PRIMITIVES, scales_k)},
                                "state_sha256": ckpt_hashes[str(epoch)],
                                "reload_max_abs_u": reload_errors["u"],
                                "reload_max_abs_u_x": reload_errors["u_x"],
                                "reload_max_abs_u_xx": reload_errors["u_xx"],
                                "reload_max_abs_u_t": reload_errors["u_t"]})

        # Phase 18 has no epoch-of-convergence definition: final recovery is the
        # declared endpoint rule; retain every epoch's loss/coefficient history.
        for seed in SEEDS:
            torch.manual_seed(seed)
            sym = SymNet().to(device)
            initial_state_hash = tensor_state_hash(sym.state_dict())
            opt = torch.optim.Adam(sym.parameters(), lr=1e-2)
            optimizer_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
            assert optimizer_param_ids == {id(p) for p in sym.parameters()}
            assert all(not p.requires_grad for p in frozen.parameters())
            theta_hash_before = tensor_state_hash(frozen.state_dict())
            min_loss, min_epoch = float("inf"), 0
            history = []
            for epoch_probe in range(1001):
                prediction = sym(features)
                loss = nn.functional.mse_loss(prediction, target)
                loss_value = float(loss.detach())
                xi = coeff_vector(sym, scales_k, scope["product_term_dict"])
                if loss_value < min_loss:
                    min_loss, min_epoch = loss_value, epoch_probe
                history.append({"checkpoint_epoch": epoch, "seed": seed, "epoch": epoch_probe,
                                "pde_loss": loss_value,
                                "coefficient_error": float(np.linalg.norm(xi - TRUTH)),
                                "spurious_l2": float(np.linalg.norm(xi[SPURIOUS_IDX])),
                                **{f"xi_{n}": float(v) for n, v in zip(FEATURE_NAMES, xi)}})
                if epoch_probe == 1000: break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            assert theta_hash_before == tensor_state_hash(frozen.state_dict()), f"Frozen theta changed at {epoch}/{seed}"
            xi = coeff_vector(sym, scales_k, scope["product_term_dict"])
            spurious = float(np.linalg.norm(xi[SPURIOUS_IDX]))
            final_rows.append({"checkpoint_epoch": epoch, "seed": seed, "final_pde_loss": loss_value,
                               "minimum_pde_loss": min_loss, "epoch_of_minimum_pde_loss": min_epoch,
                               "coefficient_error": float(np.linalg.norm(xi - TRUTH)), "spurious_l2": spurious,
                               "loose_success": success(xi, LOOSE), "strong_success": success(xi, STRONG),
                               "convergence_epoch": np.nan,
                               "symnet_initial_state_sha256": initial_state_hash,
                               "frozen_theta_sha256": theta_hash_before,
                               **{f"xi_{n}": float(v) for n, v in zip(FEATURE_NAMES, xi)}})
            history_rows.extend(history)
            if seed % 5 == 4:
                print(f"checkpoint {epoch}, seed {seed}: PDE={loss_value:.6g}, loose recovery={success(xi, LOOSE)}", flush=True)

    finals = pd.DataFrame(final_rows)
    history_df = pd.DataFrame(history_rows)
    checkpoints_df = pd.DataFrame(checkpoint_rows)
    finals.to_csv(OUT / "per_seed_results.csv", index=False)
    checkpoints_df.to_csv(OUT / "frozen_surrogate_metrics.csv", index=False)
    history_df.to_csv(OUT / "per_seed_histories.csv", index=False)
    # The Phase 19B snapshot recorder saves theta before its epoch's optimizer
    # update, so these derivative metrics are directly comparable to theta_k.
    derivative_checks = []
    for _, current in checkpoints_df.iterrows():
        epoch = int(current.checkpoint_epoch)
        checks = {}
        for quantity in ["u", "u_x", "u_xx", "u_t"]:
            actual, expected = float(current[quantity]), float(p19b_snap.loc[epoch, quantity + "_mse"])
            checks[quantity] = {"rerun": actual, "historical_phase19b": expected,
                                "abs_error": abs(actual - expected),
                                "pass": bool(np.isclose(actual, expected, rtol=2e-5, atol=3e-6))}
        derivative_checks.append({"epoch": epoch, "pass": all(v["pass"] for v in checks.values()), "metrics": checks})
    validation["phase19b_derivative_snapshot_comparison"] = {
        "source": str(P19B_SNAP.relative_to(ROOT)),
        "timing": "Phase 19B saved surrogate state before the labeled epoch update",
        "tolerance": {"rtol": 2e-5, "atol": 3e-6}, "checkpoints": derivative_checks,
        "all_pass": all(v["pass"] for v in derivative_checks)}
    validation["all_pass"] = validation["all_pass"] and validation["phase19b_derivative_snapshot_comparison"]["all_pass"]
    (OUT / "joint_reproduction_validation.json").write_text(json.dumps(validation, indent=2))
    if not validation["all_pass"]:
        (OUT / "status.json").write_text(json.dumps({"status": "stopped_derivative_validation_mismatch"}, indent=2))
        raise RuntimeError("Phase 19B derivative snapshots did not reproduce; interpretation stopped")
    summaries = []
    for _, c in checkpoints_df.iterrows():
        group = finals[finals.checkpoint_epoch == c.checkpoint_epoch]
        summaries.append({"theta_checkpoint": int(c.checkpoint_epoch),
                          **{k: float(c[k]) for k in ["u", "u_x", "u_xx", "u_t"]},
                          "recovered": int(group.loose_success.sum()), "total": int(len(group)),
                          "R_k": float(group.loose_success.mean()),
                          "coefficient_error_mean": float(group.coefficient_error.mean()),
                          "coefficient_error_median": float(group.coefficient_error.median()),
                          "coefficient_error_std": float(group.coefficient_error.std(ddof=1)),
                          "strong_recovered": int(group.strong_success.sum())})
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "recoverability_by_checkpoint.csv", index=False)
    distribution_rows = []
    for epoch in CHECKPOINTS:
        group = finals[finals.checkpoint_epoch == epoch]
        for feature in FEATURE_NAMES:
            values = group[f"xi_{feature}"].astype(float)
            distribution_rows.append({"checkpoint_epoch": epoch, "feature": feature,
                                      "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                                      "min": float(values.min()), "q25": float(values.quantile(.25)),
                                      "median": float(values.median()), "q75": float(values.quantile(.75)),
                                      "max": float(values.max()),
                                      "seed_values": json.dumps(values.tolist())})
    pd.DataFrame(distribution_rows).to_csv(OUT / "coefficient_distribution.csv", index=False)
    config = {
        "phase": "23 — Symbolic Recoverability Along a Joint Surrogate Trajectory",
        "selected_joint_source": "Phase 19B transferred-scale joint setup, validated against Phase 21 Adam checkpoint metrics",
        "optimizer": "Adam", "learning_rate": 5e-4, "lambda_data": 1.0, "lambda_pde": 0.5,
        "epochs": 1000, "diagnostic_epochs": CHECKPOINTS, "batch_size": 4096,
        "seed": 19219, "batch_seed": 19220, "dataset_seed": 0,
        "architecture": "SirenMLP(hidden_size=64, hidden_layers=3, first_omega_0=20, hidden_omega_0=1) + existing MinimalSymNet",
        "primitive_features": PRIMITIVES, "transferred_scales": scales_np.tolist(),
        "probe_protocol": phase18_control, "probe_seeds": SEEDS,
        "historical_phase21_metrics": str(P21_TABLE.relative_to(ROOT)),
        "source_phase22_matched_inputs": str(P22_INPUTS.relative_to(ROOT)),
        "reused_artifact_sha256": {
            str(P21_TABLE.relative_to(ROOT)): sha256(P21_TABLE),
            str(P19B_SNAP.relative_to(ROOT)): sha256(P19B_SNAP),
            str(P22_INPUTS.relative_to(ROOT)): sha256(P22_INPUTS),
            str(PHASE18_RUNS.relative_to(ROOT)): sha256(PHASE18_RUNS),
            "run_results/phase19b_scale_transfer/final_summary.csv": sha256(ROOT / "run_results/phase19b_scale_transfer/final_summary.csv"),
        },
        "initial_surrogate_state_sha256": tensor_state_hash(s0),
        "initial_symnet_state_sha256": tensor_state_hash(y0),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "device": "cpu", "dtype": "float32", "torch_version": torch.__version__, "threads": torch.get_num_threads(),
        "checkpoint_timing": "theta at beginning of the labeled epoch, before that epoch's optimizer step",
        "phase18_primary_success": "loose criterion at final epoch 1000; convergence epoch not defined by Phase 18",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2))
    (OUT / "status.json").write_text(json.dumps({"status": "complete", "all_joint_metrics_reproduced": True,
                                                   "theta_frozen_checks": "passed", "checkpoint_reload_checks": "passed"}, indent=2))
    print("\nRecoverability by frozen surrogate checkpoint:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved Phase 23 artifacts to {OUT}")


if __name__ == "__main__":
    run()
