"""Audit the Phase 24 smooth-data surrogate fit without changing Phase 24."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from prog.mlps import SirenMLP
from prog.minimal_symnet import MinimalSymNet
from utils.burgers_recoverability import (
    CHECKPOINTS, OUT as PHASE24, P22, P23, PRIMITIVES, surrogate,
    evaluate_diagnostic_fields, compute_error_metrics, rms,
)
from utils.diagnostic_io import sha256, write_json, inventory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "run_results/phase24_surrogate_fit_audit"


def _smooth_observations():
    x, _, _, (dense_t, dense_u) = solve_burgers(
        return_history=True, history_every=1,
        initial_condition=lambda values: 0.5 * np.sin(values),
    )
    keep = np.r_[0, np.arange(1, 501, 2), 500]
    t = dense_t[keep].astype(np.float32)
    u = dense_u[keep].astype(np.float32)
    t_flat = np.repeat(t[:, None], len(x), axis=1).reshape(-1).astype(np.float32)
    x_flat = np.repeat(np.asarray(x, dtype=np.float32)[None, :], len(t), axis=0).reshape(-1)
    u_flat = u.reshape(-1)
    return np.asarray(x, dtype=np.float32), t, u, t_flat, x_flat, u_flat


def _model_fields(model, t_flat, x_flat):
    return evaluate_diagnostic_fields(model, t_flat, x_flat, (252, 256), PRIMITIVES)


def _gradient_row(model, symnet, t, x, target, scales):
    tb = t.detach().clone().requires_grad_(True)
    xb = x.detach().clone().requires_grad_(True)
    pred = model(tb, xb)
    ut = torch.autograd.grad(pred, tb, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
    ux = torch.autograd.grad(pred, xb, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(ux, xb, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    rhs = symnet(torch.cat([pred, ux, uxx], dim=1) / scales.reshape(1, 3))
    data = nn.functional.mse_loss(pred, target)
    pde = nn.functional.mse_loss(ut, rhs)
    params = list(model.parameters())
    gd = torch.autograd.grad(data, params, retain_graph=True, allow_unused=True)
    gp = torch.autograd.grad(pde, params, retain_graph=True, allow_unused=True)
    def flat(grads):
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                          for p, g in zip(params, grads)])
    gd, gp = flat(gd), flat(gp)
    wd, wp = gd, 0.5 * gp
    return {
        "data_loss": float(data.detach()), "pde_loss": float(pde.detach()),
        "total_loss": float(data.detach() + 0.5 * pde.detach()),
        "data_grad_norm": float(gd.norm()), "weighted_pde_grad_norm": float(wp.norm()),
        "weighted_pde_over_data": float(wp.norm() / (gd.norm() + 1e-30)),
        "gradient_cosine": float(torch.dot(gd, gp) / (gd.norm() * gp.norm() + 1e-30)),
    }


def _plot_comparison(out, reference, data_only, joint):
    indices = np.linspace(0, len(reference["t"]) - 1, 5, dtype=int)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), squeeze=False)
    for col, idx in enumerate(indices):
        axes[0, col].plot(reference["x"], reference["u"][idx], label="reference")
        axes[0, col].plot(reference["x"], data_only["u"][idx], "--", label="data-only")
        axes[0, col].plot(reference["x"], joint["u"][idx], ":", label="joint")
        axes[1, col].plot(reference["x"], reference["u"][idx] - data_only["u"][idx], label="data-only error")
        axes[1, col].plot(reference["x"], reference["u"][idx] - joint["u"][idx], label="joint error")
        axes[0, col].set_title(f"t={reference['t'][idx]:.3f}")
        axes[1, col].set_title(f"error at t={reference['t'][idx]:.3f}")
        axes[0, col].set_xlabel("x"); axes[1, col].set_xlabel("x")
    axes[0, 0].set_ylabel("u"); axes[1, 0].set_ylabel("reference − prediction")
    axes[0, 0].legend(fontsize=8); axes[1, 0].legend(fontsize=8)
    fig.suptitle("Phase 24 smooth Burgers: data-only versus joint surrogate fit", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "data_only_vs_joint_slices.pdf", bbox_inches="tight")
    fig.savefig(out / "data_only_vs_joint_slices.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(out=OUT):
    out = Path(out)
    if (out / "status.json").exists():
        if json.loads((out / "status.json").read_text())["status"] == "complete":
            print("Reusing completed Phase 24 fit audit.")
            return
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete audit directory: {out}")
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    matched = torch.load(P22, map_location="cpu", weights_only=False)
    x_grid, t_grid, u_grid, t_flat, x_flat, u_flat = _smooth_observations()
    np.testing.assert_array_equal(matched["t"].numpy().ravel(), t_flat)
    np.testing.assert_array_equal(matched["x"].numpy().ravel(), x_flat)
    target = torch.tensor(u_flat, dtype=torch.float32).reshape(-1, 1)
    coordinates = pd.DataFrame({"flat_index": np.arange(len(u_flat)), "t": t_flat, "x": x_flat, "u": u_flat})
    coordinates.loc[[0, 255, 256, 257, 12345, 32768, 64511]].to_csv(out / "correspondence_samples.csv", index=False)
    coordinates.to_csv(out / "smooth_training_coordinates.csv", index=False)
    assert np.array_equal(u_grid.reshape(-1).astype(np.float32), u_flat)
    np.savez_compressed(out / "smooth_reference.npz", x=x_grid, t=t_grid, u=u_grid)
    reference = {"x": x_grid, "t": t_grid, "u": u_grid}

    phase24_joint_history = pd.read_csv(PHASE24 / "joint_history.csv")
    p23_joint = pd.read_csv(P23 / "joint_checkpoint_metrics.csv")
    p24_metrics = pd.read_csv(PHASE24 / "frozen_surrogate_metrics.csv")
    p23_metrics = pd.read_csv(P23 / "frozen_surrogate_metrics.csv")
    phase24_scales = np.array(json.loads((PHASE24 / "config.json").read_text())["transferred_scales"])
    phase23_scales = np.array(json.loads((P23 / "config.json").read_text())["transferred_scales"])
    comparison = []
    for epoch in CHECKPOINTS:
        p24 = phase24_joint_history[phase24_joint_history.epoch == epoch].iloc[0]
        p23 = p23_joint[p23_joint.epoch == epoch].iloc[0]
        m24 = p24_metrics[p24_metrics.checkpoint_epoch == epoch].iloc[0]
        m23 = p23_metrics[p23_metrics.checkpoint_epoch == epoch].iloc[0]
        comparison.append({"epoch": epoch,
            "phase23_data_loss": p23.data_loss, "phase24_data_loss": p24.data_loss,
            "phase23_pde_loss": p23.pde_loss, "phase24_pde_loss": p24.pde_loss,
            "phase23_total_loss": p23.total_loss, "phase24_total_loss": p24.total_loss,
            **{f"phase23_{q}_mse": m23[q] for q in ["u", "u_x", "u_xx", "u_t"]},
            **{f"phase24_{q}_mse": m24[q] for q in ["u", "u_x", "u_xx", "u_t"]},
            "phase23_scale_u": m23.scale_u, "phase23_scale_u_x": m23.scale_u_x, "phase23_scale_u_xx": m23.scale_u_xx,
            "phase24_scale_u": m24.scale_u, "phase24_scale_u_x": m24.scale_u_x, "phase24_scale_u_xx": m24.scale_u_xx,
            "phase23_joint_transport": p23["xi_u*u_x"], "phase24_joint_transport": p24["xi_u*u_x"],
            "phase23_joint_diffusion": p23.xi_u_xx, "phase24_joint_diffusion": p24.xi_u_xx})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(out / "phase23_phase24_comparison.csv", index=False)
    write_json(out / "scale_comparison.json", {"phase23_transferred_scales": phase23_scales.tolist(),
        "phase24_transferred_scales": phase24_scales.tolist(),
        "initial_scale_ratio_phase24_over_phase23": (phase24_scales / phase23_scales).tolist()})

    # Reproduce all five saved Phase 24 states and errors before any control runs.
    saved_refs = np.load(PHASE24 / "reference_fields.npz")
    reproduced_rows = []
    for epoch in CHECKPOINTS:
        model = surrogate(); model.load_state_dict(torch.load(PHASE24 / "checkpoints" / f"theta_epoch_{epoch:04d}.pt", weights_only=False)["surrogate"])
        fields = _model_fields(model, t_flat, x_flat)
        reproduced_rows.append({"epoch": epoch, **{q + "_mse": float(np.mean((fields[q] - saved_refs[q]) ** 2)) for q in ["u", "u_x", "u_xx", "u_t"]}})
    pd.DataFrame(reproduced_rows).to_csv(out / "reproduced_phase24_metrics.csv", index=False)

    # Matched data-only run and a diagnostic replay of the joint trajectory.
    initial = matched["surrogate"]
    data_model = surrogate(); data_model.load_state_dict(initial)
    joint_model = surrogate(); joint_model.load_state_dict(initial)
    joint_sym = MinimalSymNet(); joint_sym.load_state_dict(matched["symnet"])
    data_opt = torch.optim.Adam(data_model.parameters(), lr=5e-4)
    joint_opt = torch.optim.Adam(list(joint_model.parameters()) + list(joint_sym.parameters()), lr=5e-4)
    batches = matched["batches"]
    scales = torch.tensor(phase24_scales, dtype=torch.float32)
    data_rows, joint_rows = [], []
    data_states, joint_states = {}, {}
    for epoch in range(1001):
        idx = batches[epoch - 1] if epoch else batches[0]
        tb, xb, ub = matched["t"][idx], matched["x"][idx], target[idx]
        data_pred = data_model(tb, xb)
        data_loss = nn.functional.mse_loss(data_pred, ub)
        if epoch in CHECKPOINTS:
            data_rows.append({"epoch": epoch, "data_loss": float(data_loss.detach())})
            joint_rows.append({"epoch": epoch, **_gradient_row(joint_model, joint_sym, tb, xb, ub, scales)})
            data_states[epoch] = {k: v.detach().clone() for k, v in data_model.state_dict().items()}
            joint_states[epoch] = {k: v.detach().clone() for k, v in joint_model.state_dict().items()}
            if epoch == 1000:
                break
        data_opt.zero_grad(set_to_none=True); data_loss.backward(); data_opt.step()
        # The joint replay uses the exact Phase 24 objective and batch sequence.
        tbj = tb.detach().clone().requires_grad_(True); xbj = xb.detach().clone().requires_grad_(True)
        pred = joint_model(tbj, xbj)
        ut = torch.autograd.grad(pred, tbj, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
        ux = torch.autograd.grad(pred, xbj, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
        uxx = torch.autograd.grad(ux, xbj, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
        loss = nn.functional.mse_loss(pred, ub) + 0.5 * nn.functional.mse_loss(ut, joint_sym(torch.cat([pred, ux, uxx], 1) / scales))
        joint_opt.zero_grad(set_to_none=True); loss.backward(); joint_opt.step()
    data_rows = pd.DataFrame(data_rows); joint_rows = pd.DataFrame(joint_rows)
    data_rows.to_csv(out / "data_only_checkpoints.csv", index=False)
    joint_rows.to_csv(out / "joint_gradient_diagnostics.csv", index=False)
    data_compare = []
    for epoch in CHECKPOINTS:
        data_checkpoint = surrogate(); data_checkpoint.load_state_dict(data_states[epoch])
        joint_checkpoint = surrogate(); joint_checkpoint.load_state_dict(joint_states[epoch])
        data_fields = _model_fields(data_checkpoint, t_flat, x_flat)
        joint_fields = _model_fields(joint_checkpoint, t_flat, x_flat)
        np.savez_compressed(out / f"data_only_fields_epoch_{epoch:04d}.npz", **data_fields)
        np.savez_compressed(out / f"joint_replay_fields_epoch_{epoch:04d}.npz", **joint_fields)
        data_compare.append({"epoch": epoch,
            "data_only_field_mse": float(np.mean((data_fields["u"] - u_grid) ** 2)),
            "joint_replay_field_mse": float(np.mean((joint_fields["u"] - u_grid) ** 2)),
            **{f"data_only_{q}_mse": float(np.mean((data_fields[q] - saved_refs[q]) ** 2)) for q in ["u_x", "u_xx", "u_t"]},
            **{f"joint_replay_{q}_mse": float(np.mean((joint_fields[q] - saved_refs[q]) ** 2)) for q in ["u_x", "u_xx", "u_t"]}})
    pd.DataFrame(data_compare).to_csv(out / "data_only_joint_field_comparison.csv", index=False)
    _plot_comparison(out, reference, data_fields, joint_fields)
    config = {"phase": "Phase 24 surrogate fit audit", "matched_horizon": 1000, "checkpoints": CHECKPOINTS,
        "optimizer": "Adam", "learning_rate": 5e-4, "batch_size": 4096, "batch_source": str(P22.relative_to(ROOT)),
        "initial_state_source": str(P22.relative_to(ROOT)), "phase23_artifacts_preserved": True,
        "phase24_artifacts_preserved": True, "phase24_inventory": inventory(PHASE24),
        "phase24_artifacts_sha256": {"config": sha256(PHASE24 / "config.json"), "joint_history": sha256(PHASE24 / "joint_history.csv")},
        "smooth_data_checks": {"initial_condition": "0.5*sin(x)", "domain": [0., float(2*np.pi)], "T": 1., "Nx": 256,
                                "Nt": 252, "flattening": "time-major, spatial-minor", "exact_coordinate_match": True,
                                "exact_u_match": True}}
    write_json(out / "config.json", config)
    write_json(out / "status.json", {"status": "complete", "phase24_unchanged": True,
        "data_path_correspondence_verified": True, "matched_data_only_completed": True})
    print(comparison.to_string(index=False))
    print(data_rows.to_string(index=False))
    print(joint_rows.to_string(index=False))


if __name__ == "__main__":
    run()
