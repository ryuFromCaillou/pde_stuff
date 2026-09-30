"""Measure how primitive feature scales alter Phase 24 joint PDE pressure."""
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

from prog.minimal_symnet import MinimalSymNet, product_term_dict
from utils.burgers_recoverability import P22, PRIMITIVES, coefficients, evaluate_diagnostic_fields, surrogate
from utils.diagnostic_io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "run_results/phase24_scale_gradient_diagnostic"
PHASE24 = ROOT / "run_results/phase24_smooth_burgers_control"
P23 = ROOT / "run_results/phase23_joint_trajectory_recoverability"
CHECKPOINTS = [0, 10, 50, 100, 200]
SCALE_P24 = np.array([0.3498518168926239, 0.37613674998283386, 4.194288730621338], dtype=np.float64)
SCALE_P23 = np.array([0.8564476370811462, 2.418001890182495, 49.38141632080078], dtype=np.float64)
SCALE_UNIT = np.ones(3, dtype=np.float64)


def _flat_grads(grads, params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                      for p, g in zip(params, grads)])


def _primitives(model, t, x):
    pred = model(t, x)
    ut = torch.autograd.grad(pred, t, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
    ux = torch.autograd.grad(pred, x, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    return pred, ut, ux, uxx


def _gradient_metrics(model, sym, t, x, target, scales):
    tb = t.detach().clone().requires_grad_(True)
    xb = x.detach().clone().requires_grad_(True)
    pred, ut, ux, uxx = _primitives(model, tb, xb)
    scale_tensor = torch.tensor(scales, dtype=torch.float32)
    rhs = sym(torch.cat([pred, ux, uxx], dim=1) / scale_tensor.reshape(1, 3))
    data = nn.functional.mse_loss(pred, target)
    pde = nn.functional.mse_loss(ut, rhs)
    params = list(model.parameters())
    gd = _flat_grads(torch.autograd.grad(data, params, retain_graph=True, allow_unused=True), params)
    gp = _flat_grads(torch.autograd.grad(pde, params, retain_graph=True, allow_unused=True), params)
    weighted = 0.5 * gp
    total = gd + weighted
    return {
        "data_loss": float(data.detach()), "pde_loss": float(pde.detach()),
        "weighted_total_loss": float((data + 0.5 * pde).detach()),
        "data_grad_norm": float(gd.norm()), "weighted_pde_grad_norm": float(weighted.norm()),
        "gradient_ratio": float(weighted.norm() / (gd.norm() + 1e-30)),
        "gradient_cosine": float(torch.dot(gd, gp) / (gd.norm() * gp.norm() + 1e-30)),
        "total_surrogate_grad_norm": float(total.norm()),
    }


def _state_metrics(model, t_flat, x_flat, shape, refs):
    fields = evaluate_diagnostic_fields(model, t_flat, x_flat, shape, ["u", "u_x", "u_xx"])
    return {f"{q}_mse": float(np.mean((fields[q] - refs[q]) ** 2)) for q in ["u", "u_x", "u_xx", "u_t"]}


def _make_scale_cases():
    return {
        "phase24_smooth": SCALE_P24,
        "phase23_transferred": SCALE_P23,
        "unit_unscaled": SCALE_UNIT,
        "p23_u_from_p24": np.array([SCALE_P24[0], SCALE_P23[1], SCALE_P23[2]]),
        "p23_ux_from_p24": np.array([SCALE_P23[0], SCALE_P24[1], SCALE_P23[2]]),
        "p23_uxx_from_p24": np.array([SCALE_P23[0], SCALE_P23[1], SCALE_P24[2]]),
    }


def _plot(out, trajectory):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, group in trajectory.groupby("condition"):
        axes[0].plot(group.epoch, group.gradient_ratio, marker="o", label=name)
        axes[1].plot(group.epoch, group.u_mse, marker="o", label=name)
        axes[2].plot(group.epoch, group.data_loss, marker="o", label=f"{name}: data")
        axes[2].plot(group.epoch, 0.5 * group.pde_loss, marker="x", linestyle="--", label=f"{name}: 0.5 PDE")
    axes[0].set_yscale("log")
    axes[0].set_title("PDE-to-data surrogate gradient pressure")
    axes[0].set_ylabel(r"$R=|0.5 g_{PDE}|/|g_{data}|$")
    axes[1].set_title("Smooth Burgers field fidelity")
    axes[1].set_ylabel("field MSE")
    axes[2].set_title("Joint data/PDE loss tradeoff")
    axes[2].set_ylabel("loss")
    for ax in axes:
        ax.set_xlabel("checkpoint update"); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=7, ncol=2)
    fig.suptitle("Phase 24 feature scaling changes joint surrogate pressure", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(out / "scale_gradient_trajectory.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / "scale_gradient_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def run(out=OUT):
    out = Path(out)
    if (out / "status.json").exists() and json.loads((out / "status.json").read_text()).get("status") == "complete":
        print("Reusing completed Phase 24 scale-gradient diagnostic."); return
    if out.exists(): raise RuntimeError(f"Refusing to overwrite incomplete diagnostic directory: {out}")
    out.mkdir(parents=True); torch.set_num_threads(2)
    matched = torch.load(P22, map_location="cpu", weights_only=False)
    ref_archive = np.load(PHASE24 / "reference_fields.npz")
    refs_np = dict(ref_archive); shape = refs_np["u"].shape
    target_full = torch.tensor(refs_np["u"].reshape(-1), dtype=torch.float32).reshape(-1, 1)
    t_flat, x_flat = matched["t"], matched["x"]
    initial_model = surrogate(); initial_model.load_state_dict(matched["surrogate"])
    initial_fields = evaluate_diagnostic_fields(initial_model, t_flat.numpy().ravel(), x_flat.numpy().ravel(), shape, PRIMITIVES)
    raw_rms = np.array([np.sqrt(np.mean(initial_fields[q] ** 2)) for q in PRIMITIVES])
    cases = _make_scale_cases()
    first_rows = []
    idx = matched["batches"][0]
    for name, scales in cases.items():
        model = surrogate(); model.load_state_dict(matched["surrogate"])
        sym = MinimalSymNet(); sym.load_state_dict(matched["symnet"])
        row = {"condition": name, "scale_u": scales[0], "scale_u_x": scales[1], "scale_u_xx": scales[2],
               **{f"raw_rms_{q}": v for q, v in zip(PRIMITIVES, raw_rms)},
               **{f"normalized_rms_{q}": v / scales[i] for i, (q, v) in enumerate(zip(PRIMITIVES, raw_rms))}}
        row.update(_gradient_metrics(model, sym, matched["t"][idx], matched["x"][idx], target_full[idx], scales))
        row.update({f"xi_{k}": v for k, v in zip(product_term_dict(sym), coefficients(sym, scales))})
        first_rows.append(row)
    pd.DataFrame(first_rows).to_csv(out / "matched_initial_scale_comparison.csv", index=False)

    trajectory_rows = []; batches = matched["batches"]
    for name in ["phase24_smooth", "phase23_transferred", "unit_unscaled"]:
        scales = cases[name]
        model = surrogate(); model.load_state_dict(matched["surrogate"])
        sym = MinimalSymNet(); sym.load_state_dict(matched["symnet"])
        optimizer = torch.optim.Adam(list(model.parameters()) + list(sym.parameters()), lr=5e-4)
        for epoch in range(max(CHECKPOINTS) + 1):
            idx = batches[epoch - 1] if epoch else batches[0]
            t_batch, x_batch, y_batch = matched["t"][idx], matched["x"][idx], target_full[idx]
            row = {"condition": name, "epoch": epoch, **_gradient_metrics(model, sym, t_batch, x_batch, y_batch, scales)}
            row.update(_state_metrics(model, t_flat.numpy().ravel(), x_flat.numpy().ravel(), shape, refs_np))
            row.update({f"xi_{k}": v for k, v in zip(product_term_dict(sym), coefficients(sym, scales))})
            if epoch in CHECKPOINTS: trajectory_rows.append(row)
            if epoch == max(CHECKPOINTS): break
            tb = t_batch.detach().clone().requires_grad_(True); xb = x_batch.detach().clone().requires_grad_(True)
            pred, ut, ux, uxx = _primitives(model, tb, xb)
            rhs = sym(torch.cat([pred, ux, uxx], 1) / torch.tensor(scales, dtype=torch.float32).reshape(1, 3))
            loss = nn.functional.mse_loss(pred, y_batch) + 0.5 * nn.functional.mse_loss(ut, rhs)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    trajectory = pd.DataFrame(trajectory_rows); trajectory.to_csv(out / "short_joint_trajectories.csv", index=False); _plot(out, trajectory)
    write_json(out / "config.json", {"phase": "Phase 24 scale-gradient diagnostic", "matched_input": str(P22.relative_to(ROOT)),
        "phase24_scales": SCALE_P24.tolist(), "phase23_scales": SCALE_P23.tolist(), "unit_scales": SCALE_UNIT.tolist(),
        "conditions": list(cases), "checkpoints": CHECKPOINTS, "horizon": max(CHECKPOINTS), "lambda_data": 1., "lambda_pde": .5,
        "learning_rate": 5e-4, "phase24_config_sha256": sha256(PHASE24 / "config.json"), "phase23_config_sha256": sha256(P23 / "config.json"),
        "no_phase23_phase24_overwrite": True})
    write_json(out / "status.json", {"status": "complete", "matched_state": True, "only_scales_changed": True})
    print(pd.DataFrame(first_rows).to_string(index=False)); print(trajectory.to_string(index=False))


if __name__ == "__main__": run()
