"""Long-horizon continuation of the original Phase 19B joint trajectory."""
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

from prog.mlps import SirenMLP
from prog.minimal_symnet import MinimalSymNet, product_term_dict
from utils.burgers_recoverability import P22, PRIMITIVES, coefficients
from utils.derivative_utils import build_burgers_reference_derivative_grids, evaluate_diagnostic_fields
from utils.diagnostic_io import write_json
from Datasets.data.processed.burg_gen.burg_gen import solve_burgers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "run_results/phase19b_long_horizon_control"
TARGET_STEPS = 100_000
CHECKPOINTS = [0, 50, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
TRUTH = np.array([0., 0., .02, 0., -1., 0., 0., 0., 0.])
NAMES = ['u', 'u_x', 'u_xx', 'u^2', 'u*u_x', 'u*u_xx', 'u_x^2', 'u_x*u_xx', 'u_xx^2']


def _coefficients(sym, scales):
    s = np.asarray(scales, dtype=np.float64)
    products = np.array([s[0], s[1], s[2], s[0]**2, s[0]*s[1], s[0]*s[2], s[1]**2, s[1]*s[2], s[2]**2])
    return np.array(list(product_term_dict(sym).values())) / products


def _metrics(xi):
    spurious = float(np.linalg.norm(xi[[0, 1, 3, 5, 6, 7, 8]]))
    error = float(np.linalg.norm(xi - TRUTH))
    loose = bool(abs(xi[4] + 1) < .25 and abs(xi[2] - .02) < .02 and spurious < .25)
    strong = bool(abs(xi[4] + 1) < .10 and abs(xi[2] - .02) < .01 and spurious < .10)
    return error, spurious, loose, strong


def _model_fields(model, t_flat, x_flat, shape):
    return evaluate_diagnostic_fields(model, t_flat, x_flat, shape, PRIMITIVES)


def _plot(out, checkpoint):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(checkpoint.step, checkpoint["xi_u*u_x"], marker="o", label=r"$\xi_{u u_x}$")
    ax.plot(checkpoint.step, checkpoint.xi_u_xx, marker="o", label=r"$\xi_{u_{xx}}$")
    ax.axhline(-1., color="tab:blue", linestyle=":", label="transport target -1")
    ax.axhline(.02, color="tab:orange", linestyle=":", label="diffusion target 0.02")
    ax.set_xscale("symlog", linthresh=50)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("physical coefficient")
    ax.set_title("Phase 19B Burgers coefficients over long-horizon joint training")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "coefficient_trajectory.png", dpi=160); fig.savefig(out / "coefficient_trajectory.pdf"); plt.close(fig)


def run(out=OUT, target_steps=TARGET_STEPS):
    out = Path(out)
    status_path = out / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
        print("Reusing completed Phase 19B long-horizon control."); return
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    matched = torch.load(P22, map_location="cpu", weights_only=False)
    scales = matched["scales"].detach().clone()
    assert np.allclose(scales.numpy(), [0.8564476370811462, 2.418001890182495, 49.38141632080078], rtol=1e-7, atol=1e-9)
    t_train, x_train, u_train = matched["t"], matched["x"], matched["u"]
    batch_size = int(matched["batches"][0].numel())
    # Recreate the original shock reference and verify the archived observation grid.
    x_grid, _, _, (all_times, all_u) = solve_burgers(return_history=True, history_every=1)
    keep = np.r_[0, np.arange(1, 501, 2), 500]
    t_grid, u_grid = all_times[keep], all_u[keep]
    t_flat = np.repeat(t_grid[:, None], len(x_grid), axis=1).reshape(-1).astype(np.float32)
    x_flat = np.repeat(x_grid[None, :], len(t_grid), axis=0).reshape(-1).astype(np.float32)
    assert np.array_equal(t_flat, t_train.numpy().ravel()) and np.array_equal(x_flat, x_train.numpy().ravel())
    assert np.array_equal(u_grid.astype(np.float32).ravel(), u_train.numpy().ravel())
    ref_grid = build_burgers_reference_derivative_grids(u_grid=u_grid.astype(np.float32), x_grid=x_grid.astype(np.float32), nu=.02)
    refs = {q: ref_grid[q]["physical"] for q in ["u", "u_x", "u_xx", "u_t"]}

    # The original Phase 19B batch seed is 19220. Verify all archived batches, then continue the same RNG stream.
    batch_generator = torch.Generator(device="cpu"); batch_generator.manual_seed(19220)
    batches = []
    for old in matched["batches"]:
        new = torch.randint(0, len(t_train), (batch_size,), generator=batch_generator)
        assert torch.equal(old, new), "Archived Phase 19B batches do not match seed 19220."
        batches.append(old)

    model = SirenMLP(hidden_size=64, hidden_layers=3, first_omega_0=20., hidden_omega_0=1.)
    sym = MinimalSymNet(); model.load_state_dict(matched["surrogate"]); sym.load_state_dict(matched["symnet"])
    optimizer = torch.optim.Adam(list(model.parameters()) + list(sym.parameters()), lr=5e-4)
    history_path = out / "optimization_history.csv"
    checkpoint_path = out / "checkpoint_metrics.csv"
    rows = []; checkpoints = []
    first_loose = None; first_strong = None
    for step in range(target_steps + 1):
        if step < len(batches):
            idx = batches[step - 1] if step else batches[0]
        else:
            # The original indexing uses batch 0 twice; after the archived sequence, draw one fresh batch per step.
            if step == len(batches):
                idx = batches[-1]
            else:
                idx = torch.randint(0, len(t_train), (batch_size,), generator=batch_generator)
        tb = t_train[idx].detach().clone().requires_grad_(True); xb = x_train[idx].detach().clone().requires_grad_(True)
        up = model(tb, xb)
        ut = torch.autograd.grad(up, tb, torch.ones_like(up), create_graph=True, retain_graph=True)[0]
        ux = torch.autograd.grad(up, xb, torch.ones_like(up), create_graph=True, retain_graph=True)[0]
        uxx = torch.autograd.grad(ux, xb, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
        rhs = sym(torch.cat([up, ux, uxx], 1) / scales.reshape(1, 3))
        data_loss = nn.functional.mse_loss(up, u_train[idx]); pde_loss = nn.functional.mse_loss(ut, rhs)
        total_loss = data_loss + .5 * pde_loss
        xi = _coefficients(sym, scales.numpy()); coeff_error, spurious, loose, strong = _metrics(xi)
        if loose and first_loose is None: first_loose = step
        if strong and first_strong is None: first_strong = step
        row = {"step": step, "data_loss": float(data_loss.detach()), "pde_loss": float(pde_loss.detach()), "total_loss": float(total_loss.detach()),
               "xi_u*u_x": float(xi[4]), "xi_u_xx": float(xi[2]), "coefficient_error": coeff_error,
               "spurious_l2": spurious, "loose_recovery": loose, "strong_recovery": strong,
               **{f"xi_{n}": float(v) for n, v in zip(NAMES, xi)}}
        rows.append(row)
        if step in CHECKPOINTS:
            fields = _model_fields(model, t_flat, x_flat, u_grid.shape)
            full = {f"{q}_mse": float(np.mean((fields[q] - refs[q]) ** 2)) for q in ["u", "u_x", "u_xx", "u_t"]}
            checkpoint_row = {**row, **full}; checkpoints.append(checkpoint_row)
            torch.save({"step": step, "surrogate": model.state_dict(), "symnet": sym.state_dict(), "optimizer": optimizer.state_dict()}, out / f"checkpoint_{step:06d}.pt")
            pd.DataFrame(rows).to_csv(history_path, index=False); pd.DataFrame(checkpoints).to_csv(checkpoint_path, index=False)
            print(f"step={step}: data={row['data_loss']:.6e} PDE={row['pde_loss']:.6e} transport={row['xi_u*u_x']:+.6f} diffusion={row['xi_u_xx']:+.6f}", flush=True)
        if step == target_steps: break
        optimizer.zero_grad(set_to_none=True); total_loss.backward(); optimizer.step()
    checkpoint = pd.DataFrame(checkpoints); _plot(out, checkpoint)
    pd.DataFrame(rows).to_csv(history_path, index=False); checkpoint.to_csv(checkpoint_path, index=False)
    initial = {k: v.detach().clone() for k, v in matched["surrogate"].items()}
    write_json(out / "config.json", {"phase": "Phase 19B long-horizon joint-training control", "steps": target_steps,
        "clock": "one optimizer.step() per outer-loop iteration; batch 0 reused at steps 0 and 1",
        "architecture": "SirenMLP(64,3,20,1)+MinimalSymNet", "optimizer": "Adam", "learning_rate": 5e-4,
        "lambda_data": 1., "lambda_pde": .5, "batch_size": batch_size, "initialization_seed": 19219,
        "batch_seed": 19220, "scales": scales.numpy().tolist(), "checkpoints": CHECKPOINTS,
        "recovery_criteria": {"loose": {"transport": .25, "diffusion": .02, "spurious_l2": .25}, "strong": {"transport": .10, "diffusion": .01, "spurious_l2": .10}},
        "source_matched_inputs": str(P22.relative_to(ROOT))})
    torch.save({"surrogate": initial, "symnet": matched["symnet"], "scales": scales, "batches_verified": len(batches)}, out / "initial_state.pt")
    write_json(out / "status.json", {"status": "complete", "steps": target_steps, "first_loose_step": first_loose, "first_strong_step": first_strong,
                                      "original_artifacts_preserved": True, "later_phases_unchanged": True})
    print(checkpoint.to_string(index=False)); print("first_loose_step", first_loose, "first_strong_step", first_strong)


if __name__ == "__main__": run()
