"""Canonical Phase 19B-i scale-transfer PDE-weight sweep."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from prog.minimal_symnet import MinimalSymNet, product_term_dict
from prog.mlps import SirenMLP
from utils.derivative_utils import build_burgers_reference_derivative_grids, evaluate_diagnostic_fields

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "run_results/phase19b_i_pde_loss_weight_sweep"
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]
CHECKPOINTS = [0, 50, 200, 500, 1000]
TRUTH = np.array([0., 0., .02, 0., -1., 0., 0., 0., 0.])
NAMES = ['u', 'u_x', 'u_xx', 'u^2', 'u*u_x', 'u*u_xx', 'u_x^2', 'u_x*u_xx', 'u_xx^2']


def _scales(s):
    return np.array([s[0], s[1], s[2], s[0]**2, s[0]*s[1], s[0]*s[2], s[1]**2, s[1]*s[2], s[2]**2])


def _coefficients(sym, scales):
    return np.array(list(product_term_dict(sym).values())) / _scales(scales)


def _norm(grads):
    return float(np.sqrt(sum(float((g.detach() ** 2).sum()) for g in grads if g is not None)))


def run(out=OUT):
    out = Path(out)
    if (out / "summary.csv").exists():
        print("Reusing completed Phase 19B-i sweep artifacts."); return
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    matched = torch.load(ROOT / "run_results/phase22_sgd_control/matched_inputs.pt", map_location="cpu", weights_only=False)
    scales = matched["scales"].detach().clone()
    assert np.allclose(scales.numpy(), [0.85644764, 2.418002, 49.381416], rtol=1e-5, atol=1e-5)
    t_train, x_train, u_train = matched["t"], matched["x"], matched["u"]
    batches = matched["batches"]
    s0, y0 = matched["surrogate"], matched["symnet"]
    x, _, _, (times, clean) = solve_burgers(return_history=True, history_every=1)
    keep = np.r_[0, np.arange(1, 501, 2), 500]
    t_grid, u_grid = times[keep], clean[keep]
    t_flat = np.repeat(t_grid[:, None], len(x), axis=1).reshape(-1).astype(np.float32)
    x_flat = np.repeat(x[None, :], len(t_grid), axis=0).reshape(-1).astype(np.float32)
    ref = build_burgers_reference_derivative_grids(u_grid=u_grid.astype(np.float32), x_grid=x.astype(np.float32), nu=.02)
    refs = {q: ref[q]["physical"] for q in ["u", "u_x", "u_xx", "u_t"]}
    assert np.array_equal(u_grid.astype(np.float32).reshape(-1), u_train.numpy().reshape(-1))
    histories = {}; snapshots = []
    for lam in LAMBDAS:
        sm = SirenMLP(hidden_size=64, hidden_layers=3, first_omega_0=20., hidden_omega_0=1.)
        sm.load_state_dict(s0); ym = MinimalSymNet(); ym.load_state_dict(y0)
        opt = torch.optim.Adam(list(sm.parameters()) + list(ym.parameters()), lr=5e-4)
        rows = []; states = {}
        for ep in range(1001):
            ix = batches[ep - 1] if ep else batches[0]
            tb = t_train[ix].detach().clone().requires_grad_(True); xb = x_train[ix].detach().clone().requires_grad_(True)
            up = sm(tb, xb)
            ut = torch.autograd.grad(up, tb, torch.ones_like(up), create_graph=True, retain_graph=True)[0]
            ux = torch.autograd.grad(up, xb, torch.ones_like(up), create_graph=True, retain_graph=True)[0]
            uxx = torch.autograd.grad(ux, xb, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
            rhs = ym(torch.cat([up, ux, uxx], 1) / scales.reshape(1, 3))
            dl = nn.functional.mse_loss(up, u_train[ix]); pl = nn.functional.mse_loss(ut, rhs); total = dl + lam * pl
            xi = _coefficients(ym, scales.numpy())
            gd = torch.autograd.grad(dl, list(sm.parameters()), retain_graph=True, allow_unused=True)
            gp = torch.autograd.grad(pl, list(sm.parameters()), retain_graph=True, allow_unused=True)
            nd, npde = _norm(gd), _norm(gp)
            dot = sum(float(a.detach().flatten().dot(b.detach().flatten())) for a, b in zip(gd, gp) if a is not None and b is not None)
            row = {"lambda_pde": lam, "epoch": ep, "data_loss": float(dl), "pde_loss": float(pl), "total_loss": float(total),
                   "coeff_error": float(np.linalg.norm(xi - TRUTH)), "spurious_l2": float(np.linalg.norm(xi[[0, 1, 3, 5, 6, 7, 8]])),
                   "data_grad": nd, "pde_grad": npde, "weighted_pde_grad": lam * npde,
                   "grad_ratio": lam * npde / (nd + 1e-30), "grad_cos": dot / (nd * npde + 1e-30),
                   **{f"xi_{n}": float(v) for n, v in zip(NAMES, xi)}}
            rows.append(row)
            if ep in CHECKPOINTS:
                states[ep] = {k: v.detach().clone() for k, v in sm.state_dict().items()}
            if ep == 1000: break
            opt.zero_grad(); total.backward(); opt.step()
        histories[lam] = pd.DataFrame(rows)
        histories[lam].to_csv(out / f"lambda_{lam:g}_history.csv", index=False)
        for ep, state in states.items():
            mm = SirenMLP(hidden_size=64, hidden_layers=3, first_omega_0=20., hidden_omega_0=1.); mm.load_state_dict(state)
            fields = evaluate_diagnostic_fields(mm, t_flat, x_flat, u_grid.shape, ["u", "u_x", "u_xx"])
            snapshots.append({"lambda_pde": lam, "epoch": ep, **{f"{q}_mse": float(np.mean((fields[q] - refs[q]) ** 2)) for q in ["u", "u_x", "u_xx", "u_t"]}})
    pd.concat(histories.values(), ignore_index=True).to_csv(out / "checkpoint_metrics.csv", index=False)
    snap = pd.DataFrame(snapshots); snap.to_csv(out / "snapshot_metrics.csv", index=False)
    summary = []
    for lam, h in histories.items():
        r = h.iloc[-1]; s = snap[(snap.lambda_pde == lam) & (snap.epoch == 1000)].iloc[0]
        summary.append({"lambda_pde": lam, "final data MSE": r.data_loss, "final PDE MSE": r.pde_loss,
                        "final u_x MSE": s.u_x_mse, "final u_xx MSE": s.u_xx_mse,
                        "xi(u*u_x)": r["xi_u*u_x"], "xi(u_xx)": r.xi_u_xx,
                        "coefficient error": r.coeff_error, "spurious norm": r.spurious_l2,
                        "gradient ratio": r.grad_ratio, "gradient cosine": r.grad_cos})
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({"lambdas": LAMBDAS, "lambda_data": 1., "epochs": 1000,
        "batch_size": 4096, "learning_rate": 5e-4, "scales": scales.numpy().tolist(), "matched_inputs": "run_results/phase22_sgd_control/matched_inputs.pt"}, indent=2))


if __name__ == "__main__": run()
