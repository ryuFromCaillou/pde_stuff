from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from derivative_utils import (
    compute_error_metrics,
    fd_first_centered,
    fd_first_periodic,
    fd_second_centered,
    fd_second_periodic,
    fd_third_centered,
    fd_third_periodic,
)
from utils.fit_utils import fit_model_to_data, predict_on_grid
from prog.hlprs import savefig_atomic
from prog.mlps import SimpleMLP, SirenMLP

from Datasets.data.processed.allenc_gen.allen_cahn_gen import AllenCahnConfig, solve_allen_cahn
from Datasets.data.processed.burg_gen.burg_gen import BurgersDatasetConfig as BurgConfig, solve_burgers


SUMMARY_FIELDS = [
    "dataset",
    "seed",
    "model",
    "epochs",
    "batch_size",
    "lr",
    "noise_level",
    "u_rel_l2",
    "u_rmse",
    "u_max_abs",
    "ux_rel_l2",
    "ux_rmse",
    "ux_max_abs",
    "uxx_rel_l2",
    "uxx_rmse",
    "uxx_max_abs",
    "uxxx_rel_l2",
    "uxxx_rmse",
    "uxxx_max_abs",
    "final_train_loss",
    "ux_rel_l2_highgrad",
    "uxx_rel_l2_highgrad",
    "uxxx_rel_l2_highgrad",
]


@dataclass
class RunConfig:
    dataset: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    hidden_layers: int
    first_omega_0: float
    hidden_omega_0: float
    noise_level: float
    stride_t: int
    stride_x: int
    burgers_N: int
    burgers_L: float
    burgers_nu: float
    burgers_dt: float
    burgers_T: float
    allen_N: int
    allen_dt: float
    allen_T: float
    allen_d: float
    allen_reaction_scale: float
    allen_bc_value: float
    eval_chunk_size: int


def _affine_to_minus1_1(v: np.ndarray):
    '''
    Compute a,b such that (v - b) / a maps v to [-1,1].
    '''
    v = np.asarray(v, dtype=np.float64)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        raise ValueError("Cannot normalize: invalid range")
    a = 0.5 * (vmax - vmin)
    b = 0.5 * (vmax + vmin)
    return a, b


def _to_norm(v: np.ndarray, a: float, b: float) -> np.ndarray:
    return (np.asarray(v, dtype=np.float64) - b) / a


def _to_phys(vn: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.asarray(vn, dtype=np.float64) + b


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_train_samples(
    *,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    u_grid: np.ndarray,
    stride_t: int,
    stride_x: int,
    noise_level: float,
    seed: int,
):
    stride_t = max(1, int(stride_t)) # subsampling by stride_t means we take every stride_t'th time step; must be >=1
    stride_x = max(1, int(stride_x))

    rows = np.arange(t_grid.size)[::stride_t]
    cols = np.arange(x_grid.size)[::stride_x]

    t2d = np.repeat(t_grid[:, None], x_grid.size, axis=1)
    x2d = np.repeat(x_grid[None, :], t_grid.size, axis=0)
    u2d = u_grid

    t_s = t2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32) #(reshaping after subsampling ensures correct alignment of t,x,u values; if we reshaped first then subsampled, we’d break the alignment)
    x_s = x2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)
    y_s = u2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)

    if float(noise_level) > 0:
        rng = np.random.default_rng(int(seed))
        sigma = float(noise_level) * float(np.std(y_s)) # scale noise to data stddev so noise_level is a relative measure; if we didn't scale by stddev, then noise_level would be an absolute measure and might need to be adjusted for different datasets or even different train/test splits of the same dataset
        y_noisy = (y_s + sigma * rng.standard_normal(size=y_s.shape)).astype(np.float32)
    else:
        y_noisy = y_s

    # flattened subsampled training data; each of these is 1D with same length
    return t_s, x_s, y_s, y_noisy


def _predict_and_derivs_on_grid(
    model,
    *,
    t_norm_flat: np.ndarray,
    x_norm_flat: np.ndarray,
    device: str,
    ax_scale: float,
    chunk_size: int,
    max_order: int = 3,
):
    model = model.to(device)
    model.eval()

    t_norm_flat = np.asarray(t_norm_flat, dtype=np.float32).reshape(-1, 1)
    x_norm_flat = np.asarray(x_norm_flat, dtype=np.float32).reshape(-1, 1)
    n = int(t_norm_flat.shape[0])

    out: dict[str, list[np.ndarray]] = {"u": [], "ux": [], "uxx": [], "uxxx": []}

    chunk_size = max(1, int(chunk_size))
    max_order = int(max_order)

    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        t_b = torch.from_numpy(t_norm_flat[s:e]).to(device)
        x_b = torch.from_numpy(x_norm_flat[s:e]).to(device).requires_grad_(True)

        with torch.enable_grad():
            u = model(t_b, x_b)
            out["u"].append(u.detach().cpu().numpy().reshape(-1))

            if max_order >= 1:
                ux = torch.autograd.grad(
                    u,
                    x_b,
                    grad_outputs=torch.ones_like(u),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                out["ux"].append((ux / ax_scale).detach().cpu().numpy().reshape(-1))

            if max_order >= 2:
                uxx = torch.autograd.grad(
                    ux,
                    x_b,
                    grad_outputs=torch.ones_like(ux),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                out["uxx"].append((uxx / (ax_scale**2)).detach().cpu().numpy().reshape(-1))

            if max_order >= 3:
                uxxx = torch.autograd.grad(
                    uxx,
                    x_b,
                    grad_outputs=torch.ones_like(uxx),
                    create_graph=False,
                    retain_graph=False,
                )[0]
                out["uxxx"].append((uxxx / (ax_scale**3)).detach().cpu().numpy().reshape(-1))

    return {k: np.concatenate(v, axis=0) if len(v) else None for k, v in out.items()}


def _plot_loss(losses: list[float], path: Path, title: str):
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(len(losses)), losses)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(title)
    savefig_atomic(path)


def _plot_heatmap(t: np.ndarray, x: np.ndarray, U: np.ndarray, path: Path, title: str, cbar: str):
    plt.figure(figsize=(6, 4), constrained_layout=True)
    plt.imshow(
        U,
        extent=[float(x.min()), float(x.max()), float(t.min()), float(t.max())],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label=cbar)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    savefig_atomic(path)


def _plot_snapshots(
    t: np.ndarray,
    x: np.ndarray,
    U_pred: np.ndarray,
    U_true: np.ndarray,
    path: Path,
    title: str,
    snap_count: int = 5,
):
    Nt = int(t.size)
    idxs = np.linspace(0, Nt - 1, snap_count, dtype=int)
    ncols = min(3, idxs.size)
    nrows = int(math.ceil(idxs.size / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, idxs):
        ax.plot(x, U_true[k], label="true", linewidth=1.5)
        ax.plot(x, U_pred[k], "--", label="pred", linewidth=1.2)
        ax.set_title(f"t = {t[k]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend(fontsize=8)

    for ax in axes[idxs.size :]:
        ax.axis("off")

    fig.suptitle(title)
    savefig_atomic(path)


def _plot_derivative_overlays(
    t: np.ndarray,
    x: np.ndarray,
    D_pred: np.ndarray,
    D_ref: np.ndarray,
    path: Path,
    title: str,
    ylabel: str,
    snap_count: int = 5,
):
    Nt = int(t.size)
    idxs = np.linspace(0, Nt - 1, snap_count, dtype=int)
    ncols = min(3, idxs.size)
    nrows = int(math.ceil(idxs.size / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, idxs):
        ax.plot(x, D_ref[k], label="FD ref", linewidth=1.5)
        ax.plot(x, D_pred[k], "--", label="model", linewidth=1.2)
        ax.set_title(f"t = {t[k]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    for ax in axes[idxs.size :]:
        ax.axis("off")

    fig.suptitle(title)
    savefig_atomic(path)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _fd_derivs_burgers(U_true: np.ndarray, dx: float):
    Nt, Nx = U_true.shape
    ux = np.zeros_like(U_true)
    uxx = np.zeros_like(U_true)
    uxxx = np.zeros_like(U_true)
    for i in range(Nt):
        row = U_true[i]
        ux[i] = fd_first_periodic(row, dx)
        uxx[i] = fd_second_periodic(row, dx)
        uxxx[i] = fd_third_periodic(row, dx)
    return ux, uxx, uxxx


def _fd_derivs_allen(U_true: np.ndarray, dx: float):
    Nt, Nx = U_true.shape
    ux = np.zeros((Nt, Nx - 2), dtype=np.float64)
    uxx = np.zeros((Nt, Nx - 2), dtype=np.float64)
    uxxx = np.zeros((Nt, Nx - 4), dtype=np.float64)
    for i in range(Nt):
        row = U_true[i]
        ux[i] = fd_first_centered(row, dx)
        uxx[i] = fd_second_centered(row, dx)
        uxxx[i] = fd_third_centered(row, dx)
    return ux, uxx, uxxx


def _metrics_row(
    *,
    cfg: RunConfig,
    model_name: str,
    metrics: dict[str, dict[str, float]],
    final_train_loss: float,
    extra: dict[str, Any] | None = None,
):
    extra = extra or {}
    row: dict[str, Any] = dict(
        dataset=cfg.dataset,
        seed=int(cfg.seed),
        model=model_name,
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        lr=float(cfg.lr),
        noise_level=float(cfg.noise_level),
        final_train_loss=float(final_train_loss),
    )
    for key in ["u", "ux", "uxx", "uxxx"]:
        for m in ["rel_l2", "rmse", "max_abs"]:
            row[f"{key}_{m}"] = float(metrics[key][m])
    row.update(extra)
    return row


def run_one_dataset(cfg: RunConfig) -> list[dict[str, Any]]:
    _seed_everything(cfg.seed)

    if cfg.dataset == "burgers":
        burg_cfg = BurgConfig(
            N=int(cfg.burgers_N),
            L=float(cfg.burgers_L),
            nu=float(cfg.burgers_nu),
            dt=float(cfg.burgers_dt),
            T=float(cfg.burgers_T),
            seed=int(cfg.seed),
        )
        
        x, _u_final, _t_end, (t_grid, U_true) = solve_burgers(burg_cfg)

        x = x.astype(np.float64)
        t_grid = t_grid.astype(np.float64)
        U_true = U_true.astype(np.float64)
        dx = float(x[1] - x[0])

        #t_train below is full t mesh subsampled by stride_t; 
        # x_train is full x mesh subsampled by stride_x; 
        # y_train is corresponding u values with noise
        # so _make_train_samples provides data to be fitted to, but on a 
        # coarser mesh than the original t_grid/x_grid that we use for evaluation and plotting
        t_train, x_train, y_train_clean, y_train = _make_train_samples(
            t_grid=t_grid,
            x_grid=x,
            u_grid=U_true,
            stride_t=cfg.stride_t,
            stride_x=cfg.stride_x,
            noise_level=cfg.noise_level,
            seed=cfg.seed,
        )

        ux_ref, uxx_ref, uxxx_ref = _fd_derivs_burgers(U_true, dx)
        x_ux = x
        x_uxx = x
        x_uxxx = x
        trim = dict(ux=(slice(None), slice(None)), uxx=(slice(None), slice(None)), uxxx=(slice(None), slice(None)))

    elif cfg.dataset == "allen_cahn":
        allen_cfg = AllenCahnConfig(
            N=int(cfg.allen_N),
            dt=float(cfg.allen_dt),
            T=float(cfg.allen_T),
            d=float(cfg.allen_d),
            reaction_scale=float(cfg.allen_reaction_scale),
            bc_value=float(cfg.allen_bc_value),
            stride_t=int(cfg.stride_t),
            stride_x=int(cfg.stride_x),
            noise_level=float(cfg.noise_level),
            seed=int(cfg.seed),
        )
        x, _u_final, _t_end, (t_grid, U_true) = solve_allen_cahn(allen_cfg, return_history=True)
        x = x.astype(np.float64)
        t_grid = t_grid.astype(np.float64)
        U_true = U_true.astype(np.float64)
        dx = float(x[1] - x[0])

        t_train, x_train, y_train_clean, y_train = _make_train_samples(
            t_grid=t_grid,
            x_grid=x,
            u_grid=U_true,
            stride_t=cfg.stride_t,
            stride_x=cfg.stride_x,
            noise_level=cfg.noise_level,
            seed=cfg.seed,
        )

        ux_ref, uxx_ref, uxxx_ref = _fd_derivs_allen(U_true, dx)
        x_ux = x[1:-1]
        x_uxx = x[1:-1]
        x_uxxx = x[2:-2]
        trim = dict(
            ux=(slice(None), slice(1, -1)),
            uxx=(slice(None), slice(1, -1)),
            uxxx=(slice(None), slice(2, -2)),
        )

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    # Normalize to [-1,1] for both models
    a_t, b_t = _affine_to_minus1_1(t_grid)
    a_x, b_x = _affine_to_minus1_1(x)

    t_train_n = _to_norm(t_train, a_t, b_t)
    x_train_n = _to_norm(x_train, a_x, b_x)

    Nt, Nx = U_true.shape
    t2d = np.repeat(t_grid[:, None], Nx, axis=1)
    x2d = np.repeat(x[None, :], Nt, axis=0)
    t_flat_n = _to_norm(t2d.reshape(-1), a_t, b_t)
    x_flat_n = _to_norm(x2d.reshape(-1), a_x, b_x)

    base_dir = Path("runs") / "derivative_compare" / cfg.dataset / f"seed_{cfg.seed:03d}"
    base_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = asdict(cfg)
    cfg_json.update({"t_norm": {"a": a_t, "b": b_t}, "x_norm": {"a": a_x, "b": b_x}})
    _write_json(base_dir / "config.json", cfg_json)

    rows: list[dict[str, Any]] = []

    models = {
        "simplemlp": SimpleMLP(n_layers=int(cfg.hidden_layers) + 1, hidden_size=int(cfg.hidden_size)),
        "siren": SirenMLP(
            hidden_size=int(cfg.hidden_size),
            hidden_layers=int(cfg.hidden_layers),
            first_omega_0=float(cfg.first_omega_0),
            hidden_omega_0=float(cfg.hidden_omega_0),
        ),
    }

    for model_name, model in models.items():
        run_dir = base_dir / model_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n== {cfg.dataset} seed={cfg.seed} model={model_name} ==")
        model, hist = fit_model_to_data(
            model,
            t_train_n, # normalized t_train
            x_train_n,
            y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            device=cfg.device,
            log_every=max(1, cfg.epochs // 10),
        )
        _write_json(run_dir / "loss_history.json", {"losses": hist.losses})

        # Predict u on dense grid (aligned with t2d/x2d)
        u_pred_flat = predict_on_grid(model, t_flat_n, x_flat_n, cfg.device)
        U_pred = u_pred_flat.reshape(Nt, Nx).astype(np.float64)

        # Autograd derivatives on dense grid, converted back to physical x-derivatives.
        derivs = _predict_and_derivs_on_grid(
            model,
            t_norm_flat=t_flat_n,
            x_norm_flat=x_flat_n,
            device=cfg.device,
            ax_scale=a_x,
            chunk_size=cfg.eval_chunk_size,
            max_order=3,
        )
        UX_pred = derivs["ux"].reshape(Nt, Nx).astype(np.float64)
        UXX_pred = derivs["uxx"].reshape(Nt, Nx).astype(np.float64)
        UXXX_pred = derivs["uxxx"].reshape(Nt, Nx).astype(np.float64)

        # Match reference shapes (Allen-Cahn uses interior-only refs)
        UX_pred_cmp = UX_pred[trim["ux"]]
        UXX_pred_cmp = UXX_pred[trim["uxx"]]
        UXXX_pred_cmp = UXXX_pred[trim["uxxx"]]

        metrics = {
            "u": compute_error_metrics(U_pred, U_true),
            "ux": compute_error_metrics(UX_pred_cmp, ux_ref),
            "uxx": compute_error_metrics(UXX_pred_cmp, uxx_ref),
            "uxxx": compute_error_metrics(UXXX_pred_cmp, uxxx_ref),
        }

        extra: dict[str, Any] = {}
        if cfg.dataset == "burgers":
            ux_ref_flat = ux_ref.reshape(-1)
            thresh = np.quantile(np.abs(ux_ref_flat), 0.9)
            mask = np.abs(ux_ref_flat) >= thresh
            extra["ux_rel_l2_highgrad"] = float(
                np.linalg.norm((UX_pred.reshape(-1)[mask] - ux_ref_flat[mask]))
                / (np.linalg.norm(ux_ref_flat[mask]) + 1e-12)
            )
            uxx_ref_flat = uxx_ref.reshape(-1)
            extra["uxx_rel_l2_highgrad"] = float(
                np.linalg.norm((UXX_pred.reshape(-1)[mask] - uxx_ref_flat[mask]))
                / (np.linalg.norm(uxx_ref_flat[mask]) + 1e-12)
            )
            uxxx_ref_flat = uxxx_ref.reshape(-1)
            extra["uxxx_rel_l2_highgrad"] = float(
                np.linalg.norm((UXXX_pred.reshape(-1)[mask] - uxxx_ref_flat[mask]))
                / (np.linalg.norm(uxxx_ref_flat[mask]) + 1e-12)
            )

        _write_json(run_dir / "metrics.json", {"metrics": metrics, "extra": extra})

        # Plots
        _plot_loss(hist.losses, run_dir / "loss_curve.pdf", title=f"{model_name} train loss")
        _plot_heatmap(t_grid, x, U_pred, run_dir / "fit_heatmap.pdf", title=f"{model_name} u_pred(t,x)", cbar="u_pred")
        _plot_snapshots(
            t_grid,
            x,
            U_pred,
            U_true,
            run_dir / "fit_snapshots.pdf",
            title=f"{model_name} snapshots",
            snap_count=5,
        )

        _plot_derivative_overlays(
            t_grid,
            x_ux,
            UX_pred_cmp,
            ux_ref,
            run_dir / "ux_overlay.pdf",
            title=f"{model_name} u_x overlays",
            ylabel="u_x",
            snap_count=5,
        )
        _plot_derivative_overlays(
            t_grid,
            x_uxx,
            UXX_pred_cmp,
            uxx_ref,
            run_dir / "uxx_overlay.pdf",
            title=f"{model_name} u_xx overlays",
            ylabel="u_xx",
            snap_count=5,
        )
        _plot_derivative_overlays(
            t_grid,
            x_uxxx,
            UXXX_pred_cmp,
            uxxx_ref,
            run_dir / "uxxx_overlay.pdf",
            title=f"{model_name} u_xxx overlays",
            ylabel="u_xxx",
            snap_count=5,
        )

        # Error heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        axes = axes.ravel()

        axes[0].imshow(
            np.abs(U_pred - U_true),
            extent=[float(x.min()), float(x.max()), float(t_grid.min()), float(t_grid.max())],
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axes[0].set_title("|u_pred - u_true|")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("t")

        axes[1].imshow(
            np.abs(UX_pred_cmp - ux_ref),
            extent=[float(x_ux.min()), float(x_ux.max()), float(t_grid.min()), float(t_grid.max())],
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axes[1].set_title("|ux_pred - ux_fd|")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")

        axes[2].imshow(
            np.abs(UXX_pred_cmp - uxx_ref),
            extent=[float(x_uxx.min()), float(x_uxx.max()), float(t_grid.min()), float(t_grid.max())],
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axes[2].set_title("|uxx_pred - uxx_fd|")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("t")

        axes[3].imshow(
            np.abs(UXXX_pred_cmp - uxxx_ref),
            extent=[float(x_uxxx.min()), float(x_uxxx.max()), float(t_grid.min()), float(t_grid.max())],
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axes[3].set_title("|uxxx_pred - uxxx_fd|")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("t")

        savefig_atomic(run_dir / "error_heatmaps.pdf")

        rows.append(
            _metrics_row(
                cfg=cfg,
                model_name=model_name,
                metrics=metrics,
                final_train_loss=float(hist.losses[-1] if hist.losses else float("nan")),
                extra=extra,
            )
        )

    # Dataset-level summary
    _write_csv(base_dir / "summary.csv", rows, SUMMARY_FIELDS)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["burgers", "allen_cahn"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--hidden_layers", type=int, default=3)
    p.add_argument("--first_omega_0", type=float, default=30.0)
    p.add_argument("--hidden_omega_0", type=float, default=30.0)
    p.add_argument("--noise_level", type=float, default=0.05)
    p.add_argument("--stride_t", type=int, default=1)
    p.add_argument("--stride_x", type=int, default=1)
    p.add_argument("--eval_chunk_size", type=int, default=4096)

    # Burgers params
    p.add_argument("--burgers_N", type=int, default=256)
    p.add_argument("--burgers_L", type=float, default=2 * np.pi)
    p.add_argument("--burgers_nu", type=float, default=0.02)
    p.add_argument("--burgers_dt", type=float, default=2e-3)
    p.add_argument("--burgers_T", type=float, default=1.0)

    # Allen-Cahn params
    p.add_argument("--allen_N", type=int, default=201)
    p.add_argument("--allen_dt", type=float, default=0.01)
    p.add_argument("--allen_T", type=float, default=1.0)
    p.add_argument("--allen_d", type=float, default=0.001)
    p.add_argument("--allen_reaction_scale", type=float, default=5.0)
    p.add_argument("--allen_bc_value", type=float, default=-1.0)

    args = p.parse_args()

    all_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        dataset_rows: list[dict[str, Any]] = []
        for seed in args.seeds:
            cfg = RunConfig(
                dataset=str(dataset),
                seed=int(seed),
                device=str(args.device),
                epochs=int(160 if dataset == "burgers" else 500), 
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                hidden_size=int(args.hidden_size),
                hidden_layers=int(args.hidden_layers),
                first_omega_0=float(args.first_omega_0),
                hidden_omega_0=float(args.hidden_omega_0),
                noise_level=float(args.noise_level),
                stride_t=int(args.stride_t),
                stride_x=int(args.stride_x),
                burgers_N=int(args.burgers_N),
                burgers_L=float(args.burgers_L),
                burgers_nu=float(args.burgers_nu),
                burgers_dt=float(args.burgers_dt),
                burgers_T=float(args.burgers_T),
                allen_N=int(args.allen_N),
                allen_dt=float(args.allen_dt),
                allen_T=float(args.allen_T),
                allen_d=float(args.allen_d),
                allen_reaction_scale=float(args.allen_reaction_scale),
                allen_bc_value=float(args.allen_bc_value),
                eval_chunk_size=int(args.eval_chunk_size),
            )
            seed_rows = run_one_dataset(cfg)
            #metadata stuff
            dataset_rows.extend(seed_rows)
            all_rows.extend(seed_rows)

        # Dataset-level aggregation (across seeds)
        ds_out = Path("runs") / "derivative_compare" / str(dataset) / "summary.csv"
        if dataset_rows:
            _write_csv(ds_out, dataset_rows, SUMMARY_FIELDS)
            print(f"Wrote {ds_out}")

    out_path = Path("runs") / "derivative_compare" / "all_results.csv"
    if all_rows:
        _write_csv(out_path, all_rows, SUMMARY_FIELDS)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
