from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from prog import hlprs
from prog.hlprs import savefig_atomic
from prog.mlps import SirenMLP, SimpleMLP
from utils.derivative_utils import (
    autograd_spatial_derivatives,
    compute_error_metrics,
    fd_first_periodic,
    fd_second_periodic,
    fd_third_periodic,
)
from utils.extract_pde_ls import extract_pde_ls
from utils.fit_utils import fit_model_to_data
from utils.tv_utils import available_tv_types, dispatch_tv


DEFAULT_LAMBDAS = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]


def _fmt_value_for_path(v: float) -> str:
    s = f"{float(v):.6g}"
    return s.replace(".", "p").replace("-", "m").replace("+", "")


def _seed_everything(seed: int) -> None:
    '''
    seeding np.random, torch
    '''
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _affine_to_minus1_1(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, dtype=np.float64)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        raise ValueError("Cannot normalize: invalid range")
    a = 0.5 * (vmax - vmin)
    b = 0.5 * (vmax + vmin)
    return a, b


def _to_norm(v: np.ndarray, a: float, b: float) -> np.ndarray:
    return (np.asarray(v, dtype=np.float64) - float(b)) / float(a)


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
    stride_t = max(1, int(stride_t))
    stride_x = max(1, int(stride_x))

    rows = np.arange(t_grid.size)[::stride_t]
    cols = np.arange(x_grid.size)[::stride_x]

    t2d = np.repeat(t_grid[:, None], x_grid.size, axis=1)
    x2d = np.repeat(x_grid[None, :], t_grid.size, axis=0)
    u2d = u_grid

    t_s = t2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)
    x_s = x2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)
    y_s = u2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)

    if float(noise_level) > 0:
        rng = np.random.default_rng(int(seed))
        sigma = float(noise_level) * float(np.std(y_s))
        y_noisy = (y_s + sigma * rng.standard_normal(size=y_s.shape)).astype(np.float32)
    else:
        y_noisy = y_s

    return t_s, x_s, y_s, y_noisy


class PhysCoordWrapper(torch.nn.Module):
    """
    Wrap a model trained on normalized coords so we can evaluate on physical (t,x),
    while keeping autograd derivatives in physical units via torch-side normalization.
    """

    def __init__(self, base_model: torch.nn.Module, *, a_t: float, b_t: float, a_x: float, b_x: float):
        super().__init__()
        self.base_model = base_model
        self.a_t = float(a_t)
        self.b_t = float(b_t)
        self.a_x = float(a_x)
        self.b_x = float(b_x)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_n = (t - self.b_t) / self.a_t
        x_n = (x - self.b_x) / self.a_x
        return self.base_model(t_n, x_n)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _autograd_x_derivs_on_grid(
    model: torch.nn.Module,
    *,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    device: str,
    chunk_size: int,
    max_order: int = 3,
) -> dict[str, np.ndarray]:
    t_grid = np.asarray(t_grid, dtype=np.float32).reshape(-1)
    x_grid = np.asarray(x_grid, dtype=np.float32).reshape(-1)
    Nt = int(t_grid.size)
    Nx = int(x_grid.size)

    t2d = np.repeat(t_grid[:, None], Nx, axis=1)
    x2d = np.repeat(x_grid[None, :], Nt, axis=0)

    t_flat = t2d.reshape(-1, 1)
    x_flat = x2d.reshape(-1, 1)

    chunk_size = max(1, int(chunk_size))

    out: dict[str, list[np.ndarray]] = {"u": [], "ux": [], "uxx": [], "uxxx": []}
    model = model.to(device)
    model.eval()

    for s in range(0, t_flat.shape[0], chunk_size):
        e = min(t_flat.shape[0], s + chunk_size)
        t_b = torch.from_numpy(t_flat[s:e]).to(device)
        x_b = torch.from_numpy(x_flat[s:e]).to(device).requires_grad_(True)
        with torch.enable_grad():
            derivs = autograd_spatial_derivatives(model, t_b, x_b, max_order=max_order)
        out["u"].append(derivs["u"].detach().cpu().numpy().reshape(-1))
        out["ux"].append(derivs.get("ux", derivs["u"].new_zeros(derivs["u"].shape)).detach().cpu().numpy().reshape(-1))
        out["uxx"].append(
            derivs.get("uxx", derivs["u"].new_zeros(derivs["u"].shape)).detach().cpu().numpy().reshape(-1)
        )
        out["uxxx"].append(
            derivs.get("uxxx", derivs["u"].new_zeros(derivs["u"].shape)).detach().cpu().numpy().reshape(-1)
        )

    out_np = {k: np.concatenate(v, axis=0).reshape(Nt, Nx).astype(np.float64) for k, v in out.items()}
    return out_np


def _fd_derivs_burgers(u_grid: np.ndarray, x_grid: np.ndarray) -> dict[str, np.ndarray]:
    U = np.asarray(u_grid, dtype=np.float64)
    x = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    dx = float(x[1] - x[0])

    Ux = np.stack([fd_first_periodic(U[i], dx) for i in range(U.shape[0])], axis=0)
    Uxx = np.stack([fd_second_periodic(U[i], dx) for i in range(U.shape[0])], axis=0)
    Uxxx = np.stack([fd_third_periodic(U[i], dx) for i in range(U.shape[0])], axis=0)
    return {"u": U, "ux": Ux, "uxx": Uxx, "uxxx": Uxxx}


def _plot_vs_lambda(
    *,
    run_dir: Path,
    df_rows: list[dict[str, Any]],
    x_key: str,
    y_keys: list[str],
    title: str,
    out_name: str,
) -> None:
    xs = []
    ys: dict[str, list[float]] = {k: [] for k in y_keys}
    for r in df_rows:
        if int(r.get("status", 0)) != 1:
            continue
        xs.append(float(r[x_key]))
        for k in y_keys:
            ys[k].append(float(r.get(k, float("nan"))))
    if not xs:
        return

    order = np.argsort(np.asarray(xs))
    xs_s = np.asarray(xs, dtype=float)[order]
    plt.figure(figsize=(6, 4))
    for k in y_keys:
        y = np.asarray(ys[k], dtype=float)[order]
        plt.plot(xs_s, y, marker="o", label=k)
    plt.xscale("symlog", linthresh=1e-10)
    plt.xlabel(x_key)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    savefig_atomic(run_dir / out_name)


def _plot_derivative_overlays(
    *,
    run_dir: Path,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    pred: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    snap_no: int = 5,
    max_cols: int = 3,
) -> None:
    """
    Per-run visual: overlay predicted vs FD derivatives for selected times.

    Creates `derivative_overlays.pdf` in `run_dir`.
    """
    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    Nt = int(t_grid.size)
    snap_no = max(1, int(snap_no))
    idxs = np.linspace(0, Nt - 1, snap_no, dtype=int) if Nt > 1 else np.array([0], dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), constrained_layout=True)
    deriv_keys = [("ux", "u_x"), ("uxx", "u_xx"), ("uxxx", "u_xxx")]
    for ax, (k, title) in zip(axes, deriv_keys):
        for i in idxs:
            ax.plot(x_grid, ref[k][i], linestyle="--", linewidth=1.0, alpha=0.75, label=f"fd t={t_grid[i]:.3g}" if k == "ux" else None)
            ax.plot(x_grid, pred[k][i], linewidth=1.0, alpha=0.9, label=f"pred t={t_grid[i]:.3g}" if k == "ux" else None)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel(k)
        ax.grid(True, alpha=0.25)

    # only one legend to keep clutter down
    axes[0].legend(fontsize=7, ncol=min(int(max_cols), max(1, int(snap_no))))
    savefig_atomic(Path(run_dir) / "derivative_overlays.pdf")


@dataclass
class RunConfig:
    dataset: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    lr: float
    model: str
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
    tv_type: str
    tv_lambda: float
    eval_chunk_size: int


SUMMARY_FIELDS = [
    "dataset",
    "seed",
    "model",
    "epochs",
    "batch_size",
    "lr",
    "noise_level",
    "stride_t",
    "stride_x",
    "tv_type",
    "tv_lambda",
    "final_total_loss",
    "final_data_loss",
    "final_tv_loss",
    "ux_rel_l2",
    "uxx_rel_l2",
    "uxxx_rel_l2",
    "l2_coeff_error",
    "status",
    "error",
]


def _make_model(cfg: RunConfig) -> torch.nn.Module:
    m = str(cfg.model).lower()
    if m == "siren":
        return SirenMLP(
            hidden_size=int(cfg.hidden_size),
            hidden_layers=int(cfg.hidden_layers),
            first_omega_0=float(cfg.first_omega_0),
            hidden_omega_0=float(cfg.hidden_omega_0),
        )
    if m == "mlp":
        return SimpleMLP(n_layers=int(cfg.hidden_layers), hidden_size=int(cfg.hidden_size))
    raise ValueError(f"Unknown model='{cfg.model}' (expected 'siren' or 'mlp').")


def run_one(cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", asdict(cfg))

    try:
        _seed_everything(int(cfg.seed))

        if str(cfg.dataset).lower() != "burgers":
            raise ValueError("This sweep runner currently supports dataset='burgers' only.")

        # dataset
        x_grid, _u_final, _t_end, (t_grid, u_grid) = solve_burgers(
            N=int(cfg.burgers_N),
            L=float(cfg.burgers_L),
            nu=float(cfg.burgers_nu),
            dt=float(cfg.burgers_dt),
            T=float(cfg.burgers_T),
            seed=int(cfg.seed),
            return_history=True,
        )

        t_grid = np.asarray(t_grid, dtype=np.float64)
        x_grid = np.asarray(x_grid, dtype=np.float64)
        u_grid = np.asarray(u_grid, dtype=np.float64)
        if not np.isfinite(u_grid).all():
            raise ValueError(
                "Burgers solver returned non-finite values (nan/inf). "
                "Try adjusting --burgers_dt / --burgers_nu / --burgers_N."
            )

        # Training samples (physical coords)
        t_train, x_train, y_clean, y_noisy = _make_train_samples(
            t_grid=t_grid,
            x_grid=x_grid,
            u_grid=u_grid,
            stride_t=int(cfg.stride_t),
            stride_x=int(cfg.stride_x),
            noise_level=float(cfg.noise_level),
            seed=int(cfg.seed),
        )
        if not (np.isfinite(t_train).all() and np.isfinite(x_train).all() and np.isfinite(y_noisy).all()):
            raise ValueError("Training samples contain non-finite values (nan/inf).")

        # Normalize coords for training stability
        a_t, b_t = _affine_to_minus1_1(t_grid)
        a_x, b_x = _affine_to_minus1_1(x_grid)
        t_train_n = _to_norm(t_train, a_t, b_t).astype(np.float32)
        x_train_n = _to_norm(x_train, a_x, b_x).astype(np.float32)

        base_model = _make_model(cfg)

        # training loop
        tv_terms = None
        if float(cfg.tv_lambda) != 0.0:
            tv_fn = dispatch_tv(cfg.tv_type)
            tv_terms = [(float(cfg.tv_lambda), tv_fn)]

        base_model, hist = fit_model_to_data(
            base_model,
            t_train_n,
            x_train_n,
            y_noisy,
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            lr=float(cfg.lr),
            device=str(cfg.device),
            log_every=max(1, int(cfg.epochs) // 10),
            tv_terms=tv_terms,
        )
        if (not hist.losses) or (not np.isfinite(np.asarray(hist.losses, dtype=np.float64)).all()):
            raise ValueError("Training produced non-finite loss values (nan/inf).")

        # History CSV
        history_rows = []
        if hist.rows:
            for r in hist.rows:
                history_rows.append(
                    {
                        "epoch": int(r.epoch),
                        "total_loss": float(r.total_loss),
                        "data_loss": float(r.data_loss),
                        "pde_loss": float(r.pde_loss),
                        "tv_loss": float(r.tv_loss),
                    }
                )
        else:
            for i, l in enumerate(hist.losses):
                history_rows.append({"epoch": int(i), "total_loss": float(l), "data_loss": float("nan"), "pde_loss": 0.0, "tv_loss": float("nan")})
        _write_csv(run_dir / "history.csv", history_rows, ["epoch", "total_loss", "data_loss", "pde_loss", "tv_loss"])

        # Wrap for physical evaluation + physical-unit derivatives
        phys_model = PhysCoordWrapper(base_model, a_t=a_t, b_t=b_t, a_x=a_x, b_x=b_x)

        # Snapshot helper plots (best-effort)
        try:
            fig, fig_hm, _payload = hlprs.snapshot_comp(
                phys_model,
                int(cfg.stride_x),
                int(cfg.stride_t),
                y_noisy,
                y_clean,
                t_train,
                x_train,
                snap_no=5,
            )
            fig.savefig(run_dir / "fit_snapshots.pdf")
            fig_hm.savefig(run_dir / "fit_heatmap.pdf")
        except Exception as e:
            print(f"[warn] snapshot_comp failed: {e}")

        # Derivative error metrics (autograd vs FD control on clean grid)
        pred_derivs = _autograd_x_derivs_on_grid(
            phys_model,
            t_grid=t_grid,
            x_grid=x_grid,
            device=str(cfg.device),
            chunk_size=int(cfg.eval_chunk_size),
            max_order=3,
        )
        ref_derivs = _fd_derivs_burgers(u_grid, x_grid)

        # Derivative overlay plots (best-effort)
        try:
            _plot_derivative_overlays(
                run_dir=run_dir,
                t_grid=t_grid,
                x_grid=x_grid,
                pred=pred_derivs,
                ref=ref_derivs,
                snap_no=1,
            )
            print(f"Saved derivative overlay plots to {run_dir / 'derivative_overlays.pdf'}")
        except Exception as e:
            print(f"[warn] derivative overlay plot failed: {e}")

        ux_m = compute_error_metrics(pred_derivs["ux"], ref_derivs["ux"])
        uxx_m = compute_error_metrics(pred_derivs["uxx"], ref_derivs["uxx"])
        uxxx_m = compute_error_metrics(pred_derivs["uxxx"], ref_derivs["uxxx"])

        # PDE extraction via least squares on the full grid
        Nt = int(t_grid.size)
        Nx = int(x_grid.size)
        t2d = np.repeat(t_grid[:, None], Nx, axis=1)
        x2d = np.repeat(x_grid[None, :], Nt, axis=0)
        t_flat = t2d.reshape(-1)
        x_flat = x2d.reshape(-1)
        feature_terms = ["u", "u_x", "u_xx", "uu_x"]
        pde = extract_pde_ls(phys_model, t_flat, x_flat, feature_terms, device=str(cfg.device))

        true_coeffs = np.array([0.0, 0.0, float(cfg.burgers_nu), -1.0], dtype=float)
        coeffs = np.asarray(pde["coeffs"], dtype=float).reshape(-1)
        l2_coeff_error = float(np.linalg.norm(coeffs - true_coeffs))

        final_row = hist.rows[-1] if hist.rows else None
        summary = {
            "dataset": str(cfg.dataset),
            "seed": int(cfg.seed),
            "model": str(cfg.model),
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr),
            "noise_level": float(cfg.noise_level),
            "stride_t": int(cfg.stride_t),
            "stride_x": int(cfg.stride_x),
            "tv_type": str(cfg.tv_type),
            "tv_lambda": float(cfg.tv_lambda),
            "final_total_loss": float(final_row.total_loss if final_row else (hist.losses[-1] if hist.losses else float("nan"))),
            "final_data_loss": float(final_row.data_loss if final_row else float("nan")),
            "final_tv_loss": float(final_row.tv_loss if final_row else float("nan")),
            "ux_rel_l2": float(ux_m["rel_l2"]),
            "uxx_rel_l2": float(uxx_m["rel_l2"]),
            "uxxx_rel_l2": float(uxxx_m["rel_l2"]),
            "l2_coeff_error": float(l2_coeff_error),
            "status": 1,
            "error": "",
            "pde_names": list(pde["names"]),
            "pde_coeffs": coeffs.tolist(),
            "true_coeffs": true_coeffs.tolist(),
        }

        _write_json(run_dir / "summary.json", summary)

        # Lightweight sweep visuals (optional, best-effort)
        try:
            plt.figure(figsize=(6, 4))
            plt.plot([r["epoch"] for r in history_rows], [r["total_loss"] for r in history_rows], label="total")
            plt.plot([r["epoch"] for r in history_rows], [r["data_loss"] for r in history_rows], label="data")
            plt.plot([r["epoch"] for r in history_rows], [r["tv_loss"] for r in history_rows], label="tv")
            plt.yscale("log")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"{cfg.tv_type}  lambda={cfg.tv_lambda:g}  seed={cfg.seed}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            savefig_atomic(run_dir / "loss_curves.pdf")
        except Exception:
            pass

        return summary
    except Exception as e:
        summary = {
            "dataset": str(cfg.dataset),
            "seed": int(cfg.seed),
            "model": str(cfg.model),
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr),
            "noise_level": float(cfg.noise_level),
            "stride_t": int(cfg.stride_t),
            "stride_x": int(cfg.stride_x),
            "tv_type": str(cfg.tv_type),
            "tv_lambda": float(cfg.tv_lambda),
            "final_total_loss": float("nan"),
            "final_data_loss": float("nan"),
            "final_tv_loss": float("nan"),
            "ux_rel_l2": float("nan"),
            "uxx_rel_l2": float("nan"),
            "uxxx_rel_l2": float("nan"),
            "l2_coeff_error": float("nan"),
            "status": 0,
            "error": repr(e),
        }
        _write_json(run_dir / "summary.json", summary)
        return summary


def main() -> None:
    p = argparse.ArgumentParser(description="TV regularization lambda sweep (Burgers)")
    p.add_argument("--dataset", default="burgers")
    p.add_argument("--tv_types", nargs="+", default=list(available_tv_types()))
    p.add_argument("--tv_lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model", default="siren", choices=["siren"])
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--hidden_layers", type=int, default=3)
    p.add_argument("--first_omega_0", type=float, default=30.0)
    p.add_argument("--hidden_omega_0", type=float, default=30.0)
    p.add_argument("--noise_level", type=float, default=0.05)
    p.add_argument("--stride_t", type=int, default=1)
    p.add_argument("--stride_x", type=int, default=1)
    p.add_argument("--eval_chunk_size", type=int, default=4096)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    # Burgers params
    p.add_argument("--burgers_N", type=int, default=256)
    p.add_argument("--burgers_L", type=float, default=2 * np.pi)
    p.add_argument("--burgers_nu", type=float, default=0.02)
    p.add_argument("--burgers_dt", type=float, default=2e-3)
    p.add_argument("--burgers_T", type=float, default=1.0)

    args = p.parse_args()

    all_rows: list[dict[str, Any]] = []

    for tv_type in args.tv_types:
        tv_type = str(tv_type)
        root = Path("runs") / f"{tv_type}_lambda_sweep"
        root.mkdir(parents=True, exist_ok=True)

        for tv_lambda in args.tv_lambdas:
            for seed in args.seeds:
                cfg = RunConfig(
                    dataset=str(args.dataset),
                    seed=int(seed),
                    device=str(args.device),
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    model=str(args.model),
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
                    tv_type=str(tv_type),
                    tv_lambda=float(tv_lambda),
                    eval_chunk_size=int(args.eval_chunk_size),
                )

                run_dir = root / f"tv_lambda_{_fmt_value_for_path(float(tv_lambda))}" / f"seed_{int(seed):03d}"
                if args.dry_run:
                    print(f"[dry_run] {run_dir}")
                    continue

                summary_path = run_dir / "summary.json"
                if (not args.overwrite) and summary_path.exists():
                    try:
                        prev = json.loads(summary_path.read_text(encoding="utf-8"))
                        if int(prev.get("status", 0)) == 1:
                            all_rows.append(prev)
                            continue
                    except Exception:
                        pass

                summary = run_one(cfg, run_dir)
                all_rows.append(summary)

        # Per-tv_type sweep summary + plots
        tv_rows = [r for r in all_rows if r.get("tv_type") == tv_type]
        if tv_rows:
            _write_csv(root / "summary.csv", tv_rows, SUMMARY_FIELDS)
            _plot_vs_lambda(
                run_dir=root,
                df_rows=tv_rows,
                x_key="tv_lambda",
                y_keys=["final_data_loss", "final_tv_loss"],
                title=f"{tv_type}: final losses vs tv_lambda",
                out_name="loss_vs_lambda.pdf",
            )
            _plot_vs_lambda(
                run_dir=root,
                df_rows=tv_rows,
                x_key="tv_lambda",
                y_keys=["ux_rel_l2", "uxx_rel_l2", "uxxx_rel_l2"],
                title=f"{tv_type}: derivative error vs tv_lambda",
                out_name="deriv_error_vs_lambda.pdf",
            )
            _plot_vs_lambda(
                run_dir=root,
                df_rows=tv_rows,
                x_key="tv_lambda",
                y_keys=["l2_coeff_error"],
                title=f"{tv_type}: PDE coeff error vs tv_lambda",
                out_name="coeff_error_vs_lambda.pdf",
            )
            print(f"Wrote {root / 'summary.csv'}")

    # Global summary
    if (not args.dry_run) and all_rows:
        out_path = Path("runs") / "tv_lambda_sweep_all_results.csv"
        _write_csv(out_path, all_rows, SUMMARY_FIELDS)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
