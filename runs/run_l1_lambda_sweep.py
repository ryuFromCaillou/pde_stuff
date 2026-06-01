from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # optional dependency
    plt = None
import numpy as np
import torch

from Datasets.data.processed.allenc_gen.allen_cahn_gen import AllenCahnConfig, solve_allen_cahn
from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from Datasets.data.processed.heat_gen.heat_gen import HeatConfig, solve_heat
from prog import hlprs
from prog.hlprs import savefig_atomic
from prog.mlps import SirenMLP
from utils.derivative_utils import compute_error_metrics, fd_first_periodic, fd_second_periodic, fd_third_periodic
from utils.extract_pde_ls import extract_pde_ls
from utils.fit_utils import fit_model_to_data
from utils.tv_utils import available_tv_types, dispatch_tv

DEFAULT_L1_LAMBDAS_GENERIC = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
DEFAULT_L1_LAMBDAS_BURGERS = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]


def _fmt_value_for_path(v: float) -> str:
    s = f"{float(v):.6g}"
    return s.replace(".", "p").replace("-", "m").replace("+", "")


def _seed_everything(seed: int) -> None:
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
    model = model.to(device)
    model.eval()

    out_u = np.zeros((Nt * Nx,), dtype=np.float32)
    out_ux = np.zeros((Nt * Nx,), dtype=np.float32)
    out_uxx = np.zeros((Nt * Nx,), dtype=np.float32)
    out_uxxx = np.zeros((Nt * Nx,), dtype=np.float32)

    n = int(t_flat.shape[0])
    for i0 in range(0, n, chunk_size):
        i1 = min(n, i0 + chunk_size)
        t_b = torch.from_numpy(t_flat[i0:i1]).to(device).requires_grad_(True)
        x_b = torch.from_numpy(x_flat[i0:i1]).to(device).requires_grad_(True)
        with torch.enable_grad():
            u = model(t_b, x_b)
            ux = torch.autograd.grad(u, x_b, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            uxx = torch.autograd.grad(ux, x_b, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
            uxxx = torch.autograd.grad(uxx, x_b, grad_outputs=torch.ones_like(uxx), create_graph=True)[0]

        out_u[i0:i1] = u.detach().cpu().numpy().reshape(-1)
        out_ux[i0:i1] = ux.detach().cpu().numpy().reshape(-1)
        out_uxx[i0:i1] = uxx.detach().cpu().numpy().reshape(-1)
        out_uxxx[i0:i1] = uxxx.detach().cpu().numpy().reshape(-1)

    return {
        "u": out_u.reshape(Nt, Nx),
        "ux": out_ux.reshape(Nt, Nx),
        "uxx": out_uxx.reshape(Nt, Nx),
        "uxxx": out_uxxx.reshape(Nt, Nx),
    }


def _fd_derivs_periodic(u_grid: np.ndarray, x_grid: np.ndarray) -> dict[str, np.ndarray]:
    u_grid = np.asarray(u_grid, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    dx = float(x_grid[1] - x_grid[0]) if x_grid.size >= 2 else 1.0

    ux = fd_first_periodic(u_grid, dx)
    uxx = fd_second_periodic(u_grid, dx)
    uxxx = fd_third_periodic(u_grid, dx)
    return {"ux": ux, "uxx": uxx, "uxxx": uxxx}


def _make_model(cfg: "RunConfig") -> torch.nn.Module:
    if str(cfg.model).lower() == "siren":
        return SirenMLP(
            hidden_size=int(cfg.hidden_size),
            hidden_layers=int(cfg.hidden_layers),
            first_omega_0=float(cfg.first_omega_0),
            hidden_omega_0=float(cfg.hidden_omega_0),
        )
    raise ValueError(f"Unknown model='{cfg.model}'")


def _plot_vs_lambda(
    *,
    run_dir: Path,
    df_rows: list[dict[str, Any]],
    x_key: str,
    y_keys: list[str],
    title: str,
    out_name: str,
) -> None:
    try:
        if plt is None:
            return
        rows_ok = [r for r in df_rows if int(r.get("status", 0)) == 1]
        if not rows_ok:
            return
        xs = [float(r.get(x_key, float("nan"))) for r in rows_ok]
        plt.figure(figsize=(7, 4))
        for yk in y_keys:
            ys = [float(r.get(yk, float("nan"))) for r in rows_ok]
            plt.plot(xs, ys, marker="o", label=yk)
        plt.xscale("symlog", linthresh=1e-12)
        plt.yscale("symlog", linthresh=1e-12)
        plt.xlabel(x_key)
        plt.ylabel("metric")
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        savefig_atomic(Path(run_dir) / out_name)
    except Exception:
        pass


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

    # dataset params
    burgers_N: int = 256
    burgers_L: float = 2.0
    burgers_nu: float = 0.01
    burgers_dt: float = 1e-3
    burgers_T: float = 1.0

    allen_N: int = 256
    allen_x_min: float = -1.0
    allen_x_max: float = 1.0
    allen_dt: float = 1e-3
    allen_T: float = 1.0
    allen_d: float = 0.1
    allen_reaction_scale: float = 1.0
    allen_bc_value: float = 0.0

    heat_N: int = 256
    heat_L: float = 2.0
    heat_dt: float = 1e-3
    heat_T: float = 1.0
    heat_alpha: float = 0.01
    heat_ic_modes: int = 8

    # regularizers
    tv_type: str = "tv_u"
    tv_lambda: float = 0.0
    l1_lambda: float = 0.0  # used as lam_sparse_eql in fit_utils.fit_model_to_data
    sparse_eql_s: float = 1e-3

    eval_chunk_size: int = 20000


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
    "l1_lambda",
    "sparse_eql_s",
    "final_total_loss",
    "final_data_loss",
    "final_tv_loss",
    "final_sparse_eql_loss",
    "ux_rel_l2",
    "uxx_rel_l2",
    "uxxx_rel_l2",
    "l2_coeff_error",
    "status",
    "error",
    "pde_names",
    "pde_coeffs",
    "true_coeffs",
]


def run_one(cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", asdict(cfg))

    try:
        _seed_everything(int(cfg.seed))
        dataset = str(cfg.dataset).lower().strip()

        if dataset in {"burgers", "burger"}:
            x_grid, u_final, t_final, (t_grid, u_grid) = solve_burgers(
                N=int(cfg.burgers_N),
                L=float(cfg.burgers_L),
                nu=float(cfg.burgers_nu),
                dt=float(cfg.burgers_dt),
                T=float(cfg.burgers_T),
                return_history=True,
            )
        elif dataset in {"allen_cahn", "allen-cahn", "allencahn", "allen"}:
            allen_cfg = AllenCahnConfig(
                N=int(cfg.allen_N),
                x_min=float(cfg.allen_x_min),
                x_max=float(cfg.allen_x_max),
                dt=float(cfg.allen_dt),
                T=float(cfg.allen_T),
                d=float(cfg.allen_d),
                reaction_scale=float(cfg.allen_reaction_scale),
                bc_value=float(cfg.allen_bc_value),
                seed=int(cfg.seed),
            )
            x_grid, u_final, t_final, (t_grid, u_grid) = solve_allen_cahn(allen_cfg, return_history=True)
        elif dataset in {"heat"}:
            heat_cfg = HeatConfig(
                N=int(cfg.heat_N),
                L=float(cfg.heat_L),
                dt=float(cfg.heat_dt),
                T=float(cfg.heat_T),
                alpha=float(cfg.heat_alpha),
                seed=int(cfg.seed),
                ic_modes=int(cfg.heat_ic_modes),
            )
            x_grid, u_final, t_final, (t_grid, u_grid) = solve_heat(heat_cfg, return_history=True)
        else:
            raise ValueError("Unknown dataset. Use --dataset burgers, --dataset allen_cahn, or --dataset heat.")

        t_grid = np.asarray(t_grid, dtype=np.float64)
        x_grid = np.asarray(x_grid, dtype=np.float64)
        u_grid = np.asarray(u_grid, dtype=np.float64)
        if not np.isfinite(u_grid).all():
            raise ValueError(f"{dataset} solver returned non-finite values (nan/inf).")

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

        # Normalize coords for siren training stability
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
            lam_sparse_eql=float(cfg.l1_lambda),
            sparse_eql_s=float(cfg.sparse_eql_s),
        )
        if (not hist.losses) or (not np.isfinite(np.asarray(hist.losses, dtype=np.float64)).all()):
            raise ValueError("Training produced non-finite loss values (nan/inf).")

        # History CSV
        history_rows: list[dict[str, Any]] = []
        if hist.rows:
            for r in hist.rows:
                history_rows.append(
                    {
                        "epoch": int(r.epoch),
                        "total_loss": float(r.total_loss),
                        "data_loss": float(r.data_loss),
                        "pde_loss": float(r.pde_loss),
                        "tv_loss": float(r.tv_loss),
                        "l1_data_loss": float(r.l1_data_loss),
                        "sparse_eql_loss": float(r.sparse_eql_loss),
                    }
                )
        else:
            for i, l in enumerate(hist.losses):
                history_rows.append(
                    {
                        "epoch": int(i),
                        "total_loss": float(l),
                        "data_loss": float("nan"),
                        "pde_loss": 0.0,
                        "tv_loss": float("nan"),
                        "l1_data_loss": float("nan"),
                        "sparse_eql_loss": float("nan"),
                    }
                )

        _write_csv(
            run_dir / "history.csv",
            history_rows,
            ["epoch", "total_loss", "data_loss", "pde_loss", "tv_loss", "l1_data_loss", "sparse_eql_loss"],
        )

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
        ref_derivs = _fd_derivs_periodic(u_grid, x_grid)

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

        if dataset in {"burgers", "burger"}:
            feature_terms = ["u", "u_x", "u_xx", "uu_x"]
            pde = extract_pde_ls(phys_model, t_flat, x_flat, feature_terms, device=str(cfg.device))
            true_coeffs = np.array([0.0, 0.0, float(cfg.burgers_nu), -1.0], dtype=float)
        elif dataset in {"allen_cahn", "allen-cahn", "allencahn", "allen"}:
            feature_terms = ["u", "u_xx", "u3"]
            pde = extract_pde_ls(phys_model, t_flat, x_flat, feature_terms, device=str(cfg.device))
            r = float(cfg.allen_reaction_scale)
            true_coeffs = np.array([r, float(cfg.allen_d), -r], dtype=float)
        else:
            feature_terms = ["u_xx"]
            pde = extract_pde_ls(phys_model, t_flat, x_flat, feature_terms, device=str(cfg.device))
            true_coeffs = np.array([float(cfg.heat_alpha)], dtype=float)

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
            "l1_lambda": float(cfg.l1_lambda),
            "sparse_eql_s": float(cfg.sparse_eql_s),
            "final_total_loss": float(
                final_row.total_loss if final_row else (hist.losses[-1] if hist.losses else float("nan"))
            ),
            "final_data_loss": float(final_row.data_loss if final_row else float("nan")),
            "final_tv_loss": float(final_row.tv_loss if final_row else float("nan")),
            "final_sparse_eql_loss": float(final_row.sparse_eql_loss if final_row else float("nan")),
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

        # Lightweight visuals (best-effort)
        try:
            if plt is None:
                raise RuntimeError("matplotlib not available")
            plt.figure(figsize=(6, 4))
            epochs = [r["epoch"] for r in history_rows]
            plt.plot(epochs, [r["total_loss"] for r in history_rows], label="total")
            plt.plot(epochs, [r["data_loss"] for r in history_rows], label="data")
            plt.plot(epochs, [r["tv_loss"] for r in history_rows], label="tv")
            plt.plot(epochs, [r["sparse_eql_loss"] for r in history_rows], label="sparse_eql")
            plt.yscale("log")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"l1_lambda={cfg.l1_lambda:g}  seed={cfg.seed}")
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
            "l1_lambda": float(cfg.l1_lambda),
            "sparse_eql_s": float(cfg.sparse_eql_s),
            "final_total_loss": float("nan"),
            "final_data_loss": float("nan"),
            "final_tv_loss": float("nan"),
            "final_sparse_eql_loss": float("nan"),
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
    p = argparse.ArgumentParser(description="L1 (sparse_eql) regularization lambda sweep (Burgers / Allen–Cahn / Heat)")
    p.add_argument("--dataset", default="burgers", choices=["burgers", "allen_cahn", "heat"])
    p.add_argument("--l1_lambdas", nargs="*", type=float, default=None)
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
    p.add_argument("--noise_level", type=float, default=0.0)
    p.add_argument("--stride_t", type=int, default=1)
    p.add_argument("--stride_x", type=int, default=1)

    # Optional TV during fitting (held fixed while sweeping L1)
    p.add_argument("--tv_type", default="tv_u", choices=list(available_tv_types()))
    p.add_argument("--tv_lambda", type=float, default=0.0)
    p.add_argument("--sparse_eql_s", type=float, default=1e-3)

    p.add_argument("--eval_chunk_size", type=int, default=20000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    # Burgers params
    p.add_argument("--burgers_N", type=int, default=256)
    p.add_argument("--burgers_L", type=float, default=2.0)
    p.add_argument("--burgers_nu", type=float, default=0.01)
    p.add_argument("--burgers_dt", type=float, default=1e-3)
    p.add_argument("--burgers_T", type=float, default=1.0)

    # Allen–Cahn params
    p.add_argument("--allen_N", type=int, default=256)
    p.add_argument("--allen_x_min", type=float, default=-1.0)
    p.add_argument("--allen_x_max", type=float, default=1.0)
    p.add_argument("--allen_dt", type=float, default=1e-3)
    p.add_argument("--allen_T", type=float, default=1.0)
    p.add_argument("--allen_d", type=float, default=0.1)
    p.add_argument("--allen_reaction_scale", type=float, default=1.0)
    p.add_argument("--allen_bc_value", type=float, default=0.0)

    # Heat params
    p.add_argument("--heat_N", type=int, default=256)
    p.add_argument("--heat_L", type=float, default=2.0)
    p.add_argument("--heat_dt", type=float, default=1e-3)
    p.add_argument("--heat_T", type=float, default=1.0)
    p.add_argument("--heat_alpha", type=float, default=0.01)
    p.add_argument("--heat_ic_modes", type=int, default=8)

    args = p.parse_args()

    if not args.l1_lambdas:
        if str(args.dataset).lower() == "burgers":
            args.l1_lambdas = list(DEFAULT_L1_LAMBDAS_BURGERS)
        else:
            args.l1_lambdas = list(DEFAULT_L1_LAMBDAS_GENERIC)

    root = (
        Path("run_results")
        / "l1_lambda_sweep"
        / str(args.dataset)
        / f"tv_{str(args.tv_type)}_lam_{_fmt_value_for_path(float(args.tv_lambda))}"
    )
    root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    for l1_lambda in args.l1_lambdas:
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
                allen_N=int(args.allen_N),
                allen_x_min=float(args.allen_x_min),
                allen_x_max=float(args.allen_x_max),
                allen_dt=float(args.allen_dt),
                allen_T=float(args.allen_T),
                allen_d=float(args.allen_d),
                allen_reaction_scale=float(args.allen_reaction_scale),
                allen_bc_value=float(args.allen_bc_value),
                heat_N=int(args.heat_N),
                heat_L=float(args.heat_L),
                heat_dt=float(args.heat_dt),
                heat_T=float(args.heat_T),
                heat_alpha=float(args.heat_alpha),
                heat_ic_modes=int(args.heat_ic_modes),
                tv_type=str(args.tv_type),
                tv_lambda=float(args.tv_lambda),
                l1_lambda=float(l1_lambda),
                sparse_eql_s=float(args.sparse_eql_s),
                eval_chunk_size=int(args.eval_chunk_size),
            )

            run_dir = root / f"l1_lambda_{_fmt_value_for_path(float(l1_lambda))}" / f"seed_{int(seed):03d}"
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

    if (not args.dry_run) and all_rows:
        _write_csv(root / "summary.csv", all_rows, SUMMARY_FIELDS)
        _plot_vs_lambda(
            run_dir=root,
            df_rows=all_rows,
            x_key="l1_lambda",
            y_keys=["final_data_loss", "final_tv_loss", "final_sparse_eql_loss"],
            title="final losses vs l1_lambda",
            out_name="loss_vs_l1_lambda.pdf",
        )
        _plot_vs_lambda(
            run_dir=root,
            df_rows=all_rows,
            x_key="l1_lambda",
            y_keys=["ux_rel_l2", "uxx_rel_l2", "uxxx_rel_l2"],
            title="derivative error vs l1_lambda",
            out_name="deriv_error_vs_l1_lambda.pdf",
        )
        _plot_vs_lambda(
            run_dir=root,
            df_rows=all_rows,
            x_key="l1_lambda",
            y_keys=["l2_coeff_error"],
            title="PDE coeff error vs l1_lambda",
            out_name="coeff_error_vs_l1_lambda.pdf",
        )
        print(f"Wrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
