from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from Datasets.data.processed.allenc_gen.allen_cahn_gen import AllenCahnConfig, solve_allen_cahn
from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from Datasets.data.processed.heat_gen.heat_gen import HeatConfig, solve_heat
from prog import hlprs
from prog.hlprs import savefig_atomic
from prog.mlps import SirenMLP
from utils.extract_pde_ls import build_feature_matrix, eval_model_and_time_derivative, extract_pde_ls
from utils.derivative_utils import (
    compute_error_metrics,
    fd_first_centered,
    fd_first_periodic,
    fd_second_centered,
    fd_second_periodic,
    fd_third_centered,
    fd_third_periodic,
)
from utils.data_prep_utils import PDETrainDataset
from utils.fit_utils import fit_model_to_data


DERIVATIVE_SUMMARY_FIELDS = [
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
]

AGGREGATED_DERIVATIVE_FIELDS = [
    "u_rel_l2",
    "ux_rel_l2",
    "uxx_rel_l2",
    "uxxx_rel_l2",
]

LS_SUMMARY_FIELDS = [
    "ls_terms",
    "ls_coeffs",
    "ls_residual_rel_l2",
    "ls_residual_rmse",
    "ls_rank",
    "ls_condition_number",
    "ls_coeff_error_l2",
    "ls_active_terms",
]

AGGREGATED_LS_FIELDS = [
    "ls_residual_rel_l2",
    "ls_residual_rmse",
    "ls_condition_number",
    "ls_coeff_error_l2",
]


SUMMARY_FIELDS = [
    "dataset",
    "seed",
    "model",
    "hidden_size",
    "hidden_layers",
    "first_omega_0",
    "hidden_omega_0",
    "epochs",
    "batch_size",
    "lr",
    "noise_level",
    "stride_t",
    "stride_x",
    *DERIVATIVE_SUMMARY_FIELDS,
    *LS_SUMMARY_FIELDS,
    "final_train_loss",
    "min_train_loss",
    "status",
    "error",
]


def _fmt_value_for_path(v: float) -> str:
    s = f"{float(v):.6g}"
    return s.replace(".", "p").replace("-", "m").replace("+", "")


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PhysCoordWrapper(torch.nn.Module):
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


def _nan_derivative_summary() -> dict[str, float]:
    return {field: float("nan") for field in DERIVATIVE_SUMMARY_FIELDS}


def _empty_ls_summary() -> dict[str, Any]:
    return {
        "ls_terms": "",
        "ls_coeffs": "",
        "ls_residual_rel_l2": float("nan"),
        "ls_residual_rmse": float("nan"),
        "ls_rank": float("nan"),
        "ls_condition_number": float("nan"),
        "ls_coeff_error_l2": float("nan"),
        "ls_active_terms": "",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_dataset(cfg: "RunConfig"):
    dataset = str(cfg.dataset).lower().strip()
    if dataset in {"burgers", "burger"}:
        x_grid, _u_final, _t_final, (t_grid, u_grid) = solve_burgers(
            N=int(cfg.burgers_N),
            L=float(cfg.burgers_L),
            nu=float(cfg.burgers_nu),
            dt=float(cfg.burgers_dt),
            T=float(cfg.burgers_T),
            seed=int(cfg.seed),
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
        x_grid, _u_final, _t_final, (t_grid, u_grid) = solve_allen_cahn(allen_cfg, return_history=True)
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
        x_grid, _u_final, _t_final, (t_grid, u_grid) = solve_heat(heat_cfg, return_history=True)
    else:
        raise ValueError("Unknown dataset. Use --dataset burgers, allen_cahn, or heat.")

    return (
        np.asarray(t_grid, dtype=np.float64),
        np.asarray(x_grid, dtype=np.float64),
        np.asarray(u_grid, dtype=np.float64),
    )


def _make_model(cfg: "RunConfig") -> torch.nn.Module:
    if str(cfg.model).lower() != "siren":
        raise ValueError(f"Unknown model='{cfg.model}' (expected 'siren').")
    return SirenMLP(
        hidden_size=int(cfg.hidden_size),
        hidden_layers=int(cfg.hidden_layers),
        first_omega_0=float(cfg.first_omega_0),
        hidden_omega_0=float(cfg.hidden_omega_0),
    )


def _plot_heatmap(rows: list[dict[str, Any]], *, x_key: str, y_key: str, value_key: str, path: Path, title: str) -> None:
    if plt is None or not rows:
        return

    x_vals = sorted({float(row[x_key]) for row in rows})
    y_vals = sorted({int(row[y_key]) for row in rows})
    grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=np.float64)
    x_index = {value: idx for idx, value in enumerate(x_vals)}
    y_index = {value: idx for idx, value in enumerate(y_vals)}

    for row in rows:
        xv = float(row[x_key])
        yv = int(row[y_key])
        grid[y_index[yv], x_index[xv]] = float(row[value_key])

    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(x_vals), 1.2 + 0.6 * len(y_vals)))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([_fmt_value_for_path(v) for v in x_vals], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value_key)

    for row_idx, yv in enumerate(y_vals):
        for col_idx, xv in enumerate(x_vals):
            value = grid[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.2e}", ha="center", va="center", color="white", fontsize=7)

    fig.tight_layout()
    savefig_atomic(path)


def _plot_derivative_overlays(
    t: np.ndarray,
    x: np.ndarray,
    pred: np.ndarray,
    ref: np.ndarray,
    path: Path,
    title: str,
    ylabel: str,
    snap_count: int = 5,
) -> None:
    Nt = int(t.size)
    idxs = np.linspace(0, Nt - 1, min(max(1, snap_count), Nt), dtype=int)
    ncols = min(3, idxs.size)
    nrows = int(math.ceil(idxs.size / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, idxs):
        ax.plot(x, ref[k], label="FD ref", linewidth=1.5)
        ax.plot(x, pred[k], "--", label="model", linewidth=1.2)
        ax.set_title(f"t = {t[k]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    for ax in axes[idxs.size:]:
        ax.axis("off")

    fig.suptitle(title)
    savefig_atomic(path)


def _autograd_spatial_derivs_on_grid(
    model: torch.nn.Module,
    *,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    device: str,
    chunk_size: int = 4096,
) -> dict[str, np.ndarray]:
    t_grid = np.asarray(t_grid, dtype=np.float32).reshape(-1)
    x_grid = np.asarray(x_grid, dtype=np.float32).reshape(-1)
    Nt = int(t_grid.size)
    Nx = int(x_grid.size)

    t2d = np.repeat(t_grid[:, None], Nx, axis=1)
    x2d = np.repeat(x_grid[None, :], Nt, axis=0)
    t_flat = t2d.reshape(-1, 1)
    x_flat = x2d.reshape(-1, 1)

    out_u = np.zeros((Nt * Nx,), dtype=np.float32)
    out_ux = np.zeros((Nt * Nx,), dtype=np.float32)
    out_uxx = np.zeros((Nt * Nx,), dtype=np.float32)
    out_uxxx = np.zeros((Nt * Nx,), dtype=np.float32)

    model = model.to(device)
    model.eval()
    chunk_size = max(1, int(chunk_size))

    for i0 in range(0, t_flat.shape[0], chunk_size):
        i1 = min(t_flat.shape[0], i0 + chunk_size)
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


def _fd_derivs_grid(u_grid: np.ndarray, x_grid: np.ndarray, *, periodic: bool) -> dict[str, np.ndarray]:
    u_grid = np.asarray(u_grid, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    dx = float(x_grid[1] - x_grid[0]) if x_grid.size >= 2 else 1.0

    if periodic:
        ux = np.stack([fd_first_periodic(row, dx) for row in u_grid], axis=0)
        uxx = np.stack([fd_second_periodic(row, dx) for row in u_grid], axis=0)
        uxxx = np.stack([fd_third_periodic(row, dx) for row in u_grid], axis=0)
        return {
            "ux": ux,
            "uxx": uxx,
            "uxxx": uxxx,
            "x_ux": x_grid,
            "x_uxx": x_grid,
            "x_uxxx": x_grid,
        }

    ux = np.stack([fd_first_centered(row, dx) for row in u_grid], axis=0)
    uxx = np.stack([fd_second_centered(row, dx) for row in u_grid], axis=0)
    uxxx = np.stack([fd_third_centered(row, dx) for row in u_grid], axis=0)
    return {
        "ux": ux,
        "uxx": uxx,
        "uxxx": uxxx,
        "x_ux": x_grid[1:-1],
        "x_uxx": x_grid[1:-1],
        "x_uxxx": x_grid[2:-2],
    }


def _true_pde_spec(cfg: "RunConfig") -> tuple[list[str], np.ndarray]:
    dataset = str(cfg.dataset).lower()
    if dataset in {"burgers", "burger"}:
        return ["u", "u_x", "u_xx", "uu_x"], np.array([0.0, 0.0, float(cfg.burgers_nu), -1.0], dtype=float)
    if dataset in {"allen_cahn", "allen-cahn", "allencahn", "allen"}:
        reaction_scale = float(cfg.allen_reaction_scale)
        return ["u", "u_xx", "u3"], np.array([reaction_scale, float(cfg.allen_d), -reaction_scale], dtype=float)
    return ["u_xx"], np.array([float(cfg.heat_alpha)], dtype=float)


def _format_term(term: str) -> str:
    return {
        "uu_x": "u*u_x",
        "u3": "u^3",
    }.get(term, term)


def _build_pde_payload(
    model: torch.nn.Module,
    *,
    cfg: "RunConfig",
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    device: str,
) -> dict[str, Any]:
    feature_terms, true_coeffs = _true_pde_spec(cfg)
    Nt = int(t_grid.size)
    Nx = int(x_grid.size)
    t2d = np.repeat(t_grid[:, None], Nx, axis=1)
    x2d = np.repeat(x_grid[None, :], Nt, axis=0)
    t_flat = t2d.reshape(-1)
    x_flat = x2d.reshape(-1)

    pde = extract_pde_ls(model, t_flat, x_flat, feature_terms, device=device)
    _u_pred_np, u_t_np, _t_torch, x_torch, u_pred_torch = eval_model_and_time_derivative(model, t_flat, x_flat, device)
    F_np, names, _scales = build_feature_matrix(feature_terms, u_pred_torch, x_torch)
    coeffs = np.asarray(pde["coeffs"], dtype=float).reshape(-1)
    target = np.asarray(u_t_np, dtype=float).reshape(-1)
    pred_target = np.asarray(F_np, dtype=float) @ coeffs
    residual = pred_target - target
    coeff_error_l2 = float(np.linalg.norm(coeffs - true_coeffs))
    singular_values = np.asarray(pde["singular_values"], dtype=float).reshape(-1)
    condition_number = float(np.max(singular_values) / np.min(singular_values)) if singular_values.size and np.min(singular_values) > 0 else float("inf")

    return {
        "terms": list(names),
        "coeffs": coeffs.tolist(),
        "residual_rel_l2": float(np.linalg.norm(residual) / (np.linalg.norm(target) + 1e-12)),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "rank": int(pde["rank"]),
        "condition_number": condition_number,
        "target": "u_t",
        "method": "numpy_lstsq",
        "true_pde_terms": list(feature_terms),
        "true_pde_coeffs": true_coeffs.tolist(),
        "coeff_error_l2": coeff_error_l2,
    }


def _render_pde_text(payload: dict[str, Any]) -> str:
    terms = payload["terms"]
    coeffs = payload["coeffs"]
    pieces = [f"{float(coeff):+.4f}*{_format_term(str(term))}" for term, coeff in zip(terms, coeffs)]
    equation = " ".join(pieces).replace("+ -", "- ")
    return "\n".join(
        [
            f"u_t ~= {equation}",
            f"residual_rel_l2 = {float(payload['residual_rel_l2']):.6e}",
            f"residual_rmse = {float(payload['residual_rmse']):.6e}",
            f"coeff_error_l2 = {float(payload['coeff_error_l2']):.6e}",
            f"condition_number = {float(payload['condition_number']):.6e}",
        ]
    )


def _active_terms(terms: list[str], coeffs: list[float], *, threshold: float = 1e-8) -> list[str]:
    active: list[str] = []
    for term, coeff in zip(terms, coeffs):
        if abs(float(coeff)) > float(threshold):
            active.append(str(term))
    return active


def _extract_ls_summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    terms = list(payload.get("terms", []))
    coeffs = list(payload.get("coeffs", []))
    return {
        "ls_terms": json.dumps(terms),
        "ls_coeffs": json.dumps(coeffs),
        "ls_residual_rel_l2": float(payload.get("residual_rel_l2", float("nan"))),
        "ls_residual_rmse": float(payload.get("residual_rmse", float("nan"))),
        "ls_rank": int(payload["rank"]) if "rank" in payload and payload.get("rank") is not None else float("nan"),
        "ls_condition_number": float(payload.get("condition_number", float("nan"))),
        "ls_coeff_error_l2": float(payload.get("coeff_error_l2", float("nan"))),
        "ls_active_terms": json.dumps(_active_terms(terms, coeffs)),
    }


def _extract_pde_summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "l2_coeff_error": float(payload.get("coeff_error_l2", float("nan"))),
        "pde_names": list(payload.get("terms", [])),
        "pde_coeffs": list(payload.get("coeffs", [])),
        "true_coeffs": list(payload.get("true_pde_coeffs", [])),
    }


def _enrich_summary_from_run_artifacts(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    enriched = dict(summary)
    for key, value in _nan_derivative_summary().items():
        enriched.setdefault(key, value)
    for key, value in _empty_ls_summary().items():
        enriched.setdefault(key, value)

    ls_payload = _read_json_if_exists(Path(run_dir) / "least_squares_pde.json")
    if ls_payload is not None:
        enriched.update(_extract_ls_summary_from_payload(ls_payload))
        enriched.update(_extract_pde_summary_from_payload(ls_payload))
    return enriched


def _collect_existing_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("layers_*/*/seed_*/summary.json")):
        summary = _read_json_if_exists(summary_path)
        if not isinstance(summary, dict):
            continue
        rows.append(_enrich_summary_from_run_artifacts(summary, summary_path.parent))
    return rows


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

    burgers_N: int = 256
    burgers_L: float = 2 * np.pi
    burgers_nu: float = 0.02
    burgers_dt: float = 2e-3
    burgers_T: float = 1.0

    allen_N: int = 201
    allen_x_min: float = -1.0
    allen_x_max: float = 1.0
    allen_dt: float = 0.01
    allen_T: float = 1.0
    allen_d: float = 0.001
    allen_reaction_scale: float = 5.0
    allen_bc_value: float = -1.0

    heat_N: int = 256
    heat_L: float = 2 * np.pi
    heat_dt: float = 2e-3
    heat_T: float = 1.0
    heat_alpha: float = 0.01
    heat_ic_modes: int = 8


def run_one(cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", asdict(cfg))

    try:
        _seed_everything(int(cfg.seed))
        t_grid, x_grid, u_grid = _load_dataset(cfg)
        if not np.isfinite(u_grid).all():
            raise ValueError(f"{cfg.dataset} solver returned non-finite values (nan/inf).")

        train_ds = PDETrainDataset(
            t_grid=t_grid,
            x_grid=x_grid,
            u_grid=u_grid,
            stride_t=int(cfg.stride_t),
            stride_x=int(cfg.stride_x),
            noise_level=float(cfg.noise_level),
            seed=int(cfg.seed),
            normalize=True,
        )
        t_train_n, x_train_n, y_clean, y_noisy = train_ds.fit_arrays_with_clean()
        t_train = train_ds.t_train
        x_train = train_ds.x_train
        if not (np.isfinite(t_train_n).all() and np.isfinite(x_train_n).all() and np.isfinite(y_noisy).all()):
            raise ValueError("Training samples contain non-finite values (nan/inf).")

        model = _make_model(cfg)
        model, hist = fit_model_to_data(
            model,
            t_train_n,
            x_train_n,
            y_noisy,
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            lr=float(cfg.lr),
            device=str(cfg.device),
            log_every=max(1, int(cfg.epochs) // 10),
        )
        if (not hist.losses) or (not np.isfinite(np.asarray(hist.losses, dtype=np.float64)).all()):
            raise ValueError("Training produced non-finite loss values (nan/inf).")

        history_rows: list[dict[str, Any]] = []
        if hist.rows:
            for row in hist.rows:
                history_rows.append(
                    {
                        "epoch": int(row.epoch),
                        "total_loss": float(row.total_loss),
                        "data_loss": float(row.data_loss),
                        "pde_loss": float(row.pde_loss),
                        "tv_loss": float(row.tv_loss),
                        "l1_data_loss": float(row.l1_data_loss),
                        "sparse_eql_loss": float(row.sparse_eql_loss),
                    }
                )
        _write_csv(
            run_dir / "loss_history.csv",
            history_rows,
            ["epoch", "total_loss", "data_loss", "pde_loss", "tv_loss", "l1_data_loss", "sparse_eql_loss"],
        )

        try:
            fig, fig_hm, _payload = hlprs.snapshot_comp(
                PhysCoordWrapper(
                    model,
                    a_t=float(train_ds.t_norm.a),
                    b_t=float(train_ds.t_norm.b),
                    a_x=float(train_ds.x_norm.a),
                    b_x=float(train_ds.x_norm.b),
                ),
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

        derivative_summary = _nan_derivative_summary()

        try:
            phys_model = PhysCoordWrapper(
                model,
                a_t=float(train_ds.t_norm.a),
                b_t=float(train_ds.t_norm.b),
                a_x=float(train_ds.x_norm.a),
                b_x=float(train_ds.x_norm.b),
            )
            pred_derivs = _autograd_spatial_derivs_on_grid(
                phys_model,
                t_grid=t_grid,
                x_grid=x_grid,
                device=str(cfg.device),
                chunk_size=max(1, int(cfg.batch_size)),
            )
            ref_derivs = _fd_derivs_grid(u_grid, x_grid, periodic=str(cfg.dataset).lower() in {"burgers", "burger", "heat"})
            derivative_summary["u_rel_l2"] = float(compute_error_metrics(pred_derivs["u"], u_grid)["rel_l2"])
            derivative_summary["u_rmse"] = float(compute_error_metrics(pred_derivs["u"], u_grid)["rmse"])
            derivative_summary["u_max_abs"] = float(compute_error_metrics(pred_derivs["u"], u_grid)["max_abs"])

            if str(cfg.dataset).lower() in {"burgers", "burger", "heat"}:
                ux_pred = pred_derivs["ux"]
                uxx_pred = pred_derivs["uxx"]
                uxxx_pred = pred_derivs["uxxx"]
            else:
                ux_pred = pred_derivs["ux"][:, 1:-1]
                uxx_pred = pred_derivs["uxx"][:, 1:-1]
                uxxx_pred = pred_derivs["uxxx"][:, 2:-2]

            ux_metrics = compute_error_metrics(ux_pred, ref_derivs["ux"])
            uxx_metrics = compute_error_metrics(uxx_pred, ref_derivs["uxx"])
            uxxx_metrics = compute_error_metrics(uxxx_pred, ref_derivs["uxxx"])
            derivative_summary.update(
                {
                    "ux_rel_l2": float(ux_metrics["rel_l2"]),
                    "ux_rmse": float(ux_metrics["rmse"]),
                    "ux_max_abs": float(ux_metrics["max_abs"]),
                    "uxx_rel_l2": float(uxx_metrics["rel_l2"]),
                    "uxx_rmse": float(uxx_metrics["rmse"]),
                    "uxx_max_abs": float(uxx_metrics["max_abs"]),
                    "uxxx_rel_l2": float(uxxx_metrics["rel_l2"]),
                    "uxxx_rmse": float(uxxx_metrics["rmse"]),
                    "uxxx_max_abs": float(uxxx_metrics["max_abs"]),
                }
            )

            if str(cfg.dataset).lower() in {"burgers", "burger", "heat"}:
                _plot_derivative_overlays(
                    t_grid,
                    x_grid,
                    pred_derivs["u"],
                    u_grid,
                    run_dir / "u_overlay.pdf",
                    title=f"{cfg.model} u overlays",
                    ylabel="u",
                )
                _plot_derivative_overlays(
                    t_grid,
                    x_grid,
                    ux_pred,
                    ref_derivs["ux"],
                    run_dir / "ux_overlay.pdf",
                    title=f"{cfg.model} u_x overlays",
                    ylabel="u_x",
                )
                _plot_derivative_overlays(
                    t_grid,
                    x_grid,
                    uxx_pred,
                    ref_derivs["uxx"],
                    run_dir / "uxx_overlay.pdf",
                    title=f"{cfg.model} u_xx overlays",
                    ylabel="u_xx",
                )
                _plot_derivative_overlays(
                    t_grid,
                    x_grid,
                    uxxx_pred,
                    ref_derivs["uxxx"],
                    run_dir / "uxxx_overlay.pdf",
                    title=f"{cfg.model} u_xxx overlays",
                    ylabel="u_xxx",
                )
            else:
                _plot_derivative_overlays(
                    t_grid,
                    x_grid,
                    pred_derivs["u"],
                    u_grid,
                    run_dir / "u_overlay.pdf",
                    title=f"{cfg.model} u overlays",
                    ylabel="u",
                )
                _plot_derivative_overlays(
                    t_grid,
                    ref_derivs["x_ux"],
                    ux_pred,
                    ref_derivs["ux"],
                    run_dir / "ux_overlay.pdf",
                    title=f"{cfg.model} u_x overlays",
                    ylabel="u_x",
                )
                _plot_derivative_overlays(
                    t_grid,
                    ref_derivs["x_uxx"],
                    uxx_pred,
                    ref_derivs["uxx"],
                    run_dir / "uxx_overlay.pdf",
                    title=f"{cfg.model} u_xx overlays",
                    ylabel="u_xx",
                )
                _plot_derivative_overlays(
                    t_grid,
                    ref_derivs["x_uxxx"],
                    uxxx_pred,
                    ref_derivs["uxxx"],
                    run_dir / "uxxx_overlay.pdf",
                    title=f"{cfg.model} u_xxx overlays",
                    ylabel="u_xxx",
                )
        except Exception as e:
            print(f"[warn] derivative overlay failed: {e}")

        pde_summary: dict[str, Any] = {}
        ls_summary: dict[str, Any] = _empty_ls_summary()

        try:
            pde_payload = _build_pde_payload(
                phys_model,
                cfg=cfg,
                t_grid=t_grid,
                x_grid=x_grid,
                device=str(cfg.device),
            )
            _write_json(run_dir / "least_squares_pde.json", pde_payload)
            _write_text(run_dir / "least_squares_pde.txt", _render_pde_text(pde_payload))
            active_terms = _active_terms(list(pde_payload["terms"]), list(pde_payload["coeffs"]))
            pde_summary = {
                "l2_coeff_error": float(pde_payload["coeff_error_l2"]),
                "pde_names": list(pde_payload["terms"]),
                "pde_coeffs": list(pde_payload["coeffs"]),
                "true_coeffs": list(pde_payload["true_pde_coeffs"]),
            }
            ls_summary = _extract_ls_summary_from_payload(pde_payload)
            ls_summary["ls_active_terms"] = json.dumps(active_terms)
        except Exception as e:
            print(f"[warn] least-squares PDE extraction failed: {e}")

        final_loss = float(hist.losses[-1])
        min_loss = float(np.min(np.asarray(hist.losses, dtype=np.float64)))
        summary = {
            "dataset": str(cfg.dataset),
            "seed": int(cfg.seed),
            "model": str(cfg.model),
            "hidden_size": int(cfg.hidden_size),
            "hidden_layers": int(cfg.hidden_layers),
            "first_omega_0": float(cfg.first_omega_0),
            "hidden_omega_0": float(cfg.hidden_omega_0),
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr),
            "noise_level": float(cfg.noise_level),
            "stride_t": int(cfg.stride_t),
            "stride_x": int(cfg.stride_x),
            **derivative_summary,
            **ls_summary,
            **pde_summary,
            "final_train_loss": final_loss,
            "min_train_loss": min_loss,
            "status": 1,
            "error": "",
        }
        _write_json(run_dir / "summary.json", summary)
        return summary

    except Exception as e:
        summary = {
            "dataset": str(cfg.dataset),
            "seed": int(cfg.seed),
            "model": str(cfg.model),
            "hidden_size": int(cfg.hidden_size),
            "hidden_layers": int(cfg.hidden_layers),
            "first_omega_0": float(cfg.first_omega_0),
            "hidden_omega_0": float(cfg.hidden_omega_0),
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr),
            "noise_level": float(cfg.noise_level),
            "stride_t": int(cfg.stride_t),
            "stride_x": int(cfg.stride_x),
            **_nan_derivative_summary(),
            **_empty_ls_summary(),
            "final_train_loss": float("nan"),
            "min_train_loss": float("nan"),
            "status": 0,
            "error": repr(e),
        }
        _write_json(run_dir / "summary.json", summary)
        return summary


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        if int(row.get("status", 0)) != 1:
            continue
        key = (int(row["hidden_layers"]), float(row["hidden_omega_0"]))
        grouped.setdefault(key, []).append(row)

    agg_rows: list[dict[str, Any]] = []
    for (hidden_layers, hidden_omega_0), combo_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        final_losses = np.asarray([float(r["final_train_loss"]) for r in combo_rows], dtype=np.float64)
        min_losses = np.asarray([float(r["min_train_loss"]) for r in combo_rows], dtype=np.float64)
        agg_row = {
            "hidden_layers": int(hidden_layers),
            "hidden_omega_0": float(hidden_omega_0),
            "num_seeds": int(len(combo_rows)),
            "final_train_loss_mean": float(np.mean(final_losses)),
            "final_train_loss_std": float(np.std(final_losses, ddof=0)),
            "min_train_loss_mean": float(np.mean(min_losses)),
            "min_train_loss_std": float(np.std(min_losses, ddof=0)),
        }
        for field in AGGREGATED_DERIVATIVE_FIELDS:
            values = np.asarray([float(r.get(field, float("nan"))) for r in combo_rows], dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            agg_row[f"{field}_mean"] = float(np.mean(finite_values)) if finite_values.size else float("nan")
            agg_row[f"{field}_std"] = float(np.std(finite_values, ddof=0)) if finite_values.size else float("nan")
        for field in AGGREGATED_LS_FIELDS:
            values = np.asarray([float(r.get(field, float("nan"))) for r in combo_rows], dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            agg_row[f"{field}_mean"] = float(np.mean(finite_values)) if finite_values.size else float("nan")
            agg_row[f"{field}_std"] = float(np.std(finite_values, ddof=0)) if finite_values.size else float("nan")
        agg_rows.append(agg_row)
    return agg_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="SIREN depth/hidden-omega sweep")
    parser.add_argument("--dataset", default="burgers", choices=["burgers", "allen_cahn", "heat"])
    parser.add_argument("--hidden_layers_grid", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--hidden_omega_0s", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--first_omega_0", type=float, default=30.0)
    parser.add_argument("--noise_level", type=float, default=0.05)
    parser.add_argument("--stride_t", type=int, default=1)
    parser.add_argument("--stride_x", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--refresh_from_existing", action="store_true")

    parser.add_argument("--burgers_N", type=int, default=256)
    parser.add_argument("--burgers_L", type=float, default=2 * np.pi)
    parser.add_argument("--burgers_nu", type=float, default=0.02)
    parser.add_argument("--burgers_dt", type=float, default=2e-3)
    parser.add_argument("--burgers_T", type=float, default=1.0)

    parser.add_argument("--allen_N", type=int, default=201)
    parser.add_argument("--allen_x_min", type=float, default=-1.0)
    parser.add_argument("--allen_x_max", type=float, default=1.0)
    parser.add_argument("--allen_dt", type=float, default=0.01)
    parser.add_argument("--allen_T", type=float, default=1.0)
    parser.add_argument("--allen_d", type=float, default=0.001)
    parser.add_argument("--allen_reaction_scale", type=float, default=5.0)
    parser.add_argument("--allen_bc_value", type=float, default=-1.0)

    parser.add_argument("--heat_N", type=int, default=256)
    parser.add_argument("--heat_L", type=float, default=2 * np.pi)
    parser.add_argument("--heat_dt", type=float, default=2e-3)
    parser.add_argument("--heat_T", type=float, default=1.0)
    parser.add_argument("--heat_alpha", type=float, default=0.01)
    parser.add_argument("--heat_ic_modes", type=int, default=8)

    args = parser.parse_args()

    root = Path("run_results") / "siren_hparam_sweep" / str(args.dataset)
    root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    if args.refresh_from_existing:
        all_rows = _collect_existing_rows(root)
    else:
        for hidden_layers in args.hidden_layers_grid:
            for hidden_omega_0 in args.hidden_omega_0s:
                for seed in args.seeds:
                    cfg = RunConfig(
                        dataset=str(args.dataset),
                        seed=int(seed),
                        device=str(args.device),
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        model="siren",
                        hidden_size=int(args.hidden_size),
                        hidden_layers=int(hidden_layers),
                        first_omega_0=float(args.first_omega_0),
                        hidden_omega_0=float(hidden_omega_0),
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
                    )

                    run_dir = (
                        root
                        / f"layers_{int(hidden_layers)}"
                        / f"hidden_omega_{_fmt_value_for_path(float(hidden_omega_0))}"
                        / f"seed_{int(seed):03d}"
                    )

                    if args.dry_run:
                        print(f"[dry_run] {run_dir}")
                        continue

                    summary_path = run_dir / "summary.json"
                    if (not args.overwrite) and summary_path.exists():
                        prev = _read_json_if_exists(summary_path)
                        if isinstance(prev, dict) and int(prev.get("status", 0)) == 1:
                            all_rows.append(_enrich_summary_from_run_artifacts(prev, run_dir))
                            continue

                    summary = run_one(cfg, run_dir)
                    all_rows.append(summary)

    if not args.dry_run and all_rows:
        _write_csv(root / "summary.csv", all_rows, SUMMARY_FIELDS)
        agg_rows = _aggregate_rows(all_rows)
        if agg_rows:
            _write_csv(
                root / "summary_agg.csv",
                agg_rows,
                [
                    "hidden_layers",
                    "hidden_omega_0",
                    "num_seeds",
                    "final_train_loss_mean",
                    "final_train_loss_std",
                    "min_train_loss_mean",
                    "min_train_loss_std",
                    "u_rel_l2_mean",
                    "u_rel_l2_std",
                    "ux_rel_l2_mean",
                    "ux_rel_l2_std",
                    "uxx_rel_l2_mean",
                    "uxx_rel_l2_std",
                    "uxxx_rel_l2_mean",
                    "uxxx_rel_l2_std",
                    "ls_residual_rel_l2_mean",
                    "ls_residual_rel_l2_std",
                    "ls_residual_rmse_mean",
                    "ls_residual_rmse_std",
                    "ls_condition_number_mean",
                    "ls_condition_number_std",
                    "ls_coeff_error_l2_mean",
                    "ls_coeff_error_l2_std",
                ],
            )
            _plot_heatmap(
                agg_rows,
                x_key="hidden_omega_0",
                y_key="hidden_layers",
                value_key="final_train_loss_mean",
                path=root / "final_train_loss_heatmap.pdf",
                title="Mean final train loss",
            )
            _plot_heatmap(
                agg_rows,
                x_key="hidden_omega_0",
                y_key="hidden_layers",
                value_key="min_train_loss_mean",
                path=root / "min_train_loss_heatmap.pdf",
                title="Mean min train loss",
            )
            _plot_heatmap(
                agg_rows,
                x_key="hidden_omega_0",
                y_key="hidden_layers",
                value_key="u_rel_l2_mean",
                path=root / "u_rel_l2_heatmap.pdf",
                title="Mean u relative L2 error",
            )
            _plot_heatmap(
                agg_rows,
                x_key="hidden_omega_0",
                y_key="hidden_layers",
                value_key="ux_rel_l2_mean",
                path=root / "ux_rel_l2_heatmap.pdf",
                title="Mean ux relative L2 error",
            )
            _plot_heatmap(
                agg_rows,
                x_key="hidden_omega_0",
                y_key="hidden_layers",
                value_key="uxx_rel_l2_mean",
                path=root / "uxx_rel_l2_heatmap.pdf",
                title="Mean uxx relative L2 error",
            )
            _plot_heatmap(
                agg_rows,
                x_key="hidden_omega_0",
                y_key="hidden_layers",
                value_key="uxxx_rel_l2_mean",
                path=root / "uxxx_rel_l2_heatmap.pdf",
                title="Mean uxxx relative L2 error",
            )
        print(f"Wrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
