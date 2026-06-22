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
from prog.featlib import FeatureTensor
from prog.hlprs import savefig_atomic
from prog.mlps import EQL, SirenMLP
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
from utils.extract_pde_ls import extract_pde_ls
from utils.fit_utils import fit_data_and_pde
from utils.tv_utils import available_tv_types, dispatch_tv

DEFAULT_PDE_LAMBDAS_GENERIC = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
DEFAULT_PDE_LAMBDAS_BURGERS = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]


def _fmt_value_for_path(v: float) -> str:
    s = f"{float(v):.6g}"
    return s.replace(".", "p").replace("-", "m").replace("+", "")


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _feature_key(name: str) -> str:
    return "".join(ch for ch in str(name) if ch.isalnum())


def _feature_metric_fields(feature_names: list[str]) -> list[str]:
    fields: list[str] = []
    for name in feature_names:
        key = _feature_key(name)
        fields.extend([f"{key}_rel_l2", f"{key}_rmse", f"{key}_max_abs"])
    return fields


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


def _fd_derivs_centered(u_grid: np.ndarray, x_grid: np.ndarray) -> dict[str, np.ndarray]:
    u_grid = np.asarray(u_grid, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    dx = float(x_grid[1] - x_grid[0]) if x_grid.size >= 2 else 1.0
    ux = np.stack([fd_first_centered(row, dx) for row in u_grid], axis=0)
    uxx = np.stack([fd_second_centered(row, dx) for row in u_grid], axis=0)
    uxxx = np.stack([fd_third_centered(row, dx) for row in u_grid], axis=0)
    return {"ux": ux, "uxx": uxx, "uxxx": uxxx}


def _make_u_model(cfg: "RunConfig") -> torch.nn.Module:
    if str(cfg.model).lower() == "siren":
        return SirenMLP(
            hidden_size=int(cfg.hidden_size),
            hidden_layers=int(cfg.hidden_layers),
            first_omega_0=float(cfg.first_omega_0),
            hidden_omega_0=float(cfg.hidden_omega_0),
        )
    raise ValueError(f"Unknown model='{cfg.model}'")


def _make_v_model(*, feature_dim: int, cfg: "RunConfig") -> torch.nn.Module:
    return EQL(in_dim=int(feature_dim), prod_dim=int(cfg.eql_prod_dim), num_layers=int(cfg.eql_layers), bias=False)


def _eql_feature_terms_for_dataset(dataset: str) -> list[str]:
    dataset = str(dataset).lower().strip()
    if dataset in {"burgers", "burger"}:
        return ["u", "u_x", "u_xx"]
    if dataset in {"allen_cahn", "allen-cahn", "allencahn", "allen"}:
        return ["u", "u_x", "u_xx"]
    if dataset in {"heat"}:
        return ["u", "u_x", "u_xx"]
    raise ValueError(f"Unknown dataset='{dataset}'")


def _ls_feature_terms_for_dataset(dataset: str) -> list[str]:
    dataset = str(dataset).lower().strip()
    if dataset in {"burgers", "burger"}:
        return ["u", "u_x", "u_xx", "uu_x"]
    if dataset in {"allen_cahn", "allen-cahn", "allencahn", "allen"}:
        return ["u", "u_xx", "u3"]
    if dataset in {"heat"}:
        return ["u_xx"]
    raise ValueError(f"Unknown dataset='{dataset}'")


def _true_coeffs_for_dataset(cfg: "RunConfig", feature_terms: list[str]) -> np.ndarray:
    dataset = str(cfg.dataset).lower().strip()
    if dataset in {"burgers", "burger"}:
        return np.array([0.0, 0.0, float(cfg.burgers_nu), -1.0], dtype=float)
    if dataset in {"allen_cahn", "allen-cahn", "allencahn", "allen"}:
        r = float(cfg.allen_reaction_scale)
        return np.array([r, float(cfg.allen_d), -r], dtype=float)
    if dataset in {"heat"}:
        return np.array([float(cfg.heat_alpha)], dtype=float)
    raise ValueError(f"Unknown dataset='{dataset}'")


def _reference_feature_grids(
    *,
    dataset: str,
    u_grid: np.ndarray,
    x_grid: np.ndarray,
    feature_terms: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    dataset = str(dataset).lower().strip()
    u_grid = np.asarray(u_grid, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    if dataset in {"burgers", "burger", "heat"}:
        derivs = _fd_derivs_periodic(u_grid, x_grid)
        x_maps = {
            "u": x_grid,
            "u_x": x_grid,
            "u_xx": x_grid,
            "u_xxx": x_grid,
            "uu_x": x_grid,
            "u3": x_grid,
        }
        feats = {
            "u": u_grid,
            "u_x": derivs["ux"],
            "u_xx": derivs["uxx"],
            "u_xxx": derivs["uxxx"],
            "uu_x": u_grid * derivs["ux"],
            "u3": u_grid**3,
        }
        return {k: feats[k] for k in feature_terms}, {k: x_maps[k] for k in feature_terms}

    derivs = _fd_derivs_centered(u_grid, x_grid)
    feats = {
        "u": u_grid,
        "u_x": derivs["ux"],
        "u_xx": derivs["uxx"],
        "u_xxx": derivs["uxxx"],
        "uu_x": u_grid[:, 1:-1] * derivs["ux"],
        "u3": u_grid**3,
    }
    x_maps = {
        "u": x_grid,
        "u_x": x_grid[1:-1],
        "u_xx": x_grid[1:-1],
        "u_xxx": x_grid[2:-2],
        "uu_x": x_grid[1:-1],
        "u3": x_grid,
    }
    return {k: feats[k] for k in feature_terms}, {k: x_maps[k] for k in feature_terms}


def _predicted_feature_grids(
    model: torch.nn.Module,
    *,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    feature_terms: list[str],
    device: str,
    chunk_size: int,
) -> tuple[list[str], dict[str, np.ndarray]]:
    t_grid = np.asarray(t_grid, dtype=np.float32).reshape(-1)
    x_grid = np.asarray(x_grid, dtype=np.float32).reshape(-1)
    Nt = int(t_grid.size)
    Nx = int(x_grid.size)
    t2d = np.repeat(t_grid[:, None], Nx, axis=1)
    x2d = np.repeat(x_grid[None, :], Nt, axis=0)
    t_flat = t2d.reshape(-1, 1)
    x_flat = x2d.reshape(-1, 1)

    feat_builder = FeatureTensor(feature_terms, normalize=False, keep_raw=True)
    names: list[str] | None = None
    cols: dict[str, list[np.ndarray]] = {}

    model = model.to(device)
    model.eval()
    chunk_size = max(1, int(chunk_size))

    for i0 in range(0, t_flat.shape[0], chunk_size):
        i1 = min(t_flat.shape[0], i0 + chunk_size)
        t_b = torch.from_numpy(t_flat[i0:i1]).to(device).requires_grad_(True)
        x_b = torch.from_numpy(x_flat[i0:i1]).to(device).requires_grad_(True)
        with torch.enable_grad():
            u = model(t_b, x_b)
            feat_out = feat_builder.build(u, x=x_b)
        if names is None:
            names = list(feat_out.names)
            cols = {name: [] for name in names}
        raw = feat_out.raw_cols.detach().cpu().numpy()
        for idx, name in enumerate(names):
            cols[name].append(raw[:, idx])

    if names is None:
        raise RuntimeError("No feature names produced.")
    out = {name: np.concatenate(parts, axis=0).reshape(Nt, Nx) for name, parts in cols.items()}
    return names, out


def _plot_feature_overlay(
    *,
    t: np.ndarray,
    x: np.ndarray,
    pred: np.ndarray,
    ref: np.ndarray,
    path: Path,
    title: str,
    ylabel: str,
    snap_count: int = 5,
) -> None:
    if plt is None:
        return
    Nt = int(t.size)
    idxs = np.linspace(0, Nt - 1, min(max(1, snap_count), Nt), dtype=int)
    ncols = min(3, idxs.size)
    nrows = int(math.ceil(idxs.size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, idxs):
        ax.plot(x, ref[k], label="reference", linewidth=1.5)
        ax.plot(x, pred[k], "--", label="model", linewidth=1.2)
        ax.set_title(f"t = {t[k]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    for ax in axes[idxs.size:]:
        ax.axis("off")
    fig.suptitle(title)
    savefig_atomic(path)


def _write_simple_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_least_squares_outputs(run_dir: Path, payload: dict[str, Any]) -> None:
    ls_dir = Path(run_dir) / "pde_outputs" / "least_squares"
    _write_json(ls_dir / "pde.json", payload)
    pieces = [f"{float(c):+.4f}*{term}" for term, c in zip(payload["terms"], payload["coeffs"])]
    eqn = " ".join(pieces).replace("+ -", "- ")
    _write_text(
        ls_dir / "pde.txt",
        "\n".join(
            [
                f"{payload['target']} ~= {eqn}",
                f"residual_rel_l2 = {float(payload['residual_rel_l2']):.6e}",
                f"residual_rmse = {float(payload['residual_rmse']):.6e}",
                f"coeff_error_l2 = {float(payload.get('coeff_error_l2', float('nan'))):.6e}",
            ]
        ),
    )
    _write_simple_csv(
        ls_dir / "coefficients.csv",
        ["term", "coeff"],
        [[term, coeff] for term, coeff in zip(payload["terms"], payload["coeffs"])],
    )
    _write_json(
        ls_dir / "diagnostics.json",
        {
            "method": payload["method"],
            "target": payload["target"],
            "residual_rel_l2": payload["residual_rel_l2"],
            "residual_rmse": payload["residual_rmse"],
            "rank": payload["rank"],
            "condition_number": payload["condition_number"],
            "coeff_error_l2": payload.get("coeff_error_l2", float("nan")),
        },
    )


def _write_eql_outputs(run_dir: Path, v_model: torch.nn.Module, feature_names: list[str], true_coeffs: np.ndarray) -> dict[str, Any]:
    eql_dir = Path(run_dir) / "pde_outputs" / "eql"
    eql_dir.mkdir(parents=True, exist_ok=True)
    readout = v_model.readout.weight.detach().cpu().numpy().reshape(-1)
    base_names = list(feature_names) + [f"prod_{i}" for i in range(max(0, readout.size - len(feature_names)))]
    _write_simple_csv(
        eql_dir / "readout_coefficients.csv",
        ["term", "weight"],
        [[name, float(weight)] for name, weight in zip(base_names, readout.tolist())],
    )
    _write_simple_csv(
        eql_dir / "coefficients.csv",
        ["term", "weight"],
        [[name, float(weight)] for name, weight in zip(base_names, readout.tolist())],
    )

    diagnostics: dict[str, Any] = {
        "method": "eql",
        "eql_layers": int(len(v_model.linears)),
        "eql_prod_dim": int(v_model.linears[0].out_features) if len(v_model.linears) else 0,
        "readout_dim": int(readout.size),
    }
    try:
        matrix = v_model.effective_quadratic_matrix(symmetrize=False).detach().cpu().numpy()
        _write_simple_csv(
            eql_dir / "effective_quadratic_matrix.csv",
            ["feature", *feature_names],
            [[feature_names[i], *matrix[i].tolist()] for i in range(len(feature_names))],
        )
        diagnostics["effective_quadratic_matrix_available"] = True
    except Exception as exc:
        diagnostics["effective_quadratic_matrix_available"] = False
        diagnostics["effective_quadratic_matrix_error"] = repr(exc)

    payload = {
        "method": "eql",
        "target": "u_t",
        "feature_names": feature_names,
        "readout_terms": base_names,
        "readout_coeffs": readout.tolist(),
        "true_pde_coeffs": np.asarray(true_coeffs, dtype=float).reshape(-1).tolist(),
    }
    _write_json(eql_dir / "pde.json", payload)
    _write_text(eql_dir / "pde.txt", "EQL readout written to readout_coefficients.csv")
    _write_json(eql_dir / "diagnostics.json", diagnostics)
    return {
        "eql_method": "eql",
        "eql_feature_names": json.dumps(feature_names),
        "eql_readout_terms": json.dumps(base_names),
        "eql_readout_coeffs": json.dumps(readout.tolist()),
        "eql_readout_dim": int(readout.size),
        "eql_effective_quadratic_matrix_available": bool(diagnostics.get("effective_quadratic_matrix_available", False)),
        "eql_effective_quadratic_matrix_error": diagnostics.get("effective_quadratic_matrix_error", ""),
    }


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

    pde_lambda: float = 1.0
    data_lambda: float = 1.0

    tv_type: str = "tv_u"
    tv_lambda: float = 0.0

    eql_layers: int = 1
    eql_prod_dim: int = 2

    eval_chunk_size: int = 20000


BASE_SUMMARY_FIELDS = [
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
    "tv_type",
    "tv_lambda",
    "pde_lambda",
    "data_lambda",
    "eql_layers",
    "eql_prod_dim",
    "final_total_loss",
    "final_data_loss",
    "final_pde_loss",
    "final_tv_loss",
    "pde_method",
    "feature_names",
    "ls_method",
    "ls_terms",
    "ls_coeffs",
    "pde_terms",
    "pde_coeffs",
    "true_pde_terms",
    "true_pde_coeffs",
    "ls_true_pde_terms",
    "ls_true_pde_coeffs",
    "ls_residual_rel_l2",
    "ls_residual_rmse",
    "ls_rank",
    "ls_condition_number",
    "ls_coeff_error_l2",
    "ls_num_active_terms",
    "ls_active_terms",
    "eql_method",
    "eql_feature_names",
    "eql_readout_terms",
    "eql_readout_coeffs",
    "eql_readout_dim",
    "eql_effective_quadratic_matrix_available",
    "eql_effective_quadratic_matrix_error",
    "status",
    "error",
]


def _ordered_summary_fields(rows: list[dict[str, Any]]) -> list[str]:
    feature_fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key in BASE_SUMMARY_FIELDS:
                continue
            if key.endswith(("_rel_l2", "_rmse", "_max_abs")) and key not in feature_fields:
                feature_fields.append(key)
    return BASE_SUMMARY_FIELDS[:18] + feature_fields + BASE_SUMMARY_FIELDS[18:]


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float], list[dict[str, Any]]] = {}
    for row in rows:
        if int(row.get("status", 0)) != 1:
            continue
        key = (float(row["pde_lambda"]),)
        grouped.setdefault(key, []).append(row)

    metric_keys = set()
    for row in rows:
        for key in row.keys():
            if key.endswith(("_rel_l2", "_rmse", "_max_abs")) or key in {
                "final_train_loss",
                "final_total_loss",
                "final_data_loss",
                "final_pde_loss",
                "final_tv_loss",
                "ls_residual_rel_l2",
                "ls_residual_rmse",
                "ls_condition_number",
                "ls_coeff_error_l2",
            }:
                metric_keys.add(key)

    agg_rows: list[dict[str, Any]] = []
    for (pde_lambda,), group_rows in sorted(grouped.items(), key=lambda item: item[0][0]):
        agg = {"pde_lambda": float(pde_lambda), "num_seeds": int(len(group_rows))}
        for key in sorted(metric_keys):
            vals = np.asarray([float(r.get(key, float("nan"))) for r in group_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            agg[f"{key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            agg[f"{key}_std"] = float(np.std(vals, ddof=0)) if vals.size else float("nan")
        agg_rows.append(agg)
    return agg_rows


def run_one(cfg: RunConfig, run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", asdict(cfg))
    eql_feature_terms = _eql_feature_terms_for_dataset(str(cfg.dataset).lower().strip())
    ls_feature_terms = _ls_feature_terms_for_dataset(str(cfg.dataset).lower().strip())

    try:
        _seed_everything(int(cfg.seed))
        dataset = str(cfg.dataset).lower().strip()

        if dataset in {"burgers", "burger"}:
            x_grid, _u_final, _t_end, (t_grid, u_grid) = solve_burgers(
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
            x_grid, _u_final, _t_end, (t_grid, u_grid) = solve_allen_cahn(allen_cfg, return_history=True)
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
            x_grid, _u_final, _t_end, (t_grid, u_grid) = solve_heat(heat_cfg, return_history=True)
        else:
            raise ValueError("Unknown dataset. Use --dataset burgers, --dataset allen_cahn, or --dataset heat.")

        t_grid = np.asarray(t_grid, dtype=np.float64)
        x_grid = np.asarray(x_grid, dtype=np.float64)
        u_grid = np.asarray(u_grid, dtype=np.float64)
        if not np.isfinite(u_grid).all():
            raise ValueError(f"{dataset} solver returned non-finite values (nan/inf).")

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
        a_t = float(train_ds.t_norm.a)
        b_t = float(train_ds.t_norm.b)
        a_x = float(train_ds.x_norm.a)
        b_x = float(train_ds.x_norm.b)

        feat = FeatureTensor(terms=eql_feature_terms, normalize=False)

        u_model = _make_u_model(cfg)
        v_model = _make_v_model(feature_dim=len(eql_feature_terms), cfg=cfg)

        tv_terms = None
        if float(cfg.tv_lambda) != 0.0:
            tv_fn = dispatch_tv(cfg.tv_type)
            tv_terms = [(float(cfg.tv_lambda), tv_fn)]

        u_model, v_model, hist = fit_data_and_pde(
            u_model,
            v_model,
            t_train_n,
            x_train_n,
            y_noisy,
            feat.build,
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            lr=float(cfg.lr),
            device=str(cfg.device),
            lam_pde=float(cfg.pde_lambda),
            lam_data=float(cfg.data_lambda),
            log_every=max(1, int(cfg.epochs) // 10),
            tv_terms=tv_terms,
        )

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
                    }
                )
        else:
            for i, l in enumerate(hist.losses):
                history_rows.append(
                    {
                        "epoch": int(i),
                        "total_loss": float(l),
                        "data_loss": float("nan"),
                        "pde_loss": float("nan"),
                        "tv_loss": float("nan"),
                    }
                )
        _write_csv(run_dir / "loss_history.csv", history_rows, ["epoch", "total_loss", "data_loss", "pde_loss", "tv_loss"])

        phys_u = PhysCoordWrapper(u_model, a_t=a_t, b_t=b_t, a_x=a_x, b_x=b_x)

        try:
            fig, fig_hm, _payload = hlprs.snapshot_comp(
                phys_u,
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

        feature_names, pred_features = _predicted_feature_grids(
            phys_u,
            t_grid=t_grid,
            x_grid=x_grid,
            feature_terms=eql_feature_terms,
            device=str(cfg.device),
            chunk_size=int(cfg.eval_chunk_size),
        )
        ref_features, ref_xgrids = _reference_feature_grids(
            dataset=dataset,
            u_grid=u_grid,
            x_grid=x_grid,
            feature_terms=eql_feature_terms,
        )
        feature_summary: dict[str, Any] = {}
        for name in feature_names:
            key = _feature_key(name)
            metrics = compute_error_metrics(pred_features[name], ref_features[name])
            feature_summary[f"{key}_rel_l2"] = float(metrics["rel_l2"])
            feature_summary[f"{key}_rmse"] = float(metrics["rmse"])
            feature_summary[f"{key}_max_abs"] = float(metrics["max_abs"])
            _plot_feature_overlay(
                t=t_grid,
                x=ref_xgrids[name],
                pred=pred_features[name],
                ref=ref_features[name],
                path=run_dir / "feature_overlays" / f"{key}_overlay.pdf",
                title=f"{name} overlay",
                ylabel=name,
            )

        Nt = int(t_grid.size)
        Nx = int(x_grid.size)
        t2d = np.repeat(t_grid[:, None], Nx, axis=1)
        x2d = np.repeat(x_grid[None, :], Nt, axis=0)
        t_flat = t2d.reshape(-1)
        x_flat = x2d.reshape(-1)

        pde = extract_pde_ls(phys_u, t_flat, x_flat, ls_feature_terms, device=str(cfg.device))
        true_coeffs = _true_coeffs_for_dataset(cfg, ls_feature_terms)
        coeffs = np.asarray(pde["coeffs"], dtype=float).reshape(-1)
        l2_coeff_error = float(np.linalg.norm(coeffs - true_coeffs))
        residuals = np.asarray(pde["residuals"], dtype=float).reshape(-1)
        singular_values = np.asarray(pde["singular_values"], dtype=float).reshape(-1)
        cond = float(np.max(singular_values) / np.min(singular_values)) if singular_values.size and np.min(singular_values) > 0 else float("inf")
        ls_payload = {
            "method": "least_squares",
            "target": "u_t",
            "terms": list(pde["names"]),
            "coeffs": coeffs.tolist(),
            "residual_rel_l2": float(np.sqrt(residuals[0]) / (np.linalg.norm(coeffs) + 1e-12)) if residuals.size else float("nan"),
            "residual_rmse": float(np.sqrt(residuals[0] / max(1, t_flat.size))) if residuals.size else float("nan"),
            "rank": int(pde["rank"]),
            "condition_number": cond,
            "true_pde_terms": list(ls_feature_terms),
            "true_pde_coeffs": true_coeffs.tolist(),
            "coeff_error_l2": l2_coeff_error,
        }
        _write_least_squares_outputs(run_dir, ls_payload)
        eql_diag = _write_eql_outputs(run_dir, v_model, feature_names, true_coeffs)
        ls_active_terms = [term for term, coeff in zip(ls_payload["terms"], ls_payload["coeffs"]) if abs(float(coeff)) > 1e-8]

        final_row = hist.rows[-1] if hist.rows else None
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
            "tv_type": str(cfg.tv_type),
            "tv_lambda": float(cfg.tv_lambda),
            "pde_lambda": float(cfg.pde_lambda),
            "data_lambda": float(cfg.data_lambda),
            "eql_layers": int(cfg.eql_layers),
            "eql_prod_dim": int(cfg.eql_prod_dim),
            "pde_method": "least_squares,eql",
            "feature_names": json.dumps(feature_names),
            "final_total_loss": float(
                final_row.total_loss if final_row else (hist.losses[-1] if hist.losses else float("nan"))
            ),
            "final_train_loss": float(
                final_row.total_loss if final_row else (hist.losses[-1] if hist.losses else float("nan"))
            ),
            "min_train_loss": float(np.min(np.asarray(hist.losses, dtype=np.float64))) if hist.losses else float("nan"),
            "final_data_loss": float(final_row.data_loss if final_row else float("nan")),
            "final_pde_loss": float(final_row.pde_loss if final_row else float("nan")),
            "final_tv_loss": float(final_row.tv_loss if final_row else float("nan")),
            "ls_method": "least_squares",
            "ls_terms": json.dumps(list(pde["names"])),
            "ls_coeffs": json.dumps(coeffs.tolist()),
            "pde_terms": json.dumps(list(pde["names"])),
            "pde_coeffs": json.dumps(coeffs.tolist()),
            "true_pde_terms": json.dumps(ls_feature_terms),
            "true_pde_coeffs": json.dumps(true_coeffs.tolist()),
            "ls_true_pde_terms": json.dumps(ls_feature_terms),
            "ls_true_pde_coeffs": json.dumps(true_coeffs.tolist()),
            "ls_residual_rel_l2": float(ls_payload["residual_rel_l2"]),
            "ls_residual_rmse": float(ls_payload["residual_rmse"]),
            "ls_rank": int(ls_payload["rank"]),
            "ls_condition_number": float(ls_payload["condition_number"]),
            "ls_coeff_error_l2": float(l2_coeff_error),
            "ls_num_active_terms": int(len(ls_active_terms)),
            "ls_active_terms": "|".join(ls_active_terms),
            **eql_diag,
            "status": 1,
            "error": "",
            **feature_summary,
        }

        _write_json(run_dir / "summary.json", summary)

        try:
            if plt is None:
                raise RuntimeError("matplotlib not available")
            plt.figure(figsize=(6, 4))
            epochs = [r["epoch"] for r in history_rows]
            plt.plot(epochs, [r["total_loss"] for r in history_rows], label="total")
            plt.plot(epochs, [r["data_loss"] for r in history_rows], label="data")
            plt.plot(epochs, [r["pde_loss"] for r in history_rows], label="pde")
            plt.plot(epochs, [r["tv_loss"] for r in history_rows], label="tv")
            plt.yscale("log")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"pde_lambda={cfg.pde_lambda:g}  seed={cfg.seed}")
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
            "tv_type": str(cfg.tv_type),
            "tv_lambda": float(cfg.tv_lambda),
            "pde_lambda": float(cfg.pde_lambda),
            "data_lambda": float(cfg.data_lambda),
            "eql_layers": int(cfg.eql_layers),
            "eql_prod_dim": int(cfg.eql_prod_dim),
            "pde_method": "",
            "feature_names": json.dumps(eql_feature_terms),
            "final_total_loss": float("nan"),
            "final_train_loss": float("nan"),
            "min_train_loss": float("nan"),
            "final_data_loss": float("nan"),
            "final_pde_loss": float("nan"),
            "final_tv_loss": float("nan"),
            "ls_method": "",
            "ls_terms": "",
            "ls_coeffs": "",
            "pde_terms": "",
            "pde_coeffs": "",
            "true_pde_terms": json.dumps(ls_feature_terms),
            "true_pde_coeffs": "",
            "ls_true_pde_terms": "",
            "ls_true_pde_coeffs": "",
            "ls_residual_rel_l2": float("nan"),
            "ls_residual_rmse": float("nan"),
            "ls_rank": float("nan"),
            "ls_condition_number": float("nan"),
            "ls_coeff_error_l2": float("nan"),
            "ls_num_active_terms": float("nan"),
            "ls_active_terms": "",
            "eql_method": "",
            "eql_feature_names": json.dumps(eql_feature_terms),
            "eql_readout_terms": "",
            "eql_readout_coeffs": "",
            "eql_readout_dim": float("nan"),
            "eql_effective_quadratic_matrix_available": False,
            "eql_effective_quadratic_matrix_error": "",
            "status": 0,
            "error": repr(e),
        }
        _write_json(run_dir / "summary.json", summary)
        return summary


def main() -> None:
    p = argparse.ArgumentParser(description="PDE loss weight (lambda_pde) sweep (Burgers / Allen–Cahn / Heat)")
    p.add_argument("--dataset", default="burgers", choices=["burgers", "allen_cahn", "heat"])
    p.add_argument("--pde_lambdas", nargs="*", type=float, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model", default="siren", choices=["siren"])
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--hidden_layers", type=int, default=3)
    p.add_argument("--first_omega_0", type=float, default=30.0)
    p.add_argument("--hidden_omega_0", type=float, default=1.0)
    p.add_argument("--noise_level", type=float, default=0.0)
    p.add_argument("--stride_t", type=int, default=1)
    p.add_argument("--stride_x", type=int, default=1)
    p.add_argument("--data_lambda", type=float, default=1.0)

    p.add_argument("--tv_type", default="tv_u", choices=list(available_tv_types()))
    p.add_argument("--tv_lambda", type=float, default=0.0)

    p.add_argument("--eql_layers", type=int, default=1)
    p.add_argument("--eql_prod_dim", type=int, default=2)

    p.add_argument("--eval_chunk_size", type=int, default=20000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--sweep_coords", nargs="+", default=["pde_lambda"], help="Sweep coordinate column names (one or two).")

    p.add_argument("--burgers_N", type=int, default=256)
    p.add_argument("--burgers_L", type=float, default=2.0)
    p.add_argument("--burgers_nu", type=float, default=0.01)
    p.add_argument("--burgers_dt", type=float, default=1e-3)
    p.add_argument("--burgers_T", type=float, default=1.0)

    p.add_argument("--allen_N", type=int, default=256)
    p.add_argument("--allen_x_min", type=float, default=-1.0)
    p.add_argument("--allen_x_max", type=float, default=1.0)
    p.add_argument("--allen_dt", type=float, default=1e-3)
    p.add_argument("--allen_T", type=float, default=1.0)
    p.add_argument("--allen_d", type=float, default=0.1)
    p.add_argument("--allen_reaction_scale", type=float, default=1.0)
    p.add_argument("--allen_bc_value", type=float, default=0.0)

    p.add_argument("--heat_N", type=int, default=256)
    p.add_argument("--heat_L", type=float, default=2.0)
    p.add_argument("--heat_dt", type=float, default=1e-3)
    p.add_argument("--heat_T", type=float, default=1.0)
    p.add_argument("--heat_alpha", type=float, default=0.01)
    p.add_argument("--heat_ic_modes", type=int, default=8)

    args = p.parse_args()

    if not args.pde_lambdas:
        if str(args.dataset).lower() == "burgers":
            args.pde_lambdas = list(DEFAULT_PDE_LAMBDAS_BURGERS)
        else:
            args.pde_lambdas = list(DEFAULT_PDE_LAMBDAS_GENERIC)

    root = (
        Path("run_results")
        / "pde_loss_sweep"
        / str(args.dataset)
        / f"tv_{str(args.tv_type)}_lam_{_fmt_value_for_path(float(args.tv_lambda))}"
        / f"data_lam_{_fmt_value_for_path(float(args.data_lambda))}"
    )
    root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    for pde_lambda in args.pde_lambdas:
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
                pde_lambda=float(pde_lambda),
                data_lambda=float(args.data_lambda),
                eql_layers=int(args.eql_layers),
                eql_prod_dim=int(args.eql_prod_dim),
                eval_chunk_size=int(args.eval_chunk_size),
            )

            run_dir = root / f"pde_lambda_{_fmt_value_for_path(float(pde_lambda))}" / f"seed_{int(seed):03d}"
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
        summary_fields = _ordered_summary_fields(all_rows)
        _write_csv(root / "summary.csv", all_rows, summary_fields)
        agg_rows = _aggregate_rows(all_rows)
        if agg_rows:
            agg_fields = ["pde_lambda", "num_seeds"]
            dynamic_agg = [k for k in agg_rows[0].keys() if k not in {"pde_lambda", "num_seeds"}]
            _write_csv(root / "summary_agg.csv", agg_rows, agg_fields + dynamic_agg)
        _plot_vs_lambda(
            run_dir=root,
            df_rows=all_rows,
            x_key="pde_lambda",
            y_keys=["final_data_loss", "final_pde_loss", "final_tv_loss"],
            title="final losses vs pde_lambda",
            out_name="loss_vs_pde_lambda.pdf",
        )
        _plot_vs_lambda(
            run_dir=root,
            df_rows=all_rows,
            x_key="pde_lambda",
            y_keys=[k for k in summary_fields if k.endswith("_rel_l2")],
            title="feature error vs pde_lambda",
            out_name="feature_error_vs_pde_lambda.pdf",
        )
        _plot_vs_lambda(
            run_dir=root,
            df_rows=all_rows,
            x_key="pde_lambda",
            y_keys=["ls_coeff_error_l2", "ls_residual_rel_l2"],
            title="least-squares diagnostics vs pde_lambda",
            out_name="coeff_error_vs_pde_lambda.pdf",
        )
        print(f"Wrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
