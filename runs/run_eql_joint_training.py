from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from prog.featlib import FeatureTensor
from prog.mlps import EQL, SirenMLP, rescale_polynomial_coefficients
from utils.data_prep_utils import PDETrainDataset
from utils.derivative_utils import (
    build_burgers_reference_derivative_grids,
    build_reference_primitive_features,
    compute_primitive_feature_l2_scales,
    compute_error_metrics,
    evaluate_primitive_feature_metrics,
    primitive_feature_name_to_key,
    reconstruct_grid,
)
from utils.feature_plotting import (
    save_primitive_feature_overlays,
    save_space_time_heatmap,
    save_time_slice_snapshots,
)
from utils.fit_utils import fit_data_and_pde, fit_model_to_data, predict_on_grid
from utils.tv_utils import dispatch_tv


@dataclass
class RunConfig:
    dataset_name: str = "Allen_Cahn_generated"
    noise_level: float = 0.0
    seed: int = 0
    stride_t: int = 2
    stride_x: int = 2
    device: str = "cpu"
    pretrain_epochs: int = 150
    pretrain_lr: float = 1e-3
    epochs: int = 600
    joint_lr: float = 1e-3
    batch_size: int = 1024
    weight_decay: float = 0.0
    lam_pde: float = 0.5
    lam_data: float = 1.0
    lam_sparse_eql: float = 1e-5
    sparse_eql_s: float = 1e-3
    tv_type: str = "tv_ux"
    tv_lambda: float = 1e-6
    hidden_size: int = 64
    hidden_layers: int = 3
    first_omega_0: float = 20.0
    hidden_omega_0: float = 1.0
    eql_prod_dim: int = 2
    eql_num_layers: int = 2
    feature_terms: tuple[str, ...] = ("u", "u_x", "u_xx")
    feature_normalize: bool = False
    feature_normalize_mode: str = "per_batch"
    max_time_slices: int = 5
    log_every: int = 50
    output_root: str | None = None
    output_tag: str | None = None


def load_dataset(dataset_name: str):
    if dataset_name.startswith("Allen_Cahn"):
        path = Path("Datasets/data/raw") / f"{dataset_name}.mat"
        raw = loadmat(path)
        t = raw["t"].reshape(-1)
        x = raw["x"].reshape(-1)
        u = np.asarray(raw["u"], dtype=np.float64)
        return t, x, u, "periodic", {"dataset_name": dataset_name}

    if dataset_name == "burgers":
        path = Path("Datasets/data/raw") / f"{dataset_name}.mat"
        raw = loadmat(path)
        t = raw["t"].reshape(-1)
        x = raw["x"].reshape(-1)
        u = np.asarray(raw["usol"], dtype=np.float64).T
        return t, x, u, "periodic", {"dataset_name": dataset_name}

    if dataset_name == "burg_gen":
        x, _u_final, _t_end, (t, u) = solve_burgers(
            seed=0,
            return_history=True,
        )
        return t, x, np.asarray(u, dtype=np.float64), "periodic", {
            "dataset_name": dataset_name,
            "nu": 0.02,
            "generator": "solve_burgers",
            "time_integrator": "RK4",
            "spatial_operator": "second-order centered periodic finite differences",
        }

    raise ValueError(f"Unsupported dataset '{dataset_name}' for this run script.")


def format_float_token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def default_output_dir(cfg: RunConfig) -> Path:
    base = (
        Path("run_results")
        / "eql_joint_training"
        / cfg.dataset_name.lower()
        / f"seed_{cfg.seed:03d}"
    )
    if cfg.output_tag:
        return base.parent / f"{base.name}_{cfg.output_tag}"
    return base


def compute_augmented_history_rows(history_rows: list[dict], cfg: RunConfig) -> list[dict]:
    augmented_rows = []
    lam_data = float(cfg.lam_data)
    lam_pde = float(cfg.lam_pde)
    for row in history_rows:
        augmented = dict(row)
        augmented["raw_data_mse"] = (
            float(row["data_loss"]) / lam_data if lam_data != 0.0 else None
        )
        augmented["weighted_data_contribution"] = float(row["data_loss"])
        augmented["raw_pde_mse"] = (
            float(row["pde_loss"]) / lam_pde if lam_pde != 0.0 else None
        )
        augmented["weighted_pde_contribution"] = float(row["pde_loss"])
        augmented["tv_loss_weighted"] = float(row["tv_loss"])
        augmented["sparse_eql_loss_weighted"] = float(row["sparse_eql_loss"])
        augmented_rows.append(augmented)
    return augmented_rows


def _tensor_stats_abs(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().reshape(-1).abs()
    return {
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "max": float(flat.max().cpu()),
    }


def _tensor_stats_signed(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().reshape(-1)
    abs_flat = flat.abs()
    return {
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "max_abs": float(abs_flat.max().cpu()),
    }


def collect_pde_diagnostics(
    u_model,
    v_model,
    dataset: PDETrainDataset,
    cfg: RunConfig,
    *,
    fixed_feature_scales: dict[str, float] | None = None,
) -> dict:
    full_t_n, full_x_n = dataset.full_grid_normalized_flat()
    t = torch.from_numpy(np.asarray(full_t_n, dtype=np.float32).reshape(-1, 1)).to(cfg.device).requires_grad_(True)
    x = torch.from_numpy(np.asarray(full_x_n, dtype=np.float32).reshape(-1, 1)).to(cfg.device).requires_grad_(True)

    u_model = u_model.to(cfg.device)
    v_model = v_model.to(cfg.device)
    u_model.eval()
    v_model.eval()

    builder = FeatureTensor(
        terms=list(cfg.feature_terms),
        normalize=cfg.feature_normalize,
        keep_raw=True,
        fixed_scales=fixed_feature_scales,
    )

    with torch.enable_grad():
        u_pred = u_model(t, x)
        features = builder.build(u_pred, x=x)
        u_t = torch.autograd.grad(
            u_pred,
            t,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=False,
            retain_graph=False,
        )[0]
        v_pred = v_model(features.F)
        residual = u_t - v_pred

    raw_by_name = {
        name: features.raw_cols[:, idx:idx + 1]
        for idx, name in enumerate(features.names)
    }
    scales = {
        name: float(features.scales[idx].detach().cpu())
        for idx, name in enumerate(features.names)
    }

    readout = v_model.readout.weight.detach().cpu().reshape(-1)
    base_weights = {
        name: float(readout[idx])
        for idx, name in enumerate(cfg.feature_terms)
    }

    return {
        "feature_scales_full_grid": scales,
        "feature_normalization_mode": cfg.feature_normalize_mode if cfg.feature_normalize else "none",
        "u_t_abs_stats": _tensor_stats_abs(u_t),
        "u_abs_stats": _tensor_stats_abs(raw_by_name["u"]),
        "u_x_abs_stats": _tensor_stats_abs(raw_by_name["u_x"]),
        "u_xx_abs_stats": _tensor_stats_abs(raw_by_name["u_xx"]),
        "v_pred_abs_stats": _tensor_stats_abs(v_pred),
        "pde_residual_stats": _tensor_stats_signed(residual),
        "readout_base_weights": base_weights,
    }


def _normalized_feature_builder(cfg: RunConfig, fixed_feature_scales: dict[str, float] | None = None):
    return FeatureTensor(
        terms=list(cfg.feature_terms),
        normalize=cfg.feature_normalize,
        fixed_scales=fixed_feature_scales,
    ).build


def serialize_monomial_coefficients(coefficients):
    return {
        " * ".join(monomial) if monomial else "1": float(coeff)
        for monomial, coeff in coefficients.items()
    }


def _format_signed(value: float) -> str:
    return f"{float(value):+.12g}"


def _format_linear_form(weights: np.ndarray, feature_terms: tuple[str, ...]) -> str:
    pieces = [f"({_format_signed(weights[idx])})*{feature_terms[idx]}" for idx in range(len(feature_terms))]
    return " ".join(pieces)


def _canonicalize_monomial(monomial: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(monomial))


def _aggregate_polynomial_by_monomial(coefficients) -> dict[tuple[str, ...], float]:
    aggregated = {}
    for monomial, coeff in coefficients.items():
        key = _canonicalize_monomial(tuple(monomial))
        aggregated[key] = aggregated.get(key, 0.0) + float(coeff)
    return aggregated


def _second_degree_terms(feature_terms: tuple[str, ...]) -> list[tuple[str, ...]]:
    u, ux, uxx = feature_terms
    return [
        (u,),
        (ux,),
        (uxx,),
        tuple(sorted((u, u))),
        tuple(sorted((u, ux))),
        tuple(sorted((u, uxx))),
        tuple(sorted((ux, ux))),
        tuple(sorted((ux, uxx))),
        tuple(sorted((uxx, uxx))),
    ]


def _serialize_selected_terms(coefficients, feature_terms: tuple[str, ...]) -> dict[str, float]:
    aggregated = _aggregate_polynomial_by_monomial(coefficients)
    return {
        " * ".join(term): float(aggregated.get(term, 0.0))
        for term in _second_degree_terms(feature_terms)
    }


def build_snapshot_time_indices(num_times: int) -> list[int]:
    if int(num_times) <= 1:
        return [0]
    anchors = [0.0, 0.25, 0.5, 0.75, 1.0]
    return np.unique([int(round(a * (num_times - 1))) for a in anchors]).tolist()


def evaluate_model_solution_and_derivatives(
    model,
    t_np,
    x_np,
    *,
    device: str = "cpu",
):
    t_arr = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
    x_arr = np.asarray(x_np, dtype=np.float32).reshape(-1, 1)
    t = torch.from_numpy(t_arr).to(device).requires_grad_(True)
    x = torch.from_numpy(x_arr).to(device).requires_grad_(True)

    model = model.to(device)
    model.eval()
    with torch.enable_grad():
        u = model(t, x)
        u_t = torch.autograd.grad(
            u,
            t,
            grad_outputs=torch.ones_like(u),
            create_graph=False,
            retain_graph=True,
        )[0]
        u_x = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=False,
            retain_graph=False,
        )[0]

    return {
        "u": u.detach().cpu().numpy().reshape(-1),
        "u_t": u_t.detach().cpu().numpy().reshape(-1),
        "u_x": u_x.detach().cpu().numpy().reshape(-1),
        "u_xx": u_xx.detach().cpu().numpy().reshape(-1),
    }


def build_reference_diagnostics(
    dataset: PDETrainDataset,
    feature_terms: tuple[str, ...],
    *,
    derivative_mode: str,
    dataset_name: str,
    dataset_metadata: dict | None = None,
):
    dataset_metadata = dataset_metadata or {}
    u_ref_flat = dataset.u_grid.reshape(-1)

    if dataset_name == "burg_gen":
        nu = float(dataset_metadata.get("nu", 0.02))
        derivative_grids = build_burgers_reference_derivative_grids(
            u_grid=dataset.u_grid,
            x_grid=dataset.x_grid,
            nu=nu,
            t_coord_scale=dataset.at_scale,
            x_coord_scale=dataset.ax_scale,
        )
        feature_names = list(feature_terms)
        by_name = {
            name: np.asarray(derivative_grids[name]["normalized"], dtype=np.float64).reshape(-1)
            for name in feature_names
        }
        by_name["u_t"] = np.asarray(derivative_grids["u_t"]["normalized"], dtype=np.float64).reshape(-1)
        by_name["u*u_x"] = np.asarray(derivative_grids["u*u_x"]["normalized"], dtype=np.float64).reshape(-1)
        return {
            "t_flat": np.repeat(dataset.t_grid[:, None], dataset.x_grid.size, axis=1).reshape(-1),
            "x_flat": np.repeat(dataset.x_grid[None, :], dataset.t_grid.size, axis=0).reshape(-1),
            "u_flat": np.asarray(u_ref_flat, dtype=np.float64).reshape(-1),
            "features": by_name,
            "feature_grids_normalized": {
                "u": np.asarray(derivative_grids["u"]["normalized"], dtype=np.float64),
                "u_t": np.asarray(derivative_grids["u_t"]["normalized"], dtype=np.float64),
                "u_x": np.asarray(derivative_grids["u_x"]["normalized"], dtype=np.float64),
                "u_xx": np.asarray(derivative_grids["u_xx"]["normalized"], dtype=np.float64),
                "u*u_x": np.asarray(derivative_grids["u*u_x"]["normalized"], dtype=np.float64),
            },
            "feature_grids_physical": {
                "u": np.asarray(derivative_grids["u"]["physical"], dtype=np.float64),
                "u_t": np.asarray(derivative_grids["u_t"]["physical"], dtype=np.float64),
                "u_x": np.asarray(derivative_grids["u_x"]["physical"], dtype=np.float64),
                "u_xx": np.asarray(derivative_grids["u_xx"]["physical"], dtype=np.float64),
                "u*u_x": np.asarray(derivative_grids["u*u_x"]["physical"], dtype=np.float64),
            },
            "derivative_source": {
                "label": "reference derivatives",
                "u_t": "Clean-rollout Burgers RHS on physical grid, then scaled to normalized t.",
                "u_x": "Generator periodic centered finite differences on physical grid, then scaled to normalized x.",
                "u_xx": "Generator periodic centered second derivative on physical grid, then scaled to normalized x.",
                "coordinate_scales": {
                    "t_coord_scale": float(dataset.at_scale),
                    "x_coord_scale": float(dataset.ax_scale),
                },
                "metadata": derivative_grids["metadata"],
            },
        }

    full_t_n, full_x_n = dataset.full_grid_normalized_flat()
    ref_features = build_reference_primitive_features(
        full_t_n,
        full_x_n,
        u_ref_flat,
        feature_terms,
        derivative_mode=derivative_mode,
    )
    values = np.asarray(ref_features["values"], dtype=np.float64)
    by_name = {
        name: values[:, idx]
        for idx, name in enumerate(ref_features["feature_names"])
    }
    by_name["u_t"] = np.full_like(by_name["u"], np.nan, dtype=np.float64)
    return {
        "t_flat": np.repeat(dataset.t_grid[:, None], dataset.x_grid.size, axis=1).reshape(-1),
        "x_flat": np.repeat(dataset.x_grid[None, :], dataset.t_grid.size, axis=0).reshape(-1),
        "u_flat": np.asarray(u_ref_flat, dtype=np.float64).reshape(-1),
        "features": by_name,
        "feature_grids_normalized": {
            name: np.asarray(values[:, idx], dtype=np.float64).reshape(dataset.u_grid.shape)
            for idx, name in enumerate(ref_features["feature_names"])
        },
        "derivative_source": {
            "label": "reference derivatives",
            "u_t": "Unavailable from current fallback dataset path.",
            "u_x": "Periodic finite differences on normalized x grid.",
            "u_xx": "Periodic second finite differences on normalized x grid.",
        },
    }


def build_training_sample_grid(dataset: PDETrainDataset) -> np.ndarray:
    sample_grid = np.full(dataset.u_grid.shape, np.nan, dtype=np.float64)
    row_idx = np.arange(dataset.t_grid.size)[:: dataset.stride_t]
    col_idx = np.arange(dataset.x_grid.size)[:: dataset.stride_x]
    sample_values = np.asarray(dataset.y_train_noisy, dtype=np.float64).reshape(len(row_idx), len(col_idx))
    sample_grid[np.ix_(row_idx, col_idx)] = sample_values
    return sample_grid


def build_eql_inspection(
    v_model: EQL,
    feature_terms: tuple[str, ...],
    feature_scales: dict[str, float] | None,
):
    linears0 = v_model.linears[0].weight.detach().cpu().numpy()
    readout = v_model.readout.weight.detach().cpu().numpy()
    z1_weights = linears0[0]
    z2_weights = linears0[1]
    direct_weights = readout[0, : len(feature_terms)]
    product_weight = float(readout[0, len(feature_terms)])

    normalized_coeffs = v_model.normalized_polynomial_coefficients(feature_names=list(feature_terms))
    raw_coeffs = (
        rescale_polynomial_coefficients(normalized_coeffs, feature_scales)
        if feature_scales is not None
        else normalized_coeffs
    )

    explicit = {
        "z1": f"z1 = {_format_linear_form(z1_weights, feature_terms)}",
        "z2": f"z2 = {_format_linear_form(z2_weights, feature_terms)}",
        "p": "p = z1*z2",
        "u_t_hat": (
            "u_t_hat = "
            + " ".join(
                [f"({_format_signed(direct_weights[idx])})*{feature_terms[idx]}" for idx in range(len(feature_terms))]
                + [f"({_format_signed(product_weight)})*p"]
            )
        ),
    }

    return {
        "linears_0_weight": linears0.tolist(),
        "readout_weight": readout.tolist(),
        "explicit_equations": explicit,
        "selected_normalized_basis_coefficients": _serialize_selected_terms(normalized_coeffs, feature_terms),
        "selected_raw_basis_coefficients": _serialize_selected_terms(raw_coeffs, feature_terms),
        "full_normalized_basis_coefficients": serialize_monomial_coefficients(
            _aggregate_polynomial_by_monomial(normalized_coeffs)
        ),
        "full_raw_basis_coefficients": serialize_monomial_coefficients(
            _aggregate_polynomial_by_monomial(raw_coeffs)
        ),
    }


def write_metrics_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_scalar(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6e}"
    return str(value)


def serialize_metric_block(metrics: dict[str, dict[str, float]]) -> list[str]:
    lines = []
    for name, values in metrics.items():
        lines.append(
            f"- {name}: "
            f"MSE={format_scalar(values.get('mse'))}, "
            f"RMSE={format_scalar(values.get('rmse'))}, "
            f"rel_L2={format_scalar(values.get('rel_l2'))}, "
            f"max_abs={format_scalar(values.get('max_abs'))}"
        )
    return lines


def write_pde_outputs(out_dir: Path, eql_inspection: dict) -> dict[str, str]:
    eql_dir = out_dir / "pde_outputs" / "eql"
    eql_dir.mkdir(parents=True, exist_ok=True)

    coeffs = eql_inspection["selected_raw_basis_coefficients"]
    coeff_rows = [{"term": term, "coefficient": float(value)} for term, value in coeffs.items()]
    write_metrics_csv(coeff_rows, eql_dir / "coefficients.csv")

    recovered_text = "u_t_hat = " + " + ".join(
        [f"({float(value):+.12g})*{term.replace(' ', '')}" for term, value in coeffs.items()]
    )
    (eql_dir / "pde.txt").write_text(recovered_text + "\n")
    (eql_dir / "pde.json").write_text(
        json.dumps(
            {
                "equation": recovered_text,
                "selected_raw_basis_coefficients": coeffs,
                "selected_normalized_basis_coefficients": eql_inspection["selected_normalized_basis_coefficients"],
            },
            indent=2,
        )
    )
    (eql_dir / "diagnostics.json").write_text(json.dumps(eql_inspection, indent=2))
    return {
        "dir": str(eql_dir),
        "pde_txt": str(eql_dir / "pde.txt"),
        "pde_json": str(eql_dir / "pde.json"),
        "coefficients_csv": str(eql_dir / "coefficients.csv"),
        "diagnostics_json": str(eql_dir / "diagnostics.json"),
    }


def write_research_state(
    *,
    out_dir: Path,
    cfg: RunConfig,
    derivative_source: dict,
    metrics: dict,
    recovered_pde: str,
    summary: dict,
    important_findings: list[str],
    anomalies: list[str],
) -> str:
    content = "\n".join(
        [
            "# Research State",
            "",
            "## Experiment objective",
            "Evaluate whether low surrogate solution error also yields low derivative error for the current Burgers baseline, using surrogate autodiff derivatives against clean-rollout reference derivatives.",
            "",
            "## Exact configuration",
            "```json",
            json.dumps(asdict(cfg), indent=2),
            "```",
            "",
            "## Dataset",
            f"- dataset: `{cfg.dataset_name}`",
            "- clean solution source: `Datasets/data/processed/burg_gen/burg_gen.py::solve_burgers`",
            "- numerical method: RK4 in time with second-order centered periodic finite differences in space",
            "",
            "## Source/method for reference derivatives",
            f"- u_t_ref: {derivative_source['u_t']}",
            f"- u_x_ref: {derivative_source['u_x']}",
            f"- u_xx_ref: {derivative_source['u_xx']}",
            "",
            "## Final surrogate error",
            *serialize_metric_block({"u": metrics["solution_errors"]["u"]}),
            "",
            "## Derivative errors",
            *serialize_metric_block(metrics["derivative_errors"]),
            "",
            "## PDE-feature diagnostics",
            *serialize_metric_block(metrics["feature_diagnostics"]),
            "",
            "## Recovered PDE",
            f"`{recovered_pde}`",
            "",
            "## Important visual findings",
            *[f"- {item}" for item in important_findings],
            "",
            "## Anomalies/failures",
            *[f"- {item}" for item in anomalies],
            "",
            "## Artifact directory",
            f"`{out_dir}`",
            "",
            "## Next suggested experiment",
            "- Repeat the same diagnostic with the same baseline but compare against a higher-order time-reference estimate from dense saved states to separate generator-discretization error from surrogate derivative error.",
            "",
            "## Summary file",
            f"- summary: `{out_dir / 'summary.json'}`",
            f"- metrics: `{out_dir / 'metrics.json'}`",
        ]
    ).strip() + "\n"
    path = out_dir / "research_state.md"
    path.write_text(content)
    return content


def build_coefficient_transform_summary(
    v_model: EQL,
    feature_terms: tuple[str, ...],
    feature_scales: dict[str, float],
) -> dict:
    normalized_coeffs = v_model.normalized_polynomial_coefficients(feature_names=list(feature_terms))
    raw_coeffs = rescale_polynomial_coefficients(normalized_coeffs, feature_scales)
    scale_symbols = {name: f"s_{name.replace('_', '')}" for name in feature_terms}

    product_examples = {}
    for monomial in sorted(normalized_coeffs, key=lambda m: (len(m), m)):
        if len(monomial) <= 1:
            continue
        symbolic_denom = " ".join(scale_symbols[name] for name in monomial)
        key = " * ".join(monomial)
        product_examples[key] = (
            f"c_raw({key}) = c_norm({key}) / ({symbolic_denom})"
        )

    return {
        "normalized_feature_definition": {
            name: f"{name}_norm = {name} / {scale_symbols[name]}"
            for name in feature_terms
        },
        "scale_values": {name: float(feature_scales[name]) for name in feature_terms},
        "derivation": {
            "general_rule": (
                "For any monomial m = prod_j f_j^a_j with normalized inputs z_j = f_j / s_j, "
                "c_raw(m) = c_norm(m) / prod_j s_j^a_j."
            ),
            "linear_terms": {
                name: f"c_raw({name}) = c_norm({name}) / {scale_symbols[name]}"
                for name in feature_terms
            },
            "product_terms": product_examples,
        },
        "normalized_basis_coefficients": serialize_monomial_coefficients(normalized_coeffs),
        "raw_basis_coefficients": serialize_monomial_coefficients(raw_coeffs),
    }


def run_experiment(cfg: RunConfig, *, out_dir: Path | None = None) -> dict:
    start_time = time.perf_counter()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")

    t_grid, x_grid, u_grid, derivative_mode, dataset_metadata = load_dataset(cfg.dataset_name)
    dataset = PDETrainDataset(
        t_grid=t_grid,
        x_grid=x_grid,
        u_grid=u_grid,
        stride_t=cfg.stride_t,
        stride_x=cfg.stride_x,
        noise_level=cfg.noise_level,
        seed=cfg.seed,
        normalize=True,
    )
    t_train, x_train, y_clean, y_noisy = dataset.fit_arrays_with_clean()

    fixed_feature_scales = None
    fixed_scale_info = None
    if cfg.feature_normalize:
        mode = str(cfg.feature_normalize_mode).lower()
        if mode not in {"per_batch", "global_fixed"}:
            raise ValueError(
                f"Unsupported feature_normalize_mode '{cfg.feature_normalize_mode}'. "
                "Expected 'per_batch' or 'global_fixed'."
            )
        if mode == "global_fixed":
            fixed_scale_info = compute_primitive_feature_l2_scales(
                t_train,
                x_train,
                y_clean,
                cfg.feature_terms,
                derivative_mode=derivative_mode,
            )
            fixed_feature_scales = fixed_scale_info["scales_by_name"]

    u_model = SirenMLP(
        hidden_size=cfg.hidden_size,
        hidden_layers=cfg.hidden_layers,
        first_omega_0=cfg.first_omega_0,
        hidden_omega_0=cfg.hidden_omega_0,
    )
    v_model = EQL(
        in_dim=len(cfg.feature_terms),
        prod_dim=cfg.eql_prod_dim,
        num_layers=cfg.eql_num_layers,
        bias=False,
    )
    feature_builder = _normalized_feature_builder(cfg, fixed_feature_scales)

    tv_terms = []
    if float(cfg.tv_lambda) != 0.0:
        tv_terms.append((float(cfg.tv_lambda), dispatch_tv(cfg.tv_type)))

    if int(cfg.pretrain_epochs) > 0:
        u_model, _ = fit_model_to_data(
            u_model,
            t_train,
            x_train,
            y_noisy,
            epochs=cfg.pretrain_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.pretrain_lr,
            device=cfg.device,
            weight_decay=cfg.weight_decay,
            log_every=cfg.log_every,
            tv_terms=tv_terms,
        )

    u_model, v_model, history = fit_data_and_pde(
        u_model,
        v_model,
        t_train,
        x_train,
        y_noisy,
        feature_builder,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.joint_lr,
        device=cfg.device,
        weight_decay=cfg.weight_decay,
        lam_pde=cfg.lam_pde,
        lam_data=cfg.lam_data,
        lam_sparse_eql=cfg.lam_sparse_eql,
        sparse_eql_s=cfg.sparse_eql_s,
        log_every=cfg.log_every,
        tv_terms=tv_terms,
    )

    runtime_seconds = float(time.perf_counter() - start_time)

    full_t_n, full_x_n = dataset.full_grid_normalized_flat()
    u_pred = predict_on_grid(u_model, full_t_n, full_x_n, cfg.device)
    data_mse_full = float(np.mean((u_pred - dataset.u_grid.reshape(-1)) ** 2))

    model_derivatives = evaluate_model_solution_and_derivatives(
        u_model,
        full_t_n,
        full_x_n,
        device=cfg.device,
    )
    reference = build_reference_diagnostics(
        dataset,
        cfg.feature_terms,
        derivative_mode=derivative_mode,
        dataset_name=cfg.dataset_name,
        dataset_metadata=dataset_metadata,
    )
    derivative_errors = {
        "u": compute_error_metrics(model_derivatives["u"], reference["u_flat"]),
        "u_t": compute_error_metrics(model_derivatives["u_t"], reference["features"]["u_t"]),
        "u_x": compute_error_metrics(model_derivatives["u_x"], reference["features"]["u_x"]),
        "u_xx": compute_error_metrics(model_derivatives["u_xx"], reference["features"]["u_xx"]),
    }

    feature_metrics = evaluate_primitive_feature_metrics(
        u_model,
        full_t_n,
        full_x_n,
        dataset.u_grid.reshape(-1),
        cfg.feature_terms,
        device=cfg.device,
        derivative_mode=derivative_mode,
    )

    if out_dir is None:
        out_dir = Path(cfg.output_root) if cfg.output_root else default_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlays = save_primitive_feature_overlays(
        u_model,
        full_t_n,
        full_x_n,
        dataset.u_grid.reshape(-1),
        cfg.feature_terms,
        output_dir=out_dir / "feature_overlays",
        device=cfg.device,
        derivative_mode=derivative_mode,
        max_time_slices=cfg.max_time_slices,
    )

    t_unique = np.asarray(dataset.t_grid, dtype=np.float64).reshape(-1)
    x_unique = np.asarray(dataset.x_grid, dtype=np.float64).reshape(-1)
    u_ref_grid = np.asarray(reference["u_flat"], dtype=np.float64).reshape(dataset.u_grid.shape)
    u_pred_grid = np.asarray(model_derivatives["u"], dtype=np.float64).reshape(dataset.u_grid.shape)
    ut_ref_grid = np.asarray(reference["features"]["u_t"], dtype=np.float64).reshape(dataset.u_grid.shape)
    ut_pred_grid = np.asarray(model_derivatives["u_t"], dtype=np.float64).reshape(dataset.u_grid.shape)
    ux_ref_grid = np.asarray(reference["features"]["u_x"], dtype=np.float64).reshape(dataset.u_grid.shape)
    ux_pred_grid = np.asarray(model_derivatives["u_x"], dtype=np.float64).reshape(dataset.u_grid.shape)
    uxx_ref_grid = np.asarray(reference["features"]["u_xx"], dtype=np.float64).reshape(dataset.u_grid.shape)
    uxx_pred_grid = np.asarray(model_derivatives["u_xx"], dtype=np.float64).reshape(dataset.u_grid.shape)
    uux_ref_grid = u_ref_grid * ux_ref_grid
    uux_pred_grid = u_pred_grid * ux_pred_grid
    sample_grid = build_training_sample_grid(dataset)
    time_indices = build_snapshot_time_indices(len(t_unique))
    snapshot_files = {
        "u": [str(path) for path in save_time_slice_snapshots(
            x_grid=x_unique,
            t_grid=t_unique,
            true_grid=u_ref_grid,
            pred_grid=u_pred_grid,
            sample_grid=sample_grid,
            output_dir=out_dir / "snapshots" / "u",
            value_name="u",
            time_indices=time_indices,
            true_label="u_ref",
            pred_label="u_theta",
        )]
    }
    for feature_name, ref_grid, pred_grid in (
        ("u_t", ut_ref_grid, ut_pred_grid),
        ("u_x", ux_ref_grid, ux_pred_grid),
        ("u_xx", uxx_ref_grid, uxx_pred_grid),
    ):
        snapshot_files[feature_name] = [str(path) for path in save_time_slice_snapshots(
            x_grid=x_unique,
            t_grid=t_unique,
            true_grid=ref_grid,
            pred_grid=pred_grid,
            output_dir=out_dir / "snapshots" / primitive_feature_name_to_key(feature_name),
            value_name=feature_name,
            time_indices=time_indices,
            true_label=f"{feature_name}_ref",
            pred_label=f"{feature_name}_AD",
        )]

    abs_error_grids = {
        "u": np.abs(u_pred_grid - u_ref_grid),
        "u_t": np.abs(ut_pred_grid - ut_ref_grid),
        "u_x": np.abs(ux_pred_grid - ux_ref_grid),
        "u_xx": np.abs(uxx_pred_grid - uxx_ref_grid),
        "u*u_x": np.abs(uux_pred_grid - uux_ref_grid),
    }
    heatmap_files = {}
    for name, grid in abs_error_grids.items():
        key = primitive_feature_name_to_key(name)
        heatmap_files[name] = str(
            save_space_time_heatmap(
                x_grid=x_unique,
                t_grid=t_unique,
                value_grid=grid,
                output_path=out_dir / "heatmaps" / f"{key}_abs_error_heatmap.pdf",
                title=f"|{name}_AD - {name}_ref|",
                colorbar_label="absolute error",
            )
        )

    feature_diagnostics = {
        "u": compute_error_metrics(u_pred_grid.reshape(-1), u_ref_grid.reshape(-1)),
        "u_x": compute_error_metrics(ux_pred_grid.reshape(-1), ux_ref_grid.reshape(-1)),
        "u_xx": compute_error_metrics(uxx_pred_grid.reshape(-1), uxx_ref_grid.reshape(-1)),
        "u*u_x": compute_error_metrics(uux_pred_grid.reshape(-1), uux_ref_grid.reshape(-1)),
    }

    history_rows = [asdict(row) for row in (history.rows or [])]
    augmented_history_rows = compute_augmented_history_rows(history_rows, cfg)
    final_row = augmented_history_rows[-1]
    min_row = min(augmented_history_rows, key=lambda row: float(row["total_loss"]))
    min_data_row = min(augmented_history_rows, key=lambda row: float(row["raw_data_mse"]))
    min_pde_row = min(augmented_history_rows, key=lambda row: float(row["raw_pde_mse"]))
    product_coeffs = serialize_product_coefficients(
        v_model.product_coefficients(feature_names=list(cfg.feature_terms))
    )
    pde_diagnostics = collect_pde_diagnostics(
        u_model,
        v_model,
        dataset,
        cfg,
        fixed_feature_scales=fixed_feature_scales,
    )
    coefficient_summary = None
    if cfg.feature_normalize:
        coefficient_summary = build_coefficient_transform_summary(
            v_model,
            cfg.feature_terms,
            pde_diagnostics["feature_scales_full_grid"],
        )
    eql_inspection = build_eql_inspection(
        v_model,
        cfg.feature_terms,
        pde_diagnostics["feature_scales_full_grid"] if cfg.feature_normalize else None,
    )
    pde_output_files = write_pde_outputs(out_dir, eql_inspection)

    metrics_payload = {
        "solution_errors": {"u": derivative_errors["u"]},
        "derivative_errors": {
            "u_t": derivative_errors["u_t"],
            "u_x": derivative_errors["u_x"],
            "u_xx": derivative_errors["u_xx"],
        },
        "feature_diagnostics": feature_diagnostics,
    }

    selected_raw_coeffs = eql_inspection["selected_raw_basis_coefficients"]
    recovered_pde = "u_t_hat = " + " + ".join(
        [f"({float(value):+.12g})*{term.replace(' ', '')}" for term, value in selected_raw_coeffs.items()]
    )

    important_findings = [
        "Solution and derivative slice plots share the same physical time anchors used in the diagnostic snapshots.",
        "Absolute derivative-error heatmaps expose where surrogate fit quality and derivative quality diverge across the full rollout.",
        "The composite feature u*u_x is evaluated independently from the trained PDE readout using the surrogate solution and autodiff u_x.",
    ]
    anomalies = []
    if derivative_errors["u_t"]["rel_l2"] > derivative_errors["u"]["rel_l2"] * 2.0:
        anomalies.append("Time-derivative error is substantially larger than solution error.")
    if derivative_errors["u_xx"]["rel_l2"] > derivative_errors["u_x"]["rel_l2"]:
        anomalies.append("Second-derivative fidelity degrades relative to first-derivative fidelity.")
    if not anomalies:
        anomalies.append("No obvious anomaly threshold fired; inspect the heatmaps and time-slice plots.")

    training_metrics = {
        "final": {
            "epoch": int(final_row["epoch"]),
            "data_loss": float(final_row["raw_data_mse"]),
            "raw_pde_mse": float(final_row["raw_pde_mse"]),
            "weighted_pde_loss": float(final_row["weighted_pde_contribution"]),
            "total_loss": float(final_row["total_loss"]),
        },
        "minimum_data_loss": {
            "epoch": int(min_data_row["epoch"]),
            "data_loss": float(min_data_row["raw_data_mse"]),
        },
        "minimum_raw_pde_mse": {
            "epoch": int(min_pde_row["epoch"]),
            "raw_pde_mse": float(min_pde_row["raw_pde_mse"]),
            "weighted_pde_loss": float(min_pde_row["weighted_pde_contribution"]),
        },
        "minimum_total_loss": {
            "epoch": int(min_row["epoch"]),
            "total_loss": float(min_row["total_loss"]),
        },
    }

    summary = {
        "dataset": cfg.dataset_name,
        "seed": cfg.seed,
        "device": cfg.device,
        "num_samples": int(len(dataset)),
        "grid_shape": [int(dataset.u_grid.shape[0]), int(dataset.u_grid.shape[1])],
        "config": asdict(cfg),
        "final_total_loss": final_row["total_loss"],
        "final_data_loss": final_row["data_loss"],
        "final_pde_loss": final_row["pde_loss"],
        "final_tv_loss": final_row["tv_loss"],
        "final_sparse_eql_loss": final_row["sparse_eql_loss"],
        "final_raw_data_mse": final_row["raw_data_mse"],
        "final_weighted_data_contribution": final_row["weighted_data_contribution"],
        "final_raw_pde_mse": final_row["raw_pde_mse"],
        "final_weighted_pde_contribution": final_row["weighted_pde_contribution"],
        "min_total_loss": float(min(history.losses)),
        "min_total_loss_epoch": int(min_row["epoch"]),
        "data_mse_full_grid": data_mse_full,
        "runtime_seconds": runtime_seconds,
        "training_metrics": training_metrics,
        "final_5_epochs": augmented_history_rows[-5:],
        "derivative_errors": derivative_errors,
        "pde_diagnostics": pde_diagnostics,
        "feature_normalization": {
            "enabled": bool(cfg.feature_normalize),
            "mode": cfg.feature_normalize_mode if cfg.feature_normalize else "none",
            "training_dataset_l2_scales": fixed_feature_scales,
            "training_dataset_scale_metadata": fixed_scale_info,
        },
        "feature_metrics": feature_metrics,
        "metrics": metrics_payload,
        "eql_product_coefficients": product_coeffs,
        "eql_coefficient_recovery": coefficient_summary,
        "eql_inspection": eql_inspection,
        "recovered_pde": recovered_pde,
        "reference_derivative_source": reference["derivative_source"],
        "feature_overlay_files": [str(path) for path in overlays],
        "snapshot_time_indices": [int(idx) for idx in time_indices],
        "snapshot_times": [float(t_unique[idx]) for idx in time_indices],
        "snapshot_files": snapshot_files,
        "heatmap_files": heatmap_files,
        "pde_output_files": pde_output_files,
    }

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (out_dir / "derivative_metrics.json").write_text(json.dumps(metrics_payload["derivative_errors"], indent=2))
    (out_dir / "loss_history.json").write_text(json.dumps(augmented_history_rows, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "eql_inspection.json").write_text(json.dumps(eql_inspection, indent=2))
    write_metrics_csv(augmented_history_rows, out_dir / "loss_history.csv")
    torch.save(
        {
            "u_model_state_dict": u_model.state_dict(),
            "v_model_state_dict": v_model.state_dict(),
            "config": asdict(cfg),
        },
        out_dir / "models.pt",
    )
    research_state_text = write_research_state(
        out_dir=out_dir,
        cfg=cfg,
        derivative_source=reference["derivative_source"],
        metrics=metrics_payload,
        recovered_pde=recovered_pde,
        summary=summary,
        important_findings=important_findings,
        anomalies=anomalies,
    )
    (Path("research_state.md")).write_text(research_state_text)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name")
    parser.add_argument("--device")
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--pretrain-lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--joint-lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--stride-t", type=int)
    parser.add_argument("--stride-x", type=int)
    parser.add_argument("--lam-pde", type=float)
    parser.add_argument("--lam-data", type=float)
    parser.add_argument("--lam-sparse-eql", type=float)
    parser.add_argument("--tv-lambda", type=float)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--hidden-layers", type=int)
    parser.add_argument("--first-omega-0", type=float)
    parser.add_argument("--hidden-omega-0", type=float)
    parser.add_argument("--eql-prod-dim", type=int)
    parser.add_argument("--eql-num-layers", type=int)
    parser.add_argument("--feature-normalize", action="store_true")
    parser.add_argument("--feature-normalize-mode", choices=("per_batch", "global_fixed"))
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--output-tag")
    args = parser.parse_args()

    cfg = RunConfig()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key.replace("-", "_"), value)
    summary = run_experiment(cfg)
    print(json.dumps(summary, indent=2))


def serialize_product_coefficients(product_coeffs):
    serialized = {}
    for prod_name, poly in product_coeffs.items():
        serialized[prod_name] = {
            " * ".join(monomial) if monomial else "1": float(coeff)
            for monomial, coeff in poly.items()
        }
    return serialized


if __name__ == "__main__":
    main()
