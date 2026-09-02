from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
    compute_primitive_feature_l2_scales,
    evaluate_primitive_feature_metrics,
)
from utils.feature_plotting import save_primitive_feature_overlays
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
        return t, x, u, "periodic"

    if dataset_name == "burgers":
        path = Path("Datasets/data/raw") / f"{dataset_name}.mat"
        raw = loadmat(path)
        t = raw["t"].reshape(-1)
        x = raw["x"].reshape(-1)
        u = np.asarray(raw["usol"], dtype=np.float64).T
        return t, x, u, "periodic"

    if dataset_name == "burg_gen":
        x, _u_final, _t_end, (t, u) = solve_burgers(
            seed=0,
            return_history=True,
        )
        return t, x, np.asarray(u, dtype=np.float64), "periodic"

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

    t_grid, x_grid, u_grid, derivative_mode = load_dataset(cfg.dataset_name)
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

    history_rows = [asdict(row) for row in (history.rows or [])]
    augmented_history_rows = compute_augmented_history_rows(history_rows, cfg)
    final_row = augmented_history_rows[-1]
    min_row = min(augmented_history_rows, key=lambda row: float(row["total_loss"]))
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
        "final_5_epochs": augmented_history_rows[-5:],
        "pde_diagnostics": pde_diagnostics,
        "feature_normalization": {
            "enabled": bool(cfg.feature_normalize),
            "mode": cfg.feature_normalize_mode if cfg.feature_normalize else "none",
            "training_dataset_l2_scales": fixed_feature_scales,
            "training_dataset_scale_metadata": fixed_scale_info,
        },
        "feature_metrics": feature_metrics,
        "eql_product_coefficients": product_coeffs,
        "eql_coefficient_recovery": coefficient_summary,
        "feature_overlay_files": [str(path) for path in overlays],
    }

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "loss_history.json").write_text(json.dumps(augmented_history_rows, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        {
            "u_model_state_dict": u_model.state_dict(),
            "v_model_state_dict": v_model.state_dict(),
            "config": asdict(cfg),
        },
        out_dir / "models.pt",
    )
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
