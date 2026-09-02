from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from run_eql_joint_training import RunConfig, format_float_token, run_experiment


LAM_PDE_VALUES = [0.5, 0.1, 0.01, 0.001, 0.0001]


def flatten_feature_metrics(summary: dict) -> dict:
    feature_metrics = summary.get("feature_metrics", {})
    if isinstance(feature_metrics, dict):
        summary_flat = feature_metrics.get("summary_flat")
        if isinstance(summary_flat, dict):
            return dict(summary_flat)

        metrics_by_name = feature_metrics.get("metrics_by_name")
        if isinstance(metrics_by_name, dict):
            flat = {}
            for feature_name, metrics in metrics_by_name.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name, metric_value in metrics.items():
                    flat[f"{feature_name}_{metric_name}"] = metric_value
            return flat
    return {}


def build_summary_row(summary: dict, run_dir: Path) -> dict:
    cfg = summary["config"]
    row = {
        "dataset": summary["dataset"],
        "seed": summary["seed"],
        "model": "eql_joint_training",
        "lam_pde": cfg["lam_pde"],
        "lam_data": cfg["lam_data"],
        "tv_lambda": cfg["tv_lambda"],
        "lam_sparse_eql": cfg["lam_sparse_eql"],
        "sparse_eql_s": cfg["sparse_eql_s"],
        "hidden_size": cfg["hidden_size"],
        "hidden_layers": cfg["hidden_layers"],
        "first_omega_0": cfg["first_omega_0"],
        "hidden_omega_0": cfg["hidden_omega_0"],
        "pretrain_epochs": cfg["pretrain_epochs"],
        "epochs": cfg["epochs"],
        "pretrain_lr": cfg["pretrain_lr"],
        "joint_lr": cfg["joint_lr"],
        "batch_size": cfg["batch_size"],
        "stride_t": cfg["stride_t"],
        "stride_x": cfg["stride_x"],
        "feature_names": ",".join(cfg["feature_terms"]),
        "final_total_loss": summary["final_total_loss"],
        "min_total_loss": summary["min_total_loss"],
        "min_total_loss_epoch": summary["min_total_loss_epoch"],
        "final_raw_data_mse": summary["final_raw_data_mse"],
        "final_weighted_data_contribution": summary["final_weighted_data_contribution"],
        "final_raw_pde_mse": summary["final_raw_pde_mse"],
        "final_weighted_pde_contribution": summary["final_weighted_pde_contribution"],
        "final_tv_loss": summary["final_tv_loss"],
        "final_sparse_eql_loss": summary["final_sparse_eql_loss"],
        "data_mse_full_grid": summary["data_mse_full_grid"],
        "runtime_seconds": summary["runtime_seconds"],
        "run_dir": str(run_dir),
    }
    row.update(flatten_feature_metrics(summary))
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_agg(path: Path, rows: list[dict]) -> None:
    agg_rows = []
    for row in rows:
        agg_rows.append(
            {
                "lam_pde": row["lam_pde"],
                "num_seeds": 1,
                "final_train_loss_mean": row["final_total_loss"],
                "final_train_loss_std": 0.0,
                "min_train_loss_mean": row["min_total_loss"],
                "min_train_loss_std": 0.0,
                "final_raw_data_mse_mean": row["final_raw_data_mse"],
                "final_raw_data_mse_std": 0.0,
                "final_raw_pde_mse_mean": row["final_raw_pde_mse"],
                "final_raw_pde_mse_std": 0.0,
                "runtime_seconds_mean": row["runtime_seconds"],
                "runtime_seconds_std": 0.0,
            }
        )
    write_csv(path, agg_rows)


def main():
    cfg = RunConfig(dataset_name="burg_gen")
    sweep_root = (
        Path("run_results")
        / "lam_pde_sweep"
        / cfg.dataset_name.lower()
    )
    sweep_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    run_index = []

    for lam_pde in LAM_PDE_VALUES:
        run_cfg = RunConfig(**asdict(cfg))
        run_cfg.lam_pde = lam_pde
        lam_dir = sweep_root / f"lam_pde_{format_float_token(lam_pde)}" / f"seed_{run_cfg.seed:03d}"
        summary_path = lam_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = run_experiment(run_cfg, out_dir=lam_dir)
        summary_rows.append(build_summary_row(summary, lam_dir))
        run_index.append(
            {
                "lam_pde": lam_pde,
                "run_dir": str(lam_dir),
                "summary_file": str(lam_dir / "summary.json"),
            }
        )

    write_csv(sweep_root / "summary.csv", summary_rows)
    write_summary_agg(sweep_root / "summary_agg.csv", summary_rows)
    (sweep_root / "run_index.json").write_text(json.dumps(run_index, indent=2))


if __name__ == "__main__":
    main()
