"""Sweep over PDE loss weight (`lambda_pde`) with full artifact generation per runs/AGENTS.md.

Saves structured outputs to `run_results/pde_lambda_sweep/burgers/lambda_X/seed_###/` by default.

Per-seed outputs:
  - config.json: sweep parameters
  - loss_history.csv: training loss at each step
  - summary.json: final metrics + PDE extraction results
  - pde_outputs/least_squares/: LS-extracted coefficients
  - pde_outputs/eql/: EQL coefficients (if EQL is fit)

Aggregated outputs (across seeds):
  - summary.csv: one row per lambda/seed combination
  - summary_agg.csv: aggregated across seeds (mean/std)

Usage:
    python runs/run_pde_lambda_sweep.py --lambdas 0,0.01,0.1,1 --epochs 100 --num-seeds 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from prog.mlps import SirenMLP, EQL
from prog.trainer import TrainerConfig, PDETrainer
from Datasets.data.processed.burg_gen.burg_gen import BurgersDatasetConfig, build_dataset_from_burgers
from utils.extract_pde_ls import extract_pde_ls


def parse_lambdas(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def save_pde_extraction(extraction: dict, output_dir: str, method_name: str = "least_squares") -> None:
    """Save PDE extraction results in spec format."""
    pde_dir = Path(output_dir) / "pde_outputs" / method_name
    pde_dir.mkdir(parents=True, exist_ok=True)

    coeffs = np.asarray(extraction["coeffs"], dtype=float)
    names = extraction["names"]

    # pde.json
    residuals = np.asarray(extraction.get("residuals", [])).tolist()
    singular_values = np.asarray(extraction.get("singular_values", [])).tolist()
    pde_json = {
        "method": method_name,
        "feature_names": names,
        "coefficients": coeffs.tolist(),
        "residuals": residuals,
        "rank": int(extraction.get("rank", -1)),
        "singular_values": singular_values,
    }
    with open(pde_dir / "pde.json", "w") as f:
        json.dump(pde_json, f, indent=2)

    # pde.txt (human-readable)
    pde_txt = "PDE Extraction (Least-Squares)\n" + "=" * 40 + "\n"
    for name, coeff in zip(names, coeffs):
        pde_txt += f"{name:12s}: {coeff:12.6e}\n"
    with open(pde_dir / "pde.txt", "w") as f:
        f.write(pde_txt)

    # coefficients.csv
    with open(pde_dir / "coefficients.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_name", "coefficient"])
        writer.writeheader()
        for name, coeff in zip(names, coeffs):
            writer.writerow({"feature_name": name, "coefficient": float(coeff)})

    # diagnostics.json
    diagnostics = {
        "rank": int(extraction.get("rank", -1)),
        "singular_values": singular_values,
        "residuals": residuals,
    }
    with open(pde_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)


def run_single(
    cfg_kwargs: dict,
    data_kwargs: dict,
    lambda_pde: float,
    seed: int,
    base_out_dir: str,
) -> dict:
    """Train on a single lambda_pde value and seed."""

    # Setup output directory
    seed_dir = Path(base_out_dir) / "burgers" / f"lambda_{lambda_pde:.6f}".rstrip('0').rstrip('.') / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    data_cfg = dict(data_kwargs)
    data_cfg["seed"] = seed
    bcfg = BurgersDatasetConfig(**data_cfg)
    t_s, x_s, y_clean, y_noisy, _ = build_dataset_from_burgers(bcfg)

    # Create dataloader
    t_t = torch.from_numpy(t_s).view(-1, 1)
    x_t = torch.from_numpy(x_s).view(-1, 1)
    y_noisy_t = torch.from_numpy(y_noisy).view(-1, 1)
    y_clean_t = torch.from_numpy(y_clean).view(-1, 1)

    dataset = TensorDataset(t_t, x_t, y_noisy_t, y_clean_t)
    loader = DataLoader(dataset, batch_size=cfg_kwargs.get("batch_size", 1024), shuffle=True)

    # Models
    u_model = SirenMLP(
        hidden_size=cfg_kwargs.get("hidden_size", 64),
        hidden_layers=cfg_kwargs.get("hidden_layers", 3),
    )
    selected_derivs = tuple(cfg_kwargs.get("selected_derivs", ("u", "u_x", "u_xx")))
    v_model = EQL(in_dim=len(selected_derivs))

    # Trainer config
    cfg = TrainerConfig(lr=cfg_kwargs.get("lr", 1e-3))
    cfg.lambda_pde = float(lambda_pde)
    cfg.lambda_data = float(cfg_kwargs.get("lambda_data", 1.0))
    cfg.lambda_reg = float(cfg_kwargs.get("lambda_reg", 1e-3))
    cfg.lambda_tv = float(cfg_kwargs.get("lambda_tv", 0.0))
    cfg.selected_derivs = selected_derivs
    cfg.device = torch.device(cfg_kwargs.get("device", "cpu"))
    setattr(cfg, "feature_normalize", True)

    trainer = PDETrainer(u_model, v_model, cfg)

    epochs = int(cfg_kwargs.get("epochs", 200))

    # Save config
    config_dict = {
        "lambda_pde": float(lambda_pde),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(cfg_kwargs.get("batch_size", 1024)),
        "lr": float(cfg_kwargs.get("lr", 1e-3)),
        "lambda_data": float(cfg_kwargs.get("lambda_data", 1.0)),
        "lambda_reg": float(cfg_kwargs.get("lambda_reg", 1e-3)),
        "lambda_tv": float(cfg_kwargs.get("lambda_tv", 0.0)),
        "selected_derivs": list(selected_derivs),
        "noise_level": float(data_kwargs.get("noise_level", 0.05)),
    }
    with open(seed_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Training loop with loss history
    loss_history = []
    for epoch in range(epochs):
        for batch in loader:
            t_b, x_b, u_noisy_b, u_clean_b = batch
            metrics = trainer.step(t_b, x_b, u_noisy_b, u_clean_b)
            loss_history.append(metrics)

    # Save loss history
    if loss_history:
        with open(seed_dir / "loss_history.csv", "w", newline="") as f:
            fieldnames = list(loss_history[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in loss_history:
                # Convert numpy arrays and tensors to lists
                row = {}
                for k, v in entry.items():
                    if isinstance(v, (list, tuple)):
                        row[k] = "|".join(map(str, v))
                    else:
                        row[k] = str(v)
                writer.writerow(row)

    # Extract PDE on full grid
    extraction = extract_pde_ls(u_model, t_s, x_s, selected_derivs, device=str(cfg.device))
    save_pde_extraction(extraction, str(seed_dir), method_name="least_squares")

    # Prepare summary
    final_metrics = loss_history[-1] if loss_history else {}
    summary = {
        "lambda_pde": float(lambda_pde),
        "seed": int(seed),
        "final_loss": float(final_metrics.get("loss", float("nan"))),
        "final_loss_data": float(final_metrics.get("loss_data", float("nan"))),
        "final_loss_pde": float(final_metrics.get("loss_pde", float("nan"))),
        "l1": float(final_metrics.get("l1", float("nan"))),
        "loss_tv": float(final_metrics.get("loss_tv", float("nan"))),
        "pde_method": "least_squares",
        "feature_names": extraction["names"],
        "coefficients": extraction["coeffs"].tolist(),
        "num_samples": int(len(t_s)),
    }
    with open(seed_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def aggregate_results(base_out_dir: str, lambdas: list[float], num_seeds: int) -> None:
    """Aggregate results across seeds and create summary files."""
    base_path = Path(base_out_dir)

    all_rows = []

    for lam in lambdas:
        lam_dir = base_path / "burgers" / f"lambda_{lam:.6f}".rstrip('0').rstrip('.')
        if not lam_dir.exists():
            continue

        seed_summaries = []
        for seed in range(num_seeds):
            seed_dir = lam_dir / f"seed_{seed:03d}"
            summary_file = seed_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                seed_summaries.append(summary)
                all_rows.append(summary)

        # Aggregate across seeds for this lambda
        if seed_summaries:
            agg = {"lambda_pde": lam}
            for key in ["final_loss", "final_loss_data", "final_loss_pde", "l1", "loss_tv"]:
                values = [s.get(key, float("nan")) for s in seed_summaries]
                values = [v for v in values if not np.isnan(v)]
                if values:
                    agg[f"{key}_mean"] = float(np.mean(values))
                    agg[f"{key}_std"] = float(np.std(values))
                    agg[f"{key}_min"] = float(np.min(values))
                    agg[f"{key}_max"] = float(np.max(values))

    # Write summary.csv (one row per lambda/seed)
    summary_csv = base_path / "summary.csv"
    if all_rows:
        with open(summary_csv, "w", newline="") as f:
            fieldnames = sorted(set().union(*[r.keys() for r in all_rows]))
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                # Flatten lists/arrays to strings
                flat_row = {}
                for k, v in row.items():
                    if isinstance(v, (list, tuple)):
                        flat_row[k] = "|".join(map(str, v))
                    else:
                        flat_row[k] = str(v)
                writer.writerow(flat_row)
        print(f"Summary saved to {summary_csv}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sweep `lambda_pde` with per-seed structured outputs.")
    p.add_argument("--lambdas", type=str, default="0,0.001,0.01,0.1,1",
                   help="Comma-separated lambda_pde values")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=str, default="run_results/pde_lambda_sweep")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--noise", type=float, default=0.05)

    args = p.parse_args(argv)

    lambdas = parse_lambdas(args.lambdas)
    os.makedirs(args.out_dir, exist_ok=True)

    data_kwargs = {"noise_level": float(args.noise)}
    cfg_kwargs = {
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "device": args.device,
    }

    for lam in lambdas:
        for seed in range(int(args.num_seeds)):
            print(f"Running lambda_pde={lam:.6f}, seed={seed:03d} ...")
            run_single(cfg_kwargs, data_kwargs, lam, seed, args.out_dir)

    # Aggregate
    print("Aggregating results ...")
    aggregate_results(args.out_dir, lambdas, int(args.num_seeds))

    print(f"Sweep complete — outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
