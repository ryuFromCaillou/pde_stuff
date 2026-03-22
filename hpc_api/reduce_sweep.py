import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def reduce_sweep(sweep_root):
    """
    Generates sweep-level comparison plots.
    """

    sweep_root = Path(sweep_root)

    results_path = sweep_root / "sweep_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    results = pd.read_csv(results_path)

    results = results[results.status == 1]
    if len(results) == 0:
        plots_dir = sweep_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        (plots_dir / "README.txt").write_text("No successful runs (status==1).", encoding="utf-8")
        return results

    sweep_param = results["sweep_param_name"].iloc[0]

    plots_dir = sweep_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Best run (by PDE loss)
    if "final_pde_loss" in results.columns:
        best = results.sort_values("final_pde_loss").iloc[0].to_dict()
        (plots_dir / "best_run.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")

    for metric in [
        "final_tv_loss",
        "final_l1_loss",
        "final_pde_loss",
        "final_data_loss",
        "final_total_loss",
    ]:

        if metric not in results.columns:
            continue

        df = results.copy()
        df["_sweep_val_num"] = pd.to_numeric(df["sweep_param_value"], errors="coerce")
        if df["_sweep_val_num"].notna().any():
            df = df.sort_values("_sweep_val_num")
        else:
            df = df.sort_values("sweep_param_value")

        plt.figure(figsize=(7, 5))
        plt.plot(
            df["sweep_param_value"],
            df[metric],
            marker="o",
        )

        plt.xlabel(sweep_param)
        plt.ylabel(metric)
        plt.tight_layout()

        plt.savefig(plots_dir / f"{metric}_vs_{sweep_param}.png")
        plt.close()

    return results
