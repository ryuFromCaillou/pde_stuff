import json
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _json_dump(path: Path, obj) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _normalize_history(history):
    if history is None:
        return []
    if not isinstance(history, list):
        raise TypeError("train_fn result['history'] must be a list[dict].")
    rows = []
    for row in history:
        if not isinstance(row, dict):
            raise TypeError("history entries must be dicts.")
        rows.append(row)
    return rows


def train_one(config, run_dir, train_fn):
    """
    Runs a single training job.

    Parameters
    ----------
    config : dict
        Training configuration
    run_dir : Path
        Directory where artifacts will be written
    train_fn : callable
        User training function returning:
        {
            "history": list[dict],
            "best_epoch": int,
            "snapshot_path": str | Path | None,
            "status": 0 or 1
        }
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _json_dump(run_dir / "config.json", config)

    start = time.time()

    try:
        result = train_fn(run_dir, config)
    except Exception as e:
        runtime = time.time() - start
        tb = traceback.format_exc()
        _json_dump(run_dir / "error.json", {"error": repr(e)})
        (run_dir / "traceback.txt").write_text(tb, encoding="utf-8")

        summary = {
            "run_name": run_dir.name,
            "sweep_param_name": config.get("sweep_param_name"),
            "sweep_param_value": config.get("sweep_param_value"),
            "best_epoch": None,
            "final_epoch": None,
            "final_total_loss": float("nan"),
            "final_tv_loss": float("nan"),
            "final_l1_loss": float("nan"),
            "final_pde_loss": float("nan"),
            "final_data_loss": float("nan"),
            "runtime_sec": runtime,
            "snapshot_path": None,
            "status": 0,
        }
        _json_dump(run_dir / "summary.json", summary)
        return summary

    runtime = time.time() - start

    history = _normalize_history(result.get("history"))

    df = pd.DataFrame(history)
    df.to_csv(run_dir / "history.csv", index=False)

    if len(df) == 0:
        final_row = {}
    else:
        final_row = df.iloc[-1].to_dict()

    summary = {
        "run_name": run_dir.name,
        "sweep_param_name": config.get("sweep_param_name"),
        "sweep_param_value": config.get("sweep_param_value"),
        "best_epoch": result.get("best_epoch"),
        "final_epoch": int(final_row.get("epoch")) if final_row.get("epoch") is not None else None,
        "final_total_loss": float(final_row.get("total_loss")) if final_row.get("total_loss") is not None else float("nan"),
        "final_tv_loss": float(final_row.get("tv_loss")) if final_row.get("tv_loss") is not None else float("nan"),
        "final_l1_loss": float(final_row.get("l1_loss")) if final_row.get("l1_loss") is not None else float("nan"),
        "final_pde_loss": float(final_row.get("pde_loss")) if final_row.get("pde_loss") is not None else float("nan"),
        "final_data_loss": float(final_row.get("data_loss")) if final_row.get("data_loss") is not None else float("nan"),
        "runtime_sec": runtime,
        "snapshot_path": str(result.get("snapshot_path")) if result.get("snapshot_path") is not None else None,
        "status": int(result.get("status", 1)),
    }

    extra = result.get("summary_extra")
    if extra is not None:
        if not isinstance(extra, dict):
            raise TypeError("train_fn result['summary_extra'] must be a dict if provided.")
        summary.update(extra)

    _json_dump(run_dir / "summary.json", summary)

    plot_losses(df, run_dir)

    return summary


def plot_losses(df, run_dir):
    if df is None or len(df) == 0 or "epoch" not in df.columns:
        return
    plt.figure(figsize=(8, 5))

    for col in ["data_loss", "pde_loss", "tv_loss", "l1_loss"]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curves.png")
    plt.close()
