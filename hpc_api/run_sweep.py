import copy
import json
from pathlib import Path

import pandas as pd

from .train_one import train_one


def _fmt_value_for_path(v):
    if isinstance(v, float):
        s = f"{v:.6g}"
    else:
        s = str(v)
    s = s.replace(".", "p").replace("-", "m").replace("+", "")
    return s


def run_sweep(base_config, sweep_param, sweep_values, sweep_root, train_fn, *, overwrite: bool = False):
    """
    Executes multiple runs across sweep values.
    """

    sweep_root = Path(sweep_root)
    runs_dir = sweep_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with open(sweep_root / "sweep_config.json", "w") as f:
        json.dump(
            {
                "base_config": base_config,
                "sweep_param": sweep_param,
                "values": sweep_values,
            },
            f,
            indent=2,
        )

    summaries = []

    for value in sweep_values:

        cfg = copy.deepcopy(base_config)
        cfg[sweep_param] = value
        cfg["sweep_param_name"] = sweep_param
        cfg["sweep_param_value"] = value

        run_name = f"{sweep_param}_{_fmt_value_for_path(value)}"
        run_dir = runs_dir / run_name

        summary_path = run_dir / "summary.json"
        if not overwrite and summary_path.exists():
            try:
                prev = json.loads(summary_path.read_text(encoding="utf-8"))
                if int(prev.get("status", 0)) == 1:
                    summaries.append(prev)
                    continue
            except Exception:
                pass

        summary = train_one(cfg, run_dir, train_fn)
        summaries.append(summary)

    df = pd.DataFrame(summaries)
    df.to_csv(sweep_root / "sweep_results.csv", index=False)

    return df
