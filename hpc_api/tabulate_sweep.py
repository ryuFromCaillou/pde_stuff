from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_COLUMNS = [
    "run_name",
    "sweep_param_name",
    "sweep_param_value",
    "best_epoch",
    "final_epoch",
    "final_total_loss",
    "final_pde_loss",
    "final_data_loss",
    "final_tv_loss",
    "final_l1_loss",
    "w_u",
    "w_ux",
    "w_uxx",
    "w_prod",
    "M01",
    "M10",
    "runtime_sec",
    "snapshot_path",
    "status",
]


def tabulate_sweep(
    sweep_root: str | Path,
    *,
    columns: Iterable[str] | None = None,
    status_only: bool = True,
    sort_by: str | None = "sweep_param_value",
    out_dir: str | Path | None = None,
    basename: str = "sweep_table",
) -> pd.DataFrame:
    """
    Read sweep results and emit email-friendly tables.

    Inputs
    ------
    sweep_root : path-like
        Directory containing `sweep_results.csv`.
    columns : optional iterable[str]
        Columns to keep; defaults to `DEFAULT_COLUMNS` (missing columns are ignored).
    status_only : bool
        If True, keep only rows with `status == 1` when present.
    sort_by : optional str
        Column to sort by (numeric sort attempted).
    out_dir : optional path-like
        Where to write `{basename}.csv` and `{basename}.md`. Defaults to `sweep_root`.
    basename : str
        Output filename prefix.
    """
    sweep_root = Path(sweep_root)
    results_path = sweep_root / "sweep_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    df = pd.read_csv(results_path)

    if status_only and "status" in df.columns:
        df = df[df["status"] == 1].copy()

    cols = list(columns) if columns is not None else list(DEFAULT_COLUMNS)
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df[cols].copy()

    if sort_by is not None and sort_by in df.columns:
        s = pd.to_numeric(df[sort_by], errors="coerce")
        if s.notna().any():
            df = df.assign(_sort=s).sort_values("_sort").drop(columns=["_sort"])
        else:
            df = df.sort_values(sort_by)

    out_dir = Path(out_dir) if out_dir is not None else sweep_root
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{basename}.csv", index=False)
    (out_dir / f"{basename}.md").write_text(df.to_markdown(index=False), encoding="utf-8")

    return df

