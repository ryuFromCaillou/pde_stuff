# `hpc_api`

Small utilities for running repeatable training jobs (single runs and parameter sweeps) and producing sweep summaries (plots + tables). This is designed to be used from the **repo root**, where outputs are written into a `sweep_root/` directory you choose (often under `runs/`).

## Components

### `train_one(config, run_dir, train_fn)` (`hpc_api/train_one.py`)
Runs a single training job and writes run artifacts into `run_dir/`.

**Inputs**
- `config` (`dict`): your training configuration.
- `run_dir` (`Path | str`): output directory for this run.
- `train_fn` (`callable`): your training function; signature: `train_fn(config: dict, run_dir: Path) -> dict`.

**`train_fn` return contract**
Return a dict with (recommended) keys:
- `history`: `list[dict]` (rows are typically per-epoch metrics; each dict becomes one CSV row).
- `best_epoch`: `int | None`
- `snapshot_path`: `str | Path | None` (optional artifact pointer)
- `status`: `1` for success, `0` for failure
- `summary_extra`: `dict` (optional; merged into `summary.json`, useful for extra scalars)

**Outputs in `run_dir/`**
- `config.json`: the input config.
- `history.csv`: CSV version of `history` (if provided).
- `summary.json`: one-row summary (includes `final_*` losses if present in the last history row).
- `loss_curves.png`: line plot of `data_loss`, `pde_loss`, `tv_loss`, `l1_loss` (if present).
- On exception in `train_fn`: `error.json` + `traceback.txt` are written, and a failure `summary.json` is still produced.

---

### `run_sweep(base_config, sweep_param, sweep_values, sweep_root, train_fn, overwrite=False)` (`hpc_api/run_sweep.py`)
Runs `train_one(...)` for each value in `sweep_values` while modifying `base_config[sweep_param]`.

**Run layout**
```
<sweep_root>/
  sweep_config.json
  sweep_results.csv
  runs/
    <sweep_param>_<value>/
      config.json
      history.csv
      summary.json
      loss_curves.png
      ...
```

**Notes**
- Each run also gets `config["sweep_param_name"]` and `config["sweep_param_value"]`.
- If `overwrite=False` and `runs/<run_name>/summary.json` exists with `status==1`, that run is skipped and its prior summary is reused.

---

### `reduce_sweep(sweep_root)` (`hpc_api/reduce_sweep.py`)
Reads `sweep_root/sweep_results.csv` and writes sweep-level comparison plots into `sweep_root/plots/`.

**Outputs in `sweep_root/plots/`**
- `final_*_vs_<sweep_param>.png` for available `final_*` metrics.
- `best_run.json` (currently: best by `final_pde_loss` when that column exists).
- If there are no successful rows (`status != 1`), a `README.txt` is written explaining that.

---

### `tabulate_sweep(...)` (`hpc_api/tabulate_sweep.py`)
Reads `sweep_root/sweep_results.csv` and emits email-friendly tables.

**Common options**
- `columns`: list of columns to keep (defaults to `DEFAULT_COLUMNS`; missing columns are ignored).
- `status_only=True`: keep only rows with `status == 1` (when present).
- `sort_by="sweep_param_value"`: numeric sort attempted.
- `out_dir`: where to write outputs (default: `sweep_root`).
- `basename="sweep_table"`: output file prefix.

**Outputs**
- `<out_dir>/<basename>.csv`
- `<out_dir>/<basename>.md`

## How to use

### Importing from the repo root
These utilities are meant to be imported from the repository root (so Python can find `hpc_api/`).

In scripts you can do what `tools/` does:
```py
import sys
sys.path.append(".")  # repo root
from hpc_api.run_sweep import run_sweep
```

### Example sweep script
See `tools/run_stride_sweep.py` for a complete example that defines a `train_fn(...)` and calls `run_sweep(...)` with `sweep_root = Path("runs/...")`.

### CLI helper: tabulation
`tools/tabulate_sweep.py` is a thin CLI wrapper around `hpc_api.tabulate_sweep`:
```powershell
python tools/tabulate_sweep.py runs\hpc_stride_t_sweep
```

