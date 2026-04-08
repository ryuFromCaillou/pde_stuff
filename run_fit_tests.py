from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from Datasets.data.processed.allenc_gen.allen_cahn_gen import AllenCahnConfig, solve_allen_cahn
from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from utils.extract_pde_ls import extract_pde_ls
from utils.fit_utils import fit_model_to_data
from prog import hlprs
from prog.mlps import SimpleMLP, SirenMLP


SUMMARY_FIELDS = [
    "dataset",
    "seed",
    "model",
    "feature_names",
    "coeffs",
    "true_coeffs",
    "abs_coeff_error",
    "l2_coeff_error",
    "residuals",
    "rank",
    "singular_values",
    "final_train_loss",
]


@dataclass
class RunConfig:
    dataset: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    hidden_layers: int
    first_omega_0: float
    hidden_omega_0: float
    noise_level: float
    stride_t: int
    stride_x: int
    burgers_N: int
    burgers_L: float
    burgers_nu: float
    burgers_dt: float
    burgers_T: float
    allen_N: int
    allen_dt: float
    allen_T: float
    allen_d: float
    allen_reaction_scale: float
    allen_bc_value: float


def _affine_to_minus1_1(v: np.ndarray):
    v = np.asarray(v, dtype=np.float64)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        raise ValueError("Cannot normalize: invalid range")
    a = 0.5 * (vmax - vmin)
    b = 0.5 * (vmax + vmin)
    return a, b


def _to_norm(v: np.ndarray, a: float, b: float) -> np.ndarray:
    return (np.asarray(v, dtype=np.float64) - float(b)) / float(a)


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_train_samples(
    *,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    u_grid: np.ndarray,
    stride_t: int,
    stride_x: int,
    noise_level: float,
    seed: int,
):
    stride_t = max(1, int(stride_t))
    stride_x = max(1, int(stride_x))

    rows = np.arange(t_grid.size)[::stride_t]
    cols = np.arange(x_grid.size)[::stride_x]

    t2d = np.repeat(t_grid[:, None], x_grid.size, axis=1)
    x2d = np.repeat(x_grid[None, :], t_grid.size, axis=0)
    u2d = u_grid

    t_s = t2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)
    x_s = x2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)
    y_s = u2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)

    if float(noise_level) > 0:
        rng = np.random.default_rng(int(seed))
        sigma = float(noise_level) * float(np.std(y_s))
        y_noisy = (y_s + sigma * rng.standard_normal(size=y_s.shape)).astype(np.float32)
    else:
        y_noisy = y_s

    return t_s, x_s, y_s, y_noisy


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


def true_coeffs_for_dataset(dataset_name: str, cfg: RunConfig):
    dataset_name = str(dataset_name).lower()
    if dataset_name == "burgers":
        names = ["u", "u_x", "u_xx", "uu_x"]
        coeffs = np.array([0.0, 0.0, float(cfg.burgers_nu), -1.0], dtype=float)
        return names, coeffs
    if dataset_name == "allen_cahn":
        names = ["u", "u_xx", "u3"]
        r = float(cfg.allen_reaction_scale)
        d = float(cfg.allen_d)
        coeffs = np.array([r, d, -r], dtype=float)
        return names, coeffs
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)



def run_one_dataset(cfg: RunConfig) -> list[dict[str, Any]]:
    _seed_everything(cfg.seed)

    if cfg.dataset == "burgers":
        #provide data with numerical solution
        x, _u_final, _t_end, (t_grid, U_true) = solve_burgers(
            N=int(cfg.burgers_N),
            L=float(cfg.burgers_L),
            nu=float(cfg.burgers_nu),
            dt=float(cfg.burgers_dt),
            T=float(cfg.burgers_T),
            seed=int(cfg.seed),
            return_history=True,
        )
        #preprocess stuff
        x = x.astype(np.float64)
        t_grid = t_grid.astype(np.float64)
        U_true = U_true.astype(np.float64)

        #handles normalizing, adding noise, and striding (sparse sampling)
        t_train, x_train, y_train_clean, y_train = _make_train_samples(
            t_grid=t_grid,
            x_grid=x,
            u_grid=U_true,
            stride_t=cfg.stride_t,
            stride_x=cfg.stride_x,
            noise_level=cfg.noise_level,
            seed=cfg.seed,
        )

    elif cfg.dataset == "allen_cahn":
        allen_cfg = AllenCahnConfig(
            N=int(cfg.allen_N),
            dt=float(cfg.allen_dt),
            T=float(cfg.allen_T),
            d=float(cfg.allen_d),
            reaction_scale=float(cfg.allen_reaction_scale),
            bc_value=float(cfg.allen_bc_value),
            stride_t=int(cfg.stride_t),
            stride_x=int(cfg.stride_x),
            noise_level=float(cfg.noise_level),
            seed=int(cfg.seed),
        )
        x, _u_final, _t_end, (t_grid, U_true) = solve_allen_cahn(allen_cfg, return_history=True)
        x = x.astype(np.float64)
        t_grid = t_grid.astype(np.float64)
        U_true = U_true.astype(np.float64)

        t_train, x_train, y_train_clean, y_train = _make_train_samples(
            t_grid=t_grid,
            x_grid=x,
            u_grid=U_true,
            stride_t=cfg.stride_t,
            stride_x=cfg.stride_x,
            noise_level=cfg.noise_level,
            seed=cfg.seed,
        )

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    a_t, b_t = _affine_to_minus1_1(t_grid)
    a_x, b_x = _affine_to_minus1_1(x)
    t_train_n = _to_norm(t_train, a_t, b_t).astype(np.float32)
    x_train_n = _to_norm(x_train, a_x, b_x).astype(np.float32)

    base_dir = Path("runs") / "fit_testing" / str(cfg.dataset) / f"seed_{cfg.seed:03d}"
    base_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = asdict(cfg)
    cfg_json.update({"t_norm": {"a": a_t, "b": b_t}, "x_norm": {"a": a_x, "b": b_x}})
    _write_json(base_dir / "config.json", cfg_json)

    models = {
        "siren": SirenMLP(
            hidden_size=int(cfg.hidden_size),
            hidden_layers=int(cfg.hidden_layers),
            first_omega_0=float(cfg.first_omega_0),
            hidden_omega_0=float(cfg.hidden_omega_0),
        ),
    }

    dataset_rows: list[dict[str, Any]] = []
    for model_name, base_model in models.items():
        run_dir = base_dir / model_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n== {cfg.dataset} seed={cfg.seed} model={model_name} ==")
        base_model, hist = fit_model_to_data(
            base_model,
            t_train_n,
            x_train_n,
            y_train,  # noisy by default
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            lr=float(cfg.lr),
            device=str(cfg.device),
            log_every=max(1, int(cfg.epochs) // 10),
        )
        _write_json(run_dir / "loss_history.json", {"losses": hist.losses})


        metrics = {
            "dataset": str(cfg.dataset),
            "seed": int(cfg.seed),
            "model": str(model_name),
            "final_train_loss": float(hist.losses[-1] if hist.losses else float("nan")),
        }
        _write_json(run_dir / "metrics.json", metrics)

        # Fit snapshots
        try:
            fig, fig_hm, _payload = hlprs.snapshot_comp(
                PhysCoordWrapper(base_model, a_t=a_t, b_t=b_t, a_x=a_x, b_x=b_x),
                int(cfg.stride_x),
                int(cfg.stride_t),
                y_train,
                y_train_clean,
                t_train,
                x_train,
                snap_no=5,
            )
            fig.savefig(run_dir / "fit_snapshots.pdf")
            fig_hm.savefig(run_dir / "fit_heatmap.pdf")
        except Exception as e:
            print(f"[warn] snapshot_comp failed for {cfg.dataset}/{model_name}: {e}")

        dataset_rows.append(
            {
                "dataset": str(cfg.dataset),
                "seed": int(cfg.seed),
                "model": str(model_name),
                "final_train_loss": float(metrics["final_train_loss"]),
            }
        )

    return dataset_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["burgers", "allen_cahn"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--hidden_layers", type=int, default=3)
    p.add_argument("--first_omega_0", type=float, default=30.0)
    p.add_argument("--hidden_omega_0", type=float, default=30.0)
    p.add_argument("--noise_level", type=float, default=0.05)
    p.add_argument("--stride_t", type=int, default=32)
    p.add_argument("--stride_x", type=int, default=16)

    # Burgers params
    p.add_argument("--burgers_N", type=int, default=256)
    p.add_argument("--burgers_L", type=float, default=2 * np.pi)
    p.add_argument("--burgers_nu", type=float, default=0.02)
    p.add_argument("--burgers_dt", type=float, default=2e-3)
    p.add_argument("--burgers_T", type=float, default=1.0)

    # Allen–Cahn params
    p.add_argument("--allen_N", type=int, default=201)
    p.add_argument("--allen_dt", type=float, default=0.01)
    p.add_argument("--allen_T", type=float, default=1.0)
    p.add_argument("--allen_d", type=float, default=0.001)
    p.add_argument("--allen_reaction_scale", type=float, default=5.0)
    p.add_argument("--allen_bc_value", type=float, default=-1.0)

    args = p.parse_args()

    all_rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in args.datasets:
        dataset = str(dataset)
        by_dataset.setdefault(dataset, [])
        for seed in args.seeds:
            cfg = RunConfig(
                dataset=dataset,
                seed=int(seed),
                device=str(args.device),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
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
                allen_dt=float(args.allen_dt),
                allen_T=float(args.allen_T),
                allen_d=float(args.allen_d),
                allen_reaction_scale=float(args.allen_reaction_scale),
                allen_bc_value=float(args.allen_bc_value),
            )
            rows = run_one_dataset(cfg) # out: list of dicts with keys dataset, seed, model, final_train_loss
            all_rows.extend(rows)
            by_dataset[dataset].extend(rows)

        ds_summary = Path("runs") / "fit_testing" / dataset / "summary.csv"
        if by_dataset[dataset]:
            _write_csv(ds_summary, by_dataset[dataset], SUMMARY_FIELDS)
            print(f"Wrote {ds_summary}")

    out_path = Path("runs") / "fit_testing" / "all_results.csv"
    if all_rows:
        _write_csv(out_path, all_rows, SUMMARY_FIELDS)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
