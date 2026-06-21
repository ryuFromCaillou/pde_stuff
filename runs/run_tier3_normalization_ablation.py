from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from Datasets.data.processed.allenc_gen.allen_cahn_gen import AllenCahnConfig, solve_allen_cahn
from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from hpc_api.train_one import train_one
from tools.train_fn import train_fn


def _fmt_value_for_path(v) -> str:
    if isinstance(v, float):
        s = f"{v:.6g}"
    else:
        s = str(v)
    return s.replace(".", "p").replace("-", "m").replace("+", "")


def _true_coeffs_for_dataset(dataset: str, cfg: dict) -> tuple[list[str], list[float]]:
    d = str(dataset).lower()
    if d == "burgers":
        names = ["u", "u_x", "u_xx"]
        coeffs = [0.0, 0.0, float(cfg["burgers_nu"])]
        return names, coeffs
    if d == "allen_cahn":
        names = ["u", "u_x", "u_xx"]
        r = float(cfg["allen_reaction_scale"])
        dcoef = float(cfg["allen_d"])
        coeffs = [r, 0.0, dcoef]
        return names, coeffs
    raise ValueError(f"Unknown dataset: {dataset}")


def _build_fn_for_dataset(dataset: str):
    dataset = str(dataset).lower()

    if dataset == "burgers":
        def build_fn(cfg_obj):
            x, _u_final, _t_end, (t_grid, u_grid) = solve_burgers(
                N=int(cfg_obj.burgers_N),
                L=float(cfg_obj.burgers_L),
                nu=float(cfg_obj.burgers_nu),
                dt=float(cfg_obj.burgers_dt),
                T=float(cfg_obj.burgers_T),
                seed=int(cfg_obj.seed),
                return_history=True,
            )
            return (t_grid, x, u_grid)

        return build_fn

    if dataset == "allen_cahn":
        def build_fn(cfg_obj):
            allen_cfg = AllenCahnConfig(
                N=int(cfg_obj.allen_N),
                dt=float(cfg_obj.allen_dt),
                T=float(cfg_obj.allen_T),
                d=float(cfg_obj.allen_d),
                reaction_scale=float(cfg_obj.allen_reaction_scale),
                bc_value=float(cfg_obj.allen_bc_value),
                seed=int(cfg_obj.seed),
            )
            x, _u_final, _t_end, (t_grid, u_grid) = solve_allen_cahn(allen_cfg, return_history=True)
            return (t_grid, x, u_grid)

        return build_fn

    raise ValueError(f"Unknown dataset: {dataset}")


def _make_hpc_train_fn(dataset: str):
    build_fn = _build_fn_for_dataset(dataset)

    def hpc_train_fn(run_dir: Path, cfg: dict):
        return train_fn(run_dir=run_dir, cfg=cfg, build_fn=build_fn)

    return hpc_train_fn


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-3 feature normalization ablation runner")
    p.add_argument("--datasets", nargs="+", default=["burgers", "allen_cahn"])
    p.add_argument("--field_models", nargs="+", default=["siren"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--lam_pde", nargs="+", type=float, default=[0.5]) #0.0, 1e-3, 1e-2, 1e-1, 
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--noise_level", type=float, default=0.05)
    p.add_argument("--stride_t", type=int, default=1)
    p.add_argument("--stride_x", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    # Burgers params
    p.add_argument("--burgers_N", type=int, default=256)
    p.add_argument("--burgers_L", type=float, default=2 * 3.141592653589793)
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

    # Model params
    p.add_argument("--u_hidden", type=int, default=64)
    p.add_argument("--u_n_layers", type=int, default=4)
    p.add_argument("--u_hidden_layers", type=int, default=None)
    p.add_argument("--first_omega_0", type=float, default=30.0)
    p.add_argument("--hidden_omega_0", type=float, default=1.0)

    args = p.parse_args()

    base_cfg = dict(
        device=str(args.device),
        steps=int(args.steps),
        log_every=int(args.log_every),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lam_data=10.0,
        lam_reg=0.0,
        lam_tv=0.0,
        noise_level=float(args.noise_level),
        stride_t=int(args.stride_t),
        stride_x=int(args.stride_x),
        pde_head="eql",  # required by Tier-3 spec
        u_hidden=int(args.u_hidden),
        u_n_layers=int(args.u_n_layers),
        first_omega_0=float(args.first_omega_0),
        hidden_omega_0=float(args.hidden_omega_0),
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

    if args.u_hidden_layers is not None:
        base_cfg["u_hidden_layers"] = int(args.u_hidden_layers)

    root = Path("run_results") / "tier3"
    root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for dataset in args.datasets:
        dataset = str(dataset).lower()
        hpc_train_fn = _make_hpc_train_fn(dataset)

        if dataset == "burgers":
            feature_terms = ["u", "u_x", "u_xx"]
        elif dataset == "allen_cahn":
            feature_terms = ["u", "u_x", "u_xx"]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        for field_model in args.field_models:
            field_model = str(field_model).lower()

            for normalize in [False]:
                norm_tag = "normalize_true" if normalize else "normalize_false"
                base_dir = root / dataset / norm_tag / field_model
                base_dir.mkdir(parents=True, exist_ok=True)

                for lam_pde in args.lam_pde:
                    for seed in args.seeds:
                        cfg = copy.deepcopy(base_cfg)
                        cfg.update(
                            {
                                "dataset": dataset,
                                "field_model": field_model,
                                "normalize": bool(normalize),
                                "lam_pde": float(lam_pde),
                                "seed": int(seed),
                                "selected_derivs": list(feature_terms),
                            }
                        )

                        true_names, true_coeffs = _true_coeffs_for_dataset(dataset, cfg)
                        cfg["true_feature_names"] = true_names
                        cfg["true_coeffs"] = true_coeffs

                        run_dir = (
                            base_dir
                            / f"lam_pde_{_fmt_value_for_path(float(lam_pde))}"
                            / f"seed_{int(seed):03d}"
                        )

                        if args.dry_run:
                            print(f"[dry_run] {run_dir}")
                            continue

                        summary_path = run_dir / "summary.json"
                        if (not args.overwrite) and summary_path.exists():
                            try:
                                prev = json.loads(summary_path.read_text(encoding="utf-8"))
                                if int(prev.get("status", 0)) == 1:
                                    row = {"run_dir": str(run_dir), **cfg, **prev}
                                    all_rows.append(row)
                                    continue
                            except Exception:
                                pass

                        summary = train_one(cfg, run_dir, hpc_train_fn)
                        row = {"run_dir": str(run_dir), **cfg, **summary}
                        all_rows.append(row)

                # per-(dataset, normalize, field_model) summary
                df = pd.DataFrame([r for r in all_rows if r.get("dataset") == dataset and r.get("normalize") == normalize and r.get("field_model") == field_model])
                if len(df) > 0:
                    df.to_csv(base_dir / "summary.csv", index=False)

    # global summary
    if (not args.dry_run) and all_rows:
        pd.DataFrame(all_rows).to_csv(root / "all_results.csv", index=False)


if __name__ == "__main__":
    main()
