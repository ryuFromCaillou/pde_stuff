import sys
from pathlib import Path

sys.path.append(".")  # repo root

import Datasets.matconv as mc
from hpc_api.run_sweep import run_sweep
from tools.train_fn import train_fn


def build_fn(cfg):
    partitions = mc.build_dataset_from_burgers(
        noise_level=cfg.noise,
        nu=cfg.nu,
        stride_t=cfg.stride_t,
        stride_x=cfg.stride_x,
        seed=cfg.seed,
        quantile_splits=getattr(cfg, "part_num", 1),
        return_partitions=True,
    )

    which_part = getattr(cfg, "which_part", 1)
    key = [k for k in partitions if k.startswith(f"Q{which_part}:")][0]
    return partitions[key]


def hpc_train_fn(cfg: dict, run_dir: Path):
    return train_fn(run_dir=run_dir, cfg=cfg, build_fn=build_fn)


if __name__ == "__main__":
    base_config = dict(
        seed=1432,
        device="cpu",
        steps=1000,
        log_every=100,
        batch_size=1000,
        lr=1e-3,
        noise=0.7,
        nu=0.02,
        stride_x=5,
        stride_t=1,
        lam_pde=0.5,
        lam_data=10.0,
        lam_reg=0.0,
        lam_tv=0.0,
        selected_derivs=["u", "u_x", "u_xx"],
        part_num=1,
        which_part=1,
        u_n_layers=4,
        u_hidden=64,
    )

    sweep_root = Path("runs/hpc_stride_t_sweep")
    run_sweep(
        base_config=base_config,
        sweep_param="stride_t",
        sweep_values=[1, 5],
        sweep_root=sweep_root,
        train_fn=hpc_train_fn,
        overwrite=False,
    )
