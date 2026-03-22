import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

sys.path.append(".")  # repo root

from hpc_api.run_sweep import run_sweep
from prog import featlib, hlprs, mlps, trainer
import Datasets.matconv as mc


def train_fn(cfg: dict, run_dir: Path):
    device = torch.device(cfg.get("device", "cpu"))

    partitions = mc.build_dataset_from_burgers(
        noise_level=cfg["noise"],
        nu=cfg["nu"],
        stride_t=cfg["stride_t"],
        stride_x=cfg["stride_x"],
        seed=cfg["seed"],
        quantile_splits=cfg.get("part_num", 1),
        return_partitions=True,
    )
    key = [k for k in partitions if k.startswith(f"Q{cfg.get('which_part', 1)}:")][0]
    t_np, x_np, y_np, y_noisy_np, _N = partitions[key]

    t = torch.from_numpy(t_np).to(device)
    x = torch.from_numpy(x_np).to(device)
    y_clean = torch.from_numpy(y_np).to(device)
    y_noisy = torch.from_numpy(y_noisy_np).to(device)

    selected_derivs = tuple(cfg["selected_derivs"])

    u_model = mlps.SimpleMLP(n_layers=cfg["u_n_layers"], hidden_size=cfg["u_hidden"], act=mlps.Sin)
    v_model = mlps.EQL(in_dim=len(selected_derivs), prod_dim=2, num_layers=1, bias=False)

    cfg_t = trainer.TrainerConfig(
        lr=cfg["lr"],
        lambda_pde=cfg["lam_pde"],
        lambda_reg=cfg["lam_reg"],
        lambda_tv=cfg["lam_tv"],
        lambda_data=cfg["lam_data"],
        selected_derivs=selected_derivs,
        device=device,
    )

    ft = featlib.FeatureTensor(selected_derivs, normalize=False)
    tr = trainer.PDETrainer(u_model=u_model, v_model=v_model, cfg=cfg_t, feature_builder=ft.build)

    history = []
    best_epoch = 0
    best_pde = float("inf")

    t0 = time.time()

    for epoch in range(cfg["steps"]):
        tb, xb, u_noisy_b, u_clean_b = hlprs.make_batch(
            batch_size=cfg["batch_size"],
            t_torch=t,
            x_torch=x,
            y_clean=y_clean,
            y_noisy=y_noisy,
        )
        out = tr.step(t=tb, x=xb, u_noisy=u_noisy_b, u_clean=u_clean_b)

        row = dict(
            epoch=epoch,
            total_loss=out["loss"],
            tv_loss=out["loss_tv"],
            l1_loss=out["l1"],
            pde_loss=out["loss_pde"],
            data_loss=out["loss_data"],
        )
        history.append(row)

        if row["pde_loss"] < best_pde:
            best_pde = row["pde_loss"]
            best_epoch = epoch

        if cfg.get("log_every") and epoch % int(cfg["log_every"]) == 0:
            print(
                f"[{run_dir.name}] epoch={epoch} "
                f"data={row['data_loss']:.6e} pde={row['pde_loss']:.6e} "
                f"tv={row['tv_loss']:.6e} l1={row['l1_loss']:.6e}"
            )

    runtime_sec = time.time() - t0

    # Coefficients (for sweep_results.csv parity with notebook experiments)
    summary_extra = {}
    try:
        w = tr.v.readout.weight.detach().cpu().numpy().reshape(-1)
        M = tr.v.effective_quadratic_matrix(symmetrize=False).detach().cpu().numpy()
        summary_extra.update(
            {
                "w_u": float(w[0]),
                "w_ux": float(w[1]),
                "w_uxx": float(w[2]),
                "w_prod": float(w[-1]),
                "M01": float(M[0, 1]),
                "M10": float(M[1, 0]),
            }
        )
    except Exception:
        summary_extra = {}

    # Snapshot once per run
    snap_path = None
    try:
        fig, fig_hm, _payload = hlprs.snapshot_comp(
            tr.u,
            cfg["stride_x"],
            cfg["stride_t"],
            y_noisy_np,
            y_np,
            t_np,
            x_np,
            snap_no=10,
        )
        snap_path = Path(run_dir) / "snapshot.pdf"
        fig.savefig(snap_path)
        fig_hm.savefig(Path(run_dir) / "heatmap.pdf")
    except Exception:
        snap_path = None

    return {
        "history": history,
        "best_epoch": best_epoch,
        "snapshot_path": snap_path,
        "runtime_sec": runtime_sec,
        "summary_extra": summary_extra,
        "status": 1,
    }


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
        train_fn=train_fn,
        overwrite=False,
    )
