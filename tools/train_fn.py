import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import torch
from torch.utils.data import DataLoader

from Datasets.Datasets import PDEDataset
from prog import featlib, hlprs, mlps, trainer


def _cfg_get(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_obj(cfg):
    if isinstance(cfg, dict):
        return SimpleNamespace(**cfg)
    return cfg


def train_fn(run_dir: Path, cfg: dict, build_fn):
    """
    Standalone training function for use with `hpc_api`.

    Parameters
    ----------
    run_dir : Path
        Directory where this run's artifacts live.
    cfg : dict
        Run configuration (hyperparameters, device, etc.).
    build_fn : callable
        Dataset builder: `build_fn(cfg_obj) -> (t_np, x_np, y_np, y_noisy_np, N)`.
    """
    device = torch.device(_cfg_get(cfg, "device", "cpu"))

    cfg_obj = _as_obj(cfg)
    dataset = PDEDataset(cfg_obj, build_fn)
    loader = DataLoader(
        dataset,
        batch_size=int(_cfg_get(cfg, "batch_size")),
        shuffle=True,  # subsampling each epoch
    )

    selected_derivs = tuple(_cfg_get(cfg, "selected_derivs"))

    u_model = mlps.SimpleMLP(
        n_layers=int(_cfg_get(cfg, "u_n_layers")),
        hidden_size=int(_cfg_get(cfg, "u_hidden")),
        act=mlps.Sin,
    )
    v_model = mlps.EQL(in_dim=len(selected_derivs), prod_dim=2, num_layers=1, bias=False)

    cfg_t = trainer.TrainerConfig(
        lr=float(_cfg_get(cfg, "lr")),
        lambda_pde=float(_cfg_get(cfg, "lam_pde")),
        lambda_reg=float(_cfg_get(cfg, "lam_reg")),
        lambda_tv=float(_cfg_get(cfg, "lam_tv")),
        lambda_data=float(_cfg_get(cfg, "lam_data")),
        selected_derivs=selected_derivs,
        device=device,
    )

    ft = featlib.FeatureTensor(selected_derivs, normalize=False)
    tr = trainer.PDETrainer(u_model=u_model, v_model=v_model, cfg=cfg_t, feature_builder=ft.build)

    history = []
    best_epoch = 0
    best_pde = float("inf")

    t0 = time.time()

    loader_iter = iter(loader)
    for epoch in range(int(_cfg_get(cfg, "steps"))):
        try:
            tb, xb, u_clean_b, u_noisy_b = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            tb, xb, u_clean_b, u_noisy_b = next(loader_iter)

        tb = tb.to(device)
        xb = xb.to(device)
        u_clean_b = u_clean_b.to(device)
        u_noisy_b = u_noisy_b.to(device)

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

        log_every = _cfg_get(cfg, "log_every")
        if log_every and epoch % int(log_every) == 0:
            print(
                f"[{Path(run_dir).name}] epoch={epoch} "
                f"data={row['data_loss']:.6e} pde={row['pde_loss']:.6e} "
                f"tv={row['tv_loss']:.6e} l1={row['l1_loss']:.6e}"
            )

    runtime_sec = time.time() - t0

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

    snap_path = None
    try:
        t_full, x_full, y_full, y_noisy_full = dataset.full()
        t_np = t_full.detach().cpu().numpy()
        x_np = x_full.detach().cpu().numpy()
        y_np = y_full.detach().cpu().numpy()
        y_noisy_np = y_noisy_full.detach().cpu().numpy()

        fig, fig_hm, _payload = hlprs.snapshot_comp(
            tr.u,
            int(_cfg_get(cfg, "stride_x")),
            int(_cfg_get(cfg, "stride_t")),
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
