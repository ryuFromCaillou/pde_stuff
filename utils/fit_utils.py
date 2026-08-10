from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from prog.featlib import FeatureTensor, FeatureTensorOut


@dataclass
class FitHistoryRow:
    epoch: int
    total_loss: float
    data_loss: float
    pde_loss: float
    tv_loss: float
    param_l1_loss: float = 0.0
    l1_data_loss: float = 0.0
    l1_pde_loss: float = 0.0
    sparse_eql_loss: float = 0.0


@dataclass
class FitHistory:
    # Back-compat: many scripts expect `hist.losses` to exist and represent the
    # per-epoch total loss.
    losses: list[float]
    rows: list[FitHistoryRow] | None = None


def _to_float_tensor_2d(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach()
    else:
        t = torch.from_numpy(np.asarray(x))
    t = t.float()
    if t.ndim == 1:
        t = t[:, None]
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(f"Expected shape (N,1); got {tuple(t.shape)}")
    return t

def _grad1(u, x):
    '''
    Compute du/dx using autograd. u and x should be (B,1) tensors.
    in: u: (B,1) tensor; x: (B,1) tensor
    out: du/dx as (B,1) tensor
    '''
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return du

def _smooth_l1_param_sparsity(params, *, s: float = 1e-3) -> torch.Tensor:
    """
    PDE-Net 2.0-style smooth L1 (Huber-like) penalty on parameters.

    For each parameter value p:
      - if |p| < s: 0.5/s * p^2
      - else:       |p| - s/2
    """
    s = float(s)
    if s <= 0:
        raise ValueError("s must be > 0")
    loss = None
    for p in params:
        a = p.abs()
        term = torch.where(a < s, (0.5 / s) * a * a, a - 0.5 * s).sum()
        loss = term if loss is None else (loss + term)
    if loss is None:
        return torch.tensor(0.0)
    return loss

def fit_model_to_data(
    model,
    t_train,
    x_train,
    y_train,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    weight_decay: float = 0.0,
    log_every: int = 100,
    tv_terms: list[tuple[float, callable]] | None = None,
    lam_l1_data: float = 0.0,
    lam_sparse_eql: float = 0.0,
    sparse_eql_s: float = 1e-3,
):
    '''
    Fit a model to training data.
    Dataloader creates t_train.size//batch_size batches of (t,x,y) for each epoch, sampled randomly with replacement.
    '''
    print('Number of batches per epoch:', max(1, t_train.shape[0] // batch_size))
    t = _to_float_tensor_2d(t_train) # entire time mesh as 2d tensor
    x = _to_float_tensor_2d(x_train) # entire space mesh as 2d tensor
    y = _to_float_tensor_2d(y_train) # entire solution mesh as 2d tensor

    if not (t.shape[0] == x.shape[0] == y.shape[0]):
        raise ValueError("t_train, x_train, y_train must have same length")

    model = model.to(device)
    model.train()

    ds = TensorDataset(t, x, y)
    if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= len(ds):
        loader: Iterable = [(t, x, y)]
    else:
        #loader prebuilds batches for each epoch, which is faster but less random than sampling a new batch each iteration. 
        # so an epoch is really a pass through the prebuilt N batches, not necessarily a one-time pass as I do in make_batch
        loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)


    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = torch.nn.MSELoss()
    l1_fn = torch.nn.L1Loss()

    losses: list[float] = []
    rows: list[FitHistoryRow] = []

    has_tv = bool(tv_terms) and any(float(lam) != 0.0 for lam, _fn in tv_terms)
    for epoch in range(int(epochs)):
        epoch_total = 0.0
        epoch_data = 0.0
        epoch_tv = 0.0
        epoch_l1_data = 0.0
        epoch_sparse_eql = 0.0
        n_items = 0

        for t_b, x_b, y_b in loader:
            if has_tv:
                t_b = t_b.to(device).requires_grad_(True)
                x_b = x_b.to(device).requires_grad_(True)
            else:
                t_b = t_b.to(device)
                x_b = x_b.to(device)
            y_b = y_b.to(device)

            pred = model(t_b, x_b)

            loss_tv = pred.new_tensor(0.0)
            if tv_terms:
                for lam_tv, tv_fn in tv_terms:
                    if float(lam_tv) != 0.0:
                        loss_tv = loss_tv + float(lam_tv) * tv_fn(pred, t_b, x_b)
            loss_data = loss_fn(pred, y_b)
            loss_l1_data = pred.new_tensor(0.0)
            if float(lam_l1_data) != 0.0:
                loss_l1_data = float(lam_l1_data) * l1_fn(pred, y_b)

            loss_sparse_eql = pred.new_tensor(0.0)
            if float(lam_sparse_eql) != 0.0:
                loss_sparse_eql = float(lam_sparse_eql) * _smooth_l1_param_sparsity(
                    model.parameters(), s=float(sparse_eql_s)
                )

            loss = loss_data + loss_tv + loss_l1_data + loss_sparse_eql

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(t_b.shape[0])
            epoch_total += float(loss.detach().cpu()) * bs
            epoch_data += float(loss_data.detach().cpu()) * bs
            epoch_tv += float(loss_tv.detach().cpu()) * bs
            epoch_l1_data += float(loss_l1_data.detach().cpu()) * bs
            epoch_sparse_eql += float(loss_sparse_eql.detach().cpu()) * bs
            n_items += bs
        
        # reported loss is averaged over all items in epoch (not averaged per batch)
        epoch_total /= max(1, n_items)
        epoch_data /= max(1, n_items)
        epoch_tv /= max(1, n_items)
        epoch_l1_data /= max(1, n_items)
        epoch_sparse_eql /= max(1, n_items)

        losses.append(epoch_total)
        rows.append(
            FitHistoryRow(
                epoch=int(epoch),
                total_loss=float(epoch_total),
                data_loss=float(epoch_data),
                pde_loss=0.0,
                tv_loss=float(epoch_tv),
                l1_data_loss=float(epoch_l1_data),
                l1_pde_loss=0.0,
                sparse_eql_loss=float(epoch_sparse_eql),
            )
        )

        if log_every and (epoch % int(log_every) == 0 or epoch == int(epochs) - 1):
            print(
                f"epoch {epoch:05d}  "
                f"loss={epoch_total:.6e}  "
                f"data={epoch_data:.6e}  "
                f"tv={epoch_tv:.6e}"
                + (f"  l1_data={epoch_l1_data:.6e}" if float(lam_l1_data) != 0.0 else "")
                + (f"  sparse_eql={epoch_sparse_eql:.6e}" if float(lam_sparse_eql) != 0.0 else "")
            )
    return model, FitHistory(losses=losses, rows=rows)

def fit_data_and_pde(
    u_model,
    v_model,
    t_train,
    x_train,
    y_train,
    feat_builder,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    weight_decay: float = 0.0,
    lam_pde: float = 1.0,
    lam_data: float = 1.0,
    lam_param_l1: float = 0.0,
    lam_l1_data: float = 0.0,
    lam_l1_pde: float = 0.0,
    lam_sparse_eql: float = 0.0,
    sparse_eql_s: float = 1e-3,
    log_every: int = 100,
    params: list[torch.nn.Parameter] | None = None,
    tv_terms: list[tuple[float, callable]] | None = None
):
    """
    Fit u_model to data while fitting v_model(F) to u_t,
    where F is built from the current batch prediction u_pred.
    """

    t = _to_float_tensor_2d(t_train)
    x = _to_float_tensor_2d(x_train)
    y = _to_float_tensor_2d(y_train)

    if not (t.shape[0] == x.shape[0] == y.shape[0]):
        raise ValueError("t_train, x_train, y_train must have same length")

    u_model = u_model.to(device)
    v_model = v_model.to(device)
    u_model.train()
    v_model.train()

    ds = TensorDataset(t, x, y)
    if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= len(ds):
        loader: Iterable = [(t, x, y)]
    else:
        loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)

    print("Number of batches per epoch:", len(loader))

    if params is None:
        params = list(u_model.parameters()) + list(v_model.parameters())

    opt = torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = torch.nn.MSELoss()
    l1_fn = torch.nn.L1Loss()

    losses: list[float] = []
    rows: list[FitHistoryRow] = []

    for epoch in range(int(epochs)):
        epoch_total = 0.0
        epoch_data = 0.0
        epoch_pde = 0.0
        epoch_tv = 0.0
        epoch_param_l1 = 0.0
        epoch_l1_data = 0.0
        epoch_l1_pde = 0.0
        epoch_sparse_eql = 0.0
        n_items = 0

        for t_b, x_b, y_b in loader:
            t_b = t_b.to(device).requires_grad_(True)
            x_b = x_b.to(device).requires_grad_(True)
            y_b = y_b.to(device)

            u_pred = u_model(t_b, x_b)
            u_t = _grad1(u_pred, t_b)

            feat_out = feat_builder(u_pred, x=x_b)
            F = feat_out.F

            v_pred = v_model(F)

            loss_data = lam_data * loss_fn(u_pred, y_b)
            loss_pde = lam_pde * loss_fn(u_t, v_pred)
            loss_param_l1 = u_pred.new_tensor(0.0)
            if float(lam_param_l1) != 0.0:
                loss_param_l1 = float(lam_param_l1) * sum(p.abs().sum() for p in v_model.parameters())
            loss_l1_data = float(lam_l1_data) * l1_fn(u_pred, y_b)
            loss_l1_pde = float(lam_l1_pde) * l1_fn(u_t, v_pred)
            loss_sparse_eql = float(lam_sparse_eql) * _smooth_l1_param_sparsity(
                v_model.parameters(), s=float(sparse_eql_s)
            )

            loss_tv = u_pred.new_tensor(0.0)
            if tv_terms:
                for lam_tv, tv_fn in tv_terms:
                    if float(lam_tv) != 0.0:
                        loss_tv = loss_tv + float(lam_tv) * tv_fn(u_pred, t_b, x_b)

            loss = loss_data + loss_pde + loss_param_l1 + loss_l1_data + loss_l1_pde + loss_sparse_eql + loss_tv

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(t_b.shape[0])
            epoch_total += float(loss.detach().cpu()) * bs
            epoch_data += float(loss_data.detach().cpu()) * bs
            epoch_pde += float(loss_pde.detach().cpu()) * bs
            epoch_tv += float(loss_tv.detach().cpu()) * bs
            epoch_param_l1 += float(loss_param_l1.detach().cpu()) * bs
            epoch_l1_data += float(loss_l1_data.detach().cpu()) * bs
            epoch_l1_pde += float(loss_l1_pde.detach().cpu()) * bs
            epoch_sparse_eql += float(loss_sparse_eql.detach().cpu()) * bs
            n_items += bs

        epoch_total /= max(1, n_items)
        epoch_data /= max(1, n_items)
        epoch_pde /= max(1, n_items)
        epoch_tv /= max(1, n_items)
        epoch_param_l1 /= max(1, n_items)
        epoch_l1_data /= max(1, n_items)
        epoch_l1_pde /= max(1, n_items)
        epoch_sparse_eql /= max(1, n_items)

        losses.append(epoch_total)
        rows.append(
            FitHistoryRow(
                epoch=int(epoch),
                total_loss=float(epoch_total),
                data_loss=float(epoch_data),
                pde_loss=float(epoch_pde),
                tv_loss=float(epoch_tv),
                param_l1_loss=float(epoch_param_l1),
                l1_data_loss=float(epoch_l1_data),
                l1_pde_loss=float(epoch_l1_pde),
                sparse_eql_loss=float(epoch_sparse_eql),
            )
        )

        if log_every and (epoch % int(log_every) == 0 or epoch == int(epochs) - 1):
            msg = (
                f"epoch {epoch:05d}  "
                f"loss={epoch_total:.6e}  "
                f"data={epoch_data:.6e}  "
                f"pde={epoch_pde:.6e}  "
                f"tv={epoch_tv:.6e}"
            )
            if float(lam_l1_data) != 0.0 or float(lam_l1_pde) != 0.0:
                msg = (
                    msg
                    + f"  l1_data={epoch_l1_data:.6e}"
                    + f"  l1_pde={epoch_l1_pde:.6e}"
                )
            if float(lam_param_l1) != 0.0:
                msg = msg + f"  param_l1={epoch_param_l1:.6e}"
            if float(lam_sparse_eql) != 0.0:
                msg = msg + f"  sparse_eql={epoch_sparse_eql:.6e}"
            print(msg)

    return u_model, v_model, FitHistory(losses=losses, rows=rows)

@torch.no_grad()
def predict_on_grid(model, t_np, x_np, device: str):
    model = model.to(device)
    model.eval()

    t = _to_float_tensor_2d(t_np).to(device)
    x = _to_float_tensor_2d(x_np).to(device)

    pred = model(t, x).detach().cpu().numpy().reshape(-1)
    return pred
