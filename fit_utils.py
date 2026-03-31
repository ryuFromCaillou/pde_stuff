from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from prog.hlprs import make_batch


@dataclass
class FitHistory:
    losses: list[float]


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
):
    '''
    Fit a model to training data.
    Dataloader creates t_train.size/batch_size batches of (t,x,y) for each epoch, sampled randomly with replacement.
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

    losses: list[float] = []
    for epoch in range(int(epochs)):
        epoch_loss = 0.0
        n_items = 0

        for t_b, x_b, y_b in loader:
            t_b = t_b.to(device)
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            pred = model(t_b, x_b)
            loss = loss_fn(pred, y_b)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(t_b.shape[0])
            epoch_loss += float(loss.detach().cpu()) * bs
            n_items += bs
        
        # reported loss is averaged over all items in epoch (not averaged per batch)
        epoch_loss = epoch_loss / max(1, n_items)
        losses.append(epoch_loss)

        if log_every and (epoch % int(log_every) == 0 or epoch == int(epochs) - 1):
            print(f"epoch {epoch:05d}  loss={epoch_loss:.6e}")

    return model, FitHistory(losses=losses)


@torch.no_grad()
def predict_on_grid(model, t_np, x_np, device: str):
    model = model.to(device)
    model.eval()

    t = _to_float_tensor_2d(t_np).to(device)
    x = _to_float_tensor_2d(x_np).to(device)

    pred = model(t, x).detach().cpu().numpy().reshape(-1)
    return pred

