from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def eval_model_and_time_derivative(model, t_np, x_np, device: str):
    """
    Inputs:
        t_np, x_np: flat arrays of shape (N,)
    Returns:
        u_pred_np: shape (N,)
        u_t_np: shape (N,)
        t_torch, x_torch, u_pred_torch
    """
    t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
    x_np = np.asarray(x_np, dtype=np.float32).reshape(-1, 1)

    t = torch.from_numpy(t_np).to(device).requires_grad_(True)
    x = torch.from_numpy(x_np).to(device).requires_grad_(True)

    model = model.to(device)
    model.eval()

    with torch.enable_grad():
        u_pred = model(t, x)
        u_t = torch.autograd.grad(
            u_pred,
            t,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

    u_pred_np_out = u_pred.detach().cpu().numpy().reshape(-1)
    u_t_np_out = u_t.detach().cpu().numpy().reshape(-1)
    return u_pred_np_out, u_t_np_out, t, x, u_pred


def build_feature_matrix(feature_terms: Iterable[str], u_pred_torch: torch.Tensor, x_torch: torch.Tensor):
    """
    Returns:
        F_np: (N, K)
        names: list[str]
        scales: np.ndarray
    """
    terms = [str(term) for term in feature_terms]
    allowed = {"u", "u_x", "u_xx", "u_xxx", "uu_x", "u3"}
    unsupported = [term for term in terms if term not in allowed]
    if unsupported:
        raise ValueError(f"Unsupported LS feature terms: {unsupported}")

    def _grad1(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]

    u = u_pred_torch
    u_x = _grad1(u, x_torch) if any(term in {"u_x", "u_xx", "u_xxx", "uu_x"} for term in terms) else None
    u_xx = _grad1(u_x, x_torch) if any(term in {"u_xx", "u_xxx"} for term in terms) else None
    u_xxx = _grad1(u_xx, x_torch) if "u_xxx" in terms else None
    uu_x = u * u_x if "uu_x" in terms else None
    u3 = u ** 3 if "u3" in terms else None

    feature_map = {
        "u": u,
        "u_x": u_x,
        "u_xx": u_xx,
        "u_xxx": u_xxx,
        "uu_x": uu_x,
        "u3": u3,
    }
    cols = []
    for term in terms:
        col = feature_map[term]
        if col is None:
            raise ValueError(f"Feature term '{term}' could not be constructed from the available derivatives")
        cols.append(col if col.ndim == 2 else col[:, None])

    F = torch.cat(cols, dim=1)
    scales = np.ones(len(terms), dtype=float)
    return (F.detach().cpu().numpy(), terms, scales)


def solve_pde_ls(F_np: np.ndarray, u_t_np: np.ndarray):
    """
    Solve min ||F w - u_t||_2
    Returns coefficients and diagnostics.
    """
    F_np = np.asarray(F_np, dtype=float)
    u_t_np = np.asarray(u_t_np, dtype=float).reshape(-1)
    w, residuals, rank, singular_values = np.linalg.lstsq(F_np, u_t_np, rcond=None)
    return w, residuals, int(rank), singular_values


def extract_pde_ls(model, t_np, x_np, feature_terms: Iterable[str], device: str):
    """
    Returns dict with:
        coeffs
        names
        residuals
        rank
        singular_values
    """
    _u_pred_np, u_t_np, _t_torch, x_torch, u_pred_torch = eval_model_and_time_derivative(
        model, t_np, x_np, device
    )
    F_np, names, scales = build_feature_matrix(feature_terms, u_pred_torch, x_torch)
    w, residuals, rank, singular_values = solve_pde_ls(F_np, u_t_np)
    return {
        "coeffs": np.asarray(w, dtype=float),
        "names": names,
        "residuals": np.asarray(residuals, dtype=float),
        "rank": int(rank),
        "singular_values": np.asarray(singular_values, dtype=float),
        "scales": np.asarray(scales, dtype=float),
    }
