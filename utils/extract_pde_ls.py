from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from prog.featlib import FeatureTensor


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
    ft = FeatureTensor(feature_terms, normalize=False, keep_raw=True)
    out = ft.build(u_pred_torch, x=x_torch)
    return (
        out.F.detach().cpu().numpy(),
        list(out.names),
        out.scales.detach().cpu().numpy(),
    )


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

