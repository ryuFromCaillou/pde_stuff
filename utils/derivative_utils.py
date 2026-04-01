import numpy as np
import torch


def autograd_spatial_derivatives(model, t, x, max_order=3):
    """
    Compute spatial derivatives via autograd up to `max_order`.

    Parameters
    ----------
    model : callable
        Must satisfy model(t, x) -> u with shapes (N,1).
    t, x : torch.Tensor
        Shape (N,1). `x` should have requires_grad=True for derivatives.
    max_order : int
        0..3 supported.
    """
    max_order = int(max_order)
    if max_order < 0 or max_order > 3:
        raise ValueError("max_order must be in {0,1,2,3}")

    u = model(t, x)
    out = {"u": u}

    if max_order >= 1:
        ux = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        out["ux"] = ux
    if max_order >= 2:
        uxx = torch.autograd.grad(
            out["ux"],
            x,
            grad_outputs=torch.ones_like(out["ux"]),
            create_graph=True,
            retain_graph=True,
        )[0]
        out["uxx"] = uxx
    if max_order >= 3:
        uxxx = torch.autograd.grad(
            out["uxx"],
            x,
            grad_outputs=torch.ones_like(out["uxx"]),
            create_graph=True,
            retain_graph=True,
        )[0]
        out["uxxx"] = uxxx
    return out


def fd_first_periodic(u_row, dx):
    u = np.asarray(u_row)
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def fd_second_periodic(u_row, dx):
    u = np.asarray(u_row)
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)


def fd_third_periodic(u_row, dx):
    u = np.asarray(u_row)
    return (np.roll(u, -2) - 2.0 * np.roll(u, -1) + 2.0 * np.roll(u, 1) - np.roll(u, 2)) / (
        2.0 * dx**3
    )


def fd_first_centered(u_row, dx):
    u = np.asarray(u_row)
    return (u[2:] - u[:-2]) / (2.0 * dx)


def fd_second_centered(u_row, dx):
    u = np.asarray(u_row)
    return (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)


def fd_third_centered(u_row, dx):
    u = np.asarray(u_row)
    return (u[:-4] - 2.0 * u[1:-3] + 2.0 * u[3:-1] - u[4:]) / (2.0 * dx**3)


def compute_error_metrics(pred, ref, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = pred - ref
    ref_l2 = np.linalg.norm(ref.reshape(-1))
    rel_l2 = float(np.linalg.norm(diff.reshape(-1)) / (ref_l2 + float(eps)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    return {"rel_l2": rel_l2, "rmse": rmse, "max_abs": max_abs}


def reconstruct_grid(t_np, x_np, y_np, round_decimals=6):
    """
    Reconstruct a (Nt, Nx) rectangular grid from flattened (t, x, y) samples.
    Robust to ordering in the flattened arrays.
    """
    t_r = np.round(np.asarray(t_np).reshape(-1), int(round_decimals))
    x_r = np.round(np.asarray(x_np).reshape(-1), int(round_decimals))
    y = np.asarray(y_np).reshape(-1)

    t_unique = np.unique(t_r)
    x_unique = np.unique(x_r)
    Nt, Nx = t_unique.size, x_unique.size

    t_to_i = {v: i for i, v in enumerate(t_unique.tolist())}
    x_to_j = {v: j for j, v in enumerate(x_unique.tolist())}

    Y = np.full((Nt, Nx), np.nan, dtype=np.float64)
    for ti, xi, yi in zip(t_r, x_r, y):
        Y[t_to_i[ti], x_to_j[xi]] = float(yi)

    if np.isnan(Y).any():
        missing = int(np.isnan(Y).sum())
        raise ValueError(f"Grid reconstruction failed: missing {missing} entries (data not rectangular).")

    return t_unique, x_unique, Y
