import numpy as np
import torch

from prog.featlib import FeatureTensor


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


def primitive_feature_name_to_key(name: str) -> str:
    """
    Convert primitive feature names to deterministic summary/file keys.

    Examples
    --------
    u -> u
    u_x -> ux
    u_xx -> uxx
    """
    return str(name).replace("_", "")


def evaluate_model_primitive_features(
    model,
    t_np,
    x_np,
    feature_terms,
    *,
    device: str = "cpu",
    normalize: bool = False,
):
    """
    Evaluate primitive features from the learned model using the repository's
    feature library as the authoritative source of names, definitions, and
    ordering.

    Returns
    -------
    dict with keys:
        feature_names: list[str]
        values: np.ndarray of shape (N, K)
        scales: np.ndarray of shape (K,)
    """
    t_arr = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
    x_arr = np.asarray(x_np, dtype=np.float32).reshape(-1, 1)

    t = torch.from_numpy(t_arr).to(device).requires_grad_(True)
    x = torch.from_numpy(x_arr).to(device).requires_grad_(True)

    model = model.to(device)
    model.eval()

    builder = FeatureTensor(terms=list(feature_terms), normalize=normalize, keep_raw=not normalize)

    with torch.enable_grad():
        u_pred = model(t, x)
        features = builder.build(u_pred, x=x)

    values = features.F if normalize or features.raw_cols is None else features.raw_cols
    return {
        "feature_names": list(features.names),
        "values": values.detach().cpu().numpy(),
        "scales": features.scales.detach().cpu().numpy(),
    }


def build_reference_primitive_features(
    t_np,
    x_np,
    u_ref_np,
    feature_terms,
    *,
    derivative_mode: str = "periodic",
    round_decimals: int = 6,
):
    """
    Build reference primitive features on the clean grid and return them in the
    original flattened sample order.

    The default finite-difference path assumes a rectangular space-time grid and
    periodic spatial derivatives. Callers with analytic or dataset-specific
    reference operators should wrap/replace this function at the run layer.
    """
    terms = list(feature_terms)
    allowed = {"u", "u_x", "u_xx"}
    unsupported = [term for term in terms if term not in allowed]
    if unsupported:
        raise ValueError(
            f"Reference primitive builder only supports {sorted(allowed)}; got {unsupported}"
        )

    mode = str(derivative_mode).lower()
    if mode != "periodic":
        raise ValueError(
            f"Unsupported derivative_mode '{derivative_mode}'. "
            "Use 'periodic' or inject a dataset-specific reference builder."
        )

    t_unique, x_unique, u_grid = reconstruct_grid(t_np, x_np, u_ref_np, round_decimals=round_decimals)
    if x_unique.size < 2:
        raise ValueError("Need at least two spatial grid points to compute reference derivatives.")

    dx = float(x_unique[1] - x_unique[0])
    feature_grids = {"u": u_grid}
    if "u_x" in terms or "u_xx" in terms:
        feature_grids["u_x"] = np.stack([fd_first_periodic(row, dx) for row in u_grid], axis=0)
    if "u_xx" in terms:
        feature_grids["u_xx"] = np.stack([fd_second_periodic(row, dx) for row in u_grid], axis=0)

    t_to_i = {v: i for i, v in enumerate(t_unique.tolist())}
    x_to_j = {v: j for j, v in enumerate(x_unique.tolist())}
    t_r = np.round(np.asarray(t_np).reshape(-1), int(round_decimals))
    x_r = np.round(np.asarray(x_np).reshape(-1), int(round_decimals))

    cols = []
    for term in terms:
        grid = feature_grids[term]
        flat = np.asarray([grid[t_to_i[float(ti)], x_to_j[float(xi)]] for ti, xi in zip(t_r, x_r)], dtype=np.float64)
        cols.append(flat[:, None])

    return {
        "feature_names": terms,
        "values": np.concatenate(cols, axis=1),
        "grid_shape": tuple(u_grid.shape),
        "derivative_mode": mode,
    }


def compute_primitive_feature_l2_scales(
    t_np,
    x_np,
    u_ref_np,
    feature_terms,
    *,
    derivative_mode: str = "periodic",
    reference_feature_builder=None,
    eps: float = 1e-12,
):
    """
    Compute one fixed L2 scale per primitive feature from reference training data.
    """
    if reference_feature_builder is None:
        ref_feats = build_reference_primitive_features(
            t_np,
            x_np,
            u_ref_np,
            feature_terms,
            derivative_mode=derivative_mode,
        )
    else:
        ref_feats = reference_feature_builder(t_np, x_np, u_ref_np, list(feature_terms))

    values = np.asarray(ref_feats["values"], dtype=np.float64)
    names = list(ref_feats["feature_names"])
    scales = {}
    for idx, name in enumerate(names):
        scale = float(np.linalg.norm(values[:, idx].reshape(-1), ord=2))
        scales[name] = max(scale, float(eps))

    return {
        "feature_names": names,
        "scales_by_name": scales,
        "values_shape": list(values.shape),
        "derivative_mode": ref_feats.get("derivative_mode", derivative_mode),
    }


def evaluate_primitive_feature_metrics(
    model,
    t_np,
    x_np,
    u_ref_np,
    feature_terms,
    *,
    device: str = "cpu",
    reference_feature_builder=None,
    derivative_mode: str = "periodic",
):
    """
    Canonical primitive-feature fidelity evaluation path.

    1. Use `prog.featlib.FeatureTensor` to evaluate model-side primitive
       features, preserving the active feature library's names/ordering.
    2. Build reference features on the clean grid using either:
       - a caller-supplied `reference_feature_builder`, or
       - the default finite-difference path in this module.
    3. Compare model vs reference feature columns with `compute_error_metrics`.

    Returns
    -------
    dict with keys:
        feature_names
        feature_keys
        metrics_by_name
        summary_flat
    """
    model_feats = evaluate_model_primitive_features(
        model,
        t_np,
        x_np,
        feature_terms,
        device=device,
        normalize=False,
    )
    feature_names = list(model_feats["feature_names"])

    if reference_feature_builder is None:
        ref_feats = build_reference_primitive_features(
            t_np,
            x_np,
            u_ref_np,
            feature_names,
            derivative_mode=derivative_mode,
        )
    else:
        ref_feats = reference_feature_builder(t_np, x_np, u_ref_np, feature_names)

    if list(ref_feats["feature_names"]) != feature_names:
        raise ValueError(
            "Reference feature builder returned names/order that do not match the active "
            f"feature library. Expected {feature_names}, got {list(ref_feats['feature_names'])}."
        )

    pred_values = np.asarray(model_feats["values"], dtype=np.float64)
    ref_values = np.asarray(ref_feats["values"], dtype=np.float64)
    if pred_values.shape != ref_values.shape:
        raise ValueError(
            f"Predicted/reference feature arrays must match shape; got {pred_values.shape} vs {ref_values.shape}"
        )

    metrics_by_name = {}
    summary_flat = {}
    feature_keys = []
    for idx, name in enumerate(feature_names):
        key = primitive_feature_name_to_key(name)
        feature_keys.append(key)
        metrics = compute_error_metrics(pred_values[:, idx], ref_values[:, idx])
        metrics_by_name[name] = metrics
        summary_flat[f"{key}_rel_l2"] = metrics["rel_l2"]
        summary_flat[f"{key}_rmse"] = metrics["rmse"]
        summary_flat[f"{key}_max_abs"] = metrics["max_abs"]

    return {
        "feature_names": feature_names,
        "feature_keys": feature_keys,
        "metrics_by_name": metrics_by_name,
        "summary_flat": summary_flat,
    }
