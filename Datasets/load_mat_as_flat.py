from scipy.io import loadmat
import numpy as np
from collections import OrderedDict

def load_burgers_mat_as_flat(
    mat_path,
    noise_level=0.0,
    stride_t=1,
    stride_x=1,
    seed=0,
    quantile_splits=None,
    return_partitions=False,
):
    data = loadmat(mat_path)

    # --- pull arrays (common keys in PDE-FIND datasets) ---
    x = np.squeeze(data["x"])
    t = np.squeeze(data["t"])

    # Burgers.mat sometimes uses 'u' or 'usol'
    if "u" in data:
        u = data["u"]
    elif "usol" in data:
        u = data["usol"]
    else:
        raise KeyError(f"Couldn't find 'u' or 'usol' in {list(data.keys())}")

    # --- sanitize u: handle tiny complex roundoff, enforce float64 ---
    u = np.real_if_close(u, tol=1000)
    u = np.asarray(u, dtype=np.float64)

    # --- ensure u is shaped (Nx, Nt) ---
    Nx = x.size
    Nt = t.size
    if u.shape == (Nt, Nx):
        u = u.T
    if u.shape != (Nx, Nt):
        raise ValueError(f"Expected u shape (Nx,Nt)=({Nx},{Nt}) but got {u.shape}")

    N = Nx  # match your builder's meaning

    # build rectangular grids like your burgers builder does
    t2d = np.repeat(t[None, :], Nx, axis=0)     # (Nx, Nt)
    x2d = np.repeat(x[:, None], Nt, axis=1)     # (Nx, Nt)

    rng = np.random.default_rng(seed)

    def _pack(name, t_cols_idx):
        # t_cols_idx selects time columns; then apply stride_t over selected times
        t_cols_idx = np.asarray(t_cols_idx)
        t_cols_idx = t_cols_idx[::stride_t]

        # apply strides: x along rows, t along cols
        x_rows = np.arange(Nx)[::stride_x]

        tt = t2d[np.ix_(x_rows, t_cols_idx)].reshape(-1)
        xx = x2d[np.ix_(x_rows, t_cols_idx)].reshape(-1)
        yy = u[np.ix_(x_rows, t_cols_idx)].reshape(-1)

        if noise_level and noise_level > 0:
            yy_noisy = yy + noise_level * rng.standard_normal(size=yy.shape)
        else:
            yy_noisy = yy.copy()

        return (tt.astype(np.float32),
                xx.astype(np.float32),
                yy.astype(np.float32),
                yy_noisy.astype(np.float32),
                N)

    partitions = OrderedDict()

    if quantile_splits is not None and quantile_splits >= 1:
        # split by quantiles of time values (over columns)
        qs = np.linspace(0, 1, quantile_splits + 1)
        edges = np.quantile(t, qs)
        for i in range(len(edges) - 1):
            a, b = edges[i], edges[i + 1]
            if i < len(edges) - 2:
                mask = (t >= a) & (t < b)
            else:
                mask = (t >= a) & (t <= b)
            cols = np.where(mask)[0]
            partitions[f"Q{i+1}:{a:.4g}-{b:.4g}"] = _pack(f"Q{i+1}", cols)
    else:
        # all times
        partitions["all"] = _pack("all", np.arange(Nt))

    if return_partitions:
        return partitions

    # otherwise return a single concatenated set (like your builder does)
    if len(partitions) == 1:
        return next(iter(partitions.values()))

    ts, xs, ys, yns = [], [], [], []
    for (_k, (tt, xx, yy, yy_noisy, _N)) in partitions.items():
        ts.append(tt); xs.append(xx); ys.append(yy); yns.append(yy_noisy)
    return (np.concatenate(ts), np.concatenate(xs),
            np.concatenate(ys), np.concatenate(yns), N)