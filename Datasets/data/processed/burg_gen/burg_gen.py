import numpy as np
from dataclasses import dataclass
from collections import OrderedDict


# Burgers solver (NumPy, CPU)
# ==============================
@dataclass
class BurgersDatasetConfig:
    # solver
    N: int = 256
    L: float = 2 * np.pi
    nu: float = 0.02
    dt: float = 2e-3
    T: float = 1.0
    seed: int = 0

    # dataset construction
    noise_level: float = 0.05
    stride_t: int = 32
    stride_x: int = 16
    subset_size: int = 10

def solve_burgers(
    N=256,
    L=2*np.pi,
    nu=0.02,
    dt=2e-3,
    T=1.0,
    seed=0,
    return_history=False,
):
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, L, N, endpoint=False)
    dx = L / N

    centers = rng.uniform(0.0, L, size=5)
    heights = rng.uniform(0.5, 2.0, size=5)
    widths  = rng.uniform(0.05 * L, 0.25 * L, size=5)

    u = np.zeros_like(x)
    for c, a, s in zip(centers, heights, widths):
        dx_wrap = np.minimum(np.abs(x - c), L - np.abs(x - c))
        u += a * np.exp(-0.5 * (dx_wrap / s) ** 2)
    u -= np.mean(u)

    def dudx(u):
        return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)

    def d2udx2(u):
        return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)

    def rhs(u):
        return -u * dudx(u) + nu * d2udx2(u)

    t = 0.0
    if return_history:
        nsteps = int(np.round(T / dt))
        history_t = [t]
        history_u = [u.copy()]

    nsteps = int(np.round(T / dt))
    for n in range(nsteps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt

        if return_history and (n % max(1, nsteps // 200) == 0 or n == nsteps - 1):
            history_t.append(t)
            history_u.append(u.copy())

    if return_history:
        return x, u, t, (np.array(history_t), np.vstack(history_u))
    return x, u, t

def build_dataset_from_burgers(cfg):
    x, u_final, t_end, (history_t, history_u) = solve_burgers(
        N=cfg.N,
        L=cfg.L,
        nu=cfg.nu,
        dt=cfg.dt,
        T=cfg.T,
        seed=cfg.seed,
        return_history=True,
    )

    rng = np.random.default_rng(cfg.seed)

    T_snap, X_pts = history_u.shape

    t2d = np.repeat(history_t[:, None], X_pts, axis=1)
    x2d = np.repeat(x[None, :], T_snap, axis=0)
    u2d = history_u

    rows = slice(0, None, cfg.stride_t)
    cols = slice(0, None, cfg.stride_x)

    t_s = t2d[rows, cols].reshape(-1)
    x_s = x2d[rows, cols].reshape(-1)
    y_s = u2d[rows, cols].reshape(-1)

    if cfg.noise_level > 0:
        sigma = cfg.noise_level * np.std(y_s)
        y_noisy = y_s + sigma * rng.standard_normal(y_s.shape)
    else:
        y_noisy = y_s.copy()

    return (
        t_s.astype(np.float32),
        x_s.astype(np.float32),
        y_s.astype(np.float32),
        y_noisy.astype(np.float32),
        cfg.N,
    )
