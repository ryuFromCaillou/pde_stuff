import numpy as np
from dataclasses import dataclass


@dataclass
class AllenCahnConfig:
    # Discretization / domain
    N: int = 201
    x_min: float = -1.0
    x_max: float = 1.0
    dt: float = 0.01
    T: float = 1.0

    # PDE params
    d: float = 0.001
    reaction_scale: float = 5.0  # u_t = d u_xx + r*u - r*u^3
    bc_value: float = -1.0       # Dirichlet boundary value at x_min/x_max

    # Dataset sampling / noise
    stride_t: int = 1
    stride_x: int = 1
    noise_level: float = 0.0     # as fraction of std(y_s); set 0 for noise-free
    seed: int = 0


def solve_allen_cahn(cfg: AllenCahnConfig, *, return_history: bool = False):
    """
    1D Allen–Cahn (explicit FD in time):
        u_t = d * u_xx + r*u - r*u^3

    Domain: [x_min, x_max] with Dirichlet BCs u = bc_value at endpoints.
    Space: 2nd-order centered differences for u_xx.
    Time: forward Euler.
    """
    N, dt, T = int(cfg.N), float(cfg.dt), float(cfg.T)
    d, r = float(cfg.d), float(cfg.reaction_scale)

    # Grids (include endpoints for Dirichlet BCs)
    x = np.linspace(float(cfg.x_min), float(cfg.x_max), N, endpoint=True)
    dx = float(x[1] - x[0])
    t = np.arange(0.0, T + 0.5 * dt, dt)
    Nt = int(t.size)

    # Solution array
    u = np.zeros((Nt, N), dtype=np.float64)

    # Initial condition
    u[0, :] = x**2 * np.cos(np.pi * x)

    # Boundary conditions
    u[0, 0] = float(cfg.bc_value)
    u[0, -1] = float(cfg.bc_value)

    inv_dx2 = 1.0 / (dx * dx)

    for n in range(Nt - 1):
        u_n = u[n]
        u_xx = (u_n[2:] - 2.0 * u_n[1:-1] + u_n[:-2]) * inv_dx2
        reaction = r * u_n[1:-1] - r * (u_n[1:-1] ** 3)
        u[n + 1, 1:-1] = u_n[1:-1] + dt * (d * u_xx + reaction)

        u[n + 1, 0] = float(cfg.bc_value)
        u[n + 1, -1] = float(cfg.bc_value)

    if return_history:
        return x, u[-1].copy(), float(t[-1]), (t.astype(np.float64), u.astype(np.float64))
    return x, u[-1].copy(), float(t[-1])


def build_dataset_from_allen(cfg: AllenCahnConfig):
    """
    Build a flat dataset from an Allen–Cahn rollout.

    Returns
    -------
    t_s, x_s, y_s, y_noisy, N
        Flattened arrays (float32) after applying `stride_t` and `stride_x`.
        `N` is `cfg.N` (number of spatial grid points).
    """
    x, _u_final, _t_end, (history_t, history_u) = solve_allen_cahn(cfg, return_history=True)

    T_snap, X_pts = history_u.shape
    t2d = np.repeat(history_t[:, None], X_pts, axis=1)
    x2d = np.repeat(x[None, :], T_snap, axis=0)
    u2d = history_u

    rows = np.arange(T_snap)[:: int(cfg.stride_t)]
    cols = slice(0, None, int(cfg.stride_x))

    t_s = t2d[rows][:, cols].flatten()
    x_s = x2d[rows][:, cols].flatten()
    y_s = u2d[rows][:, cols].flatten()

    if cfg.noise_level and float(cfg.noise_level) > 0:
        rng = np.random.default_rng(int(cfg.seed))
        sigma = float(cfg.noise_level) * float(np.std(y_s))
        y_noisy = y_s + sigma * rng.standard_normal(size=y_s.shape)
    else:
        y_noisy = y_s

    return (
        t_s.astype(np.float32),
        x_s.astype(np.float32),
        y_s.astype(np.float32),
        y_noisy.astype(np.float32),
        int(cfg.N),
    )


def _save_mat(path: str, *, t: np.ndarray, x: np.ndarray, u: np.ndarray) -> None:
    import scipy.io as sio

    sio.savemat(
        path,
        {
            "t": t.reshape(1, -1),
            "x": x.reshape(1, -1),
            "u": u,
        },
    )


if __name__ == "__main__":
    cfg = AllenCahnConfig(N=201, dt=0.01, T=1.0, d=0.001, reaction_scale=5.0, bc_value=-1.0)
    x, _u_final, _t_end, (t, u) = solve_allen_cahn(cfg, return_history=True)
    _save_mat("Allen_Cahn_generated.mat", t=t, x=x, u=u)
    print("Saved Allen_Cahn_generated.mat")
    print("t shape:", (1, t.size))
    print("x shape:", (1, x.size))
    print("u shape:", u.shape)

