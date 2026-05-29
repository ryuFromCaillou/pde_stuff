import numpy as np
from dataclasses import dataclass


@dataclass
class HeatConfig:
    """
    1D heat equation on a periodic domain:
        u_t = alpha * u_xx
    Domain: x in [0, L)
    """

    # Discretization / domain
    N: int = 256
    L: float = 2 * np.pi
    dt: float = 2e-3
    T: float = 1.0

    # PDE param
    alpha: float = 0.01

    # IC randomness
    seed: int = 0
    ic_modes: int = 8


def solve_heat(cfg: HeatConfig, *, return_history: bool = False):
    """
    Periodic explicit FD with RK4 time integration.

    Returns
    -------
    x, u_final, t_end, (t_hist, u_hist) when return_history=True
    x, u_final, t_end otherwise
    """
    N = int(cfg.N)
    L = float(cfg.L)
    dt = float(cfg.dt)
    T = float(cfg.T)
    alpha = float(cfg.alpha)

    if N < 8:
        raise ValueError("HeatConfig.N must be >= 8")
    if dt <= 0 or T <= 0:
        raise ValueError("HeatConfig.dt and HeatConfig.T must be > 0")
    if alpha < 0:
        raise ValueError("HeatConfig.alpha must be >= 0")

    rng = np.random.default_rng(int(cfg.seed))

    x = np.linspace(0.0, L, N, endpoint=False, dtype=np.float64)
    dx = float(L) / float(N)

    # Smooth-ish random initial condition via truncated Fourier series
    ic_modes = max(1, int(cfg.ic_modes))
    ks = np.arange(1, ic_modes + 1, dtype=np.float64)
    a = rng.standard_normal(size=ic_modes)
    b = rng.standard_normal(size=ic_modes)
    u0 = np.zeros_like(x)
    for k, ak, bk in zip(ks, a, b):
        u0 = u0 + ak * np.cos(2.0 * np.pi * k * x / L) + bk * np.sin(2.0 * np.pi * k * x / L)
    u0 = u0 - float(np.mean(u0))

    def u_xx(u_row: np.ndarray) -> np.ndarray:
        u_row = np.asarray(u_row, dtype=np.float64)
        return (np.roll(u_row, -1) - 2.0 * u_row + np.roll(u_row, 1)) / (dx * dx)

    def rhs(u_row: np.ndarray) -> np.ndarray:
        return alpha * u_xx(u_row)

    u = np.asarray(u0, dtype=np.float64).copy()
    t = 0.0
    nsteps = int(np.round(T / dt))

    if return_history:
        history_t = [t]
        history_u = [u.copy()]

    for n in range(nsteps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt

        if return_history and (n % max(1, nsteps // 200) == 0 or n == nsteps - 1):
            history_t.append(t)
            history_u.append(u.copy())

    if return_history:
        return x, u.copy(), float(t), (np.array(history_t, dtype=np.float64), np.vstack(history_u).astype(np.float64))
    return x, u.copy(), float(t)

