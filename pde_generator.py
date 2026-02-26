# ==== Imports ====

# main stuff
import argparse
import time
from dataclasses import dataclass

# scientific computing
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# file keeping stuff
import os, tempfile
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Iterable, Set, Union

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x, speeds GEMMs
scaler = torch.amp.GradScaler("cuda")

# ==============================
# Device selection
# ==============================
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("0902; 10-08-25")

# ==============================
# Model definitions
# ==============================
# === MLP, EQL, FEATLIB ===

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class SimpleMLP(nn.Module):
    def __init__(self, n_layers, hidden_size):
        super(SimpleMLP, self).__init__()
        assert n_layers >= 2, "n_layers must be at least 2"

        layers = []
        # First layer: input (2,) -> hidden_size
        layers.append(nn.Linear(2, hidden_size))
        layers.append(Sin())

        # Hidden layers: (m -> m)
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(Sin())

        # Final layer: (m -> 1)
        layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, t, x):
        input_tensor = torch.cat((t, x), dim=1)  # Concatenate along feature axis
        return self.model(input_tensor)
    
class symMLP(nn.Module):
    """
    Readout-only baseline, EQL-swappable contract.

    v(F) = readout([F])

    - Keeps readout shape (K -> 1), bias=False
    - Simple linear transformation of features acted on by optimizer (GD, Lasso, etc.)

    """
    def __init__(self, in_dim: int, prod_dim: int = 2, bias: bool = False):
        super().__init__()
        self.readout = nn.Linear(in_dim, 1, bias=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        Y = torch.cat([feats], dim=1)  # (N, K+1)
        return self.readout(Y)              # (N, 1)

# === Feature Library ===

@dataclass
class FeatureTensorOut:
    F: torch.Tensor                  # (B,K) normalized if normalize=True
    names: List[str]                 # length K
    scales: torch.Tensor             # (K,) detached; 1.0 if not normalized
    raw_cols: Optional[torch.Tensor] = None  # (B,K) raw (unnormalized) if keep_raw=True

class FeatureTensor:
    """
    Builds feature columns: "u", "u_x", "u_xx", ...
    Any requested non-primitive terms are ignored (with a warning).

    Expects u_out shape (B,1) and x tensor (B,1) with requires_grad=True
    when derivative terms are requested. Returns F shaped (B,K).
    """

    def __init__(
        self,
        terms: Iterable[str],
        normalize: bool = True,
        eps: float = 1e-12,
        keep_raw: bool = False,
    ) -> None:
        self.terms = list(terms)
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.keep_raw = bool(keep_raw)

        # runtime artifacts
        self.names: List[str] = []
        self.scales: Optional[torch.Tensor] = None

    def _l2_detached(self, col: torch.Tensor) -> torch.Tensor:
        # scalar (detached)
        return col.detach().reshape(-1).norm(p=2).clamp_min(self.eps)

    def _grad1(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]

    def build(
        self,
        u_out: torch.Tensor,
        *,
        t: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> FeatureTensorOut:
        _ = (t, y)  # explicit ignore (keeps signature stable)

        allowed = {"u", "u_x", "u_xx", "uu_x"}
        requested = [s for s in self.terms if s in allowed]
        ignored = [s for s in self.terms if s not in allowed]
        if ignored:
            print(f"Ignoring non-primitive terms: {ignored} [c:FeatureTensor]")

        if not requested:
            raise RuntimeError(f"No primitive features requested (allowed: {sorted(allowed)})")

        need = set(requested)

        feats: List[torch.Tensor] = []       # list of (B,1) normalized columns
        raw_list: List[torch.Tensor] = []    # list of (B,1) raw columns if keep_raw
        names: List[str] = []
        scales_list: List[torch.Tensor] = [] # list of scalar tensors (detached)

        def add(name: str, raw: torch.Tensor, normalize_col: bool = True) -> None:
            raw_ = raw[:, None] if raw.ndim == 1 else raw
            if raw_.ndim != 2 or raw_.shape[1] != 1:
                raise ValueError(f"Feature '{name}' must be (B,1); got {tuple(raw_.shape)}")

            if self.normalize and normalize_col:
                s = self._l2_detached(raw_)     # scalar
                col = raw_ / s
            else:
                s = torch.tensor(1.0, device=raw_.device, dtype=raw_.dtype)
                col = raw_

            feats.append(col)
            names.append(name)
            scales_list.append(s.detach())

            if self.keep_raw:
                raw_list.append(raw_)

        # Cache derivatives
        if ("u_x" in need or "u_xx" in need or "uu_x" in need) and x is None:
            raise ValueError("Requested x-derivative feature but x is None.")

        
        u_x = self._grad1(u_out, x)
        u_xx = self._grad1(u_x, x)
        uu_x = u_out * u_x

        self.u_raw = u_out
        self.ux_raw = u_x
        self.uxx_raw = u_xx
        self.uux_raw = uu_x
            
        # Build primitives
        if "u" in need:
            add("u", u_out, normalize_col=True)
        if "u_x" in need:
            add("u_x", u_x, normalize_col=True)
        if "u_xx" in need:
            add("u_xx", u_xx, normalize_col=True)
        if "uu_x" in need: 
            add("uu_x", uu_x, normalize_col=True)
        if not feats:
            raise RuntimeError("No features produced. Check 'terms' and provided coords.")

        F = torch.cat(feats, dim=1)  # (B,K)
        scales = torch.stack(scales_list).reshape(-1).detach()  # (K,)

        raw_cols = torch.cat(raw_list, dim=1) if self.keep_raw else None  # (B,K) or None

        # save artifacts and primitives for inspection
        self.scales = scales.detach()  # save scales for inspection (detached)

        return FeatureTensorOut(F=F, names=names, scales=scales, raw_cols=raw_cols)

# === EQL ===

class EQL(nn.Module):
    def __init__(self, in_dim, prod_dim=2, bias=False):
        """
        in_dim: number of input features (e.g. u, ux, uxx)

        prod_dim: number of linear combinations to create for pairwise products
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, prod_dim, bias=bias)
        self.readout = nn.Linear(in_dim + 1, 1, bias=False)


    def forward(self, feats):
        Z = self.linear(feats)  # (N, prod_dim)
        self.preop_ns = Z  # save for inspection
        prod_neuron = torch.prod(Z, dim=1, keepdim=True)  # (N,1)
        self.postop_ns = prod_neuron  # save for inspection
        Y_layer = torch.cat([feats, prod_neuron], dim=1)        # (N, in_dim+1)
        self.Y = Y_layer  # save for inspection
        return self.readout(Y_layer)


# === Trainer ===
@dataclass
class TrainerConfig:
    lr: float = 1e-3
    lambda_pde: float = 1.0
    lambda_reg: float = 1e-3
    lambda_tv: float = 1e-4
    lambda_data: float = 1.0
    selected_derivs: tuple[str, ...] = ()
    device: torch.device = torch.device("cpu")

class PDETrainer:
    """
    Consumes existing models; does not construct them.
    Think of it as an operator on (u_model, v_model).
    """
    def __init__(
        self,
        u_model: nn.Module,
        v_model: nn.Module,
        cfg: TrainerConfig,
        *,
        feature_builder: Optional[Callable] = None
    ):
        self.cfg = cfg
        self.device = cfg.device

        self.u = u_model.to(self.device)
        self.v = v_model.to(self.device)

        self.selected_derivs = cfg.selected_derivs

        # If caller didn't inject a feature_builder, build a default one.
        # WARNING: this only supports primitive terms
        if feature_builder is None:
            self.feature_tens = FeatureTensor(
                terms=self.selected_derivs,
                normalize=cfg.feature_normalize,
            )
            self.feature_builder = self.feature_tens.build
        else:
            self.feature_tens = None
            self.feature_builder = feature_builder

        params = list(self.u.parameters()) + list(self.v.parameters())
        self.optimizer = optim.Adam(params, lr=cfg.lr)
        self.mse = nn.MSELoss()

    def step(self, t, x, u_noisy, u_clean, tv_fn: Optional[Callable] = None):
        t = t.to(self.device).requires_grad_(True)
        x = x.to(self.device).requires_grad_(True)
        u_noisy = u_noisy.to(self.device)
        u_clean = u_clean.to(self.device)

        u_out = self.u(t, x)  

        # data loss
        loss_data = self.mse(u_out, u_noisy)
    
        # library + PDE loss
        features = self.feature_builder(u_out,x=x)
        self.F, self.feature_names, self.feature_scales = features.F, features.names, features.scales
       
        self.u_t = torch.autograd.grad(u_out, t, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
        v_out = self.v(self.F)
        loss_pde = self.mse(self.u_t, v_out)

        # L1 on v
        l1 = sum(p.abs().sum() for p in self.v.parameters())

        # TV term (optional injection)
        loss_tv = torch.tensor(0.0, device=self.device)
        if tv_fn is not None:
            loss_tv = tv_fn(self.u, t, x)

        loss = (
            self.cfg.lambda_data * loss_data
            + self.cfg.lambda_pde * loss_pde
            + self.cfg.lambda_reg * l1
            + self.cfg.lambda_tv * loss_tv
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_data": float(loss_data.item()),
            "loss_pde": float(loss_pde.item()),
            "l1": float(l1.item()),
            "loss_tv": float(loss_tv.item()),
            "feature_names": features.names,
            "feature_scales": features.scales,
        }
    
# Burgers solver (NumPy, CPU)
# ==============================
@dataclass
class BurgersConfig:
    N: int       # grid points
    L: float     # domain length
    nu: float    # viscosity
    dt: float    # time step
    T: float     # final time
    seed: int    # random seed for ICs and noise

def solve_burgers(cfg: BurgersConfig, return_history: bool = False):
    """
    1D viscous Burgers' equation on [0, L) with periodic BCs:
        u_t + u * u_x = nu * u_xx

    Space: 2nd-order centered differences for u_x and u_xx.
    Time: 4th-order Runge–Kutta (explicit).
    """
    rng = np.random.default_rng(cfg.seed)
    N, L, nu, dt, T = cfg.N, cfg.L, cfg.nu, cfg.dt, cfg.T

    # Grid
    x = np.linspace(0.0, L, N, endpoint=False)
    dx = L / N

    # Random Gaussian bump ICs, periodic-aware distance; zero-mean
    centers = rng.uniform(0.0, L, size=5)
    heights = rng.uniform(0.5, 2.0, size=5)
    widths  = rng.uniform(0.05*L, 0.25*L, size=5)
    u = np.zeros_like(x)
    for c, a, s in zip(centers, heights, widths):
        dx_wrap = np.minimum(np.abs(x - c), L - np.abs(x - c))
        u += a * np.exp(-0.5 * (dx_wrap / s)**2)
    u -= np.mean(u)

    # Periodic centered differences
    def dudx(u):
        return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)

    def d2udx2(u):
        return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)

    # RHS of Burgers
    def rhs(u):
        return -u * dudx(u) + nu * d2udx2(u)

    # History buffers
    t = 0.0
    if return_history:
        steps = int(np.round(T / dt))
        store_every = max(1, steps // 200)
        history_t = [t]
        history_u = [u.copy()]

    # Time loop: classic RK4
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
    else:
        return x, u, t

# ==============================
# Dataset builder from Burgers history
# ==============================

def build_dataset_from_burgers(
    N=256, L=2*np.pi, nu=0.02, dt=2e-3, T=1.0,
    noise_level=0.05, stride_t=32, stride_x=16, seed=0,
    # --- new controls ---
    time_ranges=None,          # e.g. (0.0, 0.2) or [(0.0,0.2),(0.2,0.5)]
    quantile_splits=None,      # e.g. 3 -> thirds over time (ignored if time_ranges given)
    shock_bins=None,           # e.g. [0.5, 1.0, 2.0] thresholds for max|u_x| (ignored if time_ranges/quantile_splits given)
    return_partitions=False,   # if True, return dict of partitions; else return a single concatenated set
    central_diff=True          # used only for shock_bins (u_x estimation)
):
    """
    If return_partitions=False (default): returns (t_s, x_s, y_s, y_noisy, N) for the selected times (or all).
    If return_partitions=True: returns an OrderedDict[name] -> (t_s, x_s, y_s, y_noisy, N)
    """
    cfg = BurgersConfig(N=N, L=L, nu=nu, dt=dt, T=T, seed=seed)
    x, u_final, t_end, (history_t, history_u) = solve_burgers(cfg, return_history=True)
    np.random.seed(cfg.seed)  # for reproducibility of noise

    T_snap, X_pts = history_u.shape
    # 2D grids (rows=time, cols=space)
    t2d = np.repeat(history_t[:, None], X_pts, axis=1)      # (T_snap, N)
    x2d = np.repeat(x[None, :], T_snap, axis=0)             # (T_snap, N)
    u2d = history_u                                         # (T_snap, N)

    # ------ build time partitions ------
    from collections import OrderedDict
    partitions = OrderedDict()

    def _pack(name, t_mask):
        # apply spatial stride first (on columns), then time mask (on rows), then time stride (on rows)
        cols = slice(0, None, stride_x)
        # If t_mask is an array of indices/bool mask, we’ll subsample those rows by stride_t order.
        if t_mask.dtype == bool:
            rows = np.where(t_mask)[0][::stride_t]
        else:
            rows = np.asarray(t_mask)[::stride_t]

        t_s = t2d[rows][:, cols].flatten()
        x_s = x2d[rows][:, cols].flatten()
        y_s = u2d[rows][:, cols].flatten()

        if noise_level and noise_level > 0:
            y_noisy = y_s + noise_level * np.random.randn(*y_s.shape)
        else:
            y_noisy = y_s

        partitions[name] = (
            t_s.astype(np.float32),
            x_s.astype(np.float32),
            y_s.astype(np.float32),
            y_noisy.astype(np.float32),
            N,
        )

    if time_ranges is not None:
        # absolute ranges in time
        if isinstance(time_ranges, tuple):
            time_ranges = [time_ranges]
        for (a, b) in time_ranges:
            mask = (history_t >= a) & (history_t <= b)
            _pack(f"[{a:.4g},{b:.4g}]", mask)

    elif quantile_splits is not None and quantile_splits >= 1:
        # split evenly by quantiles of time
        qs = np.linspace(0, 1, quantile_splits + 1)
        edges = np.quantile(history_t, qs)
        for i in range(len(edges) - 1):
            a, b = edges[i], edges[i+1]
            # include right edge on last bin so all points are captured
            if i < len(edges) - 2:
                mask = (history_t >= a) & (history_t < b)
            else:
                mask = (history_t >= a) & (history_t <= b)
            _pack(f"Q{i+1}:{a:.4g}-{b:.4g}", mask)

    elif shock_bins is not None and len(shock_bins) > 0:
        # simple shock proxy: max_x |u_x(t,·)| per time; then bin by thresholds
        # central differences on periodic domain
        if central_diff:
            dx = L / N
            # roll-based periodic centered difference
            ux = (np.roll(u2d, -1, axis=1) - np.roll(u2d, 1, axis=1)) / (2 * dx)
        else:
            # forward difference periodic
            dx = L / N
            ux = (np.roll(u2d, -1, axis=1) - u2d) / dx

        shock_metric = np.max(np.abs(ux), axis=1)  # shape (T_snap,)
        # build bins like: (-inf, t1], (t1, t2], ..., (tk, +inf)
        edges = [-np.inf] + list(shock_bins) + [np.inf]
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i+1]
            # left-open except first, right-closed
            if i == 0:
                mask = (shock_metric <= hi)
            else:
                mask = (shock_metric > lo) & (shock_metric <= hi)
            _pack(f"shock∈({lo:.3g},{hi:.3g}]", mask)

    else:
        # default: everything
        all_rows = np.arange(T_snap)
        _pack("all", all_rows)

    # ------ return style ------
    if return_partitions:
        return partitions
    else:
        # if multiple partitions were created but caller wants a single set,
        # concatenate them in time order
        if len(partitions) == 1:
            return next(iter(partitions.values()))
        else:
            ts, xs, ys, ns = [], [], [], []
            for (_name, (t_s, x_s, y_s, y_noisy, _N)) in partitions.items():
                ts.append(t_s); xs.append(x_s); ys.append(y_s); ns.append(y_noisy)
            return (np.concatenate(ts), np.concatenate(xs),
                    np.concatenate(ys), np.concatenate(ns), N)

# Helpers
# =======================
def print_args(args, title="Run config"):
    d = vars(args)
    # stringify lists nicely, keep other types as-is
    def _fmt(v):
        return " ".join(map(str, v)) if isinstance(v, (list, tuple)) else v
    d = {k: _fmt(v) for k, v in d.items()}
    width = max(len(k) for k in d)
    lines = [f"{k.rjust(width)} : {d[k]}" for k in sorted(d)]
    bar = "-" * (width + 2 + max(len(str(v)) for v in d.values()))
    print(f"\n{title}\n{bar}\n" + "\n".join(lines) + f"\n{bar}\n")

def make_tag(args, keys=("epochs","batch_size","noise","stride_t","stride_x","part_num","which_part","lam_tv","tv_type","lam_reg","lam_pde","lam_data")):
    parts = []
    for k in keys:
        v = getattr(args, k)
        if isinstance(v, (list, tuple)):
            v = "-".join(map(str, v))
        parts.append(f"{k}={v}")
    return "__".join(parts)

def savefig_atomic(fig_path):
    fig_path = Path(fig_path)
    if fig_path.suffix.lower() != ".pdf":
        fig_path = fig_path.with_suffix(".pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # Get a temp pathname, then close the fd so savefig can open it on Windows
    fd, tmp_name = tempfile.mkstemp(dir=fig_path.parent, suffix=".pdf")
    os.close(fd)

    # Note: dpi is ignored for vector elements in PDFs
    plt.savefig(tmp_name, format="pdf", bbox_inches="tight")
    os.replace(tmp_name, fig_path)  # atomic replace on same filesystem
    plt.close()

def save_npy_atomic(path: Path, arr):
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp path, close fd so numpy can open it on Windows
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".npy")
    os.close(fd)

    with open(tmp_name, "wb") as f:
        np.save(f, arr)

    os.replace(tmp_name, str(path))

def u_over_domain(trainer_u, t_np, x_np):
    device = next(trainer_u.parameters()).device
    t_flat = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32, device=device)
    x_flat = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        U_pred = trainer_u(t_flat, x_flat).detach().cpu().numpy().reshape(-1)
    return U_pred

def u_minus_true(u_pred, t_np, x_np, y_np, round_decimals=6):
    t_r = np.round(t_np, round_decimals)
    x_r = np.round(x_np, round_decimals)
    order = np.lexsort((x_r, t_r))  # primary key = t, secondary = x
    y_sorted = y_np.reshape(-1)[order]
    up_sorted = u_pred[order]
    max_abs_diff = float(np.max(np.abs(up_sorted - y_sorted)))
    return up_sorted - y_sorted, max_abs_diff

def snapshot_comp(
    trainer_u,
    stride_x,
    stride_t,
    y_noisy_np,
    y_np,
    t_np,
    x_np,
    snap_no=5,
    snap_which=None,
    round_decimals=6,
):
    """
    Single-figure version: left = heatmap of u_pred(t,x),
    right = snapshots (pred vs true) at selected times.
    Returns (fig, payload_dict).
    """

    u_model = trainer_u
    device = next(u_model.parameters()).device

    # 1) Predict at provided points
    t_flat = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32, device=device)
    x_flat = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        U_pred = u_model(t_flat, x_flat).detach().cpu().numpy().reshape(-1)

    # 2) Robust grid reconstruction
    t_r = np.round(t_np, round_decimals)
    x_r = np.round(x_np, round_decimals)
    order = np.lexsort((x_r, t_r))  # primary key = t, secondary = x
    t_sorted  = t_r[order]
    x_sorted  = x_r[order]
    yp_sorted = y_noisy_np.reshape(-1)[order]
    y_sorted = y_np.reshape(-1)[order]
    up_sorted = U_pred[order]
    max_abs_diff = float(np.max(np.abs(up_sorted - y_sorted)))
    print(f"max_abs_diff = {max_abs_diff:.6g}")
    
    t_unique = np.unique(t_sorted)
    x_unique = np.unique(x_sorted)
    Nt, Nx = len(t_unique), len(x_unique)

    if t_sorted.size != Nt * Nx:
        raise ValueError(
            f"Data are not a full rectangular grid: len={t_sorted.size}, Nt={Nt}, Nx={Nx}."
            " Interpolate to a rectangular grid before plotting."
        )

    U_true = yp_sorted.reshape(Nt, Nx)
    U_no_noise = y_sorted.reshape(Nt,Nx)
    U_pred_grid = up_sorted.reshape(Nt, Nx)
    
    # 3) Select snapshot times
    # --- Figure 1: heatmap ---
    fig_hm, ax0 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    im = ax0.imshow(
        U_pred_grid,
        extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    cbar = fig_hm.colorbar(im, ax=ax0)
    cbar.set_label('u_pred')
    ax0.set_xlabel('x')
    ax0.set_ylabel('t')
    ax0.set_title('Predicted u(t,x)')

    # choose idxs first (same logic you already have)
    if snap_which is None:
        idxs = np.linspace(0, Nt - 1, snap_no, dtype=int)
    else:
        idxs = np.asarray(snap_which, dtype=int)
        idxs = idxs[(idxs >= 0) & (idxs < Nt)]
        if idxs.size == 0:
            idxs = np.array([0], dtype=int)

    n_snaps = len(idxs)

    # layout: up to 3 columns per row
    max_cols = 3
    n_cols = min(n_snaps, max_cols)
    n_rows = int(np.ceil(n_snaps / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),  # tune to your LaTeX textwidth
        constrained_layout=True
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, idxs):
        ax.plot(x_unique, U_pred_grid[k, :], label="pred")
        ax.plot(x_unique, U_true[k, :], '--', alpha=0.9, label="noisy")
        ax.plot(x_unique, U_no_noise[k, :], '--', alpha=0.9, label="true")

        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f't = {t_unique[k]:.3f}')
        ax.legend(fontsize=7)

    # hide unused axes if grid bigger than number of snapshots
    for ax in axes[len(idxs):]:
        ax.axis('off')

    
    payload = {
        "t_unique": t_unique,
        "x_unique": x_unique,
        "U_true": U_true,
        "U_pred": U_pred_grid,
        "order": order,
        "Nt": Nt,
        "Nx": Nx,
        "snap_indices": idxs,
    }

    return fig, fig_hm, payload

def coeff_err_plot(CE, run_dir, names=None):
    '''
    coeffs are provided by the v-net. We compare them to the target coeffs provided by user.
    Input: plotting boolean, CE: np.ndarray [epochs, n_terms], names: list of str
    Output: coeff error plot saved to run_dir
    '''
    if CE is None:
        return
    plt.figure()
    for j in range(CE.shape[1]):
        label = names[j] if names and j < len(names) else f"term{j}"
        plt.plot(CE[:, j], label=f"|Δ {label}|")
    # overall L2
    l2 = np.linalg.norm(CE, axis=1)
    plt.plot(l2, linestyle="--", label="L2(all terms)")
    plt.xlabel("epoch"); plt.ylabel("coefficient error")
    plt.legend(); plt.tight_layout()
    savefig_atomic(run_dir/"coef_errors.pdf")

def general_plot(list_of_lists, ylabel, title, filename, run_dir):
    plt.figure()
    assert all(len(lst) == len(list_of_lists[0]) for lst in list_of_lists), \
        f"All lists must have the same length, got {[len(lst) for lst in list_of_lists]}"
    for lst in list_of_lists:
        plt.plot(torch.arange(0, len(lst)), torch.tensor(lst).view(len(lst), -1).mean(dim=1)[:len(lst)])
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    savefig_atomic(run_dir/filename)

def make_nonlinear_batch(batch_size=1000, t_torch=None, x_torch=None, y_torch=None):
    assert t_torch is not None and x_torch is not None and y_torch is not None, "Input tensors cannot be None"
    N = t_torch.shape[0]
    idx = torch.randint(0, N, (batch_size,), device=device)
    t_batch = t_torch[idx][:, None]
    x_batch = x_torch[idx][:, None]
    y_src   = y_torch
    y_batch = y_src[idx][:, None]
    
    return t_batch.float(), x_batch.float(), y_batch.float(), y_torch[idx][:, None].float()

# ==============================
# Main: training setup
# ==============================

def main():
    parser = argparse.ArgumentParser(description="PDE learner with CUDA support (Burgers dataset)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.012742749857031322, help="Learning rate (default: 1e-4)")
    parser.add_argument("--lam_reg", type=float, default=1e-5, help="Regularizer on l1 loss")
    parser.add_argument("--lam_tv", type=float, default=1e-4, help="Regularizer on tv denoising")
    parser.add_argument("--lam_pde", type=float, default=1, help="Regularizer on pde residual")
    parser.add_argument("--lam_data", type=float, default=1.0, help="Regularizer on data loss")
    parser.add_argument("--lam_a", type=float, default=1e-2, help="Regularizer on a_hat")
    parser.add_argument("--noise", type=float, default=0.5, help="Noise level added to targets (default: 0.05)")
    parser.add_argument("--log_every", type=int, default=2000, help="Logging interval (default: 2000)")
    parser.add_argument("--stride_t", type=int, default=16, help="stride in t")
    parser.add_argument("--stride_x", type=int, default=16, help="stride in x")
    parser.add_argument("--selected_derivs", nargs="+", default=["u", "u_x", "u_xx"],
        help="Derivatives to include (space-separated list, e.g. --selected_derivs uu u_x u_xx)"
    )
    parser.add_argument("--coeff_plot", action="store_true")
    parser.add_argument("--no_coeff_plot", dest="coeff_plot", action="store_false")
    parser.set_defaults(coeff_plot=True)
    parser.add_argument("--nu", type=float, default=0.02, help="viscosity")
    parser.add_argument("--target", nargs="+", default=[0,0.02,0,-1,0], help="target of selected library")
    parser.add_argument("--seed", type=int, default=0, help="training data RNG" )
    parser.add_argument("--part_num", type=int, default=1, help="How many splits" )
    parser.add_argument("--which_part", type=int, default=1, help="time partition selection" )
    parser.add_argument("--tv_type", type=str, default="u", choices=["u", "ux"], help="TV regularization type")
    parser.add_argument("--true_pde", type=dict, default={"uu_x": -1.0, "u_xx": 0.02}, help="True PDE terms for coeff error calculation")
    args = parser.parse_args()

    print_args(args) # display run config (hyperparamters, etc.)

    # save file setup
    tag = make_tag(args)  # e.g., "batch_size=32__epochs=10000__noise_std=0.05"
    SCRIPT_DIR = Path(__file__).resolve().parent
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(SCRIPT_DIR/"runs") / f"{args.noise}" / tag / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures saved to: {run_dir}\n")

    # === Main (no parse) ===

    # Build dataset once (CPU -> tensors -> device)
    partitions = build_dataset_from_burgers(noise_level=args.noise, stride_t=args.stride_t, stride_x=args.stride_x, quantile_splits=args.part_num, return_partitions=True, nu=args.nu)
    t_np, x_np, y_np, y_noisy_np, N = partitions[[k for k in partitions if k.startswith(f'Q{args.which_part}:')][0]]

    t_torch       = torch.from_numpy(t_np).to(device)
    x_torch       = torch.from_numpy(x_np).to(device)
    y_torch       = torch.from_numpy(y_np).to(device)
    y_noisy_torch = torch.from_numpy(y_noisy_np).to(device)

    u_config = dict(n_layers=4, hidden_size=64)
    v_config = dict(in_dim=len(args.selected_derivs), prod_dim=2)
    trainer_nl = PDETrainerNonlinear(
        u_config, v_config, lr=args.lr,
        lambda_reg=args.lam_reg, lambda_tv=args.lam_tv, lambda_pde=args.lam_pde, lambda_data=args.lam_data, device=device, selected_derivs=args.selected_derivs, true_pde=args.true_pde
    )
 
    # Train
    # Phase A: u only pretrain 
    lt, ld, lp, ll, ltv, pt_resid, pt_data, grad_u, grad_v, CE, names, grad_pred_hist, theta_norms_hist, residual_norm_hist = trainer_nl.train(
        epochs=args.epochs,
        batch_fn=make_nonlinear_batch,
        coeff_plot=True,
        log_every=args.log_every,
        coef_gt=args.target, 
        t_dom = t_torch, x_dom = x_torch, y_dom = y_torch, 
        batch_size=args.batch_size
    )

    save_npy_atomic(run_dir/"loss_total.npy", lt)
    save_npy_atomic(run_dir/"loss_data.npy",  ld)
    save_npy_atomic(run_dir/"loss_pde.npy",   lp)
    save_npy_atomic(run_dir/"loss_l1.npy",    ll)
    save_npy_atomic(run_dir/"loss_l1.npy",    ltv)

    w1,b1,names1=trainer_nl._pdenet_get()

    # metadata (overwrite)
    meta = dict(
        params=dict(vars(args)),  # <— convert Namespace -> dict
        feature_names=getattr(trainer_nl, "feature_names", []),
        feature_powers=getattr(trainer_nl, "feature_powers", []),
        feature_coeffs=str([w1,b1]),
        a_hat=float(torch.exp(trainer_nl.log_a).detach().cpu()),
        saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    with open(run_dir/"meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    #plots for losses and gradients  
    list_of_losses = [lt, lp, ld, ll, ltv]
    general_plot(list_of_losses, 'loss', 'Loss components over epochs', 'loss_components.png', run_dir)
    list_of_grads = [grad_pred_hist, grad_u, grad_v]
    general_plot(list_of_grads, 'gradient norm', 'Gradient norms over epochs', 'gradients.png', run_dir)
    coeff_err_plot(CE, run_dir, names)
    fig, fig_hm, load = snapshot_comp(
        trainer_nl.u, args.stride_x, args.stride_t,
        y_noisy_np, y_np, t_np, x_np,
        snap_no=5
    )
    fig.savefig(run_dir/"snap_comps.pdf")  # or fig.savefig(...)
    fig_hm.savefig(run_dir/"heat_map.pdf")  # or fig.savefig(...)

if __name__ == "__main__":
    main()
