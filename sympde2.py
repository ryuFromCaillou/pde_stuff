# ==== Imports ====
import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random

import os, json, itertools, tempfile, shutil
from pathlib import Path
from collections import OrderedDict

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x, speeds GEMMs
scaler = torch.amp.GradScaler("cuda")

# ==============================
# Device selection
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("0902; 10-08-25")

# ==============================
# Model definitions
# ==============================
# === MLPs ===

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

def _tensor_stats(x):
    x = x.detach()
    return dict(
        l1=float(x.abs().sum()),
        l2=float(torch.linalg.norm(x).item()),
        mean=float(x.mean()),
        std=float(x.std()),
        max=float(x.max()),
        min=float(x.min()),
        nnz=int((x!=0).sum().item()),
        numel=x.numel(),
    )

class SimpleMLP(nn.Module):
    def __init__(self, n_layers, hidden_size):
        super(SimpleMLP, self).__init__()
        assert n_layers >= 2, "n_layers must be at least 2"

        layers = []
        # First layer: input (2,) -> hidden_size
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers: (m -> m)
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Final layer: (m -> 1)
        layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, t, x):
        input_tensor = torch.cat((t, x), dim=1)  # Concatenate along feature axis
        return self.model(input_tensor)

class symMLP(nn.Module):
    def __init__(self, input_size, n_layers, hidden_size, linear_only=False):
        super(symMLP, self).__init__()
        assert n_layers >= 1, "n_layers must be at least 1"

        layers = []
        if n_layers == 1:
            # Direct input -> output
            layers.append(nn.Linear(input_size, 1))
        else:
            # First layer: input -> hidden
            layers.append(nn.Linear(input_size, hidden_size))
            if not linear_only:
                layers.append(nn.ReLU())
            # Intermediate layers
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                if not linear_only:
                    layers.append(nn.ReLU())
            # Final layer
            layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):   
        return self.model(x)
   
    def stats(self, with_grads=False):
        """Layer stats + effective input→output coefficients and their stats."""
        out = OrderedDict()
        for i, m in enumerate(self.model):
            if isinstance(m, nn.Linear):
                out[f"layer{i}.weight"] = _tensor_stats(m.weight)
                if m.bias is not None:
                    out[f"layer{i}.bias"] = _tensor_stats(m.bias)
                if with_grads and m.weight.grad is not None:
                    out[f"layer{i}.weight_grad"] = _tensor_stats(m.weight.grad)
                if with_grads and m.bias is not None and m.bias.grad is not None:
                    out[f"layer{i}.bias_grad"] = _tensor_stats(m.bias.grad)
        eff = self.effective_coeffs()
        out["effective_coeffs_vector"] = {
            "values": eff.detach().cpu().numpy().tolist(),
            **_tensor_stats(eff)
        }
        return out

    def effective_coeffs(self):
        """Return a single Linear-equivalent weight vector for input features (ignores bias comp)."""
        layers = [L for L in self.model if isinstance(L, nn.Linear)]
        W = layers[0].weight.detach()
        for L in layers[1:]:
            W = L.weight.detach() @ W
        return W.flatten()  # shape [n_features]

# ==== PDE Extractor ====
def extract_symbolic_pde_nl(trainer, tol=1e-3, return_dict=False):
    """Extractlinear weights from the first layer of v and map to PDE terms."""
    lin = trainer.v.model[0]
    w = lin.weight.detach().cpu().numpy().flatten()
    b = float(lin.bias.detach().cpu()) if lin.bias is not None else 0.0
    a_hat = float(torch.exp(trainer.log_a).detach().cpu())

    # expecting 4 inputs: [u_x/a, (u_x/a)^2, u_xx/a^2, u*u_x/a]
    if w.shape[0] < 4:
        print("Warning: v has fewer than 4 inputs; symbolic mapping may be invalid.")
        w0 = w[0] if w.shape[0] > 0 else 0.0
        w1 = w[1] if w.shape[0] > 1 else 0.0
        w2 = w[2] if w.shape[0] > 2 else 0.0
        w3 = w[3] if w.shape[0] > 3 else 0.0
    else:
        w0, w1, w2, w3 = w[:4]

    # Map normalized features back to PDE coefficients
    c1 =  a_hat * w0         # coefficient on u_x
    c2 =        w1           # coefficient on (u_x)^2
    c3 =        w2           # coefficient on u_xx
    c4 =  a_hat * w3         # coefficient on u*u_x
    c0 = (a_hat**2) * b      # constant term

    terms = []
    if abs(c1) > tol: terms.append(f"{c1:.4g}·u_x")
    if abs(c2) > tol: terms.append(f"{c2:.4g}·(u_x)^2")
    if abs(c3) > tol: terms.append(f"{c3:.4g}·u_xx")
    if abs(c4) > tol: terms.append(f"{c4:.4g}·(u·u_x)")
    if abs(c0) > tol: terms.append(f"{c0:.4g}")

    rhs = " + ".join(terms) if terms else "0"
    print(f"Learned scale a = {a_hat:.6g}")
    print(f"Reconstructed PDE:  u_t ≈ {rhs}")

    if return_dict:
        return {"a": a_hat, "c_u_x": c1, "c_ux2": c2, "c_uxx": c3, "c_u_ux": c4, "c_const": c0}

def extract_symbolic_pde_deep(trainer, show_physical=True):
    """
    Extract effective linear map v(F) = w^T F (+ b), print both training-space
    and (optionally) 'physical' coefficients on raw features.
    Assumes trainer.feature_names / feature_powers are from the latest step().
    If features were standardized before v, expects trainer._feat_mu/_feat_std.
    """

    # 1) Compose all Linear layers into a single W_eff, b_eff
    W_eff, b_eff = None, None
    for m in trainer.v.model:
        if isinstance(m, nn.Linear):
            W, b = m.weight.detach().cpu(), m.bias.detach().cpu()
            if W_eff is None:
                W_eff, b_eff = W, b
            else:
                # compose: y = W @ (W_eff x + b_eff) + b
                W_eff = W @ W_eff
                b_eff = W @ b_eff + b
        elif isinstance(m, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
            raise ValueError("Nonlinear activation in v; linear coefficient extraction is invalid.")

    w = W_eff.flatten().numpy()                         # shape [n_features]
    b = float(b_eff.item())                             # scalar
    names  = getattr(trainer, "feature_names", [f"f{i}" for i in range(len(w))])
    powers = getattr(trainer, "feature_powers", [0]*len(w))

    # 2) If features were standardized before v: Fz = (F - mu)/std
    #    then v(F) = w_z^T Fz + b_z = (w_z/std)^T F + (b_z - (w_z/std)^T mu).
    mu  = getattr(trainer, "_feat_mu", None)
    std = getattr(trainer, "_feat_std", None)
    if std is not None:
        std_np = std.detach().cpu().numpy().flatten().clip(min=1e-12)
        mu_np  = mu.detach().cpu().numpy().flatten() if mu is not None else np.zeros_like(std_np)
        w_true = w / std_np
        b_true = b - float((w_true * mu_np).sum())
    else:
        w_true = w
        b_true = b


    # 4) Also print the unstandardized head (still on the scaled features if you use 'a' scaling)
    if std is not None:
        print("\nAfter unstandardizing features (still training scaling wrt 'a'):")
        print(f"  b'  = {b_true:.6f}")
        for wi, nm in zip(w_true, names):
            print(f"  {wi:+.6f} · {nm}")

    if not show_physical:
        return


    # 5) Convert to physical coefficients on *raw* features if you use a-scaling:
    # Training used: u_t / a^2 ≈ Σ w_i · (raw_i / a^{p_i}) + b
    # => u_t ≈ Σ [a^{2 - p_i} w_i] · raw_i  +  [a^2 b]
    a_hat = float(torch.exp(trainer.log_a).detach().cpu()) if hasattr(trainer, "log_a") else 1.0

    w_phys = []
    for wi, p in zip(w_true, powers):
        w_phys.append((a_hat ** (2 - p)) * wi)
    b_phys = (a_hat ** 2) * b_true

    return w_phys, b_phys, names

# === Trainer ===
def _soft_thresh(x, lam):
    # elementwise soft-threshold
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)

class PDETrainerNonlinear:
    def __init__(self, u_config, v_config, lr=1e-3, lambda_pde=1.0, lambda_tv=1e-4, lambda_reg=1e-3, lambda_a=1e-3, selected_derivs=(), device=device):
        self.device = device
        self.u = SimpleMLP(**u_config).to(self.device)
        self.v = symMLP(**v_config).to(self.device)
        self.log_a = nn.Parameter(torch.zeros(1, device=self.device))
        self.lambda_pde = lambda_pde
        self.lambda_reg = lambda_reg
        self.lambda_a   = lambda_a
        self.lambda_tv = lambda_tv
        self.selected_derivs = list(selected_derivs)
        params = list(self.u.parameters()) + list(self.v.parameters()) + [self.log_a]
        #self.optimizer = optim.Adam(params, lr=lr)
        self.optimizer = optim.Adam([
            {"params": list(self.u.parameters()), "lr": lr},
            {"params": list(self.v.parameters()), "lr": lr},
            {"params": [self.log_a],            "lr": lr},
        ])
        self.mse = nn.MSELoss()

    def _build_features(self, u_out, a_hat, x=None, y=None):
        """
        Build an ordered feature matrix F along with names and powers.
        Returns: F [batch, n_feat], names [list[str]], powers [list[int]]
        """
        feats, names, powers = [], [], []

        def add(name, tensor, power):
            feats.append(tensor)
            names.append(name)
            powers.append(power)
        if 'u' in self.selected_derivs and x is not None:
            add('u', u_out, 1)

        if 'u_x' in self.selected_derivs and x is not None:
            u_x = torch.autograd.grad(u_out, x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
            add('u_x', u_x / a_hat, 1)
        
        if 'u_x2' in self.selected_derivs and x is not None:
            u_x = torch.autograd.grad(u_out, x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
            u_x2 = (u_x/a_hat)**2
            add('u_x2', u_x2, 1)

        if '2u_x' in self.selected_derivs and x is not None: 
            u_x = torch.autograd.grad(u_out, x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
            add('2u_x', 2*u_x/a_hat, 1)

        if 'u_xx' in self.selected_derivs and x is not None:
            u_x = torch.autograd.grad(u_out, x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            add('u_xx', u_xx / (a_hat**2), 2)

        if y is not None and 'u_y' in self.selected_derivs:
            u_y = torch.autograd.grad(u_out, y, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
            add('u_y', u_y / a_hat, 1)

        if y is not None and 'u_yy' in self.selected_derivs:
            u_y = torch.autograd.grad(u_out, y, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
            add('u_yy', u_yy / (a_hat**2), 2)

        if 'uu' in self.selected_derivs:
            add('uu', u_out * u_out, 0)

        if 'u_x_x' in self.selected_derivs: 
            u_x = torch.autograd.grad(u_out, x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0] / a_hat
            add('u_x_x', u_x*u_x, 2)

        if 'u_x_xx' in self.selected_derivs and x is not None:
            u_x = torch.autograd.grad(u_out, x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0] / a_hat
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] / (a_hat**2)
            add('u_x_xx', u_x * u_xx, 3)

        if 'uu_x' in self.selected_derivs and x is not None:
            u_x = torch.autograd.grad(u_out, x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0] / a_hat
            add('uu_x', u_out * u_x, 1)
        
        if '2uu_x' in self.selected_derivs and x is not None:
            u_x = torch.autograd.grad(u_out, x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0] / a_hat
            add('2uu_x', 2*u_out * u_x, 1)            

        F = torch.cat(feats, dim=1) if feats else None
        if F is not None and F.device != self.device:
            F = F.to(self.device)
        return F, names, powers

    def _pdenet_get(self): 
        return extract_symbolic_pde_deep(self)
    
    def set_lrs(self, lr_u=None, lr_v=None, lr_a=None):
    # param_groups: 0=u, 1=v, 2=a
        if lr_u is not None: self.optimizer.param_groups[0]["lr"] = lr_u
        if lr_v is not None: self.optimizer.param_groups[1]["lr"] = lr_v
        if lr_a is not None: self.optimizer.param_groups[2]["lr"] = lr_a

    def elastic_flash(
        self,
        batch_fn,
        num_batches: int = 300,
        lambda1: float = 1e-3,       # L1 strength (sparsity)
        lambda2: float = 1e-4,       # L2 (ridge) strength
        standardize: bool = True,    # z-score features, center target
        max_iter: int = 2000,
        tol: float = 1e-6,
        verbose: bool = True,
    ):
        """
        Elastic-net initializer for v:
        - Freeze u and a
        - Cache (F, y) with y = u_t / a^2 and F = selected feature stack from u
        - Solve elastic-net via coordinate descent on the standardized problem
        - Write (w, b) into the first Linear of v (expects Linear(in=n_feat, out=1))
        """
        device = next(self.u.parameters()).device
        dtype  = torch.float64  # more stable for Gram/CD; cast back when writing

        # 0) freeze u and a; keep v untouched while solving
        for p in self.u.parameters(): p.requires_grad_(False)
        self.log_a.requires_grad_(False)
        self.u.eval(); self.v.eval()

        # 1) cache features/targets from the frozen u
        F_list, y_list = [], []
        feat_names = None; feat_powers = None

        for _ in range(num_batches):
            out = batch_fn()
            # minimal tolerance for different batch_fns: (t, x, y) or (t, x, _, y)
            if len(out) == 3:
                t, x, _ydata = out
            elif len(out) == 4:
                t, x, _z, _ydata = out
            else:
                raise ValueError("batch_fn must return (t,x,y) or (t,x,*,y)")

            t = t.to(device).requires_grad_(True)
            x = x.to(device).requires_grad_(True)

            u_out = self.u(t, x)

            a_hat = torch.exp(self.log_a).detach()
            # Try to call the feature builder with create_graph=False if available
            try:
                F, names, powers = self._build_features(u_out, a_hat, x=x, y=None, create_graph=False)
            except TypeError:
                # fallback for older signature
                F, names, powers = self._build_features(u_out, a_hat, x=x, y=None)

            if F is None:
                raise RuntimeError("No features were produced; check selected_derivs.")

            # target y = u_t / a^2
            u_t = torch.autograd.grad(u_out, t, grad_outputs=torch.ones_like(u_out), create_graph=False)[0]
            y_tgt = (u_t / (a_hat ** 2)).squeeze(1)  # [N]

            F_list.append(F.detach())
            y_list.append(y_tgt.detach())
            feat_names, feat_powers = names, powers   # keep last (they're the same)

        X = torch.cat(F_list, dim=0).to(device=device, dtype=dtype)  # [N, d]
        y = torch.cat(y_list, dim=0).to(device=device, dtype=dtype)  # [N]

        N, d = X.shape

        # 2) standardize features / center target; solve on standardized problem
        mu = X.mean(0)                   # [d]
        if standardize:
            std = X.std(0).clamp_min(1e-12)
            Xs = (X - mu) / std
        else:
            std = torch.ones_like(mu)
            Xs = X - mu
        y_mean = y.mean()
        yc = y - y_mean

        # 3) coordinate descent on the Gram system
        # Problem (centered): minimize ||yc - Xs w||^2 + lambda2 ||w||^2 + lambda1 ||w||_1
        # CD update (standard): w_j <- S(c_j - sum_{k != j} G_{jk} w_k, lambda1) / (G_{jj} + lambda2),
        # where G = Xs^T Xs, c = Xs^T yc
        G = (Xs.T @ Xs)                  # [d, d]
        c = (Xs.T @ yc)                  # [d]

        w = torch.zeros(d, dtype=dtype, device=device)
        for it in range(max_iter):
            w_old = w.clone()
            # cyclic coordinate updates
            for j in range(d):
                # r_j = c_j - (G_j * w) + G_jj * w_j
                Gj = G[j, :]                       # [d]
                rj = c[j] - (Gj @ w) + Gj[j] * w[j]
                denom = Gj[j] + lambda2
                w[j] = _soft_thresh(rj, lambda1) / denom

            # check sup-norm decrease
            if torch.max(torch.abs(w - w_old)).item() < tol:
                break

        # 4) map back to raw-feature scale and compute bias
        w_raw = w / std
        b_raw = (y_mean - (mu * w_raw).sum()).to(dtype)

        # 5) write into first Linear of v
        lin = None
        for layer in self.v.model:
            if isinstance(layer, nn.Linear):
                lin = layer
                break
        if lin is None or lin.in_features != d or lin.out_features != 1:
            raise RuntimeError("elastic_flash expects v to start with Linear(in=n_feat,out=1).")

        # cast to v's dtype
        w_out = w_raw.to(lin.weight.dtype).reshape(1, -1)
        b_out = b_raw.to(lin.bias.dtype).reshape(())

        lin.weight.data.copy_(w_out)
        lin.bias.data.copy_(b_out)

        if verbose:
            # quick diagnostics
            with torch.no_grad():
                Xt = X - X.mean(0, keepdim=True)
                Gt = (Xt.T @ Xt) / max(N - 1, 1)
                # condition number (use SVD for stability)
                svals = torch.linalg.svdvals(Gt)
                condG = (svals[0] / svals[-1].clamp_min(1e-18)).item()
                # crude max correlation
                Xn = (Xt / (Xt.std(0).clamp_min(1e-12))).T  # [d, N]
                C = (Xn @ Xn.T) / (N - 1)
                C.fill_diagonal_(0)
                maxcorr = C.abs().max().item()

            print(f"[elastic_flash] N={N}, d={d}, λ1={lambda1:g}, λ2={lambda2:g}, iters={it+1}")
            print(f"[elastic_flash] cond(F^T F)≈{condG:.2e}, max|corr|≈{maxcorr:.3f}")
            print(f"[elastic_flash] wrote weights to v. ||w||_0={(w.abs() > 0).sum().item()} "
                  f"(nonzeros), bias={float(b_out):.3e}")

        # 6) unfreeze and return to train mode
        for p in self.u.parameters(): p.requires_grad_(True)
        self.log_a.requires_grad_(True)
        self.u.train(); self.v.train()

        # stash names for later inspection (optional)
        self.feature_names = feat_names
        self.feature_powers = feat_powers

        return w_out.detach().cpu().numpy(), float(b_out)

    def tv1d_space(self, u_fn, t, x, eps=1e-6, reduce="mean"):
        """
        Smoothed TV  in space:  E[ sqrt(u_x^2 + eps^2) ]
        u_fn : callable(t, x) -> u  (N,1)
        t,x  : (N,1) tensors with requires_grad as needed
        eps  : Charbonnier smoothing (avoid nondifferentiability at 0)
        reduce : "mean" | "sum" | None  (returns tensor if None)
        """
        u = u_fn(t, x)
        ones = torch.ones_like(u)
        ux = torch.autograd.grad(u, x, grad_outputs=ones,
                                 create_graph=True, retain_graph=True)[0]
        tv = torch.sqrt(ux**2 + eps**2)
        if reduce == "mean":  return tv.mean()
        if reduce == "sum":   return tv.sum()
        return tv
    
    def step(self, t, x, y, train_data=True, train_pde=True, coeff_plot=True, target=[]):
        # Ensure inputs on the correct device
        t = t.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)

        t.requires_grad_(True)
        x.requires_grad_(True)

        u_out = self.u(t, x)
        #u_t  = torch.autograd.grad(u_out, t, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
        #u_x  = torch.autograd.grad(u_out, x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
        #u_xx = torch.autograd.grad(u_x,  x, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
        #uu_x = u_x * u_out

        a_hat = torch.exp(self.log_a)

        # normalized features
        #u_x_norm  = u_x / a_hat
        #u_x2_norm = u_x_norm**2
        #u_xx_norm = u_xx / (a_hat**2)
        #uu_x_norm = u_out * u_x_norm

        loss      = torch.tensor(0.0, device=self.device)
        l1        = torch.tensor(0.0, device=self.device)
        loss_data = torch.tensor(0.0, device=self.device)
        loss_pde  = torch.tensor(0.0, device=self.device)
        l2_a      = torch.tensor(0.0, device=self.device)
        L_tv      = torch.tensor(0.0, device=self.device)

        if train_data: 
            loss_data = self.mse(u_out, y)
            loss += loss_data
    
        if train_pde: 
            # build features ONLY when using PDE head
            F, names, powers = self._build_features(u_out, a_hat, x=x, y=None)
            self.feature_names, self.feature_powers = names, powers

            u_t = torch.autograd.grad(u_out, t, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
            v_out = self.v(F)

            loss_pde = self.mse(u_t / (a_hat**2), v_out)
            loss += loss_pde

            # regularize v only when it's active
            l1 = sum(p.abs().sum() for p in self.v.parameters())
            l2_a = (self.log_a**2).sum()
            loss += self.lambda_reg * l1
            loss += self.lambda_a * l2_a
            loss += self.lambda_tv * L_tv

    #v_in  = torch.cat([u_x_norm, u_ x2_norm, u_xx_norm, uu_x_norm], dim=1)
    #v_out = self.v(v_in)
    #loss_data = self.mse(u_out, y)
    #loss_pde  = self.mse(u_t / (a_hat**2), v_out)
    #l1   = sum(p.abs().sum() for p in self.v.parameters())
    #l2_a = torch.sum(self.log_a ** 2)
    #loss = loss_data + self.lambda_pde * loss_pde + self.lambda_reg * l1 + self.lambda_a * l2_a
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        coeff_error = None
        if coeff_plot:
            assert target is not None, "No target coeff provided"
            # extract current physical coefficients
            w_phys, b_phys, names = self._pdenet_get()  # ensure this RETURNS, not prints
            curr_co = list(map(float, w_phys)) + [float(b_phys)]     # list[float]
            target  = list(map(float, target))
            assert len(curr_co) == len(target), "target length must match learned coeff length"
            coeff_error = np.abs(np.array(curr_co, dtype=np.float64) -
                                 np.array(target,  dtype=np.float64))  # shape [n_terms]

        return {
            "total_loss": float(loss.item()),
            "data_loss":  float(loss_data.item()),
            "pde_loss":   float(loss_pde.item()),
            "l1_penalty": float(l1.item()),
            "l2_log_a":   float(l2_a.item()),
            "a_hat":      float(a_hat.item()),
            "coeff_error": coeff_error,   # None or np.ndarray [n_terms]
            "coeff_names": names if coeff_plot else None,
        }

    def train(self, epochs, batch_fn, train_data=True, train_pde=True, log_every=100, coef_gt=None, coeff_plot=False):
        lt, ld, lp, ll = [], [], [], []
        coeff_err_hist = []   # list of np.ndarray [n_terms] per epoch
        coeff_names_ref = None

        for epoch in range(epochs):
            t, x, y = batch_fn()
            losses = self.step(t, x, y,
                               train_data=train_data, train_pde=train_pde,
                               target=coef_gt, coeff_plot=coeff_plot)
    
            lt.append(losses["total_loss"])
            ld.append(losses["data_loss"])
            lp.append(losses["pde_loss"])
            ll.append(losses["l1_penalty"])
    
            if coeff_plot and (losses["coeff_error"] is not None):
                if coeff_names_ref is None and losses.get("coeff_names") is not None:
                    coeff_names_ref = list(losses["coeff_names"]) + ["bias"]
                coeff_err_hist.append(losses["coeff_error"])
    
            if epoch % log_every == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: total={lt[-1]:.4e}, data={ld[-1]:.4e}, pde={lp[-1]:.4e}, l1={ll[-1]:.4e}")
                w,b,names=self._pdenet_get()
                print("v(F)  =  b  +  Σ w_i · f_i    (training space)")
                print(f"  b   = {b:.6f}")
                for wi, nm in zip(w, names):
                    print(f"  {wi:+.6f} · {nm}")

        # Turn coeff_err_hist into an array [epochs, n_terms] if collected
        coeff_err_hist = (np.vstack(coeff_err_hist) if coeff_err_hist else None)
    
        return lt, ld, lp, ll, coeff_err_hist, coeff_names_ref

# ==============================
# Burgers solver (NumPy, CPU)
# ==============================
@dataclass
class BurgersConfig:
    N: int       # grid points
    L: float     # domain length
    nu: float    # viscosity
    dt: float    # time step
    T: float     # final time
    seed: int = 0

def solve_burgers(cfg: BurgersConfig, return_history: bool = False):
    """
    1D viscous Burgers' equation on [0, L) with periodic BCs:
        u_t + u * u_x = nu * u_xx

    Space: 2nd-order centered differences for u_x and u_xx.
    Time: 4th-order Runge–Kutta (explicit).
    """
    rng = np.random.default_rng(int(time.time_ns() % (2**32)))
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

# =======================
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

def make_tag(params: dict, keys=None):
    """Stable tag from selected hyperparams (sorted keys)."""
    if keys is None:
        keys = sorted(params.keys())
    parts = [f"{k}={params[k]}" for k in keys]
    return "__".join(parts).replace(" ", "")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig_atomic(fig_path):
    fig_path = Path(fig_path)
    if fig_path.suffix.lower() != ".pdf":
        fig_path = fig_path.with_suffix(".pdf")
    ensure_dir(fig_path.parent)

    # Get a temp pathname, then close the fd so savefig can open it on Windows
    fd, tmp_name = tempfile.mkstemp(dir=fig_path.parent, suffix=".pdf")
    os.close(fd)

    # Note: dpi is ignored for vector elements in PDF; keep it only if you rasterize parts
    plt.savefig(tmp_name, format="pdf", bbox_inches="tight")
    os.replace(tmp_name, fig_path)  # atomic replace on same filesystem
    plt.close()

def save_npy_atomic(path: Path, arr):
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".npy")
    ensure_dir(path.parent)
    # Create temp path, close fd so numpy can open it on Windows
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".npy")
    os.close(fd)
    with open(tmp_name, "wb") as f:
        np.save(f, arr)
    os.replace(tmp_name, str(path))

def run_one(trainer_factory, batch_fn_factory, *, params, out_root="runs", tag_keys=("epochs","batch_size","noise_std")):
    tag = make_tag(params, keys=tag_keys)  # e.g., "batch_size=32__epochs=10000__noise_std=0.05"
    SCRIPT_DIR = Path(__file__).resolve().parent
    run_dir = Path(SCRIPT_DIR/out_root) / tag            # <- no timestamp; will overwrite files in here
    ensure_dir(run_dir)
    print(f"cwd: {run_dir}\n")

    trainer = trainer_factory()
    batch_fn = batch_fn_factory(batch_size=params["batch_size"], noise_std=params["noise_std"])

    lt, ld, lp, ll = trainer.train(
        epochs=params["epochs"],
        batch_fn=batch_fn,
        train_data=True,
        train_pde=True,
        log_every=max(1, params["epochs"] // 10),
    )

    final_oper = extract_symbolic_pde_deep(trainer)
    print(final_oper)
    # save raw arrays (overwrite)
    save_npy_atomic(run_dir/"loss_total.npy", lt)
    save_npy_atomic(run_dir/"loss_data.npy",  ld)
    save_npy_atomic(run_dir/"loss_pde.npy",   lp)
    save_npy_atomic(run_dir/"loss_l1.npy",    ll)

    # metadata (overwrite)
    meta = dict(params=params,
                feature_names=getattr(trainer, "feature_names", []),
                feature_powers=getattr(trainer, "feature_powers", []),
                a_hat=float(torch.exp(trainer.log_a).detach().cpu()),
                saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                extracted_pde=final_oper
                )
    with open(run_dir/"meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # plot (smoothed)
    def plot_smoothed(arr, label, target_points=1000):
        arr = torch.tensor(arr)
        n = arr.numel()
        k = max(1, n // target_points)
        m = (n // k) * k
        if m < n:
            arr = arr[:m]
        arr = arr.view(-1, k).mean(dim=1)
        x = torch.arange(arr.numel()) * k
        plt.plot(x, arr, label=label)

    plt.figure()
    plot_smoothed(lt, "total")
    plot_smoothed(lp, "pde")
    plot_smoothed(ld, "data")
    plot_smoothed(ll, "l1 penalty")
    plt.xlabel("epoch"); plt.ylabel("loss (chunk-avg)")
    plt.title(tag)
    plt.legend(loc="best"); plt.grid(True, alpha=0.3); plt.tight_layout()

    # single, fixed filename -> overwrite when params repeat
    savefig_atomic(run_dir/"loss_plot.png")

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

    # 3) Figure with two panels
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax0, ax1 = axs

    # Left: Heatmap of prediction
    im = ax0.imshow(
        U_pred_grid,
        extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    cbar = fig.colorbar(im, ax=ax0)
    cbar.set_label('u_pred')
    ax0.set_xlabel('x')
    ax0.set_ylabel('t')
    ax0.set_title('Predicted u(t,x)')

    # Right: Snapshots (pred vs true)
    if snap_which is None:
        idxs = np.linspace(0, Nt - 1, snap_no, dtype=int)
    else:
        idxs = np.asarray(snap_which, dtype=int)
        idxs = idxs[(idxs >= 0) & (idxs < Nt)]
        if idxs.size == 0:
            idxs = np.array([0], dtype=int)

    for k in idxs:
        ax1.plot(x_unique, U_pred_grid[k, :], label=f"pred t={t_unique[k]:.3f}")
        ax1.plot(x_unique, U_true[k, :], '--', alpha=0.9, label=f"Noisy t={t_unique[k]:.3f}")
        ax1.plot(x_unique, U_no_noise[k, :], '--', alpha=0.9, label=f"True t={t_unique[k]:.3f}")
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.set_title('Snapshots: pred vs true')
    ax1.legend(ncol=2, fontsize=8)

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

    return fig, payload

# ==============================
# Main: training setup
# ==============================
def main():
    parser = argparse.ArgumentParser(description="PDE learner with CUDA support (Burgers dataset)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.012742749857031322, help="Learning rate (default: 1e-4)")
    parser.add_argument("--lam_reg", type=float, default=1e-5, help="Regularizer on l1 loss")
    parser.add_argument("--noise", type=float, default=0.5, help="Noise level added to targets (default: 0.05)")
    parser.add_argument("--log_every", type=int, default=2000, help="Logging interval (default: 2000)")
    parser.add_argument("--stride_t", type=int, default=16, help="stride in t")
    parser.add_argument("--stride_x", type=int, default=16, help="stride in x")
    parser.add_argument("--selected_derivs", nargs="+", default=["uu", "u_x", "u_xx", "uu_x"],
        help="Derivatives to include (space-separated list, e.g. --selected_derivs uu u_x u_xx)"
    )
    parser.add_argument("--train_pde", type=bool, default=True, help="train pde?")
    parser.add_argument("--train_data", type=bool, default=True, help="train data?")
    parser.add_argument("--coeff_plot", type=bool, default=True, help="plot coeff errors?")
    parser.add_argument("--target", nargs="+", default=[0,0.02,0,-1,0], help="target of selected library")
    parser.add_argument("--seed", type=int, default=0, help="training data RNG" )
    parser.add_argument("--part_num", type=int, default=1, help="How many splits" )
    parser.add_argument("--which_part", type=int, default=1, help="time partition selection" )

    args = parser.parse_args()

    print_args(args)

    def make_tag(args, keys=("epochs","batch_size","noise","selected_derivs","lr","stride_t","stride_x","part_num","which_part")):
        parts = []
        for k in keys:
            v = getattr(args, k)
            if isinstance(v, (list, tuple)):
                v = "-".join(map(str, v))
            parts.append(f"{k}={v}")
        return "__".join(parts)
    
    tag = make_tag(args)  # e.g., "batch_size=32__epochs=10000__noise_std=0.05"
    SCRIPT_DIR = Path(__file__).resolve().parent
    run_dir = Path(SCRIPT_DIR/"runs") / f"{args.noise}" / tag            # <- no timestamp; will overwrite files in here
    ensure_dir(run_dir)
    print(f"Figures saved to: {run_dir}\n")


    # === Main (no parse) ===

    # Build dataset once (CPU -> tensors -> device)
    partitions = build_dataset_from_burgers(noise_level=args.noise, stride_t=args.stride_t, stride_x=args.stride_x, quantile_splits=args.part_num, return_partitions=True)
    t_np, x_np, y_np, y_noisy_np, N = partitions[[k for k in partitions if k.startswith(f'Q{args.which_part}:')][0]]

    t_torch       = torch.from_numpy(t_np).to(device)
    x_torch       = torch.from_numpy(x_np).to(device)
    y_torch       = torch.from_numpy(y_np).to(device)
    y_noisy_torch = torch.from_numpy(y_noisy_np).to(device)

    def make_nonlinear_batch(batch_size=args.batch_size, noise=args.noise):
        N = t_torch.shape[0]
        idx = torch.randint(0, N, (batch_size,), device=device)
        t_batch = t_torch[idx][:, None]
        x_batch = x_torch[idx][:, None]
        y_src   = y_noisy_torch if (noise and noise > 0) else y_torch
        y_batch = y_src[idx][:, None]
        return t_batch.float(), x_batch.float(), y_batch.float()

    u_config = dict(n_layers=3, hidden_size=64)
    v_config = dict(input_size=len(args.selected_derivs), n_layers=1, hidden_size=1, linear_only=True)
    trainer_nl = PDETrainerNonlinear(
        u_config, v_config, lr=args.lr,
        lambda_reg=args.lam_reg, lambda_a=1e-3, device=device, selected_derivs=args.selected_derivs
    )

    # Train
    # Phase A: u only pretrain 
    lt, ld, lp, ll, CE, names = trainer_nl.train(
        epochs=10000,
        batch_fn=make_nonlinear_batch,
        train_pde=False,   # <— u-only
        train_data=True,
        coeff_plot=False,
        log_every=args.log_every
    )

    # Phase B: Flash coeffs with least square
    #trainer_nl.elastic_flash(
    #    batch_fn= make_nonlinear_batch,
    #    num_batches=512,     # bigger cache → steadier coefficients
    #    lambda1=5e-4,        # try 1e-4..1e-2, higher → sparser
    #    lambda2=1e-4,        # small ridge to tame collinearity
    #    standardize=True,
    #    max_iter=2000,
    #    tol=1e-6,
    #    verbose=True
    #)
    
    #trainer_nl.set_lrs(lr_u=1e-3, lr_v=5e-4, lr_a=5e-4)

    ##Phase C: resume training w/ low lr
    #lt, ld, lp, ll, CE, names = trainer_nl.train(
    #    epochs=args.epochs,
    #    batch_fn=make_nonlinear_batch,
    #    log_every=args.log_every, 
    #    train_pde=args.train_pde, 
    #    train_data=args.train_data, 
    #    coef_gt=args.target,
    #    coeff_plot=args.coeff_plot
    #)

    save_npy_atomic(run_dir/"loss_total.npy", lt)
    save_npy_atomic(run_dir/"loss_data.npy",  ld)
    save_npy_atomic(run_dir/"loss_pde.npy",   lp)
    save_npy_atomic(run_dir/"loss_l1.npy",    ll)

    w,b,names=trainer_nl._pdenet_get()
    print("v(F)  =  b  +  Σ w_i · f_i    (training space)")
    print(f"  b   = {b:.6f}")
    for wi, nm in zip(w, names):
        print(f"  {wi:+.6f} · {nm}")


    # metadata (overwrite)
    meta = dict(
        params=dict(vars(args)),  # <— convert Namespace -> dict
        feature_names=getattr(trainer_nl, "feature_names", []),
        feature_powers=getattr(trainer_nl, "feature_powers", []),
        feature_coeffs=str([w,b]),
        a_hat=float(torch.exp(trainer_nl.log_a).detach().cpu()),
        saved_at=time.strftime("%Y-%m-%d %H:%M:%S")
    )

    with open(run_dir/"meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # plot (smoothed)
    def plot_smoothed(arr, label, target_points=1000):
        arr = torch.tensor(arr)
        n = arr.numel()
        k = max(1, n // target_points)
        m = (n // k) * k
        if m < n:
            arr = arr[:m]
        arr = arr.view(-1, k).mean(dim=1)
        x = torch.arange(arr.numel()) * k
        plt.plot(x, arr, label=label)

    plt.figure()
    plot_smoothed(lt, "total")
    plot_smoothed(lp, "pde")
    plot_smoothed(ld, "data")
    plot_smoothed(ll, "l1 penalty")
    plt.xlabel("epoch"); plt.ylabel("loss (chunk-avg)")
    plt.yscale("log")
    plt.title(tag)
    plt.legend(loc="best"); plt.grid(True, alpha=0.3); plt.tight_layout()

    # single, fixed filename -> overwrite when params repeat
    savefig_atomic(run_dir/"loss_plot.png")
    if CE is not None:
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
    
    snapshot_comp(
        trainer_nl.u, args.stride_x, args.stride_t,
        y_noisy_np, y_np, t_np, x_np,
        snap_no=2, snap_which=[0,-1]
    )
    savefig_atomic(run_dir/"snap_comps_init.pdf")  # or fig.savefig(...)
 
if __name__ == "__main__":
    main()
