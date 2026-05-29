from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional, Callable

import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from .featlib import FeatureTensorOut, FeatureTensor

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
    beta: float = 0.99

    # --- Optional pruning controls (disabled by default) ---
    prune_enabled: bool = False
    prune_amount: float = 0.2
    prune_coeff_mag_max: Optional[float] = None
    prune_ema_grad_max: Optional[float] = None
    prune_ema_drift_max: Optional[float] = None

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
        # this only supports primitive terms
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

        # --- Pruning indicators (EMA state) ---
        self.ema_grad: dict[str, torch.Tensor] = {}
        self.ema_drift: dict[str, torch.Tensor] = {}
        self._prev_params: dict[str, torch.Tensor] = {}
        self._prune_applied: bool = False

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
        
        #############################
        ### L1 Pruning Indicators ###
        #############################
        beta = float(self.cfg.beta)

        coeff_mag_norm = torch.tensor(0.0, device=self.device)
        ema_grad_norm = torch.tensor(0.0, device=self.device)
        ema_drift_norm = torch.tensor(0.0, device=self.device)

        for name, p in self.v.named_parameters():
            if p.grad is None:
                continue

            grad_mag = p.grad.detach().abs()
            coeff_mag = p.detach().abs()

            prev = self._prev_params.get(name)
            drift_mag = (p.detach() - prev).abs() if prev is not None else torch.zeros_like(coeff_mag)

            if name not in self.ema_grad:
                self.ema_grad[name] = grad_mag.clone()
            else:
                self.ema_grad[name] = beta * self.ema_grad[name] + (1 - beta) * grad_mag

            if name not in self.ema_drift:
                self.ema_drift[name] = drift_mag.clone()
            else:
                self.ema_drift[name] = beta * self.ema_drift[name] + (1 - beta) * drift_mag

            self._prev_params[name] = p.detach().clone()

            coeff_mag_norm = coeff_mag_norm + coeff_mag.pow(2).sum()
            ema_grad_norm = ema_grad_norm + self.ema_grad[name].pow(2).sum()
            ema_drift_norm = ema_drift_norm + self.ema_drift[name].pow(2).sum()

        coeff_mag_norm = torch.sqrt(coeff_mag_norm)
        ema_grad_norm = torch.sqrt(ema_grad_norm)
        ema_drift_norm = torch.sqrt(ema_drift_norm)

        # Trigger pruning once when indicators are below configured thresholds.
        prune_ready = self.cfg.prune_enabled and (
            (self.cfg.prune_coeff_mag_max is None or coeff_mag_norm.item() <= self.cfg.prune_coeff_mag_max)
            and (self.cfg.prune_ema_grad_max is None or ema_grad_norm.item() <= self.cfg.prune_ema_grad_max)
            and (self.cfg.prune_ema_drift_max is None or ema_drift_norm.item() <= self.cfg.prune_ema_drift_max)
        )

        if (not self._prune_applied) and prune_ready:
            try:
                # prune all hidden linear layers
                for layer in self.v.linears:
                    if isinstance(layer, nn.Linear):
                        prune.l1_unstructured(
                            layer,
                            name="weight",
                            amount=float(self.cfg.prune_amount),
                        )

                # prune readout
                prune.l1_unstructured(
                    self.v.readout,
                    name="weight",
                    amount=float(self.cfg.prune_amount),
                )

                self._prune_applied = True
            except Exception:
                pass

        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_data": float(loss_data.item()),
            "loss_pde": float(loss_pde.item()),
            "l1": float(l1.item()),
            "loss_tv": float(loss_tv.item()),
            "coeff_mag": float(coeff_mag_norm.item()),
            "ema_grad": float(ema_grad_norm.item()),
            "ema_drift": float(ema_drift_norm.item()),
            "prune_applied": int(self._prune_applied),
            "feature_names": features.names,
            "feature_scales": features.scales,
        }
  
