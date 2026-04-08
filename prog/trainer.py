import torch
from dataclasses import dataclass
from typing import Optional, Callable

import torch.nn as nn
import torch.optim as optim
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
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
  