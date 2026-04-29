import torch
from dataclasses import dataclass
from typing import Optional, Callable

import torch.nn as nn
import torch.optim as optim
from .featlib import FeatureTensorOut, FeatureTensor
# TODO add coeff error to step output
# === Trainer ===
@dataclass
class TrainerConfig:
    lr: float = 1e-3
    lr_tv: float = 1e-3
    lr_pde: float = 1e-3
    lambda_pde: float = 1e-3
    lambda_reg: float = 1e-3
    lambda_tv: float = 1e-4
    lambda_data: float = 1.0
    feature_normalize: bool = False
    selected_derivs: tuple[str, ...] = ()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAPINNScalarTrainer:
    """
    Implements the training loop for a simplified SAPINN based on McClenny et al's 
    "Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism" (https://arxiv.org/abs/2211.13227).
    The main difference is that we use a single scalar lambda for the PDE and TV losses, rather than a vector of lambdas for each (x, t) point.
    This is mainly for simplicity and practicality. Recall that our original motivation for SAPINN was to find lambda_tv*,
    and hopefully find some correlation between lambda_tv* and the noise level, curvature of initial condition, etc.
    This simplified version allows us to do that.
    """

    def __init__(
        self,
        u_model: nn.Module,
        v_model: nn.Module,
        cfg: TrainerConfig,
        *,
        feature_builder: Optional[Callable] = None,
    ):
        self.cfg = cfg
        self.device = cfg.device

        self.lambda_pde = nn.Parameter(torch.tensor([cfg.lambda_pde], device=self.device))
        self.lambda_tv = nn.Parameter(torch.tensor([cfg.lambda_tv], device=self.device))

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

        self.params = list(self.u.parameters()) + list(self.v.parameters())
        self.adam = optim.Adam(self.params, lr=cfg.lr)
        self.lbfgs = optim.LBFGS(self.params,
                                lr=0.1,
                                max_iter=20,
                                max_eval=25,
                                history_size=20,
                                line_search_fn="strong_wolfe",)
        self.lam_tv_optimizer = optim.Adam([self.lambda_tv], lr=cfg.lr_tv, maximize=True)
        self.lam_pde_optimizer = optim.Adam([self.lambda_pde], lr=cfg.lr_pde, maximize=True)

        
        self.mse = nn.MSELoss()

    def mask_tv(self, lam_tv):
        return 1e-5 * (1 / (1 + torch.exp(-(lam_tv-1))))
    
    def mask_pde(self, lam_pde):
        return lam_pde**2

    def compute_loss(self, t, x, u_noisy, u_clean, tv_fn: Optional[Callable] = None):
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
        
            # PDE term
            self.u_t = torch.autograd.grad(u_out, t, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
            v_out = self.v(self.F)
            loss_pde_raw = self.mse(self.u_t, v_out)
            mask_pde = self.mask_pde(self.lambda_pde)
            loss_pde_weighted = mask_pde * loss_pde_raw

            # L1 on v
            l1 = sum(p.abs().sum() for p in self.v.parameters())

            # TV term (optional injection)
            loss_tv_raw = torch.tensor(0.0, device=self.device)
            if tv_fn is not None:
                loss_tv_raw = tv_fn(self.u, t, x)
                mask_tv = self.mask_tv(self.lambda_tv)
                loss_tv_weighted = mask_tv * loss_tv_raw

            loss = (
                self.cfg.lambda_data * loss_data
                + loss_pde_weighted
                + self.cfg.lambda_reg * l1
                + loss_tv_weighted
            )
            aux = {
                "loss": float(loss.item()),
                "loss_data": float(loss_data.item()),

                "loss_pde_raw": float(loss_pde_raw.item()),
                "lambda_pde": float(self.lambda_pde.item()),
                "mask_pde": float(mask_pde.item()),
                "loss_pde_weighted": float(loss_pde_weighted.item()),

                "l1": float(l1.item()),

                "loss_tv_raw": float(loss_tv_raw.item()),
                "lambda_tv": float(self.lambda_tv.item()),
                "mask_tv": float(mask_tv.item()),
                "loss_tv_weighted": float(loss_tv_weighted.item()),

                "feature_names": features.names,
                "feature_scales": features.scales,
                }
            
            return loss, aux

    def step_adam(self, t, x, u_noisy, u_clean, tv_fn: Optional[Callable] = None):
        # zero out gradients for all optimizers
        self.adam.zero_grad()
        self.lam_pde_optimizer.zero_grad()
        if tv_fn is not None:
            self.lam_tv_optimizer.zero_grad()

        # compute loss and gradients
        loss, aux = self.compute_loss(t, x, u_noisy, u_clean, tv_fn)
        loss.backward()

        # gradient descent step for model parameters
        self.adam.step()
        # gradient ascent on the lambdas
        self.lam_pde_optimizer.step()
        if tv_fn is not None:
            self.lam_tv_optimizer.step()
            
        return aux
    
    def step_lbfgs(self, t, x, u_noisy, u_clean, tv_fn: Optional[Callable] = None):
        self.lambda_pde.requires_grad_(False)
        self.lambda_tv.requires_grad_(False)
        aux = {} 

        # computes the loss and auxiliary info without modifying any parameters
        def closure():
            self.lbfgs.zero_grad()

            loss, aux_temp = self.compute_loss(t, x, u_noisy, u_clean, tv_fn)
            loss.backward()

            aux.clear()
            aux.update(aux_temp)  # update the outer aux with the values computed in this closure
            return loss
        
        # gradient descent on the model parameters
        loss = self.lbfgs.step(closure)
        # update loss in aux after LBFGS step
        aux["loss"] = float(loss.item())
        return aux
    

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
            "lambda_data": float(self.cfg.lambda_data),
            "loss_pde": float(loss_pde.item()),
            "lambda_pde": float(self.cfg.lambda_pde),
            "l1": float(l1.item()),
            "lambda_reg": float(self.cfg.lambda_reg),
            "loss_tv": float(loss_tv.item()),
            "lambda_tv": float(self.cfg.lambda_tv),
            "feature_names": features.names,
            "feature_scales": features.scales,
        }
