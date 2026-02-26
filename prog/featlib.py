from dataclasses import dataclass
from typing import Iterable, List, Optional
import torch



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
