from __future__ import annotations

from typing import Callable, Iterable
import torch


def grad1(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    First derivative dy/dx for tensors shaped (B,1).
    """
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def smooth_abs(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable |z| approximation.
    """
    return torch.sqrt(z * z + float(eps))


def tv_u(u: torch.Tensor, x: torch.Tensor, eps: float = 1e-8, reduce: str = "mean") -> torch.Tensor:
    """
    TV over u in space:
        TV(u) ~= E[ sqrt(u_x^2 + eps) ]
    """
    ux = grad1(u, x)
    val = smooth_abs(ux, eps=eps)
    if reduce == "mean":
        return val.mean()
    if reduce == "sum":
        return val.sum()
    raise ValueError(f"Unknown reduce='{reduce}'")


def tv_ux(u: torch.Tensor, x: torch.Tensor, eps: float = 1e-8, reduce: str = "mean") -> torch.Tensor:
    """
    TV over u_x in space:
        TV(u_x) ~= E[ sqrt(u_xx^2 + eps) ]
    """
    ux = grad1(u, x)
    uxx = grad1(ux, x)
    val = smooth_abs(uxx, eps=eps)
    if reduce == "mean":
        return val.mean()
    if reduce == "sum":
        return val.sum()
    raise ValueError(f"Unknown reduce='{reduce}'")


def tv_uxx(u: torch.Tensor, x: torch.Tensor, eps: float = 1e-8, reduce: str = "mean") -> torch.Tensor:
    """
    TV over u_xx in space:
        TV(u_xx) ~= E[ sqrt(u_xxx^2 + eps) ]
    """
    ux = grad1(u, x)
    uxx = grad1(ux, x)
    uxxx = grad1(uxx, x)
    val = smooth_abs(uxxx, eps=eps)
    if reduce == "mean":
        return val.mean()
    if reduce == "sum":
        return val.sum()
    raise ValueError(f"Unknown reduce='{reduce}'")


def laplacian_xt_sq(
    u: torch.Tensor,
    t: torch.Tensor,
    x: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Squared spacetime Laplacian penalty:
        E[(u_xx + u_tt)^2]
    """
    ut = grad1(u, t)
    utt = grad1(ut, t)

    ux = grad1(u, x)
    uxx = grad1(ux, x)

    val = (uxx + utt) ** 2
    if reduce == "mean":
        return val.mean()
    if reduce == "sum":
        return val.sum()
    raise ValueError(f"Unknown reduce='{reduce}'")


def combine_tv_terms(
    terms: Iterable[tuple[float, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]],
    u: torch.Tensor,
    t: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Generic combiner for weighted TV/regularization terms.

    Each item is:
        (lambda_k, fn)

    where fn has signature fn(u, t, x) -> scalar tensor
    """
    total = u.new_tensor(0.0)
    for lam, fn in terms:
        lam = float(lam)
        if lam == 0.0:
            continue
        total = total + lam * fn(u, t, x)
    return total


def dispatch_tv(tv_type: str) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Map a string `tv_type` to a callable with signature:
        fn(u, t, x) -> scalar tensor
    """
    tv_type = str(tv_type).lower().strip()
    if tv_type in {"tv_u", "u"}:
        def _fn(u: torch.Tensor, _t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return tv_u(u, x)

        return _fn
    if tv_type in {"tv_ux", "ux"}:
        def _fn(u: torch.Tensor, _t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return tv_ux(u, x)

        return _fn
    if tv_type in {"tv_uxx", "uxx"}:
        def _fn(u: torch.Tensor, _t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return tv_uxx(u, x)

        return _fn
    if tv_type in {"laplacian_xt_sq", "laplacian", "lap"}:
        return laplacian_xt_sq

    raise ValueError(f"Unknown tv_type='{tv_type}'. Expected one of: tv_u, tv_ux, tv_uxx, laplacian_xt_sq")


def available_tv_types() -> tuple[str, ...]:
    return ("tv_u", "tv_ux", "tv_uxx", "laplacian_xt_sq")
