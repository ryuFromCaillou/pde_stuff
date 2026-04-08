from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class AffineNormalizer:
    a: float
    b: float

    def to_norm(self, v: np.ndarray) -> np.ndarray:
        return (np.asarray(v, dtype=np.float64) - self.b) / self.a

    def to_phys(self, vn: np.ndarray) -> np.ndarray:
        return self.a * np.asarray(vn, dtype=np.float64) + self.b

    @staticmethod
    def from_array(v: np.ndarray) -> "AffineNormalizer":
        v = np.asarray(v, dtype=np.float64)
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
            raise ValueError("Cannot normalize: invalid range")
        a = 0.5 * (vmax - vmin)
        b = 0.5 * (vmax + vmin)
        return AffineNormalizer(a=a, b=b)


class PDETrainDataset(Dataset):
    """
    Holds a full rectangular PDE rollout and produces a subsampled training set.

    Inputs
    ------
    t_grid : (Nt,)
    x_grid : (Nx,)
    u_grid : (Nt, Nx)

    Outputs exposed
    ---------------
    t_train, x_train, y_train_clean, y_train_noisy
    t_train_n, x_train_n
    t_norm, x_norm
    """

    def __init__(
        self,
        *,
        t_grid: np.ndarray,
        x_grid: np.ndarray,
        u_grid: np.ndarray,
        stride_t: int = 1,
        stride_x: int = 1,
        noise_level: float = 0.0,
        seed: int = 0,
        normalize: bool = True,
    ) -> None:
        self.t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
        self.x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
        self.u_grid = np.asarray(u_grid, dtype=np.float64)

        if self.u_grid.shape != (self.t_grid.size, self.x_grid.size):
            raise ValueError(
                f"u_grid shape {self.u_grid.shape} does not match "
                f"(Nt, Nx)=({self.t_grid.size}, {self.x_grid.size})"
            )

        self.stride_t = max(1, int(stride_t))
        self.stride_x = max(1, int(stride_x))
        self.noise_level = float(noise_level)
        self.seed = int(seed)
        self.normalize = bool(normalize)

        rows = np.arange(self.t_grid.size)[::self.stride_t]
        cols = np.arange(self.x_grid.size)[::self.stride_x]

        t2d = np.repeat(self.t_grid[:, None], self.x_grid.size, axis=1)
        x2d = np.repeat(self.x_grid[None, :], self.t_grid.size, axis=0)

        self.t_train = t2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)
        self.x_train = x2d[np.ix_(rows, cols)].reshape(-1).astype(np.float32)
        self.y_train_clean = self.u_grid[np.ix_(rows, cols)].reshape(-1).astype(np.float32)

        if self.noise_level > 0.0:
            rng = np.random.default_rng(self.seed)
            sigma = self.noise_level * float(np.std(self.y_train_clean))
            self.y_train_noisy = (
                self.y_train_clean + sigma * rng.standard_normal(size=self.y_train_clean.shape)
            ).astype(np.float32)
        else:
            self.y_train_noisy = self.y_train_clean.copy()

        self.t_norm = AffineNormalizer.from_array(self.t_grid)
        self.x_norm = AffineNormalizer.from_array(self.x_grid)

        if self.normalize:
            self.t_train_n = self.t_norm.to_norm(self.t_train).astype(np.float32)
            self.x_train_n = self.x_norm.to_norm(self.x_train).astype(np.float32)
        else:
            self.t_train_n = self.t_train.copy()
            self.x_train_n = self.x_train.copy()

    def __len__(self) -> int:
        return int(self.t_train.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.t_train_n[idx], dtype=torch.float32).view(1),
            torch.tensor(self.x_train_n[idx], dtype=torch.float32).view(1),
            torch.tensor(self.y_train_noisy[idx], dtype=torch.float32).view(1),
        )

    @property
    def ax_scale(self) -> float:
        return float(self.x_norm.a)

    @property
    def at_scale(self) -> float:
        return float(self.t_norm.a)

    def full_grid_normalized_flat(self):
        Nt, Nx = self.u_grid.shape
        t2d = np.repeat(self.t_grid[:, None], Nx, axis=1)
        x2d = np.repeat(self.x_grid[None, :], Nt, axis=0)
        t_flat_n = self.t_norm.to_norm(t2d.reshape(-1)).astype(np.float32)
        x_flat_n = self.x_norm.to_norm(x2d.reshape(-1)).astype(np.float32)
        return t_flat_n, x_flat_n

    def fit_arrays(self):
        return self.t_train_n, self.x_train_n, self.y_train_noisy

    def fit_arrays_with_clean(self):
        return self.t_train_n, self.x_train_n, self.y_train_clean, self.y_train_noisy