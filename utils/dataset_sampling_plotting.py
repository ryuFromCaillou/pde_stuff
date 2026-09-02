from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_space_time_coordinate_scatter(
    *,
    x_coords: np.ndarray,
    t_coords: np.ndarray,
    png_path: Path,
    vector_path: Path,
    dpi: int = 300,
) -> None:
    x = np.asarray(x_coords, dtype=np.float64).reshape(-1)
    t = np.asarray(t_coords, dtype=np.float64).reshape(-1)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(
        x,
        t,
        s=8.0,
        c="#1f1f1f",
        alpha=0.8,
        linewidths=0.0,
        marker="o",
        rasterized=False,
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("t", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for path in (png_path, vector_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(vector_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
