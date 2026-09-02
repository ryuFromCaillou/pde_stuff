from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.derivative_utils import (
    build_reference_primitive_features,
    evaluate_model_primitive_features,
    primitive_feature_name_to_key,
    reconstruct_grid,
)


def savefig_atomic(fig, fig_path) -> Path:
    fig_path = Path(fig_path)
    if fig_path.suffix.lower() != ".pdf":
        fig_path = fig_path.with_suffix(".pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=fig_path.parent, suffix=".pdf")
    os.close(fd)
    try:
        fig.savefig(tmp_name, format="pdf", bbox_inches="tight")
        os.replace(tmp_name, fig_path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        plt.close(fig)
    return fig_path


def select_evenly_spaced_time_indices(num_times: int, max_slices: int = 5) -> list[int]:
    num_times = int(num_times)
    if num_times <= 0:
        return []
    count = min(int(max_slices), num_times)
    if count == num_times:
        return list(range(num_times))
    return np.unique(np.linspace(0, num_times - 1, num=count, dtype=int)).tolist()


def save_primitive_feature_overlays(
    model,
    t_np,
    x_np,
    u_ref_np,
    feature_terms,
    *,
    output_dir,
    device: str = "cpu",
    reference_feature_builder=None,
    derivative_mode: str = "periodic",
    max_time_slices: int = 5,
):
    """
    Save one overlay PDF per active primitive feature.

    Each PDF overlays model vs clean-grid reference feature curves on evenly
    spaced time slices selected from the available time domain.
    """
    model_eval = evaluate_model_primitive_features(
        model,
        t_np,
        x_np,
        feature_terms,
        device=device,
        normalize=False,
    )
    feature_names = list(model_eval["feature_names"])

    if reference_feature_builder is None:
        ref_eval = build_reference_primitive_features(
            t_np,
            x_np,
            u_ref_np,
            feature_names,
            derivative_mode=derivative_mode,
        )
    else:
        ref_eval = reference_feature_builder(t_np, x_np, u_ref_np, feature_names)

    pred_values = np.asarray(model_eval["values"], dtype=np.float64)
    ref_values = np.asarray(ref_eval["values"], dtype=np.float64)
    if pred_values.shape != ref_values.shape:
        raise ValueError(
            f"Predicted/reference feature arrays must match shape; got {pred_values.shape} vs {ref_values.shape}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_unique, x_unique, _ = reconstruct_grid(t_np, x_np, u_ref_np)
    time_indices = select_evenly_spaced_time_indices(len(t_unique), max_slices=max_time_slices)
    if not time_indices:
        return []

    saved_paths = []
    n_panels = len(time_indices)
    fig_cols = min(3, n_panels)
    fig_rows = int(np.ceil(n_panels / fig_cols))

    for feature_idx, feature_name in enumerate(feature_names):
        _, _, pred_grid = reconstruct_grid(t_np, x_np, pred_values[:, feature_idx])
        _, _, ref_grid = reconstruct_grid(t_np, x_np, ref_values[:, feature_idx])
        feature_key = primitive_feature_name_to_key(feature_name)

        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(5 * fig_cols, 3.5 * fig_rows), squeeze=False)
        axes_flat = axes.reshape(-1)

        for ax_idx, time_idx in enumerate(time_indices):
            ax = axes_flat[ax_idx]
            ax.plot(x_unique, ref_grid[time_idx], label="reference", linewidth=2.0)
            ax.plot(x_unique, pred_grid[time_idx], label="model", linewidth=1.8, linestyle="--")
            ax.set_title(f"{feature_name} at t={float(t_unique[time_idx]):.6g}")
            ax.set_xlabel("x")
            ax.set_ylabel(feature_name)
            ax.grid(True, alpha=0.25)
            if ax_idx == 0:
                ax.legend()

        for ax in axes_flat[n_panels:]:
            ax.axis("off")

        fig.suptitle(f"Feature overlay: {feature_name}", fontsize=13)
        fig.tight_layout()
        saved_paths.append(savefig_atomic(fig, output_dir / f"{feature_key}_overlay.pdf"))

    return saved_paths


def save_time_slice_snapshots(
    *,
    x_grid,
    t_grid,
    true_grid,
    pred_grid,
    output_dir,
    value_name,
    sample_grid=None,
    time_indices=None,
):
    """
    Save one PDF per selected time slice for true vs learned 1D snapshots.
    """
    x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    t_grid = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    true_grid = np.asarray(true_grid, dtype=np.float64)
    pred_grid = np.asarray(pred_grid, dtype=np.float64)
    if true_grid.shape != pred_grid.shape:
        raise ValueError(
            f"true_grid and pred_grid must share shape; got {true_grid.shape} vs {pred_grid.shape}"
        )
    if true_grid.ndim != 2:
        raise ValueError(f"Expected 2D grids; got ndim={true_grid.ndim}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if time_indices is None:
        time_indices = select_evenly_spaced_time_indices(len(t_grid), max_slices=5)

    saved_paths = []
    for time_idx in time_indices:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(x_grid, true_grid[time_idx], label="true", linewidth=2.0)
        ax.plot(x_grid, pred_grid[time_idx], label="learned", linewidth=1.8, linestyle="--")
        if sample_grid is not None:
            sample_values = np.asarray(sample_grid, dtype=np.float64)
            if sample_values.shape != true_grid.shape:
                raise ValueError(
                    f"sample_grid must match true_grid shape; got {sample_values.shape} vs {true_grid.shape}"
                )
            ax.scatter(
                x_grid,
                sample_values[time_idx],
                label="train samples",
                s=10,
                alpha=0.6,
                zorder=3,
            )
        ax.set_title(f"{value_name} at t={float(t_grid[time_idx]):.6g}")
        ax.set_xlabel("x")
        ax.set_ylabel(value_name)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        filename = f"{primitive_feature_name_to_key(value_name)}_tidx_{int(time_idx):03d}.pdf"
        saved_paths.append(savefig_atomic(fig, output_dir / filename))

    return saved_paths
