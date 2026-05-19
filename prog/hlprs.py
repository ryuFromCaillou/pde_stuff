

import matplotlib.pyplot as plt
from pathlib import Path
import os,tempfile
import numpy as np
import torch

device = 'cpu'
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

def general_plot(list_of_lists, name_list, ylabel, title, filename, run_dir):
    plt.figure()
    assert all(len(lst) == len(list_of_lists[0]) for lst in list_of_lists), \
        f"All lists must have the same length, got {[len(lst) for lst in list_of_lists]}"
    for lst, name in zip(list_of_lists, name_list):
        plt.plot(torch.arange(0, len(lst)), torch.tensor(lst).view(len(lst), -1).mean(dim=1)[:len(lst)], label=name)
    plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    savefig_atomic(f"{run_dir}/{filename}")

def make_batch(batch_size=1000, t_torch=None, x_torch=None, y_clean=None, y_noisy=None):
    idx = torch.randint(0, t_torch.shape[0], (batch_size,), device=device)
    t = t_torch[idx][:, None].float().requires_grad_(True)
    x = x_torch[idx][:, None].float().requires_grad_(True)
    u_noisy = y_noisy[idx][:, None].float()
    u_clean = y_clean[idx][:, None].float()
    return t, x, u_noisy, u_clean

# inputs are trainer, data, and correct coeffs(for error calculation).
# saves a JSON with: 
def save_to_json(trainer, aux, correct_coeffs, path):
    import json
    import inspect
    learned_coeffs = trainer.v.readout.weight.detach().cpu().numpy().flatten().tolist()
    list_difference = [x - y for x, y in zip(correct_coeffs, learned_coeffs)]
    coeff_error = np.linalg.norm(list_difference).item()

    data = aux

    try:
        data["mask_pde_source"] = inspect.getsource(trainer.cfg.lambda_pde_mask_fn)
        data["mask_tv_source"] = inspect.getsource(trainer.cfg.lambda_tv_mask_fn)
    except AttributeError:
        print("couldn't save masking functions. Likely cause is trainer instance being PDETRainer, which doesn't have them as attributes.")
    data['learned_coeffs'] = learned_coeffs
    data['correct_coeffs'] = correct_coeffs
    data['coeff_error'] = coeff_error
    data["feature_scales"] = data["feature_scales"].tolist() if type(data["feature_scales"]) is torch.Tensor else data["feature_scales"]
    # print(f"key{key} is of type {type(data[key])} and shape {np.shape(data[key])}" for key in data.keys())
    with open(path, "w") as f:
        json.dump(data, f, indent=4)