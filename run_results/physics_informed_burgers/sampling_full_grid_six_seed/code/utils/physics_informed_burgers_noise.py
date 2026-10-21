"""Fixed-observation-noise robustness for the validated physical Burgers head."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.physics_informed_burgers import (
    FEATURES,
    PhysicalHead,
    coefficients,
    evaluate,
    initialize,
    objective,
    write_json,
)


SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
SEEDS = (0, 1)


def make_condition(*, sigma: float, init_seed: int, noise_seed: int):
    """Build paired observations; noise is generated once and never in training."""
    from Datasets.data.processed.burg_gen.burg_gen import solve_burgers

    x, _, _, (t, u) = solve_burgers(seed=0, nu=.02, T=.1, return_history=True)
    ti = np.linspace(0, len(t) - 1, 10, dtype=int)
    xi = np.linspace(0, len(x) - 1, 64, dtype=int)
    tm, xm = np.meshgrid(t, x, indexing="ij")
    full = tuple(torch.tensor(a.reshape(-1, 1), dtype=torch.float32) for a in (tm, xm, u))
    indices = (ti[:, None] * len(x) + xi[None, :]).ravel()
    clean_train = tuple(a[indices] for a in full)
    clean_targets = clean_train[2].numpy().ravel().astype(np.float64)
    scale = float(np.std(clean_targets, ddof=0))
    noise_rng = np.random.default_rng(noise_seed)
    epsilon = noise_rng.standard_normal(clean_targets.shape)
    noisy_targets = clean_targets + float(sigma) * scale * epsilon
    train = (clean_train[0], clean_train[1], torch.tensor(noisy_targets[:, None], dtype=torch.float32))
    ref = __import__("utils.physics_informed_burgers", fromlist=["dataset"]).dataset()[2]
    grid = dict(t=t, x=x, u=u, indices=indices, time_indices=ti, space_indices=xi)
    model, head = initialize(init_seed)
    initial = dict(surrogate=model.state_dict(), head=head.state_dict())
    return train, clean_train, full, ref, grid, epsilon, scale, initial


def validate_noise_protocol():
    """Protocol checks before any long trajectory is launched."""
    checks = {}
    a = make_condition(sigma=0.0, init_seed=0, noise_seed=7000)
    b = make_condition(sigma=0.0, init_seed=0, noise_seed=7001)
    torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
    torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)
    torch.testing.assert_close(a[0][2], b[0][2], rtol=0, atol=0)
    np.testing.assert_array_equal(a[4]["indices"], b[4]["indices"])
    checks["sigma_zero_targets_independent_of_noise_seed"] = True
    np.testing.assert_allclose(a[6], np.std(a[1][2].numpy(), ddof=0), rtol=1e-8, atol=1e-10)
    checks["scale_is_population_std_of_640_clean_targets"] = a[6]
    c = make_condition(sigma=.10, init_seed=0, noise_seed=7010)
    d = make_condition(sigma=.10, init_seed=0, noise_seed=7010)
    np.testing.assert_array_equal(c[5], d[5])
    np.testing.assert_array_equal(c[0][2].numpy(), d[0][2].numpy())
    checks["noise_is_fixed_and_reproducible"] = True
    e = make_condition(sigma=.01, init_seed=0, noise_seed=7000)
    f = make_condition(sigma=.20, init_seed=0, noise_seed=7000)
    np.testing.assert_array_equal(e[5], f[5])
    expected = .01 * e[6] * e[5]
    np.testing.assert_allclose(e[0][2].numpy().ravel() - e[1][2].numpy().ravel(), expected, rtol=1e-6, atol=1e-7)
    checks["paired_standardized_noise_across_sigma"] = True
    for p, q in zip(e[7]["surrogate"].values(), f[7]["surrogate"].values()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    checks["initialization_independent_of_noise"] = True
    # The model-side PDE objective uses autograd derivatives and does not read noisy targets.
    loss = objective(*initialize(0), e[0])
    assert all(torch.isfinite(v).item() for v in loss.values())
    checks["pde_objective_finite_and_autodiff_based"] = True
    return checks


def train_condition(out: Path, sigma: float, init_seed: int, noise_seed: int, steps: int):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    train, clean_train, full, ref, grid, epsilon, scale, initial = make_condition(
        sigma=sigma, init_seed=init_seed, noise_seed=noise_seed
    )
    model, head = initialize(init_seed)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=5e-4)
    batch_rng = torch.Generator().manual_seed(10000 + init_seed)
    checkpoints = sorted(set([0, 1, 10, 50, 100, 200, 500, 1000, 2000, 5000] + list(range(10000, steps + 1, 10000)) + [steps]))
    first = {"first_loose_step": None, "first_strong_step": None}
    clean_target_mse = float(torch.mean((train[2] - clean_train[2]) ** 2))
    noisy_target_mse = float(torch.mean((train[2] - train[2]) ** 2))
    write_json(out / "condition.json", dict(sigma=sigma, init_seed=init_seed, noise_seed=noise_seed,
        noise_scale=scale, noise_std=float(np.std(sigma * scale * epsilon, ddof=0)),
        clean_target_mse=clean_target_mse, epsilon_sha256=hashlib.sha256(epsilon.tobytes()).hexdigest(),
        initial_surrogate_sha256=hashlib.sha256(torch.cat([p.reshape(-1) for p in initial["surrogate"].values()]).numpy().tobytes()).hexdigest()))
    np.savez_compressed(out / "observations.npz", indices=grid["indices"], epsilon=epsilon,
        clean_targets=clean_train[2].numpy().ravel(), noisy_targets=train[2].numpy().ravel())
    torch.save(initial, out / "initial.pt")
    start = time.monotonic()
    rows = []
    with (out / "history.csv").open("w") as f:
        writer = None
        for step in range(steps + 1):
            idx = torch.randint(0, len(train[0]), (640,), generator=batch_rng)
            losses = objective(model, head, tuple(v[idx] for v in train))
            row = dict(step=step, sigma=sigma, init_seed=init_seed, noise_seed=noise_seed,
                       **{k: float(v.detach()) for k, v in losses.items()}, **coefficients(head))
            for level in ("loose", "strong"):
                if row[level] and first[f"first_{level}_step"] is None:
                    first[f"first_{level}_step"] = step
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
            if step in checkpoints:
                metrics, fields = evaluate(model, head, train, full, ref, grid)
                clean_metrics, _ = evaluate(model, head, clean_train, full, ref, grid)
                metrics.update({f"clean_{k}": v for k, v in clean_metrics.items() if k.startswith("observation_")})
                metrics["noisy_observation_mse"] = metrics["observation_mse"]
                metrics["clean_observation_mse"] = clean_metrics["observation_mse"]
                rows.append(dict(step=step, **metrics))
                write_json(out / "checkpoint_metrics.json", rows)
                np.savez_compressed(out / f"fields_{step:06d}.npz", **fields)
                torch.save(dict(step=step, surrogate=model.state_dict(), head=head.state_dict(),
                                optimizer=optimizer.state_dict(), batch_rng=batch_rng.get_state(), next_batch_indices=idx),
                           out / f"state_{step:06d}.pt")
            if step % 1000 == 0 or step == steps:
                f.flush()
                write_json(out / "progress.json", dict(**row, **first, elapsed_seconds=time.monotonic() - start))
                print(f"sigma={sigma:g} seed={init_seed} step={step} xi=({row['transport']:.6g},{row['diffusion']:.6g}) loss={row['total_loss']:.5g}", flush=True)
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward(); optimizer.step()
    summary = {**row, **metrics, **first,
               "status": "strong" if row["strong"] else "loose" if row["loose"] else "not_recovered",
               "elapsed_seconds": time.monotonic() - start}
    write_json(out / "summary.json", summary)
    return summary


def finalize_condition(out: Path, sigma: float, init_seed: int, noise_seed: int, steps: int):
    """Reconstruct a summary after a completed trajectory if reporting was interrupted."""
    history = pd.read_csv(out / "history.csv", float_precision="round_trip")
    if len(history) != steps + 1 or int(history.step.iloc[-1]) != steps:
        raise ValueError(f"Incomplete trajectory cannot be finalized: {out}")
    metrics_rows = json.loads((out / "checkpoint_metrics.json").read_text())
    metrics = metrics_rows[-1]
    row = history.iloc[-1].to_dict()
    first = {}
    for level in ("loose", "strong"):
        matches = history.loc[history[level].astype(bool), "step"]
        first[f"first_{level}_step"] = int(matches.iloc[0]) if len(matches) else None
    summary = {**row, **metrics, **first,
        "status": "strong" if bool(row["strong"]) else "loose" if bool(row["loose"]) else "not_recovered",
        "sigma": float(sigma), "init_seed": int(init_seed), "noise_seed": int(noise_seed),
        "elapsed_seconds": json.loads((out / "progress.json").read_text()).get("elapsed_seconds")}
    write_json(out / "summary.json", summary)
    return summary


def make_figures(root: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    histories = []
    metrics = []
    for sigma in SIGMAS:
        for seed in SEEDS:
            folder = root / f"sigma_{sigma:g}" / f"seed_{seed}"
            summary = json.loads((folder / "summary.json").read_text())
            records.append(summary)
            histories.append((sigma, seed, pd.read_csv(folder / "history.csv")))
            metrics.extend([{**m, "sigma": sigma, "seed": seed} for m in json.loads((folder / "checkpoint_metrics.json").read_text())])
    pd.DataFrame(records).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(metrics).to_csv(root / "checkpoint_metrics.csv", index=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(SIGMAS)))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for color, sigma in zip(colors, SIGMAS):
        for seed in SEEDS:
            h = next(v for s, se, v in histories if s == sigma and se == seed)
            ax[0].plot(h.step, h.transport, color=color, alpha=.9, label=f"σ={sigma:g}, seed {seed}")
            ax[1].plot(h.step, h.diffusion, color=color, alpha=.9)
    ax[0].axhline(-1, color="black", ls="--"); ax[1].axhline(.02, color="black", ls="--")
    for a in ax: a.set_xlabel("Optimizer steps"); a.grid(alpha=.2)
    ax[0].set_ylabel("Transport coefficient"); ax[1].set_ylabel("Diffusion coefficient")
    ax[0].legend(fontsize=7, ncol=2); fig.suptitle("Observation noise changes physical Burgers coefficient trajectories")
    fig.tight_layout(); fig.savefig(root / "coefficient_trajectories.png", dpi=160); fig.savefig(root / "coefficient_trajectories.pdf"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for color, sigma in zip(colors, SIGMAS):
        for seed in SEEDS:
            h = next(v for s, se, v in histories if s == sigma and se == seed)
            ax[0].plot(h.step, h.transport_abs_error, color=color, label=f"σ={sigma:g}, seed {seed}")
            ax[1].plot(h.step, h.diffusion_abs_error, color=color)
    ax[0].axhline(.10, color="black", ls="--"); ax[0].axhline(.25, color="gray", ls=":")
    ax[1].axhline(.01, color="black", ls="--"); ax[1].axhline(.02, color="gray", ls=":")
    for a in ax: a.set_xlabel("Optimizer steps"); a.set_yscale("log"); a.grid(alpha=.2)
    ax[0].set_ylabel("Absolute transport error"); ax[1].set_ylabel("Absolute diffusion error")
    ax[0].legend(fontsize=7, ncol=2); fig.suptitle("Noise-dependent physical coefficient errors")
    fig.tight_layout(); fig.savefig(root / "coefficient_errors.png", dpi=160); fig.savefig(root / "coefficient_errors.pdf"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for color, sigma in zip(colors, SIGMAS):
        for seed in SEEDS:
            m = pd.DataFrame([x for x in metrics if x["sigma"] == sigma and x["seed"] == seed])
            ax[0].plot(m.step, m.u_rel_l2, color=color, label=f"σ={sigma:g}, seed {seed}")
            ax[1].plot(m.step, m.ut_rel_l2, color=color, label=f"σ={sigma:g}, seed {seed}")
    for a in ax: a.set_xlabel("Optimizer steps"); a.set_ylabel("Clean-reference relative L2"); a.set_yscale("log"); a.grid(alpha=.2)
    ax[0].set_title("u"); ax[1].set_title("u_t, canonical autodiff")
    ax[0].legend(fontsize=7, ncol=2); fig.suptitle("Clean-reference field and temporal-derivative fidelity under observation noise")
    fig.tight_layout(); fig.savefig(root / "field_derivative_fidelity.png", dpi=160); fig.savefig(root / "field_derivative_fidelity.pdf"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for color, sigma in zip(colors, SIGMAS):
        for seed in SEEDS:
            m = pd.DataFrame([x for x in metrics if x["sigma"] == sigma and x["seed"] == seed])
            ax[0].plot(m.step, m.observation_mse, color=color, label=f"σ={sigma:g}, seed {seed}")
            ax[1].plot(m.step, m.independent_pde_mse, color=color)
    for a in ax: a.set_xlabel("Optimizer steps"); a.set_yscale("log"); a.grid(alpha=.2)
    ax[0].set_ylabel("Noisy-training observation MSE"); ax[1].set_ylabel("Independent clean-grid PDE residual MSE")
    ax[0].legend(fontsize=7, ncol=2); fig.suptitle("Training losses and clean-reference PDE residual under observation noise")
    fig.tight_layout(); fig.savefig(root / "loss_trajectories.png", dpi=160); fig.savefig(root / "loss_trajectories.pdf"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    df = pd.DataFrame(records)
    for seed in SEEDS:
        part = df[df.init_seed == seed]
        ax[0].plot(part.sigma, part.transport, "o-", label=f"seed {seed}")
        ax[1].plot(part.sigma, part.diffusion, "o-", label=f"seed {seed}")
    ax[0].axhline(-1, color="black", ls="--"); ax[1].axhline(.02, color="black", ls="--")
    for a in ax: a.set_xlabel("Noise level σ"); a.grid(alpha=.2); a.legend()
    ax[0].set_ylabel("Final transport coefficient"); ax[1].set_ylabel("Final diffusion coefficient")
    fig.suptitle("Final physical coefficient accuracy versus observation noise")
    fig.tight_layout(); fig.savefig(root / "final_coefficients.png", dpi=160); fig.savefig(root / "final_coefficients.pdf"); plt.close(fig)

    lines = ["# Burgers observation-noise robustness", "", "Two initialization seeds (0, 1) provide exploratory evidence only. Noise is applied once to the 640 training targets; all evaluation references are clean.", "", "|sigma|seed|transport|diffusion|transport error|diffusion error|field rel L2|ut rel L2|first loose|first strong|final|", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in sorted(records, key=lambda x: (x["sigma"], x["init_seed"])):
        lines.append(f"|{s['sigma']:g}|{s['init_seed']}|{s['transport']:.7g}|{s['diffusion']:.7g}|{s['transport_abs_error']:.7g}|{s['diffusion_abs_error']:.7g}|{s['u_rel_l2']:.7g}|{s['ut_rel_l2']:.7g}|{s['first_loose_step']}|{s['first_strong_step']}|{s['status']}|")
    lines += ["", "The noisy training observation MSE and independent clean-reference residual are reported separately. PDE residuals use surrogate autodiff derivatives and the physical head; they never use finite differences of noisy observations. Do not interpret two-seed outcomes as recovery probabilities. Sampling sparsity and crossed noise-sampling experiments were not run."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
