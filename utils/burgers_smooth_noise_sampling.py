"""Smooth sinusoidal Burgers noise x sampling experiment (nu=0.2, T=1).

This module is deliberately independent of pandas so that it can run in the
existing HPC2 ``pdenet`` environment without modifying that environment.

Scientific design (see BURGERS_NOISE_SAMPLING_SPEC.md):

* PDE: u_t = -u*u_x + nu*u_xx, nu=0.2, periodic [0,2*pi), t in [0,1].
* IC: u(x,0)=sin(x). Numerical grid 256 x 501 (dt=.002, RK4, centered
  periodic differences), float64.
* Fixed observation population N_base=13056, drawn uniformly without
  replacement from the 128256 numerical grid points, once per replicate.
* Sampling ratios r in {1,.75,.5,.25,.1}; counts floor(r*13056).
* Noise sigmas in {0,.01,.02,.05,.10,.20}; y = u + sigma*std(U)*epsilon.
* Six replicate seeds 0..5 with separate deterministic streams.

Training reuses ``utils.physics_informed_burgers`` unchanged: SIREN (3 hidden
layers, width 64, omega 1/1), zero-initialized two-coefficient physical head,
Adam lr=5e-4, batch 640 with replacement, unit data/PDE losses.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from utils.derivative_utils import build_burgers_reference_derivative_grids, compute_error_metrics
from utils.physics_informed_burgers import (
    FEATURES,
    PhysicalHead,
    evaluate,
    initialize,
    objective,
    write_json,
)

# ---------------------------------------------------------------------------
# Fixed scientific configuration
# ---------------------------------------------------------------------------
NU = 0.2
T_FINAL = 1.0
DT = 0.002
N_X = 256
N_T = 501
N_FULL = N_T * N_X  # 128256
POPULATION = 13056
RATIOS = (1.0, 0.75, 0.5, 0.25, 0.10)
COUNTS = tuple(int(math.floor(r * POPULATION)) for r in RATIOS)
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
SEEDS = (0, 1, 2, 3, 4, 5)

TARGETS = (-1.0, 0.2)
# Established absolute recovery limits (unchanged from the historical Burgers
# physical-head study). For nu=0.2 these are 10%/5% relative on diffusion.
LOOSE = (0.25, 0.02)
STRONG = (0.10, 0.01)
# Secondary, viscosity-scaled sensitivity criterion only. It preserves the
# historical *relative* diffusion tolerance (100%/50%) and is not primary.
LOOSE_RELATIVE = (0.25, 0.20)
STRONG_RELATIVE = (0.10, 0.10)
SUSTAINED_WINDOW = 10000

BATCH_SIZE = 640
LEARNING_RATE = 5e-4
DEFAULT_STEPS = 100000

# Independent random streams within a replicate.
ROLE_POPULATION = 101
ROLE_SUBSAMPLE = 102
ROLE_NOISE = 103
BATCH_SEED_OFFSET = 10000

CHECKPOINTS = tuple(sorted(set(
    [0, 1, 10, 50, 100, 200, 500, 1000, 2000, 5000]
    + list(range(10000, DEFAULT_STEPS + 1, 10000))
    + [DEFAULT_STEPS]
)))
FIELD_CHECKPOINTS = (0, 1000, 10000, 50000, DEFAULT_STEPS)

ROOT = Path(__file__).resolve().parents[1]
STRING_COLUMNS = {"run_id"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def count_for(ratio: float) -> int:
    return int(math.floor(float(ratio) * POPULATION))


def run_id(ratio: float, sigma: float, seed: int) -> str:
    return f"burgers_nu{NU:g}_r{ratio:.2f}_s{sigma:.2f}_seed{seed}"


def run_directory(root: Path, ratio: float, sigma: float, seed: int) -> Path:
    return Path(root) / "raw" / f"ratio_{ratio:g}" / f"sigma_{sigma:g}" / f"seed_{seed}"


def derive_rng(seed: int, role: int) -> np.random.Generator:
    """Separate deterministic NumPy stream for one replicate/role pair."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(role)]))


def checkpoints_for(steps: int) -> tuple[int, ...]:
    values = {s for s in CHECKPOINTS if s <= steps}
    values.add(int(steps))
    return tuple(sorted(values))


def field_checkpoints_for(steps: int) -> tuple[int, ...]:
    values = {s for s in FIELD_CHECKPOINTS if s <= steps}
    values.add(int(steps))
    return tuple(sorted(values))


# ---------------------------------------------------------------------------
# Numerical reference
# ---------------------------------------------------------------------------
def reference_solution(*, dt: float = DT, n_x: int = N_X, history_every: int = 1):
    """Clean float64 reference rollout with u(x,0)=sin(x)."""
    x, _, _, (t, u) = solve_burgers(
        N=int(n_x),
        nu=NU,
        dt=float(dt),
        T=T_FINAL,
        seed=0,
        return_history=True,
        history_every=int(history_every),
        initial_condition=lambda x: np.sin(x),
    )
    return np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64), np.asarray(u, dtype=np.float64)


def reference_bundle():
    """Clean solution, reference derivatives and the fixed noise scale."""
    x, t, u = reference_solution()
    if u.shape != (N_T, N_X):
        raise ValueError(f"Unexpected reference shape {u.shape}; expected {(N_T, N_X)}")
    if not np.isfinite(u).all():
        raise FloatingPointError("Non-finite reference solution")
    ref = build_burgers_reference_derivative_grids(u_grid=u, x_grid=x, nu=NU)
    scale = float(np.std(u, ddof=0))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Invalid clean-solution noise scale")
    return dict(x=x, t=t, u=u, ref=ref, scale=scale, dx=float(x[1] - x[0]))


def flatten_grid(bundle) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tm, xm = np.meshgrid(bundle["t"], bundle["x"], indexing="ij")
    return tuple(
        torch.tensor(a.reshape(-1, 1), dtype=torch.float32)
        for a in (tm, xm, bundle["u"])
    )


# ---------------------------------------------------------------------------
# Sampling and noise design
# ---------------------------------------------------------------------------
def replicate_design(seed: int) -> dict:
    """One reference population, one nested order and one noise realization.

    All three streams are independent of the sampling ratio and noise level, so
    conditions within a replicate are paired.
    """
    population = derive_rng(seed, ROLE_POPULATION).choice(
        N_FULL, size=POPULATION, replace=False
    ).astype(np.int64)
    order = derive_rng(seed, ROLE_SUBSAMPLE).permutation(POPULATION).astype(np.int64)
    epsilon = derive_rng(seed, ROLE_NOISE).standard_normal(POPULATION).astype(np.float64)
    return dict(
        seed=int(seed),
        population=population,
        subsample_order=order,
        epsilon=epsilon,
        streams=dict(
            population=f"SeedSequence([{seed},{ROLE_POPULATION}])",
            subsample=f"SeedSequence([{seed},{ROLE_SUBSAMPLE}])",
            noise=f"SeedSequence([{seed},{ROLE_NOISE}])",
            initialization=f"torch.manual_seed({seed})",
            minibatch=f"torch.Generator().manual_seed({BATCH_SEED_OFFSET + seed})",
        ),
    )


def selected_positions(design: dict, ratio: float) -> np.ndarray:
    count = count_for(ratio)
    return design["subsample_order"][:count]


def condition_arrays(bundle, design: dict, ratio: float, sigma: float) -> dict:
    """Return sampled coordinates/targets and all provenance for one condition."""
    positions = selected_positions(design, ratio)
    indices = design["population"][positions]
    clean = bundle["u"].reshape(-1)[indices]
    epsilon = design["epsilon"][positions]
    noise = float(sigma) * bundle["scale"] * epsilon
    noisy = clean + noise
    t = bundle["t"][indices // N_X]
    x = bundle["x"][indices % N_X]
    return dict(
        positions=positions,
        indices=indices,
        t=t,
        x=x,
        clean=clean,
        epsilon=epsilon,
        noise=noise,
        noisy=noisy,
        count=len(indices),
    )


def coverage(indices: np.ndarray, bundle) -> dict:
    ti, xi = np.divmod(indices, N_X)
    tc = np.bincount(ti, minlength=N_T)
    xc = np.bincount(xi, minlength=N_X)
    return dict(
        temporal_snapshots=int((tc > 0).sum()),
        spatial_coordinates=int((xc > 0).sum()),
        per_time_counts=tc.tolist(),
        per_space_counts=xc.tolist(),
        t_min=float(bundle["t"][ti].min()),
        t_max=float(bundle["t"][ti].max()),
        x_min=float(bundle["x"][xi].min()),
        x_max=float(bundle["x"][xi].max()),
        actual_percentage=100.0 * len(indices) / POPULATION,
    )


# ---------------------------------------------------------------------------
# Condition construction (training inputs)
# ---------------------------------------------------------------------------
def make_condition(bundle, ratio: float, sigma: float, seed: int):
    design = replicate_design(seed)
    arrays = condition_arrays(bundle, design, ratio, sigma)
    full = flatten_grid(bundle)
    train = (
        torch.tensor(arrays["t"].reshape(-1, 1), dtype=torch.float32),
        torch.tensor(arrays["x"].reshape(-1, 1), dtype=torch.float32),
        torch.tensor(arrays["noisy"].reshape(-1, 1), dtype=torch.float32),
    )
    clean_train = (
        train[0],
        train[1],
        torch.tensor(arrays["clean"].reshape(-1, 1), dtype=torch.float32),
    )
    grid = dict(
        t=bundle["t"],
        x=bundle["x"],
        u=bundle["u"],
        indices=arrays["indices"],
        time_indices=np.unique(arrays["indices"] // N_X),
        space_indices=np.unique(arrays["indices"] % N_X),
    )
    model, head = initialize(seed)
    initial = {"surrogate": model.state_dict(), "head": head.state_dict()}
    return dict(
        bundle=bundle,
        design=design,
        arrays=arrays,
        train=train,
        clean_train=clean_train,
        full=full,
        ref=bundle["ref"],
        grid=grid,
        initial=initial,
    )


# ---------------------------------------------------------------------------
# Recovery metrics
# ---------------------------------------------------------------------------
def coefficient_metrics(head) -> dict:
    """Physical-head coefficients and errors against the nu=0.2 targets."""
    a, b = (float(v) for v in head.xi.detach().tolist())
    transport_error = abs(a - TARGETS[0])
    diffusion_error = abs(b - TARGETS[1])
    return dict(
        transport=a,
        diffusion=b,
        transport_abs_error=transport_error,
        diffusion_abs_error=diffusion_error,
        transport_relative_error=transport_error / abs(TARGETS[0]),
        diffusion_relative_error=diffusion_error / abs(TARGETS[1]),
        loose=transport_error < LOOSE[0] and diffusion_error < LOOSE[1],
        strong=transport_error < STRONG[0] and diffusion_error < STRONG[1],
    )


def recovery_metrics(history: dict, window: int = SUSTAINED_WINDOW) -> dict:
    """First-crossing, endpoint and sustained recovery from a dense history."""
    out = {}
    for level in ("loose", "strong"):
        flags = np.asarray(history[level], dtype=bool)
        steps = np.asarray(history["step"], dtype=np.int64)
        hit = np.flatnonzero(flags)
        out[f"first_{level}_step"] = int(steps[hit[0]]) if len(hit) else None
        out[f"ever_{level}"] = bool(len(hit) > 0)
        out[f"endpoint_{level}"] = bool(flags[-1]) if len(flags) else False
        width = min(int(window), len(flags))
        sustained = bool(width > 0 and flags[-width:].all())
        out[f"sustained_{level}"] = sustained
        if sustained:
            i = len(flags) - 1
            while i >= 0 and flags[i]:
                i -= 1
            out[f"sustained_{level}_from_step"] = int(steps[i + 1])
            out[f"sustained_{level}_duration"] = int(steps[-1] - steps[i + 1])
        else:
            out[f"sustained_{level}_from_step"] = None
            out[f"sustained_{level}_duration"] = 0
    return out


def load_history(path: Path) -> dict:
    """Read a dense history CSV into NumPy arrays without pandas."""
    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        columns = list(reader.fieldnames)
        raw = {c: [] for c in columns}
        for row in reader:
            for c in columns:
                raw[c].append(row[c])
    out = {}
    for c in columns:
        if c in STRING_COLUMNS:
            out[c] = np.asarray(raw[c], dtype=object)
        elif c in ("loose", "strong", "loose_relative", "strong_relative"):
            out[c] = np.asarray([v in ("True", "true", "1", "1.0") for v in raw[c]], dtype=bool)
        else:
            out[c] = np.asarray(raw[c], dtype=np.float64)
    return out


def load_summary(folder: Path) -> dict:
    return json.loads((Path(folder) / "summary.json").read_text())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_state(v) for v in value)
    return value


def _write_history_row(writer, f, row):
    if writer is None:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
    writer.writerow(row)
    return writer


def _row(step, ratio, sigma, seed, losses, head):
    return dict(
        step=int(step),
        ratio=float(ratio),
        sigma=float(sigma),
        count=int(count_for(ratio)),
        seed=int(seed),
        run_id=run_id(ratio, sigma, seed),
        **{k: float(v.detach()) for k, v in losses.items()},
        **coefficient_metrics(head),
    )


def _check_finite(row):
    for key, value in row.items():
        if key in STRING_COLUMNS:
            continue
        if not np.isfinite(value):
            raise RuntimeError(f"Non-finite training value at step {row['step']}: {key}={value}")


def _latest_state(folder: Path, steps: int):
    candidates = sorted(Path(folder).glob("state_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def train_condition(
    out: Path,
    ratio: float,
    sigma: float,
    seed: int,
    steps: int = DEFAULT_STEPS,
    *,
    resume: bool = False,
    device: str = "cpu",
    checkpoints: tuple[int, ...] | None = None,
    field_checkpoints: tuple[int, ...] | None = None,
    verbose: bool = True,
):
    """Train one condition into a fresh (or resumable) directory.

    Checkpoint semantics match the established runner: ``state_k.pt`` is saved
    at step ``k`` after the loss for the sampled batch is recorded but before
    the optimizer update. It stores that batch and the post-draw batch RNG, so
    an interrupted run can be resumed exactly by recomputing the step-k loss
    and applying the step-k update.
    """
    torch.set_num_threads(1)
    out = Path(out)
    steps = int(steps)
    ratio = float(ratio)
    sigma = float(sigma)
    seed = int(seed)
    checkpoints = tuple(sorted(set(checkpoints if checkpoints is not None else checkpoints_for(steps))))
    field_checkpoints = tuple(sorted(set(field_checkpoints if field_checkpoints is not None else field_checkpoints_for(steps))))
    status_path = out / "status.json"

    if status_path.exists():
        status = json.loads(status_path.read_text())
        completed_steps = int(status.get("steps", -1))
        if status.get("state") == "complete" and completed_steps >= steps:
            return load_summary(out)
        if not resume:
            raise FileExistsError(
                f"{out} exists with state={status.get('state')} at {completed_steps} steps; "
                f"pass --resume to continue to {steps}"
            )

    bundle = reference_bundle()
    condition = make_condition(bundle, ratio, sigma, seed)
    design = condition["design"]
    arrays = condition["arrays"]
    train = condition["train"]
    full = condition["full"]
    ref = condition["ref"]
    grid = condition["grid"]

    out.mkdir(parents=True, exist_ok=True)
    condition_record = dict(
        run_id=run_id(ratio, sigma, seed),
        ratio=ratio,
        sigma=sigma,
        count=arrays["count"],
        population=POPULATION,
        numerical_grid=[N_T, N_X],
        numerical_points=N_FULL,
        nu=NU,
        T=T_FINAL,
        dt=DT,
        initial_condition="sin(x)",
        seed=seed,
        init_seed=seed,
        batch_seed=BATCH_SEED_OFFSET + seed,
        batch_size=BATCH_SIZE,
        batch_sampling="with replacement",
        learning_rate=LEARNING_RATE,
        steps=steps,
        targets=list(TARGETS),
        loose_absolute=list(LOOSE),
        strong_absolute=list(STRONG),
        loose_relative_sensitivity=list(LOOSE_RELATIVE),
        strong_relative_sensitivity=list(STRONG_RELATIVE),
        sustained_window=SUSTAINED_WINDOW,
        noise_scale=bundle["scale"],
        noise_scale_definition="population std (ddof=0) of the full 501x256 clean solution",
        noise_std=float(np.std(arrays["noise"], ddof=0)),
        clean_target_mse=float(np.mean((arrays["noisy"] - arrays["clean"]) ** 2)),
        streams=design["streams"],
        coverage=coverage(arrays["indices"], bundle),
        population_sha256=array_sha(design["population"]),
        subsample_order_sha256=array_sha(design["subsample_order"]),
        epsilon_sha256=array_sha(design["epsilon"]),
        selected_positions_sha256=array_sha(arrays["positions"]),
        selected_indices_sha256=array_sha(arrays["indices"]),
        initial_surrogate_sha256=hashlib.sha256(
            torch.cat([p.reshape(-1) for p in condition["initial"]["surrogate"].values()]).numpy().tobytes()
        ).hexdigest(),
    )
    write_json(out / "condition.json", condition_record)
    np.savez_compressed(
        out / "observations.npz",
        population=design["population"],
        subsample_order=design["subsample_order"],
        positions=arrays["positions"],
        indices=arrays["indices"],
        t=arrays["t"],
        x=arrays["x"],
        clean=arrays["clean"],
        epsilon=arrays["epsilon"],
        noise=arrays["noise"],
        noisy=arrays["noisy"],
    )
    torch.save(condition["initial"], out / "initial.pt")

    device = torch.device(device)
    if device.type == "cuda":
        # Keep float32 matmuls at full precision for comparability with CPU.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, head = initialize(seed)
    model.to(device)
    head.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=LEARNING_RATE)
    batch_rng = torch.Generator().manual_seed(BATCH_SEED_OFFSET + seed)
    train_dev = tuple(v.to(device) for v in train)

    history_path = out / "history.csv"
    metrics_path = out / "checkpoint_metrics.json"
    metrics_rows = []
    first = {"first_loose_step": None, "first_strong_step": None}
    start_step = 0
    previous_row = None
    start = time.monotonic()

    if resume and history_path.exists():
        history = load_history(history_path)
        state_path = _latest_state(out, steps)
        if state_path is None:
            raise FileNotFoundError(f"No checkpoint state in {out}; cannot resume")
        state = torch.load(state_path, map_location=device, weights_only=False)
        start_step = int(state["step"])
        model.load_state_dict(state["surrogate"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        batch_rng.set_state(state["batch_rng"])
        saved_idx = state["next_batch_indices"].to(device)
        # Keep history rows up to and including start_step.
        keep = history["step"] <= start_step
        with history_path.open("w", newline="") as f:
            writer = None
            for i in np.flatnonzero(keep):
                row = {c: (history[c][i].item() if hasattr(history[c][i], "item") else history[c][i])
                       for c in history}
                row["loose"] = bool(row["loose"])
                row["strong"] = bool(row["strong"])
                writer = _write_history_row(writer, f, row)
        if metrics_path.exists():
            metrics_rows = [r for r in json.loads(metrics_path.read_text()) if int(r["step"]) <= start_step]
        kept = np.flatnonzero(keep)
        previous_row = {c: history[c][kept[-1]] for c in history}
        retained = {c: history[c][keep] for c in history}
        restored = recovery_metrics(retained)
        first["first_loose_step"] = restored["first_loose_step"]
        first["first_strong_step"] = restored["first_strong_step"]
        if start_step < steps:
            # Recompute and apply the update that was pending when state was saved.
            losses = objective(model, head, tuple(v[saved_idx] for v in train_dev))
            row = _row(start_step, ratio, sigma, seed, losses, head)
            _check_finite(row)
            for key in ("data_loss", "pde_loss", "total_loss", "transport", "diffusion"):
                reference = float(previous_row[key])
                if not np.isclose(row[key], reference, rtol=1e-6, atol=1e-8):
                    raise RuntimeError(
                        f"Resume mismatch at step {start_step} for {key}: {row[key]} vs {reference}"
                    )
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()
            start_step += 1
        else:
            start_step = steps + 1
        del state
    elif resume:
        start_step = 0

    if verbose:
        print(
            f"[{run_id(ratio, sigma, seed)}] start={start_step} steps={steps} "
            f"count={arrays['count']} sigma={sigma:g} device={device}",
            flush=True,
        )

    writer = None
    with history_path.open("a" if resume and history_path.exists() else "w", newline="") as f:
        if resume and history_path.stat().st_size > 0:
            writer = csv.DictWriter(f, fieldnames=list(previous_row) if previous_row is not None else None)
        for step in range(start_step, steps + 1):
            idx = torch.randint(0, arrays["count"], (BATCH_SIZE,), generator=batch_rng)
            losses = objective(model, head, tuple(v[idx.to(device)] for v in train_dev))
            row = _row(step, ratio, sigma, seed, losses, head)
            _check_finite(row)
            for level in ("loose", "strong"):
                if row[level] and first[f"first_{level}_step"] is None:
                    first[f"first_{level}_step"] = step
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if step in checkpoints:
                # Evaluation happens on CPU with the canonical helper.
                model.to("cpu")
                head.to("cpu")
                metrics, fields = evaluate(model, head, train, full, ref, grid)
                model.to(device)
                head.to(device)
                metrics_rows = [r for r in metrics_rows if int(r["step"]) != step]
                metrics_rows.append(dict(step=step, **metrics))
                metrics_rows.sort(key=lambda r: int(r["step"]))
                write_json(metrics_path, metrics_rows)
                if step in field_checkpoints:
                    np.savez_compressed(out / f"fields_{step:06d}.npz", **fields)
                torch.save(
                    dict(
                        step=step,
                        surrogate=_cpu_state(model.state_dict()),
                        head=_cpu_state(head.state_dict()),
                        optimizer=_cpu_state(optimizer.state_dict()),
                        batch_rng=batch_rng.get_state(),
                        next_batch_indices=idx.cpu(),
                    ),
                    out / f"state_{step:06d}.pt",
                )
            if step % 1000 == 0 or step == steps:
                f.flush()
                write_json(
                    out / "progress.json",
                    dict(
                        **row,
                        **first,
                        elapsed_seconds=time.monotonic() - start,
                        pid=None,
                    ),
                )
                if verbose:
                    print(
                        f"[{run_id(ratio, sigma, seed)}] step={step} "
                        f"xi=({row['transport']:.6g},{row['diffusion']:.6g}) "
                        f"loss={row['total_loss']:.5g} seconds={time.monotonic()-start:.1f}",
                        flush=True,
                    )
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()

    # Finalize from the dense history so sustained/endpoint metrics are exact.
    history = load_history(history_path)
    last = {c: history[c][-1] for c in history}
    recovery = recovery_metrics(history)
    summary = dict(
        **{k: (v.item() if hasattr(v, "item") else v) for k, v in last.items()},
        **recovery,
        **{k: metrics_rows[-1][k] for k in metrics_rows[-1] if k != "step"},
        steps=steps,
        elapsed_seconds=time.monotonic() - start,
        status=(
            "strong" if last["strong"] else "loose" if last["loose"] else "not_recovered"
        ),
        sustained_status=(
            "strong" if recovery["sustained_strong"] else "loose" if recovery["sustained_loose"] else "not_sustained"
        ),
    )
    write_json(out / "summary.json", summary)
    write_json(out / "recovery.json", recovery)
    write_json(
        status_path,
        dict(
            state="complete",
            run_id=run_id(ratio, sigma, seed),
            steps=steps,
            completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            elapsed_seconds=summary["elapsed_seconds"],
        ),
    )
    return summary


# ---------------------------------------------------------------------------
# Validation of one completed run
# ---------------------------------------------------------------------------
def validate_run(
    folder: Path,
    ratio: float,
    sigma: float,
    seed: int,
    steps: int = DEFAULT_STEPS,
    checkpoints: tuple[int, ...] | None = None,
):
    folder = Path(folder)
    history = load_history(folder / "history.csv")
    np.testing.assert_array_equal(history["step"], np.arange(steps + 1))
    assert np.isfinite(
        np.column_stack([history[c] for c in history if c not in STRING_COLUMNS])
    ).all()
    assert (history["run_id"] == run_id(ratio, sigma, seed)).all()
    assert (history["ratio"] == ratio).all()
    assert (history["sigma"] == sigma).all()
    assert (history["count"] == count_for(ratio)).all()
    assert (history["seed"] == seed).all()
    for name, target in zip(("transport", "diffusion"), TARGETS):
        np.testing.assert_allclose(history[name + "_abs_error"], np.abs(history[name] - target), rtol=1e-14, atol=1e-16)
        np.testing.assert_allclose(
            history[name + "_relative_error"], np.abs(history[name] - target) / abs(target), rtol=1e-14, atol=1e-16
        )
    np.testing.assert_allclose(history["total_loss"], history["data_loss"] + history["pde_loss"], rtol=2e-7, atol=1e-12)
    assert (history["data_loss"] >= 0).all() and (history["pde_loss"] >= 0).all()
    np.testing.assert_array_equal(
        history["loose"],
        (history["transport_abs_error"] < LOOSE[0]) & (history["diffusion_abs_error"] < LOOSE[1]),
    )
    np.testing.assert_array_equal(
        history["strong"],
        (history["transport_abs_error"] < STRONG[0]) & (history["diffusion_abs_error"] < STRONG[1]),
    )
    summary = load_summary(folder)
    for key in history:
        if key in STRING_COLUMNS:
            continue
        assert np.isclose(summary[key], history[key][-1], rtol=1e-12, atol=1e-14), key
    expected = recovery_metrics(history)
    for key, value in expected.items():
        assert summary[key] == value, (key, summary[key], value)
    condition = json.loads((folder / "condition.json").read_text())
    for key, value in dict(
        ratio=ratio,
        sigma=sigma,
        count=count_for(ratio),
        seed=seed,
        population=POPULATION,
        numerical_points=N_FULL,
        nu=NU,
        T=T_FINAL,
        dt=DT,
        batch_size=BATCH_SIZE,
        steps=steps,
    ).items():
        assert condition[key] == value, key
    metrics = json.loads((folder / "checkpoint_metrics.json").read_text())
    expected_checkpoints = checkpoints_for(steps) if checkpoints is None else tuple(sorted(set(checkpoints)))
    np.testing.assert_array_equal(
        [r["step"] for r in metrics], np.array(expected_checkpoints)
    )
    assert np.isfinite(np.array([[v for k, v in r.items() if k != "step"] for r in metrics], dtype=float)).all()
    with np.load(folder / "observations.npz") as obs:
        design = replicate_design(seed)
        positions = selected_positions(design, ratio)
        np.testing.assert_array_equal(obs["population"], design["population"])
        np.testing.assert_array_equal(obs["subsample_order"], design["subsample_order"])
        np.testing.assert_array_equal(obs["positions"], positions)
        np.testing.assert_array_equal(obs["epsilon"], design["epsilon"][positions])
        np.testing.assert_allclose(obs["noise"], sigma * condition["noise_scale"] * obs["epsilon"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(obs["noisy"], obs["clean"] + obs["noise"], rtol=1e-12, atol=1e-14)
        assert len(obs["indices"]) == count_for(ratio)
        assert len(np.unique(obs["indices"])) == count_for(ratio)
    assert json.loads((folder / "status.json").read_text())["state"] == "complete"
    return history, metrics


# ---------------------------------------------------------------------------
# Numerical validation and identifiability
# ---------------------------------------------------------------------------
def numerical_validation(destination: Path) -> dict:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    x, t, u = reference_solution()
    bundle = reference_bundle()
    dx = bundle["dx"]
    checks = dict(
        shape=list(u.shape),
        initial_condition_max_abs_error=float(np.max(np.abs(u[0] - np.sin(x)))),
        finite=bool(np.isfinite(u).all()),
        t_start=float(t[0]),
        t_end=float(t[-1]),
        n_steps=len(t) - 1,
        dx=dx,
        dt=float(t[1] - t[0]),
        domain=[float(x[0]), float(x[-1] + dx)],
        periodic_grid=bool(np.isclose(x[0], 0.0) and np.isclose(x[-1] + dx, 2 * np.pi)),
        noise_scale=bundle["scale"],
    )
    # Periodicity / derivative operator cross-check.
    ux_fd = bundle["ref"]["u_x"]["physical"]
    wave = 2 * np.pi * np.fft.fftfreq(N_X, d=dx)
    ux_sp = np.fft.ifft(np.fft.fft(u, axis=1) * (1j * wave), axis=1).real
    checks["periodic_derivative_fd_vs_spectral_rel_l2"] = float(
        np.linalg.norm(ux_fd - ux_sp) / np.linalg.norm(ux_sp)
    )
    checks["boundary_adjacent_value_difference"] = float(abs(u[0, 0] - u[0, -1]))
    checks["periodicity_note"] = (
        "The grid uses endpoint=False and every spatial operator wraps with np.roll. "
        "The FD-vs-spectral derivative comparison is the quantitative periodicity check."
    )
    # Temporal refinement: halve dt, compare rows at matching physical times.
    _, _, u_half = reference_solution(dt=DT / 2, history_every=1)
    u_half = u_half[2::2]
    checks["temporal_refinement_rel_l2"] = float(
        np.linalg.norm(u_half - u[1:]) / np.linalg.norm(u[1:])
    )
    checks["temporal_refinement_max_abs"] = float(np.max(np.abs(u_half - u[1:])))
    # Spatial refinement: fine grid subsampled to the coarse grid.
    _, _, uf = reference_solution(dt=DT / 4, n_x=2 * N_X, history_every=1)
    fine = uf[4::4, ::2]
    checks["spatial_refinement_rel_l2"] = float(
        np.linalg.norm(fine - u[1:]) / np.linalg.norm(u[1:])
    )
    checks["spatial_refinement_max_abs"] = float(np.max(np.abs(fine - u[1:])))
    # Magnitudes.
    ux = bundle["ref"]["u_x"]["physical"]
    uxx = bundle["ref"]["u_xx"]["physical"]
    uu_x = u * ux
    checks.update(
        max_abs_u=float(np.max(np.abs(u))),
        max_abs_u_initial=float(np.max(np.abs(u[0]))),
        max_abs_u_final=float(np.max(np.abs(u[-1]))),
        amplitude_decay=float(np.max(np.abs(u[-1])) / np.max(np.abs(u[0]))),
        l2_decay=float(np.linalg.norm(u[-1]) / np.linalg.norm(u[0])),
        max_abs_u_x=float(np.max(np.abs(ux))),
        max_abs_u_xx=float(np.max(np.abs(uxx))),
        max_abs_u_u_x=float(np.max(np.abs(uu_x))),
        max_abs_u_t=float(np.max(np.abs(bundle["ref"]["u_t"]["physical"]))),
        rms_u_x=float(np.sqrt(np.mean(ux**2))),
        rms_u_xx=float(np.sqrt(np.mean(uxx**2))),
        rms_u_u_x=float(np.sqrt(np.mean(uu_x**2))),
    )
    # Identifiability over the actual sampled domains.
    identifiability = []
    design = replicate_design(0)
    for ratio in RATIOS:
        arrays = condition_arrays(bundle, design, ratio, 0.0)
        idx = arrays["indices"]
        A = np.column_stack([uu_x.reshape(-1)[idx], uxx.reshape(-1)[idx]])
        y = bundle["ref"]["u_t"]["physical"].reshape(-1)[idx]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        residual = y - A @ coef
        singular = np.linalg.svd(A, compute_uv=False)
        scale = np.linalg.norm(A, axis=0)
        normalized = A / scale
        singular_n = np.linalg.svd(normalized, compute_uv=False)
        identifiability.append(
            dict(
                ratio=ratio,
                count=arrays["count"],
                least_squares_transport=float(coef[0]),
                least_squares_diffusion=float(coef[1]),
                relative_residual=float(np.linalg.norm(residual) / np.linalg.norm(y)),
                condition_number=float(singular[0] / singular[-1]),
                normalized_condition_number=float(singular_n[0] / singular_n[-1]),
                singular_values=singular.tolist(),
                column_norms=scale.tolist(),
            )
        )
    # Noise sensitivity of the same clean-derivative least-squares diagnostic.
    noise_sensitivity = []
    for sigma in SIGMAS:
        arrays = condition_arrays(bundle, design, 1.0, sigma)
        idx = arrays["indices"]
        A = np.column_stack([uu_x.reshape(-1)[idx], uxx.reshape(-1)[idx]])
        y = bundle["ref"]["u_t"]["physical"].reshape(-1)[idx] + sigma * bundle["scale"] * arrays["epsilon"]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        noise_sensitivity.append(
            dict(
                sigma=sigma,
                least_squares_transport=float(coef[0]),
                least_squares_diffusion=float(coef[1]),
                transport_abs_error=float(abs(coef[0] - TARGETS[0])),
                diffusion_abs_error=float(abs(coef[1] - TARGETS[1])),
            )
        )
    checks["identifiability"] = identifiability
    checks["noise_sensitivity_clean_derivative_ls"] = noise_sensitivity
    checks["identifiability_note"] = (
        "Least squares on clean numerical derivatives is an upper-bound diagnostic, "
        "not a neural recovery result. It tests whether the two columns are distinguishable "
        "over each sampled domain."
    )
    write_json(destination / "numerical_validation.json", checks)
    _plot_numerical(destination, bundle)
    return checks


def _plot_numerical(destination: Path, bundle) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, t, u = bundle["x"], bundle["t"], bundle["u"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(
        u,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="viridis",
    )
    axes[0].set(xlabel="x", ylabel="t", title="Clean reference u(x,t)")
    fig.colorbar(im, ax=axes[0], label="u")
    for time in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = int(np.argmin(np.abs(t - time)))
        axes[1].plot(x, u[i], label=f"t={t[i]:.2f}")
    axes[1].set(xlabel="x", ylabel="u(x,t)", title="Representative time slices")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("Sinusoidal Burgers reference solution, nu=0.2, 256x501")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"reference_solution.{ext}", dpi=160)
    plt.close(fig)

    ux = bundle["ref"]["u_x"]["physical"]
    uxx = bundle["ref"]["u_xx"]["physical"]
    uu_x = u * ux
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(t, np.max(np.abs(ux), axis=1), label="max |u_x|")
    axes[0].plot(t, np.max(np.abs(uxx), axis=1), label="max |u_xx|")
    axes[0].plot(t, np.max(np.abs(uu_x), axis=1), label="max |u u_x|")
    axes[0].set(xlabel="t", ylabel="magnitude", yscale="log", title="Derivative and nonlinear-term magnitudes")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(t, np.max(np.abs(u), axis=1), label="max |u|")
    axes[1].plot(t, np.linalg.norm(u, axis=1) / np.sqrt(N_X), label="RMS u")
    axes[1].set(xlabel="t", ylabel="amplitude", title="Solution amplitude decay")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("Smooth Burgers regularity diagnostics")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"regularity.{ext}", dpi=160)
    plt.close(fig)

    validation = json.loads((destination / "numerical_validation.json").read_text())
    ident = validation["identifiability"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ratios = [r["ratio"] * 100 for r in ident]
    axes[0].plot(ratios, [r["condition_number"] for r in ident], "o-", label="raw")
    axes[0].plot(ratios, [r["normalized_condition_number"] for r in ident], "s--", label="column-normalized")
    axes[0].set(xlabel="Sampling fraction (%)", ylabel="Condition number", yscale="log", title="Feature-matrix conditioning")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    noise = validation["noise_sensitivity_clean_derivative_ls"]
    axes[1].plot([n["sigma"] for n in noise], [n["transport_abs_error"] for n in noise], "o-", label="transport")
    axes[1].plot([n["sigma"] for n in noise], [n["diffusion_abs_error"] for n in noise], "s-", label="diffusion")
    axes[1].set(xlabel="Noise sigma", ylabel="Absolute coefficient error", yscale="log", title="Clean-derivative LS sensitivity")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("Identifiability of (-1, 0.2) on the sampled domains")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"identifiability.{ext}", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _condition_rows(root: Path, steps: int = DEFAULT_STEPS):
    rows = []
    missing = []
    for sigma in SIGMAS:
        for ratio in RATIOS:
            for seed in SEEDS:
                folder = run_directory(root, ratio, sigma, seed)
                summary_path = folder / "summary.json"
                status_path = folder / "status.json"
                if summary_path.exists() and status_path.exists():
                    status = json.loads(status_path.read_text())
                    if status.get("state") == "complete":
                        rows.append(load_summary(folder))
                        continue
                missing.append(dict(ratio=ratio, sigma=sigma, seed=seed, path=str(folder)))
    return rows, missing


def aggregate(
    root: Path,
    *,
    require_all: bool = False,
    steps: int = DEFAULT_STEPS,
    plot: bool = True,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows, missing = _condition_rows(root, steps)
    by_condition = {}
    for row in rows:
        by_condition.setdefault((float(row["ratio"]), float(row["sigma"])), []).append(row)

    per_seed = []
    summaries = []
    for sigma in SIGMAS:
        for ratio in RATIOS:
            subset = sorted(by_condition.get((ratio, sigma), []), key=lambda r: r["seed"])
            n = len(subset)
            record = dict(
                ratio=ratio,
                sigma=sigma,
                count=count_for(ratio),
                n_complete=n,
                n_expected=len(SEEDS),
                complete=n == len(SEEDS),
            )
            for level in ("loose", "strong"):
                record[f"ever_{level}_count"] = sum(bool(r[f"ever_{level}"]) for r in subset)
                record[f"endpoint_{level}_count"] = sum(bool(r[f"endpoint_{level}"]) for r in subset)
                record[f"sustained_{level}_count"] = sum(bool(r[f"sustained_{level}"]) for r in subset)
                record[f"ever_{level}_fraction"] = record[f"ever_{level}_count"] / n if n else None
                record[f"endpoint_{level}_fraction"] = record[f"endpoint_{level}_count"] / n if n else None
                record[f"sustained_{level}_fraction"] = record[f"sustained_{level}_count"] / n if n else None
                times = [r[f"first_{level}_step"] for r in subset if r[f"first_{level}_step"] is not None]
                record[f"first_{level}_step_median"] = float(np.median(times)) if times else None
                record[f"first_{level}_step_min"] = float(np.min(times)) if times else None
                record[f"first_{level}_step_max"] = float(np.max(times)) if times else None
            for column in (
                "transport",
                "diffusion",
                "transport_abs_error",
                "diffusion_abs_error",
                "transport_relative_error",
                "diffusion_relative_error",
            ):
                values = np.asarray([r[column] for r in subset], dtype=float)
                record[f"endpoint_mean_{column}"] = float(values.mean()) if n else None
                record[f"endpoint_std_{column}"] = float(values.std(ddof=0)) if n else None
            record["elapsed_seconds_mean"] = float(np.mean([r["elapsed_seconds"] for r in subset])) if n else None
            summaries.append(record)
            for r in subset:
                per_seed.append(
                    dict(
                        ratio=ratio,
                        sigma=sigma,
                        seed=r["seed"],
                        count=count_for(ratio),
                        transport=r["transport"],
                        diffusion=r["diffusion"],
                        transport_abs_error=r["transport_abs_error"],
                        diffusion_abs_error=r["diffusion_abs_error"],
                        transport_relative_error=r["transport_relative_error"],
                        diffusion_relative_error=r["diffusion_relative_error"],
                        loose=r["loose"],
                        strong=r["strong"],
                        ever_loose=r["ever_loose"],
                        ever_strong=r["ever_strong"],
                        endpoint_loose=r["endpoint_loose"],
                        endpoint_strong=r["endpoint_strong"],
                        sustained_loose=r["sustained_loose"],
                        sustained_strong=r["sustained_strong"],
                        first_loose_step=r["first_loose_step"],
                        first_strong_step=r["first_strong_step"],
                        sustained_loose_from_step=r["sustained_loose_from_step"],
                        sustained_strong_from_step=r["sustained_strong_from_step"],
                        elapsed_seconds=r["elapsed_seconds"],
                    )
                )
    _write_csv(root / "per_seed_recovery.csv", per_seed)
    _write_csv(root / "recovery_summary.csv", summaries)
    write_json(root / "recovery_summary.json", summaries)
    completion = dict(
        expected_runs=len(SIGMAS) * len(RATIOS) * len(SEEDS),
        completed_runs=len(rows),
        missing_runs=len(missing),
        complete_conditions=sum(1 for s in summaries if s["complete"]),
        expected_conditions=len(SIGMAS) * len(RATIOS),
        missing=missing,
        complete=not missing,
    )
    write_json(root / "completion.json", completion)
    if plot:
        _plot_aggregate(root, summaries, by_condition, per_seed, steps)
    if require_all and missing:
        raise ValueError(f"Incomplete study: {len(missing)} runs missing")
    return completion


def _write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _heatmap(ax, matrix, title, *, vmin=0.0, vmax=1.0, cmap="viridis", fmt="{:.0f}"):
    im = ax.imshow(matrix, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(RATIOS)), [f"{int(r*100)}%" for r in RATIOS])
    ax.set_yticks(range(len(SIGMAS)), [f"{s:.2f}" for s in SIGMAS])
    ax.set(xlabel="Sampling fraction", ylabel="Noise sigma", title=title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, fmt.format(value), ha="center", va="center", color="white" if value < 0.6 * vmax else "black", fontsize=8)
    return im


def _plot_aggregate(root, summaries, by_condition, per_seed, steps):
    import matplotlib.pyplot as plt

    index = {(s["sigma"], s["ratio"]): s for s in summaries}
    # Recovery heatmaps.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, level, label in zip(
        axes,
        ("endpoint_loose", "endpoint_strong", "sustained_strong"),
        ("Endpoint loose recovery", "Endpoint strong recovery", "Sustained strong recovery"),
    ):
        matrix = np.array(
            [[index[(s, r)][f"{level}_fraction"] if index[(s, r)][f"{level}_fraction"] is not None else np.nan
              for r in RATIOS] for s in SIGMAS],
            dtype=float,
        )
        im = _heatmap(ax, matrix, label, vmin=0, vmax=1)
        for i, s in enumerate(SIGMAS):
            for j, r in enumerate(RATIOS):
                n = index[(s, r)]["n_complete"]
                ax.text(j, i + 0.32, f"n={n}", ha="center", va="center", fontsize=6, color="white" if matrix[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=axes, label="Recovery frequency (fraction of completed seeds)")
    fig.suptitle("Burgers noise x sampling recovery — sinusoidal IC, nu=0.2")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"noise_sampling_recovery_heatmap.{ext}", dpi=160)
    plt.close(fig)

    # Final coefficient errors.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, title in zip(
        axes,
        ("endpoint_mean_transport_abs_error", "endpoint_mean_diffusion_abs_error"),
        ("Mean |transport + 1|", "Mean |diffusion - 0.2|"),
    ):
        matrix = np.array(
            [[index[(s, r)][column] if index[(s, r)][column] is not None else np.nan for r in RATIOS] for s in SIGMAS],
            dtype=float,
        )
        im = _heatmap(ax, matrix, title, vmin=0, vmax=float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1, cmap="magma_r", fmt="{:.3f}")
        fig.colorbar(im, ax=ax, label="Absolute error")
    fig.suptitle("Final coefficient error by noise and sampling condition")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"final_coefficient_error_heatmap.{ext}", dpi=160)
    plt.close(fig)

    # Mean coefficient trajectories with seed variability (6x5 panels).
    for column, target in (("transport", -1.0), ("diffusion", 0.2)):
        fig, axes = plt.subplots(len(SIGMAS), len(RATIOS), figsize=(19, 16), sharex=True)
        for i, sigma in enumerate(SIGMAS):
            for j, ratio in enumerate(RATIOS):
                ax = axes[i, j]
                subset = sorted(by_condition.get((ratio, sigma), []), key=lambda r: r["seed"])
                if subset:
                    histories = []
                    for r in subset:
                        folder = run_directory(root, ratio, sigma, r["seed"])
                        histories.append(load_history(folder / "history.csv"))
                    common = np.arange(0, steps + 1, 100)
                    values = np.stack([h[column][common] for h in histories])
                    mean = values.mean(axis=0)
                    sd = values.std(axis=0, ddof=0)
                    ax.plot(common, mean, color="C0", lw=1)
                    ax.fill_between(common, mean - sd, mean + sd, color="C0", alpha=0.25, lw=0)
                    ax.axhline(target, color="black", ls="--", lw=0.8)
                    ax.set_title(f"sigma={sigma:g}, {int(ratio*100)}% (n={len(subset)})", fontsize=8)
                else:
                    ax.set_title(f"sigma={sigma:g}, {int(ratio*100)}% (n=0)", fontsize=8)
                ax.grid(alpha=0.2)
                if j == 0:
                    ax.set_ylabel("coefficient")
                if i == len(SIGMAS) - 1:
                    ax.set_xlabel("optimizer steps")
        fig.suptitle(f"{column.capitalize()} coefficient trajectories — mean ± population SD across completed seeds")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(root / f"coefficient_trajectories_{column}.{ext}", dpi=140)
        plt.close(fig)

    # Representative loss curves.
    selected = [(0.0, 1.0), (0.0, 0.1), (0.05, 1.0), (0.05, 0.1), (0.20, 1.0), (0.20, 0.1)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (sigma, ratio) in zip(axes.flat, selected):
        subset = sorted(by_condition.get((ratio, sigma), []), key=lambda r: r["seed"])
        for r in subset:
            h = load_history(run_directory(root, ratio, sigma, r["seed"]) / "history.csv")
            common = np.arange(0, steps + 1, 100)
            ax.plot(common, h["data_loss"][common], lw=0.7, alpha=0.5, label=f"seed {r['seed']} data")
            ax.plot(common, h["pde_loss"][common], lw=0.7, alpha=0.5, ls="--", label=f"seed {r['seed']} PDE")
        ax.set(xlabel="optimizer steps", ylabel="loss", yscale="log", title=f"sigma={sigma:g}, {int(ratio*100)}% (n={len(subset)})")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=5, ncol=2)
    fig.suptitle("Representative data and PDE loss trajectories")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"representative_loss_curves.{ext}", dpi=140)
    plt.close(fig)

    # Final coefficient error summary with seed spread.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, column in zip(axes, ("transport_abs_error", "diffusion_abs_error")):
        for sigma in SIGMAS:
            xs = [r * 100 for r in RATIOS]
            means = [index[(sigma, r)][f"endpoint_mean_{column}"] for r in RATIOS]
            stds = [index[(sigma, r)][f"endpoint_std_{column}"] for r in RATIOS]
            means = [np.nan if v is None else v for v in means]
            stds = [np.nan if v is None else v for v in stds]
            ax.errorbar(xs, means, yerr=stds, fmt="o-", capsize=3, label=f"sigma={sigma:g}")
        ax.set(xlabel="Sampling fraction (%)", ylabel="Absolute coefficient error", yscale="log", title=column.replace("_", " "))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Final coefficient error summaries — mean ± population SD across completed seeds")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"final_coefficient_errors.{ext}", dpi=160)
    plt.close(fig)


def clean_noisy_visualization(root: Path, sigma: float = 0.10, seed: int = 0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    bundle = reference_bundle()
    design = replicate_design(seed)
    arrays = condition_arrays(bundle, design, 1.0, sigma)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(bundle["u"], aspect="auto", origin="lower", extent=[0, 2 * np.pi, 0, 1], cmap="viridis")
    axes[0].set(xlabel="x", ylabel="t", title="Clean numerical solution")
    sc = axes[1].scatter(arrays["x"], arrays["t"], c=arrays["noisy"] - arrays["clean"], s=2, cmap="coolwarm")
    axes[1].set(xlabel="x", ylabel="t", title=f"Observation noise residual, sigma={sigma:g}")
    fig.colorbar(sc, ax=axes[1], label="noisy - clean")
    fig.suptitle("Clean solution and paired observation-noise realization")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"clean_vs_noisy.{ext}", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest / protocol helpers
# ---------------------------------------------------------------------------
SOURCE_FILES = (
    "utils/burgers_smooth_noise_sampling.py",
    "utils/physics_informed_burgers.py",
    "utils/derivative_utils.py",
    "prog/featlib.py",
    "prog/mlps.py",
    "Datasets/data/processed/burg_gen/burg_gen.py",
    "runs/run_burgers_smooth_noise_sampling.py",
    "BURGERS_NOISE_SAMPLING_SPEC.md",
)


def source_hashes() -> dict:
    return {name: sha(ROOT / name) for name in SOURCE_FILES if (ROOT / name).exists()}


def run_manifest(steps: int = DEFAULT_STEPS) -> list[dict]:
    records = []
    for sigma in SIGMAS:
        for ratio in RATIOS:
            for seed in SEEDS:
                records.append(
                    dict(
                        run_id=run_id(ratio, sigma, seed),
                        ratio=ratio,
                        sigma=sigma,
                        seed=seed,
                        count=count_for(ratio),
                        steps=steps,
                        directory=str(Path("raw") / f"ratio_{ratio:g}" / f"sigma_{sigma:g}" / f"seed_{seed}"),
                    )
                )
    return records
