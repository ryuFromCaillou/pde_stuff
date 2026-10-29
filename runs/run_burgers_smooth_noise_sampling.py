"""Smooth sinusoidal Burgers noise x sampling experiment (nu=0.2, T=1).

Modes
-----
validate   numerical reference checks, identifiability and protocol checks
prepare    create an immutable experiment directory with the 180-run manifest
smoke      two short engineering runs (not scientific evidence)
run-one    train one (ratio, sigma, seed) condition
shard      run a deterministic subset of the 180-run manifest
aggregate  incremental per-condition recovery tables and figures
report     aggregate plus the presentation report and clean/noisy figure
status     print completion counts

See BURGERS_NOISE_SAMPLING_SPEC.md and runs/BURGERS_SMOOTH_NOISE_SAMPLING.md.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.burgers_smooth_noise_sampling import (
    BATCH_SEED_OFFSET,
    COUNTS,
    DEFAULT_STEPS,
    DT,
    N_FULL,
    N_T,
    N_X,
    NU,
    POPULATION,
    RATIOS,
    SEEDS,
    SIGMAS,
    SOURCE_FILES,
    T_FINAL,
    aggregate,
    clean_noisy_visualization,
    condition_arrays,
    count_for,
    load_history,
    make_condition,
    numerical_validation,
    reference_bundle,
    replicate_design,
    run_directory,
    run_id,
    run_manifest,
    selected_positions,
    sha,
    source_hashes,
    train_condition,
    validate_run,
    write_json,
)
from utils.physics_informed_burgers import initialize


DEFAULT_OUTPUT = ROOT / "run_results/physics_informed_burgers/smooth_noise_sampling_six_seed"


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def protocol_checks() -> dict:
    """Deterministic checks of counts, nesting, pairing and seed separation."""
    bundle = reference_bundle()
    checks = dict(
        counts=list(COUNTS),
        expected_counts=[13056, 9792, 6528, 3264, 1305],
        population=POPULATION,
        numerical_points=N_FULL,
        sigmas=list(SIGMAS),
        ratios=list(RATIOS),
        seeds=list(SEEDS),
        noise_scale=bundle["scale"],
        run_count=len(run_manifest()),
        unique_run_ids=len({r["run_id"] for r in run_manifest()}),
        unique_directories=len({r["directory"] for r in run_manifest()}),
    )
    assert list(COUNTS) == checks["expected_counts"]
    assert checks["run_count"] == 180
    assert checks["unique_run_ids"] == 180
    assert checks["unique_directories"] == 180
    populations = []
    for seed in SEEDS:
        design = replicate_design(seed)
        assert len(design["population"]) == POPULATION
        assert len(np.unique(design["population"])) == POPULATION
        assert design["population"].min() >= 0 and design["population"].max() < N_FULL
        assert len(np.unique(design["subsample_order"])) == POPULATION
        assert len(design["epsilon"]) == POPULATION
        populations.append(design["population"])
        previous = set(range(POPULATION))
        for ratio in RATIOS:
            positions = selected_positions(design, ratio)
            assert len(positions) == count_for(ratio)
            assert len(np.unique(positions)) == count_for(ratio)
            assert set(positions.tolist()) <= previous
            previous = set(positions.tolist())
            clean = condition_arrays(bundle, design, ratio, 0.0)
            np.testing.assert_array_equal(clean["noisy"], clean["clean"])
            noisy = condition_arrays(bundle, design, ratio, 0.10)
            np.testing.assert_array_equal(noisy["t"], clean["t"])
            np.testing.assert_array_equal(noisy["x"], clean["x"])
            np.testing.assert_allclose(noisy["noise"], 0.10 * bundle["scale"] * noisy["epsilon"], rtol=1e-14, atol=1e-15)
        # Same epsilon is reused across sampling ratios.
        eps_full = condition_arrays(bundle, design, 1.0, 0.10)["epsilon"]
        eps_small = condition_arrays(bundle, design, 0.10, 0.10)["epsilon"]
        np.testing.assert_array_equal(eps_small, eps_full[: len(eps_small)])
        # Noise scales linearly and shares one realization across sigma.
        a = condition_arrays(bundle, design, 1.0, 0.01)
        b = condition_arrays(bundle, design, 1.0, 0.20)
        np.testing.assert_allclose(b["noise"], 20.0 * a["noise"], rtol=1e-13, atol=1e-15)
        # Initialization is reproducible and independent of noise/sampling.
        m1, h1 = initialize(seed)
        m2, h2 = initialize(seed)
        for p, q in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(h1.xi, h2.xi, rtol=0, atol=0)
        generator = torch.Generator().manual_seed(BATCH_SEED_OFFSET + seed)
        idx = torch.randint(0, count_for(1.0), (640,), generator=generator)
        assert idx.shape == (640,) and int(idx.min()) >= 0 and int(idx.max()) < count_for(1.0)
    assert len({p.tobytes() for p in populations}) == len(SEEDS), "Replicate populations are not distinct"
    checks.update(
        passed=True,
        nested=True,
        noise_only_on_targets=True,
        paired_noise_across_sigma=True,
        paired_epsilon_across_ratios=True,
        distinct_replicate_populations=True,
        initialization_reproducible=True,
        fixed_batch_size=640,
    )
    return checks


def prepare(output: Path, steps: int) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        experiment="Smooth sinusoidal Burgers noise x sampling",
        specification="BURGERS_NOISE_SAMPLING_SPEC.md",
        pde="u_t = -u*u_x + nu*u_xx",
        nu=NU,
        T=T_FINAL,
        dt=DT,
        numerical_grid=[N_T, N_X],
        numerical_points=N_FULL,
        initial_condition="sin(x)",
        population=POPULATION,
        ratios=list(RATIOS),
        counts=list(COUNTS),
        sigmas=list(SIGMAS),
        seeds=list(SEEDS),
        runs=len(run_manifest(steps)),
        steps=steps,
        training=dict(
            architecture="SirenMLP hidden_size=64 hidden_layers=3 first_omega=1 hidden_omega=1",
            optimizer="Adam",
            learning_rate=5e-4,
            batch_size=640,
            batch_sampling="with replacement",
            data_weight=1,
            pde_weight=1,
            device="cpu",
            dtype="float32",
            threads=1,
            head="two unconstrained physical coefficients, zero initialized",
        ),
        recovery=dict(
            targets=[-1.0, 0.2],
            loose_absolute=[0.25, 0.02],
            strong_absolute=[0.10, 0.01],
            loose_relative_sensitivity=[0.25, 0.20],
            strong_relative_sensitivity=[0.10, 0.10],
            sustained_window=10000,
            note=(
                "Primary criteria reuse the established absolute limits. Because nu increased "
                "tenfold, the diffusion limits are 10%/5% relative, stricter than the historical "
                "100%/50%. The viscosity-scaled relative criteria are a reported sensitivity only."
            ),
        ),
        seed_streams=dict(
            population="numpy SeedSequence([seed,101])",
            subsample="numpy SeedSequence([seed,102])",
            noise="numpy SeedSequence([seed,103])",
            initialization="torch.manual_seed(seed)",
            minibatch=f"torch.Generator().manual_seed({BATCH_SEED_OFFSET}+seed)",
        ),
        source_hashes=source_hashes(),
        git_head=git("rev-parse", "HEAD"),
        git_branch=git("branch", "--show-current"),
        git_remote=git("remote", "get-url", "origin"),
        environment=dict(
            python=sys.version,
            torch=torch.__version__,
            numpy=np.__version__,
            platform=platform.platform(),
            hostname=platform.node(),
            device="cpu",
            dtype="float32",
            threads=1,
        ),
        historical_reference=(
            "Separate six-seed noise (fixed 640 observations) and full-grid sampling "
            "(13,056 of a 51x256 grid, zero noise) studies. Neither is a crossed matrix."
        ),
        deviations=[
            "10% count follows the order's floor rule (1305); the historical full-grid study rounded to 1306.",
            "The reference population is drawn from the 128,256-point numerical grid, not the full grid itself.",
            "Sustained recovery is defined before execution as a criterion holding for the final 10,000 steps.",
        ],
    )
    write_json(output / "config.json", config)
    write_json(output / "runs.json", run_manifest(steps))
    (output / "SOURCE_COMMIT.txt").write_text(
        f"commit={config['git_head']}\nbranch={config['git_branch']}\nremote={config['git_remote']}\n"
    )
    code = output / "code"
    for name in SOURCE_FILES:
        source = ROOT / name
        if not source.exists():
            continue
        destination = code / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    # Hash pre-existing artifacts without modifying them.
    protected = {}
    results = ROOT / "run_results/physics_informed_burgers"
    if results.exists():
        for path in results.rglob("*"):
            if path.is_file() and output.resolve() not in path.resolve().parents:
                protected[str(path.relative_to(ROOT))] = sha(path)
    write_json(output / "protected_artifacts.json", protected)
    write_json(output / "protocol.json", protocol_checks())
    (output / "reproduction.txt").write_text(
        "\n".join(
            [
                f"{sys.executable} runs/run_burgers_smooth_noise_sampling.py --mode validate --output {output}",
                f"{sys.executable} runs/run_burgers_smooth_noise_sampling.py --mode prepare --output {output}",
                f"{sys.executable} runs/run_burgers_smooth_noise_sampling.py --mode smoke --output {output}",
                f"{sys.executable} runs/run_burgers_smooth_noise_sampling.py --mode run-one --output {output} --ratio 1.0 --sigma 0 --seed 0",
                f"{sys.executable} runs/run_burgers_smooth_noise_sampling.py --mode shard --output {output} --shard-index 0 --shards 12",
                f"{sys.executable} runs/run_burgers_smooth_noise_sampling.py --mode aggregate --output {output}",
                f"{sys.executable} runs/run_burgers_smooth_noise_sampling.py --mode report --output {output}",
            ]
        )
        + "\n"
    )


def smoke(output: Path, steps: int = 200) -> None:
    output = Path(output)
    destination = output / "smoke"
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    for ratio, sigma in ((1.0, 0.0), (0.10, 0.10)):
        folder = destination / f"ratio_{ratio:g}" / f"sigma_{sigma:g}" / "seed_0"
        summary = train_condition(folder, ratio, sigma, 0, steps, verbose=True)
        validate_run(folder, ratio, sigma, 0, steps)
        results.append(dict(ratio=ratio, sigma=sigma, summary=summary))
    write_json(destination / "smoke_validation.json", dict(passed=True, steps=steps, runs=results))
    # Smoke runs are engineering checks, not members of the 180-run matrix.
    write_json(
        destination / "completion.json",
        dict(expected_runs=180, completed_runs=len(results), missing_runs=180 - len(results),
             complete_conditions=0, expected_conditions=30, complete=False,
             note="Engineering smoke runs only; do not aggregate into the research matrix."),
    )
    print(json.dumps(dict(passed=True, runs=len(results), steps=steps), indent=2))


def _run_record(output: Path, record: dict, steps: int, resume: bool, device: str = "cpu") -> dict:
    folder = Path(output) / record["directory"]
    try:
        if (folder / "summary.json").exists():
            validate_run(folder, record["ratio"], record["sigma"], record["seed"], steps)
            return dict(run_id=record["run_id"], state="validated")
        train_condition(
            folder,
            record["ratio"],
            record["sigma"],
            record["seed"],
            steps,
            resume=resume,
            device=device,
            verbose=True,
        )
        validate_run(folder, record["ratio"], record["sigma"], record["seed"], steps)
        return dict(run_id=record["run_id"], state="validated")
    except Exception as exc:  # preserve partial work and continue the shard
        return dict(run_id=record["run_id"], state=f"failed: {type(exc).__name__}: {exc}")


def run_shard(output: Path, shard_index: int, shards: int, steps: int, resume: bool, workers: int = 1, device: str = "cpu") -> None:
    output = Path(output)
    manifest = json.loads((output / "runs.json").read_text())
    assigned = [r for i, r in enumerate(manifest) if i % shards == shard_index]
    status = []
    if workers <= 1:
        for record in assigned:
            result = _run_record(output, record, steps, resume, device)
            status.append(result)
            write_json(output / f"shard_{shard_index:02d}_status.json", dict(shard_index=shard_index, shards=shards, runs=status))
            print(f"[shard {shard_index}] {result['run_id']}: {result['state']}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {
                pool.submit(_run_record, output, record, steps, resume, device): record
                for record in assigned
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                status.append(result)
                write_json(output / f"shard_{shard_index:02d}_status.json", dict(shard_index=shard_index, shards=shards, runs=status))
                print(f"[shard {shard_index}] {result['run_id']}: {result['state']}", flush=True)
    write_json(
        output / f"shard_{shard_index:02d}_complete.json",
        dict(shard_index=shard_index, shards=shards, assigned=len(assigned), runs=status),
    )


def benchmark(output: Path, ratio: float, sigma: float, seed: int, steps: int, device: str) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if device.startswith("cuda") and not torch.cuda.is_available():
        write_json(output / "benchmark.json", dict(device=device, available=False, error="CUDA unavailable"))
        print(json.dumps(dict(device=device, available=False), indent=2))
        return
    start = __import__("time").monotonic()
    summary = train_condition(
        output / "run",
        ratio,
        sigma,
        seed,
        steps,
        device=device,
        checkpoints=(0, steps),
        field_checkpoints=(0, steps),
        verbose=True,
    )
    elapsed = __import__("time").monotonic() - start
    record = dict(
        device=device,
        available=True,
        steps=steps,
        elapsed_seconds=elapsed,
        seconds_per_step=elapsed / max(steps, 1),
        projected_100k_seconds=elapsed / max(steps, 1) * 100000,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.startswith("cuda") else None,
        final_loss=summary["total_loss"],
        finite=bool(np.isfinite(summary["total_loss"])),
    )
    write_json(output / "benchmark.json", record)
    print(json.dumps(record, indent=2))


def report(output: Path, steps: int) -> None:
    output = Path(output)
    completion = aggregate(output, require_all=False, steps=steps)
    clean_noisy_visualization(output)
    summaries = json.loads((output / "recovery_summary.json").read_text())
    lines = [
        "# Smooth sinusoidal Burgers noise x sampling",
        "",
        f"Completed runs: **{completion['completed_runs']}/{completion['expected_runs']}**; "
        f"complete conditions: **{completion['complete_conditions']}/{completion['expected_conditions']}**.",
        "",
        "This report is generated from completed runs only. It must not be read as the completed "
        "180-run study while `completion.json` reports missing runs.",
        "",
        "## Design",
        "",
        "PDE `u_t = -u u_x + 0.2 u_xx`, periodic `[0,2*pi)`, `u(x,0)=sin(x)`, `t in [0,1]`, "
        "256 spatial points, `dt=0.002`, 501 snapshots. A fixed population of 13,056 points is drawn "
        "uniformly without replacement from the 128,256 numerical points once per replicate. Nested "
        "prefixes give the five sampling fractions. One paired Gaussian realization per replicate is "
        "scaled by `sigma * std(U)` and reused across all conditions.",
        "",
        "Training reuses the established SIREN (3x64, omega 1/1), zero-initialized two-coefficient "
        "head, Adam 5e-4, batch 640 with replacement, unit data/PDE weights, 100,000 steps, CPU float32.",
        "",
        "## Recovery criteria (predeclared)",
        "",
        "Loose: `|c1+1| < 0.25` and `|c2-0.2| < 0.02`. Strong: `< 0.10` and `< 0.01`. "
        "Sustained: the criterion holds for the final 10,000 optimizer steps. First crossings need not persist.",
        "",
        "## Per-condition summary",
        "",
        "| sigma | sampling | count | n | endpoint loose | endpoint strong | sustained strong | mean c1 | mean c2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {sigma:g} | {pct:g}% | {count} | {n_complete}/{n_expected} | {el}/{en} | {es}/{en} | {ss}/{en} | {c1:.4f} | {c2:.4f} |".format(
                sigma=s["sigma"],
                pct=100 * s["ratio"],
                count=s["count"],
                n_complete=s["n_complete"],
                n_expected=s["n_expected"],
                el=s["endpoint_loose_count"],
                en=s["n_complete"],
                es=s["endpoint_strong_count"],
                ss=s["sustained_strong_count"],
                c1=s["endpoint_mean_transport"] if s["endpoint_mean_transport"] is not None else float("nan"),
                c2=s["endpoint_mean_diffusion"] if s["endpoint_mean_diffusion"] is not None else float("nan"),
            )
        )
    lines += [
        "",
        "## Figures",
        "",
        "- `noise_sampling_recovery_heatmap.png/pdf`",
        "- `coefficient_trajectories_transport.png/pdf`, `coefficient_trajectories_diffusion.png/pdf`",
        "- `final_coefficient_errors.png/pdf`, `final_coefficient_error_heatmap.png/pdf`",
        "- `representative_loss_curves.png/pdf`",
        "- `clean_vs_noisy.png/pdf`",
        "",
        "## Limitations",
        "",
        "- Known-support parameter estimation, not symbolic discovery.",
        "- Noise is added to observations only; coordinates and numerical references remain clean.",
        "- The PDE loss is evaluated at the sampled observation coordinates, so sampling changes both "
        "data density and PDE collocation coverage.",
        "- Incomplete conditions are reported with their completed seed count and are not extrapolated.",
        "- CPU float32 results; device and thread count are recorded per run.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "completion.json", completion)


def status(output: Path, steps: int) -> None:
    output = Path(output)
    manifest = json.loads((output / "runs.json").read_text())
    complete = 0
    partial = 0
    missing = 0
    for record in manifest:
        folder = output / record["directory"]
        if (folder / "status.json").exists() and json.loads((folder / "status.json").read_text()).get("state") == "complete":
            complete += 1
        elif folder.exists():
            partial += 1
        else:
            missing += 1
    print(json.dumps(dict(expected=len(manifest), complete=complete, partial=partial, missing=missing), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", required=True, choices=["validate", "prepare", "smoke", "run-one", "shard", "benchmark", "aggregate", "report", "status"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)

    if args.mode == "validate":
        args.output.mkdir(parents=True, exist_ok=True)
        checks = numerical_validation(args.output / "validation")
        protocol = protocol_checks()
        write_json(args.output / "validation/protocol.json", protocol)
        print(json.dumps(dict(numerical=checks, protocol=protocol), indent=2))
    elif args.mode == "prepare":
        prepare(args.output, args.steps)
    elif args.mode == "smoke":
        smoke(args.output, steps=min(args.steps, 200) if args.steps != DEFAULT_STEPS else 200)
    elif args.mode == "run-one":
        if None in (args.ratio, args.sigma, args.seed):
            parser.error("--ratio, --sigma and --seed are required for run-one")
        folder = run_directory(args.output, args.ratio, args.sigma, args.seed)
        summary = train_condition(
            folder,
            args.ratio,
            args.sigma,
            args.seed,
            args.steps,
            resume=args.resume,
            device=args.device,
        )
        validate_run(folder, args.ratio, args.sigma, args.seed, args.steps)
        print(json.dumps(summary, indent=2, default=str))
    elif args.mode == "shard":
        run_shard(args.output, args.shard_index, args.shards, args.steps, args.resume, args.workers, args.device)
    elif args.mode == "benchmark":
        if None in (args.ratio, args.sigma, args.seed):
            parser.error("--ratio, --sigma and --seed are required for benchmark")
        benchmark(args.output, args.ratio, args.sigma, args.seed, args.steps, args.device)
    elif args.mode == "aggregate":
        completion = aggregate(args.output, require_all=args.require_all, steps=args.steps)
        print(json.dumps(completion, indent=2))
    elif args.mode == "report":
        report(args.output, args.steps)
    elif args.mode == "status":
        status(args.output, args.steps)


if __name__ == "__main__":
    main()
