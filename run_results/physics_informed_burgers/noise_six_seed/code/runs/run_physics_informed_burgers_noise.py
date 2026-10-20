"""Execute the paired two-seed Burgers observation-noise experiment."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.physics_informed_burgers import NOTEBOOKS, hashes, write_json
from utils.physics_informed_burgers_noise import (
    SEEDS,
    SIGMAS,
    make_figures,
    finalize_condition,
    train_condition,
    validate_noise_protocol,
)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, default=Path("run_results/physics_informed_burgers/noise_robustness"))
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    if not 1 <= args.steps <= 100000:
        p.error("steps must be between 1 and 100000")
    # Existing directories are allowed only for the explicit resume workflow;
    # completed condition folders are detected and reused below.
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    source_hashes = hashes((*NOTEBOOKS, "utils/physics_informed_burgers.py", "utils/physics_informed_burgers_noise.py",
                            "runs/run_physics_informed_burgers.py", "runs/run_physics_informed_burgers_noise.py",
                            "prog/mlps.py", "prog/featlib.py", "utils/derivative_utils.py",
                            "Datasets/data/processed/burg_gen/burg_gen.py"))
    config = dict(experiment="Burgers observation-noise robustness", sigmas=list(SIGMAS), seeds=list(SEEDS),
        steps=args.steps, workers=args.workers, dataset=dict(seed=0, nu=.02, T=.1, N=256, dt=.002, observations=640),
        noise=dict(formula="u_clean + sigma*std(clean_640_targets)*epsilon", std_ddof=0,
                   paired_standard_normal=True, noise_seed_by_initialization={str(seed): 700000 + seed for seed in SEEDS}),
        training=dict(lr=5e-4, lambda_data=1., lambda_pde=1., l1=0., optimizer="Adam",
                      batch_size=640, batch_sampling="with replacement", feature_normalization=False,
                      fresh_siren=True, physical_head=True, no_least_squares=True),
        evaluation=dict(reference="clean 51x256 solver grid", derivatives="canonical FeatureTensor/autodiff; u_t from model autodiff",
                        coefficients=[-1., .02]),
        execution=dict(noise_applied_once_before_training=True, clean_reference_never_noisy=True,
                        no_early_stopping=True, no_per_condition_tuning=True, sparsity_executed=False,
                        crossed_experiment_executed=False), source_hashes=source_hashes,
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    write_json(args.output / "config.json", config)
    (args.output / "reproduction.txt").write_text(
        f"{sys.executable} runs/run_physics_informed_burgers_noise.py --output {args.output} --steps {args.steps} --workers {args.workers}\n"
    )
    validation = validate_noise_protocol()
    validation.update(notebooks_unchanged=hashes(NOTEBOOKS) == {k: source_hashes[k] for k in NOTEBOOKS})
    write_json(args.output / "validation.json", validation)
    if args.validate_only:
        print("Noise protocol validation passed")
        return
    jobs = []
    results = []
    for sigma in SIGMAS:
        for seed in SEEDS:
            noise_seed = 700000 + seed
            folder = args.output / f"sigma_{sigma:g}" / f"seed_{seed}"
            if (folder / "summary.json").exists():
                results.append(json.loads((folder / "summary.json").read_text()))
            elif (folder / "history.csv").exists():
                results.append(finalize_condition(folder, sigma, seed, noise_seed, args.steps))
            else:
                jobs.append((sigma, seed, noise_seed))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {
            pool.submit(train_condition, args.output / f"sigma_{sigma:g}" / f"seed_{seed}", sigma, seed, noise_seed, args.steps): (sigma, seed)
            for sigma, seed, noise_seed in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            write_json(args.output / "completed_conditions.json", sorted(results, key=lambda x: (x["sigma"], x["init_seed"])))
    make_figures(args.output)
    validation.update(completed_conditions=len(results), expected_conditions=len(jobs), all_conditions_complete=len(results) == len(jobs))
    write_json(args.output / "validation.json", validation)
    print("Noise experiment complete", flush=True)


if __name__ == "__main__":
    main()
