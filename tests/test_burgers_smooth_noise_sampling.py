"""Engineering tests for the smooth sinusoidal Burgers noise x sampling study.

These tests validate the implementation, not scientific recovery. Long training
is deliberately excluded; the smoke test uses a few dozen optimizer steps.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.burgers_smooth_noise_sampling import (
    COUNTS,
    LOOSE,
    N_FULL,
    N_T,
    N_X,
    POPULATION,
    RATIOS,
    SEEDS,
    SIGMAS,
    STRONG,
    SUSTAINED_WINDOW,
    TARGETS,
    aggregate,
    coefficient_metrics,
    condition_arrays,
    count_for,
    load_history,
    make_condition,
    numerical_validation,
    recovery_metrics,
    reference_bundle,
    reference_solution,
    replicate_design,
    run_directory,
    run_id,
    run_manifest,
    selected_positions,
    train_condition,
    validate_run,
)
from utils.physics_informed_burgers import initialize


TMP = Path("/tmp/opencode/burgers_tests")


def fresh(name: str) -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


class TestReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = reference_bundle()

    def test_initial_condition_and_grid(self):
        x, t, u = reference_solution()
        self.assertEqual(u.shape, (N_T, N_X))
        self.assertEqual(len(t), N_T)
        np.testing.assert_allclose(u[0], np.sin(x), rtol=0, atol=0)
        np.testing.assert_allclose(np.diff(x), x[1] - x[0], rtol=0, atol=1e-15)
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[-1] + (x[1] - x[0]), 2 * np.pi, places=14)
        self.assertTrue(np.isfinite(u).all())

    def test_periodic_derivative_consistency(self):
        bundle = self.bundle
        dx = bundle["dx"]
        ux_fd = bundle["ref"]["u_x"]["physical"]
        wave = 2 * np.pi * np.fft.fftfreq(N_X, d=dx)
        ux_sp = np.fft.ifft(np.fft.fft(bundle["u"], axis=1) * (1j * wave), axis=1).real
        rel = np.linalg.norm(ux_fd - ux_sp) / np.linalg.norm(ux_sp)
        self.assertLess(rel, 1e-3)

    def test_numerical_convergence(self):
        destination = fresh("numerical")
        checks = numerical_validation(destination)
        self.assertEqual(checks["initial_condition_max_abs_error"], 0.0)
        self.assertLess(checks["temporal_refinement_rel_l2"], 1e-8)
        self.assertLess(checks["spatial_refinement_rel_l2"], 1e-3)
        self.assertLess(checks["max_abs_u_x"], 3.0)
        self.assertLess(checks["max_abs_u_xx"], 5.0)
        self.assertAlmostEqual(checks["amplitude_decay"], 0.8096627945671375, places=8)
        for row in checks["identifiability"]:
            self.assertLess(abs(row["least_squares_transport"] + 1), 1e-8)
            self.assertLess(abs(row["least_squares_diffusion"] - 0.2), 1e-8)
            self.assertLess(row["normalized_condition_number"], 5.0)


class TestDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = reference_bundle()

    def test_counts(self):
        self.assertEqual(COUNTS, (13056, 9792, 6528, 3264, 1305))
        self.assertEqual([count_for(r) for r in RATIOS], list(COUNTS))
        self.assertEqual(count_for(0.10), 1305)
        self.assertEqual(POPULATION, 13056)

    def test_manifest_isolation(self):
        manifest = run_manifest()
        self.assertEqual(len(manifest), 180)
        self.assertEqual(len({r["run_id"] for r in manifest}), 180)
        self.assertEqual(len({r["directory"] for r in manifest}), 180)
        for record in manifest:
            expected = run_directory(Path("."), record["ratio"], record["sigma"], record["seed"])
            self.assertEqual(Path(record["directory"]), expected)
            self.assertEqual(record["run_id"], run_id(record["ratio"], record["sigma"], record["seed"]))

    def test_seed_path_formatting(self):
        # Summaries round-trip seed as a float; paths and IDs must stay integral.
        self.assertEqual(
            run_directory(Path("x"), 1.0, 0.0, 0.0),
            Path("x/raw/ratio_1/sigma_0/seed_0"),
        )
        self.assertEqual(
            run_directory(Path("x"), 0.1, 0.2, 5.0),
            Path("x/raw/ratio_0.1/sigma_0.2/seed_5"),
        )
        self.assertEqual(run_id(1.0, 0.0, 0.0), "burgers_nu0.2_r1.00_s0.00_seed0")

    def test_nested_sampling(self):
        design = replicate_design(0)
        self.assertEqual(len(design["population"]), POPULATION)
        self.assertEqual(len(np.unique(design["population"])), POPULATION)
        self.assertTrue((design["population"] >= 0).all())
        self.assertTrue((design["population"] < N_FULL).all())
        previous = set(range(POPULATION))
        for ratio in RATIOS:
            positions = selected_positions(design, ratio)
            self.assertEqual(len(positions), count_for(ratio))
            self.assertTrue(set(positions.tolist()) <= previous)
            previous = set(positions.tolist())

    def test_noise_is_target_only_and_paired(self):
        design = replicate_design(1)
        for ratio in RATIOS:
            clean = condition_arrays(self.bundle, design, ratio, 0.0)
            noisy = condition_arrays(self.bundle, design, ratio, 0.10)
            np.testing.assert_array_equal(clean["noisy"], clean["clean"])
            np.testing.assert_array_equal(clean["t"], noisy["t"])
            np.testing.assert_array_equal(clean["x"], noisy["x"])
            np.testing.assert_allclose(
                noisy["noise"], 0.10 * self.bundle["scale"] * noisy["epsilon"], rtol=1e-14, atol=1e-15
            )
            np.testing.assert_allclose(noisy["noisy"], noisy["clean"] + noisy["noise"], rtol=0, atol=0)
        eps_full = condition_arrays(self.bundle, design, 1.0, 0.10)["epsilon"]
        eps_small = condition_arrays(self.bundle, design, 0.10, 0.10)["epsilon"]
        np.testing.assert_array_equal(eps_small, eps_full[: len(eps_small)])
        low = condition_arrays(self.bundle, design, 1.0, 0.01)["noise"]
        high = condition_arrays(self.bundle, design, 1.0, 0.20)["noise"]
        np.testing.assert_allclose(high, 20.0 * low, rtol=1e-13, atol=1e-15)

    def test_seed_streams_and_reproducibility(self):
        a = replicate_design(3)
        b = replicate_design(3)
        c = replicate_design(4)
        np.testing.assert_array_equal(a["population"], b["population"])
        np.testing.assert_array_equal(a["subsample_order"], b["subsample_order"])
        np.testing.assert_array_equal(a["epsilon"], b["epsilon"])
        self.assertFalse(np.array_equal(a["population"], c["population"]))
        self.assertFalse(np.array_equal(a["epsilon"], c["epsilon"]))
        # The three roles are independent streams.
        self.assertFalse(np.array_equal(a["population"][:100], a["subsample_order"][:100]))
        self.assertFalse(np.array_equal(a["population"][:100], np.argsort(a["epsilon"])[:100]))
        m1, h1 = initialize(3)
        m2, h2 = initialize(3)
        for p, q in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(h1.xi, h2.xi, rtol=0, atol=0)


class TestRecoveryMetrics(unittest.TestCase):
    def test_first_and_sustained(self):
        steps = np.arange(0, SUSTAINED_WINDOW + 1001)
        loose = np.zeros(len(steps), dtype=bool)
        strong = np.zeros(len(steps), dtype=bool)
        loose[100:150] = True
        loose[1000:] = True
        strong[1000:] = True
        history = dict(step=steps, loose=loose, strong=strong)
        result = recovery_metrics(history)
        self.assertEqual(result["first_loose_step"], 100)
        self.assertTrue(result["ever_loose"])
        self.assertTrue(result["endpoint_loose"])
        self.assertTrue(result["sustained_loose"])
        self.assertEqual(result["sustained_loose_from_step"], 1000)
        self.assertEqual(result["first_strong_step"], 1000)
        self.assertTrue(result["sustained_strong"])

    def test_non_recovery_and_short_window(self):
        steps = np.arange(0, 101)
        history = dict(step=steps, loose=np.zeros(101, dtype=bool), strong=np.zeros(101, dtype=bool))
        result = recovery_metrics(history, window=10)
        self.assertIsNone(result["first_loose_step"])
        self.assertFalse(result["ever_loose"])
        self.assertFalse(result["sustained_loose"])

    def test_coefficient_metrics(self):
        class Head:
            xi = torch.nn.Parameter(torch.tensor([-1.0, 0.2]))

        metrics = coefficient_metrics(Head())
        self.assertLess(metrics["transport_abs_error"], 1e-7)
        self.assertLess(metrics["diffusion_abs_error"], 1e-7)
        self.assertTrue(metrics["loose"])
        self.assertTrue(metrics["strong"])
        Head.xi = torch.nn.Parameter(torch.tensor([-0.70, 0.19]))
        metrics = coefficient_metrics(Head())
        self.assertFalse(metrics["loose"])
        self.assertFalse(metrics["strong"])


class TestTrainingAndValidation(unittest.TestCase):
    def test_smoke_run_and_validation(self):
        root = fresh("smoke_run")
        folder = root / "run"
        summary = train_condition(
            folder, 0.10, 0.10, 0, 20, checkpoints=(0, 1, 10, 20), field_checkpoints=(0, 20), verbose=False
        )
        self.assertEqual(summary["steps"], 20)
        self.assertTrue(np.isfinite(summary["total_loss"]))
        history, metrics = validate_run(folder, 0.10, 0.10, 0, 20, checkpoints=(0, 1, 10, 20))
        self.assertEqual(len(history["step"]), 21)
        self.assertEqual(len(metrics), 4)
        self.assertTrue((folder / "observations.npz").exists())
        self.assertTrue((folder / "status.json").exists())

    def test_resume_is_exact(self):
        root = fresh("resume")
        checkpoints = (0, 5, 10)
        train_condition(root / "fresh", 1.0, 0.05, 2, 10, checkpoints=checkpoints, field_checkpoints=(0, 10), verbose=False)
        train_condition(root / "partial", 1.0, 0.05, 2, 5, checkpoints=(0, 5), field_checkpoints=(0, 5), verbose=False)
        train_condition(
            root / "partial", 1.0, 0.05, 2, 10, checkpoints=checkpoints, field_checkpoints=(0, 10), resume=True, verbose=False
        )
        a = load_history(root / "fresh" / "history.csv")
        b = load_history(root / "partial" / "history.csv")
        for column in a:
            if column == "run_id":
                continue
            np.testing.assert_allclose(a[column], b[column], rtol=0, atol=0)
        validate_run(root / "partial", 1.0, 0.05, 2, 10, checkpoints=checkpoints)

    def test_validation_rejects_corruption(self):
        root = fresh("corrupt")
        folder = root / "run"
        train_condition(folder, 1.0, 0.0, 0, 5, checkpoints=(0, 5), field_checkpoints=(0, 5), verbose=False)
        lines = (folder / "history.csv").read_text().splitlines()
        lines[2] = lines[2].replace(lines[2].split(",")[6], "not-a-number", 1)
        (folder / "history.csv").write_text("\n".join(lines) + "\n")
        with self.assertRaises(Exception):
            validate_run(folder, 1.0, 0.0, 0, 5, checkpoints=(0, 5))


class TestAggregation(unittest.TestCase):
    def _summary(self, ratio, sigma, seed, transport, diffusion):
        return dict(
            ratio=ratio,
            sigma=sigma,
            seed=seed,
            count=count_for(ratio),
            transport=transport,
            diffusion=diffusion,
            transport_abs_error=abs(transport - TARGETS[0]),
            diffusion_abs_error=abs(diffusion - TARGETS[1]),
            transport_relative_error=abs(transport - TARGETS[0]),
            diffusion_relative_error=abs(diffusion - TARGETS[1]) / abs(TARGETS[1]),
            loose=abs(transport - TARGETS[0]) < LOOSE[0] and abs(diffusion - TARGETS[1]) < LOOSE[1],
            strong=abs(transport - TARGETS[0]) < STRONG[0] and abs(diffusion - TARGETS[1]) < STRONG[1],
            ever_loose=True,
            ever_strong=True,
            endpoint_loose=True,
            endpoint_strong=True,
            sustained_loose=True,
            sustained_strong=True,
            first_loose_step=100,
            first_strong_step=200,
            sustained_loose_from_step=500,
            sustained_strong_from_step=600,
            elapsed_seconds=10.0,
        )

    def test_aggregation_counts_and_means(self):
        root = fresh("aggregate")
        for seed in range(6):
            folder = run_directory(root, 1.0, 0.0, seed)
            folder.mkdir(parents=True)
            transport = -1.0 + 0.01 * seed
            diffusion = 0.2 + 0.001 * seed
            (folder / "summary.json").write_text(json.dumps(self._summary(1.0, 0.0, seed, transport, diffusion)))
            (folder / "status.json").write_text(json.dumps(dict(state="complete", steps=100000)))
        completion = aggregate(root, require_all=False, plot=False)
        self.assertEqual(completion["completed_runs"], 6)
        self.assertEqual(completion["missing_runs"], 174)
        summary = json.loads((root / "recovery_summary.json").read_text())
        row = next(r for r in summary if r["ratio"] == 1.0 and r["sigma"] == 0.0)
        self.assertEqual(row["n_complete"], 6)
        self.assertEqual(row["endpoint_strong_count"], 6)
        self.assertAlmostEqual(row["endpoint_mean_transport"], -0.975)
        self.assertAlmostEqual(row["endpoint_mean_diffusion"], 0.2025)

    def test_incomplete_runs_are_excluded(self):
        root = fresh("incomplete")
        for seed in range(6):
            folder = run_directory(root, 0.5, 0.1, seed)
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps(self._summary(0.5, 0.1, seed, -1.0, 0.2)))
            state = "complete" if seed < 3 else "running"
            (folder / "status.json").write_text(json.dumps(dict(state=state, steps=100000)))
        completion = aggregate(root, require_all=False, plot=False)
        self.assertEqual(completion["completed_runs"], 3)
        row = next(r for r in json.loads((root / "recovery_summary.json").read_text()) if r["ratio"] == 0.5 and r["sigma"] == 0.1)
        self.assertEqual(row["n_complete"], 3)
        self.assertFalse(row["complete"])


if __name__ == "__main__":
    unittest.main()
