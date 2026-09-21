"""Regression checks for the matched-control interfaces (no training sweeps)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from Datasets.data.processed.burg_gen.burg_gen import solve_burgers
from prog.minimal_symnet import MinimalSymNet
from utils.burgers_recoverability import (P22, TRUTH, PRIMITIVES, coefficients,
    coordinate_check, recovered)
from utils.derivative_utils import evaluate_diagnostic_fields
from utils.diagnostic_io import inventory, verify_inventory


class BurgersControlTests(unittest.TestCase):
    def test_default_solver_preserves_archived_phase23_observations(self):
        x, _, _, (t, u) = solve_burgers(seed=0, return_history=True)
        archive = torch.load(P22, weights_only=False)
        np.testing.assert_array_equal(u.astype(np.float32).ravel(), archive['u'].numpy().ravel())
        np.testing.assert_array_equal(np.repeat(t.astype(np.float32), len(x)), archive['t'].numpy().ravel())
        np.testing.assert_array_equal(np.tile(x.astype(np.float32), len(t)), archive['x'].numpy().ravel())

    def test_custom_initial_condition_preserves_solver_cadence(self):
        initial = lambda x: .5 * np.sin(x)
        _, _, _, (t, u) = solve_burgers(initial_condition=initial, return_history=True)
        x, _, _, (dense_t, dense_u) = solve_burgers(initial_condition=initial, return_history=True, history_every=1)
        keep = np.r_[0, np.arange(1, 501, 2), 500]
        np.testing.assert_array_equal(t, dense_t[keep])
        np.testing.assert_array_equal(u, dense_u[keep])
        np.testing.assert_array_equal(u[0], initial(x))
        self.assertEqual(len(dense_t), 501)

    def test_canonical_physical_derivatives_against_analytic_field(self):
        class AnalyticField(torch.nn.Module):
            def forward(self, t, x):
                return torch.exp(-t) * torch.sin(x)
        t = np.repeat(np.linspace(0, 1, 5, dtype=np.float32), 16)
        x = np.tile(np.linspace(0, 2*np.pi, 16, endpoint=False, dtype=np.float32), 5)
        fields = evaluate_diagnostic_fields(AnalyticField(), t, x, (5, 16), PRIMITIVES, chunk_size=17)
        u = (np.exp(-t)*np.sin(x)).reshape(5,16)
        np.testing.assert_allclose(fields['u'], u, atol=2e-7)
        np.testing.assert_allclose(fields['u_x'], (np.exp(-t)*np.cos(x)).reshape(5,16), atol=2e-7)
        np.testing.assert_allclose(fields['u_xx'], -u, atol=2e-7)
        np.testing.assert_allclose(fields['u_t'], -u, atol=2e-7)

    def test_physical_coefficients_and_recovery_thresholds(self):
        scales = np.array([.35, .36, 4.2], dtype=np.float32)
        model = MinimalSymNet()
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
            model.left.weight[0,0] = float(scales[0])
            model.right.weight[0,1] = float(scales[1])
            model.product_readout.weight[0,0] = -1
            model.linear.weight[0,2] = float(.02*scales[2])
        xi = coefficients(model, scales)
        np.testing.assert_allclose(xi, TRUTH, atol=2e-7)
        self.assertTrue(recovered(xi, True))
        self.assertFalse(recovered(np.zeros(9)))
        boundary = TRUTH.copy(); boundary[4] = -.75
        self.assertFalse(recovered(boundary))
        coordinate_check(model, np.random.default_rng(1).normal(size=(100,3)).astype(np.float32), scales)

    def test_artifact_integrity_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'artifact.txt'; p.write_text('original')
            expected = inventory(directory); verify_inventory(directory, expected)
            p.write_text('changed')
            with self.assertRaises(RuntimeError): verify_inventory(directory, expected)


if __name__ == '__main__':
    unittest.main()
