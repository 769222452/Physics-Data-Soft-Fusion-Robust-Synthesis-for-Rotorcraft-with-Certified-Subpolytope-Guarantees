import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.linalg as la


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from normalized_coordinates import (  # noqa: E402
    DEFAULT_SCALES,
    allocate_stratified_budget,
    build_normalized_augmented_matrices,
    build_normalized_performance_matrices,
    build_normalized_residual_bound,
    build_fusion_diagnostics_payload,
    build_physical_augmented_matrices,
    build_physical_performance_matrices,
    classify_surrogate_status,
    compute_inconsistency_quota,
    gain_to_normalized,
    gain_to_physical,
    increment_to_normalized,
    project_disturbance_physical,
    project_increment_physical,
    sample_capped_physical_residual,
    verify_realized_record_residual,
)


PARAMETERS = {
    "sigma_t": 0.9,
    "Jx": 0.015,
    "Jy": 0.014,
    "Jz": 0.030,
    "kx": 0.20,
    "ky": 0.22,
    "kz": 0.18,
    "kp": 0.25,
    "kq": 0.27,
    "kr": 0.30,
}


class CoordinateTests(unittest.TestCase):
    def test_physical_and_normalized_updates_are_equivalent(self):
        rng = np.random.default_rng(4)
        A, B, S = build_physical_augmented_matrices(PARAMETERS, 0.1, 9.81)
        Abar, Bbar, Sbar = build_normalized_augmented_matrices(
            PARAMETERS, 0.1, 9.81
        )
        xbar = rng.standard_normal(16)
        ubar = rng.standard_normal(4)
        dbar = rng.standard_normal(6)
        x = DEFAULT_SCALES.T_xc @ xbar
        u = DEFAULT_SCALES.T_du @ ubar
        d = DEFAULT_SCALES.T_d @ dbar
        physical_next = A @ x + B @ u + S @ d
        normalized_next = Abar @ xbar + Bbar @ ubar + Sbar @ dbar
        np.testing.assert_allclose(
            physical_next,
            DEFAULT_SCALES.T_xc @ normalized_next,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_gain_round_trip(self):
        rng = np.random.default_rng(5)
        Kphysical = rng.standard_normal((4, 16))
        Kbar = gain_to_normalized(Kphysical)
        np.testing.assert_allclose(gain_to_physical(Kbar), Kphysical, atol=1e-13)

    def test_physical_weighted_output_is_preserved(self):
        rng = np.random.default_rng(51)
        q = (20, 20, 40, 5, 5, 5, 50, 50, 20, 1, 1, 1)
        r = (5, 10, 10, 10)
        Cphys, Dphys = build_physical_performance_matrices(q, r)
        Cbar, Dbar = build_normalized_performance_matrices(q, r)
        xbar = rng.standard_normal(16)
        ubar = rng.standard_normal(4)
        zphys = Cphys @ (DEFAULT_SCALES.T_xc @ xbar) + Dphys @ (
            DEFAULT_SCALES.T_du @ ubar
        )
        np.testing.assert_allclose(Cbar @ xbar + Dbar @ ubar, zphys, atol=1e-13)

    def test_increment_projection_uses_normalized_unit_ball(self):
        command = np.array([7.0, 0.0, 0.0, 0.0])
        applied, factor = project_increment_physical(command)
        self.assertLess(factor, 1.0)
        self.assertLessEqual(np.linalg.norm(increment_to_normalized(applied)), 1.0 + 1e-12)

    def test_total_disturbance_projection(self):
        raw = DEFAULT_SCALES.T_d @ np.array([[1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])
        applied, stats = project_disturbance_physical(raw)
        inv_Td = np.diag(1.0 / np.diag(DEFAULT_SCALES.T_d))
        self.assertAlmostEqual(np.linalg.norm(inv_Td @ applied[:, 0]), 1.0, places=12)
        self.assertEqual(stats["projection_count"], 1)
        self.assertGreater(stats["raw_peak_normalized"], 1.0)

    def test_realized_residual_is_bounded(self):
        rng = np.random.default_rng(6)
        L = 300
        residuals = np.column_stack(
            [sample_capped_physical_residual(rng)[0] for _ in range(L)]
        )
        inv_Tx = np.diag(1.0 / np.diag(DEFAULT_SCALES.T_x))
        E = inv_Tx @ residuals
        W = build_normalized_residual_bound(L)[:12, :12]
        self.assertGreaterEqual(la.eigvalsh(W - E @ E.T)[0], -1e-12)

    def test_diagnostic_payload_contains_no_object_arrays(self):
        rng = np.random.default_rng(7)
        A, B, S = build_normalized_augmented_matrices(PARAMETERS, 0.1, 9.81)
        L = 4
        X = rng.standard_normal((16, L))
        U = rng.standard_normal((4, L))
        D = rng.standard_normal((6, L)) * 0.1
        C = np.eye(16)
        Dout = np.zeros((16, 4))
        Xnext = A @ X + B @ U + S @ D
        batch = {
            "X_t": X, "X_tp1": Xnext, "U_t": U, "Z_t": C @ X,
            "Dbar_t": D, "A": A, "B": B, "S_c": S, "Cc": C,
            "Dc": Dout,
        }
        W_E = build_normalized_residual_bound(L)
        batch["record_residual_check"] = verify_realized_record_residual(
            batch, W_E
        )
        Delta = np.block([[A, B], [C, Dout]])
        vertex = {"p": PARAMETERS, "Delta": Delta, "S": S, "s_raw": 0.0, "s": 0.0}
        payload = build_fusion_diagnostics_payload(
            batch, [vertex], np.eye(52), np.eye(16), W_E,
            {"alpha": 1.0}, {"n_consistent": 1},
            list(PARAMETERS), PARAMETERS,
        )
        self.assertIn("batch_S_c", payload)
        self.assertTrue(all(value.dtype != object for value in payload.values()))


class EdgeCaseTests(unittest.TestCase):
    def test_stratified_budget_handles_small_and_empty_tiers(self):
        self.assertEqual(allocate_stratified_budget(0, (0.4, 0.2, 0.4), (5, 5, 5)), (0, 0, 0))
        self.assertEqual(sum(allocate_stratified_budget(1, (0.4, 0.2, 0.4), (5, 5, 5))), 1)
        self.assertEqual(sum(allocate_stratified_budget(2, (0.4, 0.2, 0.4), (5, 5, 5))), 2)
        self.assertEqual(allocate_stratified_budget(3, (0.4, 0.2, 0.4), (0, 3, 0)), (0, 3, 0))

    def test_positive_score_quota(self):
        self.assertEqual(compute_inconsistency_quota([0.0, 0.0], 8, 0.2), 0)
        self.assertEqual(compute_inconsistency_quota([0.0, 0.3], 8, 0.2), 1)
        self.assertEqual(compute_inconsistency_quota([0.1, 0.2, 0.3], 8, 0.2), 2)

    def test_status_boundaries(self):
        self.assertEqual(classify_surrogate_status(True, 1e-4), "Surrogate pass")
        self.assertEqual(classify_surrogate_status(True, 1.5e-4), "Near surrogate")
        self.assertEqual(classify_surrogate_status(True, 2.1e-4), "Failed")
        self.assertEqual(classify_surrogate_status(False, -1.0), "Failed")


if __name__ == "__main__":
    unittest.main()
