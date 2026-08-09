import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.linalg as la


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import time_domain_standard as model  # noqa: E402
from raw_qmi_scores import (  # noqa: E402
    dynamics_qmi_raw_score,
    dynamics_residual_matrix,
    full_qmi_raw_score,
)


class AlgebraTests(unittest.TestCase):
    def test_synthesis_limits_match_shared_scale_configuration(self):
        syn = model.SynthesisParams()
        scales = model.DEFAULT_SCALES
        self.assertTrue(np.allclose(scales.disturbance, syn.d_max))
        self.assertTrue(np.allclose(scales.increment, syn.du_max))
        np.testing.assert_allclose(syn.u_abs_max, scales.input_memory)
        np.testing.assert_allclose(syn.u_abs_min, -np.asarray(scales.input_memory))
        self.assertTrue(np.allclose(scales.residual_std, syn.sigma_E_x))
        self.assertTrue(np.allclose(scales.residual_cap, syn.nu_E_max))

    def test_build_psi_requires_explicit_wc_and_we(self):
        length = 3
        batch = {
            "X_t": np.zeros((16, length)),
            "X_tp1": np.zeros((16, length)),
            "U_t": np.zeros((4, length)),
            "Z_t": np.zeros((16, length)),
            "S_c": np.zeros((16, 6)),
        }
        with self.assertRaisesRegex(ValueError, "requires W_c"):
            model.build_psi_data(batch, model.SynthesisParams())
        batch["W_c"] = np.eye(16)
        with self.assertRaisesRegex(ValueError, "requires.*W_E"):
            model.build_psi_data(batch, model.SynthesisParams())

    def test_discrete_time_sum_is_rectangular_rule(self):
        self.assertAlmostEqual(
            model.discrete_time_sum(np.array([1.0, 2.0, 3.0]), 0.1),
            0.6,
        )

    def test_qmi_expansion_matches_residual_gram(self):
        rng = np.random.default_rng(10)
        V = rng.standard_normal((20, 40))
        Xnext = rng.standard_normal((32, 40))
        Delta = rng.standard_normal((32, 20))
        R = rng.standard_normal((32, 32))
        R = R @ R.T
        Psi = np.block(
            [[V @ V.T, -V @ Xnext.T], [-Xnext @ V.T, Xnext @ Xnext.T - R]]
        )
        multiplier = np.hstack([Delta, np.eye(32)])
        expanded = multiplier @ Psi @ multiplier.T
        residual = Delta @ V - Xnext
        np.testing.assert_allclose(expanded, residual @ residual.T - R, atol=1e-10)

    def test_structural_output_zero_block_preserves_qmi_feasibility(self):
        rng = np.random.default_rng(101)
        regressor = rng.standard_normal((20, 50))
        state_input = rng.standard_normal((16, 20))
        output_matrix = rng.standard_normal((16, 20))
        state_error = 0.01 * rng.standard_normal((16, 50))
        successor = state_input @ regressor + state_error
        output = output_matrix @ regressor
        observations = np.vstack((successor, output))
        successor_bound = state_error @ state_error.T + 0.5 * np.eye(16)
        full_bound = la.block_diag(successor_bound, np.zeros((16, 16)))
        psi = np.block(
            [
                [regressor @ regressor.T, -regressor @ observations.T],
                [
                    -observations @ regressor.T,
                    observations @ observations.T - full_bound,
                ],
            ]
        )
        delta = np.vstack((state_input, output_matrix))
        _, dynamics = dynamics_residual_matrix(
            state_input, regressor, successor, successor_bound
        )
        dynamics_score = dynamics_qmi_raw_score(delta, psi)
        full_score = full_qmi_raw_score(delta, psi)

        self.assertLess(dynamics_score, 0.0)
        self.assertAlmostEqual(dynamics_score, la.eigvalsh(dynamics)[-1])
        self.assertAlmostEqual(full_score, 0.0, places=10)
        self.assertAlmostEqual(full_score, max(dynamics_score, 0.0), places=10)
        self.assertAlmostEqual(
            model.compute_vertex_score_scalar(delta, psi), dynamics_score
        )
        self.assertAlmostEqual(
            model.compute_vertex_score_scalar(
                delta, psi, mode="full_qmi_lambda_max"
            ),
            full_score,
        )

    def test_zero_score_removes_surrogate_slack_exactly(self):
        rng = np.random.default_rng(11)
        A = 0.8 * np.eye(16)
        B = rng.standard_normal((16, 4)) * 0.01
        C = np.eye(16)
        D = np.zeros((16, 4))
        S = rng.standard_normal((16, 6)) * 0.01
        vertex = {"A": A, "B": B, "C": C, "D": D, "S": S, "s": 0.0}
        Q = np.eye(16)
        K = np.zeros((4, 16))
        v0 = model.eval_vertex_lmi_violation(vertex, K, Q, 2.0, 0.0, 0.95)
        vb = model.eval_vertex_lmi_violation(vertex, K, Q, 2.0, 50.0, 0.95)
        self.assertEqual(v0, vb)

    def test_output_and_increment_schur_complements(self):
        rng = np.random.default_rng(12)
        Q = np.diag(np.linspace(1.0, 2.0, 16))
        K = rng.standard_normal((4, 16)) * 0.02
        Y = K @ Q
        C = np.eye(16) * 0.1
        D = np.zeros((16, 4))
        H = C @ Q + D @ Y
        mu = 1.0
        output_block = np.block([[Q, H.T], [H, mu * np.eye(16)]])
        output_schur = Q - H.T @ H / mu
        self.assertEqual(la.eigvalsh(output_block)[0] >= -1e-12,
                         la.eigvalsh(output_schur)[0] >= -1e-12)

        increment_block = np.block([[Q, Y.T], [Y, np.eye(4)]])
        increment_schur = Q - Y.T @ Y
        self.assertEqual(la.eigvalsh(increment_block)[0] >= -1e-12,
                         la.eigvalsh(increment_schur)[0] >= -1e-12)

    def test_manuscript_matrix_dimensions(self):
        bounds = model.ParamBounds(
            sigma_t=(0.65, 1.10), Jx=(0.010, 0.020), Jy=(0.010, 0.020),
            Jz=(0.022, 0.038), kx=(0.08, 0.35), ky=(0.08, 0.35),
            kz=(0.08, 0.35), kp=(0.08, 0.45), kq=(0.08, 0.45),
            kr=(0.08, 0.45),
        )
        p = {name: float(np.mean(getattr(bounds, name)))
             for name in bounds.__dataclass_fields__}
        A, B, S = model.build_vertex_matrices(p, 0.1, 9.81)
        C, D = model.build_performance_matrices(model.SynthesisParams())
        Delta = np.block([[A, B], [C, D]])
        self.assertEqual(A.shape, (16, 16))
        self.assertEqual(B.shape, (16, 4))
        self.assertEqual(S.shape, (16, 6))
        self.assertEqual(C.shape, (16, 16))
        self.assertEqual(D.shape, (16, 4))
        self.assertEqual(Delta.shape, (32, 20))

    def test_stratified_selector_handles_ties_and_small_budgets(self):
        bounds = model.ParamBounds(
            sigma_t=(0.65, 1.10), Jx=(0.010, 0.020), Jy=(0.010, 0.020),
            Jz=(0.022, 0.038), kx=(0.08, 0.35), ky=(0.08, 0.35),
            kz=(0.08, 0.35), kp=(0.08, 0.45), kq=(0.08, 0.45),
            kr=(0.08, 0.45),
        )
        vertices = []
        for idx in range(6):
            p = {
                name: getattr(bounds, name)[idx % 2]
                for name in bounds.__dataclass_fields__
            }
            vertices.append({"p": p, "s": 0.0, "Delta": np.full((32, 20), idx)})
        one = model.select_vertices_stratified_farthest(vertices, bounds, 1)
        self.assertEqual(len(one), 1)
        self.assertEqual(float(one[0]["Delta"][0, 0]), 0.0)
        self.assertEqual(len(model.select_vertices_stratified_farthest(vertices, bounds, 2)), 2)

    def test_score_pipeline_returns_all_zero_when_no_positive_excess_exists(self):
        vertices = [{"s": 0.0, "p": {}} for _ in range(4)]
        processed, diagnostics = model.compute_si_from_vi(vertices, L=10)
        self.assertTrue(all(vertex["s"] == 0.0 for vertex in processed))
        self.assertTrue(diagnostics["uninformative"])
        self.assertEqual(diagnostics["n_consistent"], 4)

    def test_wc_dominates_vertices_and_batch_generator(self):
        bounds = model.ParamBounds(
            sigma_t=(0.65, 1.10), Jx=(0.010, 0.020), Jy=(0.010, 0.020),
            Jz=(0.022, 0.038), kx=(0.08, 0.35), ky=(0.08, 0.35),
            kz=(0.08, 0.35), kp=(0.08, 0.45), kq=(0.08, 0.45),
            kr=(0.08, 0.45),
        )
        generator = {
            name: float(np.mean(getattr(bounds, name)))
            for name in bounds.__dataclass_fields__
        }
        _, info = model.build_disturbance_psd_bound(
            bounds, model.SynthesisParams(), batch_generator=generator,
            n_internal_samples=5, seed=3,
        )
        self.assertGreaterEqual(info["min_psd_gap_vertex"], -1e-12)
        self.assertGreaterEqual(info["generator_psd_gap"], -1e-12)

    def test_batch_residual_contains_only_capped_recording_error(self):
        generator = {
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
        batch = model.simulate_batch_data(
            generator,
            model.SynthesisParams(),
            L=80,
            excite_scale=0.2,
            meas_noise_std=5e-4,
        )
        W_E = model.build_structured_residual_bound(80, 3e-3)
        check = model.verify_realized_record_residual(batch, W_E)
        self.assertGreaterEqual(check["minimum_psd_gap"], -1e-12)
        self.assertLessEqual(check["maximum_output_residual"], 1e-12)
        self.assertLessEqual(check["maximum_input_memory_residual"], 1e-12)
        u_memory = batch["X_phys"][12:16, :]
        lower = np.asarray(model.SynthesisParams().u_abs_min).reshape(4, 1)
        upper = np.asarray(model.SynthesisParams().u_abs_max).reshape(4, 1)
        self.assertTrue(np.all(u_memory >= lower - 1e-12))
        self.assertTrue(np.all(u_memory <= upper + 1e-12))
        np.testing.assert_allclose(
            np.diff(u_memory, axis=1), batch["U_phys"], atol=1e-12
        )
        self.assertIn("input_saturation_stats", batch)


if __name__ == "__main__":
    unittest.main()
