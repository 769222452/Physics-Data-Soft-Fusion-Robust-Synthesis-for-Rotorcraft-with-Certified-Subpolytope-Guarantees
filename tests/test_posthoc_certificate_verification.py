import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import posthoc_certificate_verification as verify  # noqa: E402


class PosthocCertificateVerificationTests(unittest.TestCase):
    def setUp(self):
        self.A = 0.2 * np.eye(16)
        self.B = np.zeros((16, 4))
        self.C = np.zeros((16, 16))
        self.D = np.zeros((16, 4))
        self.S = np.zeros((16, 6))
        self.Q = np.eye(16)
        self.Y = np.zeros((4, 16))

    def test_vertex_lmi_dimensions_and_zero_score_identity(self):
        standard = verify.build_vertex_lmi(
            self.A,
            self.B,
            self.C,
            self.D,
            self.S,
            self.Q,
            self.Y,
            2.0,
            0.95,
        )
        score_zero = verify.build_vertex_lmi(
            self.A,
            self.B,
            self.C,
            self.D,
            self.S,
            self.Q,
            self.Y,
            2.0,
            0.95,
            beta=50.0,
            score=0.0,
        )
        self.assertEqual(standard.shape, (54, 54))
        np.testing.assert_array_equal(standard, score_zero)

    def test_minimum_coefficient_uses_passing_bisection_endpoint(self):
        library = verify.VertexLibrary(
            A=(0.9 * np.eye(16))[None, :, :],
            B=self.B[None, :, :],
            C=self.C[None, :, :],
            D=self.D[None, :, :],
            S=self.S[None, :, :],
            raw_scores=np.array([-1.0]),
            processed_scores=np.array([0.0]),
        )
        solution = verify.SavedSolution(
            Q=self.Q,
            Y=self.Y,
            K=self.Y,
            beta=0.0,
            gamma2=2.0,
            mu=1.0,
            decay_rate=0.5,
            used_saved_y=True,
            resolved_keys={},
        )
        result = verify.find_minimum_common_coefficient(
            library,
            solution,
            np.array([0]),
            strict_tol=1e-8,
            bar_lambda_tol=1e-10,
            max_iterations=80,
        )
        self.assertTrue(result["found"])
        self.assertGreater(result["bar_lambda"], 0.81)
        self.assertLess(result["bar_lambda"], 0.811)
        self.assertLessEqual(result["upper_residual_final"], -1e-8)

    def test_minimum_coefficient_reports_empty_verification_set(self):
        library = verify.VertexLibrary(
            A=(0.9 * np.eye(16))[None, :, :],
            B=self.B[None, :, :],
            C=self.C[None, :, :],
            D=self.D[None, :, :],
            S=self.S[None, :, :],
            raw_scores=np.array([-1.0]),
            processed_scores=np.array([0.0]),
        )
        solution = verify.SavedSolution(
            Q=self.Q,
            Y=self.Y,
            K=self.Y,
            beta=0.0,
            gamma2=2.0,
            mu=1.0,
            decay_rate=0.5,
            used_saved_y=True,
            resolved_keys={},
        )
        result = verify.find_minimum_common_coefficient(
            library,
            solution,
            np.array([], dtype=int),
            strict_tol=1e-8,
            bar_lambda_tol=1e-10,
            max_iterations=80,
        )
        self.assertFalse(result["found"])
        self.assertEqual(result["status"], "empty verification set")
        self.assertIsNone(result["bar_lambda"])

    def test_minimum_coefficient_reports_absent_strict_solution(self):
        library = verify.VertexLibrary(
            A=(1.2 * np.eye(16))[None, :, :],
            B=self.B[None, :, :],
            C=self.C[None, :, :],
            D=self.D[None, :, :],
            S=self.S[None, :, :],
            raw_scores=np.array([1.0]),
            processed_scores=np.array([1.0]),
        )
        solution = verify.SavedSolution(
            Q=self.Q,
            Y=self.Y,
            K=self.Y,
            beta=0.0,
            gamma2=2.0,
            mu=1.0,
            decay_rate=0.5,
            used_saved_y=True,
            resolved_keys={},
        )
        result = verify.find_minimum_common_coefficient(
            library,
            solution,
            np.array([0]),
            strict_tol=1e-8,
            bar_lambda_tol=1e-10,
            max_iterations=80,
        )
        self.assertFalse(result["found"])
        self.assertIn("no common coefficient", result["status"])
        self.assertIsNone(result["bar_lambda"])

    def test_dual_run_writes_required_outputs(self):
        theta = np.block([[self.A, self.B], [self.C, self.D]])[None, :, :]
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temporary_root)) as directory:
            root = Path(directory)
            solution = root / "solution.npz"
            diagnostics = root / "diagnostics.npz"
            output = root / "posthoc" / "synthetic"
            np.savez(
                solution,
                proposed_Q=self.Q,
                proposed_Y=self.Y,
                proposed_K=self.Y,
                proposed_beta=np.array([0.0]),
                proposed_gamma2=np.array([2.0]),
                proposed_mu=np.array([1.0]),
                proposed_decay_rate=np.array([0.95]),
                baselineB_Q=self.Q,
                baselineB_Y=self.Y,
                baselineB_K=self.Y,
                baselineB_gamma2=np.array([2.0]),
                baselineB_mu=np.array([1.0]),
                baselineB_decay_rate=np.array([0.95]),
            )
            np.savez(
                diagnostics,
                vertex_delta=theta,
                vertex_disturbance_injection=self.S[None, :, :],
                raw_scores=np.array([-1.0]),
                processed_scores=np.array([0.0]),
            )
            args = argparse.Namespace(
                solution_proposed=solution,
                solution_baseline=solution,
                diagnostics=diagnostics,
                output_dir=output,
                scenario="Synthetic",
                proposed_prefix="proposed",
                baseline_prefix="baselineB",
                strict_tol=1e-8,
                bar_lambda_tol=1e-10,
                max_iterations=80,
                force=False,
            )
            summaries = verify.run(args)
            self.assertEqual(
                summaries["proposed"]["verified_intersection_count"], 1
            )
            self.assertEqual(
                summaries["baseline"]["verified_intersection_count"], 1
            )
            for name in (
                "proposed_summary.json",
                "proposed_vertex_residuals.csv",
                "baseline_summary.json",
                "baseline_vertex_residuals.csv",
                "posthoc_certificate_summary.tex",
            ):
                self.assertTrue((output / name).is_file(), name)
            written = json.loads(
                (output / "proposed_summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(Path(written["solution_file"]).is_absolute())
            self.assertTrue(written["used_saved_Y"])


if __name__ == "__main__":
    unittest.main()
