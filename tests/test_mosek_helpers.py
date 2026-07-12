import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.linalg as la


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from mosek_helpers import (  # noqa: E402
    fusion_matrix_level,
    mosek_exception_payload,
    validate_mosek_solution,
)


class MosekHelperTests(unittest.TestCase):
    def test_exception_payload_without_status_is_pickle_free(self):
        payload = mosek_exception_payload(RuntimeError("test failure"))
        self.assertEqual(payload["solver_acceptance_category"][0], "exception")
        self.assertIn("RuntimeError", payload["solver_status_reason"][0])
        self.assertTrue(all(value.dtype.kind != "O" for value in payload.values()))

    def test_matrix_extraction_matches_indexed_entries(self):
        try:
            import mosek.fusion as mf
        except ImportError as exc:
            self.skipTest(str(exc))

        q_target = np.array(
            [[2.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 1.2]]
        )
        y_target = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model = mf.Model("matrix_extraction_test")
        try:
            q_var = model.variable("Q", mf.Domain.inPSDCone(3))
            y_var = model.variable("Y", [2, 3], mf.Domain.unbounded())
            for i in range(3):
                for j in range(3):
                    model.constraint(
                        q_var.index(i, j),
                        mf.Domain.equalsTo(float(q_target[i, j])),
                    )
            for i in range(2):
                for j in range(3):
                    model.constraint(
                        y_var.index(i, j),
                        mf.Domain.equalsTo(float(y_target[i, j])),
                    )
            model.objective(mf.ObjectiveSense.Minimize, mf.Expr.sum(q_var))
            model.solve()

            status = validate_mosek_solution(model, allow_feasible=True)
            self.assertTrue(status["accepted"], status["reason"])
            self.assertEqual(status["acceptance_category"], "accepted_optimal")

            q_value = fusion_matrix_level(q_var, 3, 3)
            y_value = fusion_matrix_level(y_var, 2, 3)
            q_indexed = np.array(
                [
                    [float(q_var.index(i, j).level()[0]) for j in range(3)]
                    for i in range(3)
                ]
            )
            y_indexed = np.array(
                [
                    [float(y_var.index(i, j).level()[0]) for j in range(3)]
                    for i in range(2)
                ]
            )

            np.testing.assert_allclose(q_value, q_indexed, atol=1e-12)
            np.testing.assert_allclose(y_value, y_indexed, atol=1e-12)
            np.testing.assert_allclose(q_value, q_value.T, atol=1e-12)
            self.assertGreater(la.eigvalsh(q_value)[0], 0.0)
            k_extracted = la.solve(q_value.T, y_value.T).T
            k_indexed = la.solve(q_indexed.T, y_indexed.T).T
            np.testing.assert_allclose(k_extracted, k_indexed, atol=1e-12)
        finally:
            model.dispose()

    def test_infeasible_solution_is_rejected(self):
        try:
            import mosek.fusion as mf
        except ImportError as exc:
            self.skipTest(str(exc))

        model = mf.Model("status_rejection_test")
        try:
            x = model.variable("x", mf.Domain.unbounded())
            model.constraint(x, mf.Domain.greaterThan(1.0))
            model.constraint(x, mf.Domain.lessThan(0.0))
            model.objective(mf.ObjectiveSense.Minimize, x)
            model.solve()
            status = validate_mosek_solution(model, allow_feasible=True)
            self.assertFalse(status["accepted"])
            self.assertEqual(status["acceptance_category"], "rejected_solver_status")
        finally:
            model.dispose()


if __name__ == "__main__":
    unittest.main()
