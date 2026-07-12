"""Shared MOSEK Fusion solution validation and variable extraction."""

from typing import Any, Dict

import numpy as np


def fusion_matrix_level(var: Any, rows: int, cols: int) -> np.ndarray:
    """Return a Fusion matrix variable using the tested row-major order."""
    values = np.asarray(var.level(), dtype=float).reshape(-1)
    expected = int(rows) * int(cols)
    if values.size != expected:
        raise ValueError(
            f"Unexpected Fusion level size {values.size}; expected {expected} "
            f"for a {rows} by {cols} variable"
        )
    return values.reshape((int(rows), int(cols)), order="C")


def _safe_double_info(model: Any, name: str) -> float:
    try:
        return float(model.getSolverDoubleInfo(name))
    except Exception:
        return float("nan")


def _safe_int_info(model: Any, name: str) -> int:
    try:
        return int(model.getSolverIntInfo(name))
    except Exception:
        return -1


def _safe_objective(model: Any, method_name: str) -> float:
    try:
        return float(getattr(model, method_name)())
    except Exception:
        return float("nan")


def validate_mosek_solution(
    model: Any,
    *,
    allow_feasible: bool = True,
) -> Dict[str, Any]:
    """Apply the common acceptance policy after ``Model.solve()``.

    The problem must be primal-and-dual feasible. Both primal and dual
    solution statuses must be optimal, or feasible when ``allow_feasible``
    is enabled. Unknown, infeasible, certificate, ill-posed, and undefined
    statuses are rejected before any decision-variable level is accessed.
    A posteriori LMI residual checks remain a separate required stage.
    """
    import mosek.fusion as mf

    try:
        problem_status = model.getProblemStatus()
        primal_status = model.getPrimalSolutionStatus()
        dual_status = model.getDualSolutionStatus()
    except Exception as exc:
        return {
            "accepted": False,
            "acceptance_category": "rejected_status_query",
            "problem_status": "unavailable",
            "primal_status": "unavailable",
            "dual_status": "unavailable",
            "reason": f"status query failed: {type(exc).__name__}: {exc}",
            "primal_objective": float("nan"),
            "dual_objective": float("nan"),
            "optimizer_time": float("nan"),
            "interior_point_iterations": -1,
            "primal_feasibility": float("nan"),
            "dual_feasibility": float("nan"),
            "optimize_response": -1,
        }

    accepted_solution_statuses = {mf.SolutionStatus.Optimal}
    if allow_feasible:
        accepted_solution_statuses.add(mf.SolutionStatus.Feasible)

    accepted_problem = problem_status == mf.ProblemStatus.PrimalAndDualFeasible
    accepted_primal = primal_status in accepted_solution_statuses
    accepted_dual = dual_status in accepted_solution_statuses
    accepted = accepted_problem and accepted_primal and accepted_dual

    if accepted:
        both_optimal = (
            primal_status == mf.SolutionStatus.Optimal
            and dual_status == mf.SolutionStatus.Optimal
        )
        category = "accepted_optimal" if both_optimal else "accepted_feasible"
        reason = "problem and primal/dual solution statuses satisfy the policy"
    else:
        category = "rejected_solver_status"
        reason = (
            "requires ProblemStatus.PrimalAndDualFeasible and primal/dual "
            + ("Optimal or Feasible" if allow_feasible else "Optimal")
            + " solution statuses"
        )

    return {
        "accepted": accepted,
        "acceptance_category": category,
        "problem_status": str(problem_status),
        "primal_status": str(primal_status),
        "dual_status": str(dual_status),
        "reason": reason,
        "primal_objective": _safe_objective(model, "primalObjValue")
        if accepted_primal else float("nan"),
        "dual_objective": _safe_objective(model, "dualObjValue")
        if accepted_dual else float("nan"),
        "optimizer_time": _safe_double_info(model, "optimizerTime"),
        "interior_point_iterations": _safe_int_info(model, "intpntIter"),
        "primal_feasibility": _safe_double_info(model, "intpntPrimalFeas"),
        "dual_feasibility": _safe_double_info(model, "intpntDualFeas"),
        "optimize_response": _safe_int_info(model, "optimizeResponse"),
    }


def mosek_status_payload(validation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a validation result to pickle-free diagnostic arrays."""
    return {
        "solver_problem_status": np.asarray([validation["problem_status"]]),
        "solver_primal_status": np.asarray([validation["primal_status"]]),
        "solver_dual_status": np.asarray([validation["dual_status"]]),
        "solver_acceptance_category": np.asarray(
            [validation["acceptance_category"]]
        ),
        "solver_status_reason": np.asarray([validation["reason"]]),
        "solver_primal_objective": np.asarray(
            [validation["primal_objective"]], dtype=float
        ),
        "solver_dual_objective": np.asarray(
            [validation["dual_objective"]], dtype=float
        ),
        "solver_optimizer_time": np.asarray(
            [validation["optimizer_time"]], dtype=float
        ),
        "solver_interior_point_iterations": np.asarray(
            [validation["interior_point_iterations"]], dtype=int
        ),
        "solver_primal_feasibility": np.asarray(
            [validation["primal_feasibility"]], dtype=float
        ),
        "solver_dual_feasibility": np.asarray(
            [validation["dual_feasibility"]], dtype=float
        ),
        "solver_optimize_response": np.asarray(
            [validation["optimize_response"]], dtype=int
        ),
    }


def mosek_exception_payload(
    exc: Exception,
    validation: Dict[str, Any] = None,
) -> Dict[str, np.ndarray]:
    """Build diagnostics for a solve, status, extraction, or level exception."""
    if validation is None:
        validation = {
            "accepted": False,
            "acceptance_category": "exception",
            "problem_status": "unavailable",
            "primal_status": "unavailable",
            "dual_status": "unavailable",
            "reason": "solver or status query raised an exception",
            "primal_objective": float("nan"),
            "dual_objective": float("nan"),
            "optimizer_time": float("nan"),
            "interior_point_iterations": -1,
            "primal_feasibility": float("nan"),
            "dual_feasibility": float("nan"),
            "optimize_response": -1,
        }
    failed = dict(validation)
    if validation.get("acceptance_category") == "exception":
        failed["reason"] = f"{type(exc).__name__}: {exc}"
    elif validation.get("accepted", False):
        failed["acceptance_category"] = "exception"
        failed["reason"] = f"{type(exc).__name__}: {exc}"
    else:
        failed["reason"] = (
            f"{validation.get('reason', 'solver result rejected')}; "
            "decision-variable levels were not accessed"
        )
    return mosek_status_payload(failed)
