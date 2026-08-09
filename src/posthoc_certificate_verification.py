"""Verify saved proposed and baseline controllers without resynthesis.

The verification evaluates the standard bounded-real LMI directly at every
saved vertex. It does not solve an SDP, regenerate data, run Monte Carlo, or
execute a time-domain simulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

from raw_qmi_scores import dynamics_qmi_raw_score, full_qmi_raw_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRICT_TOL_GRID = (1.0e-8, 1.0e-7, 1.0e-6)
BASELINE_DISPLAY_NAME = (
    "Matched-active-set unrelaxed ablation (full-library residual evaluated)"
)


@dataclass(frozen=True)
class SavedSolution:
    Q: np.ndarray
    Y: np.ndarray
    K: np.ndarray
    beta: float
    gamma2: float
    mu: float
    decay_rate: float
    used_saved_y: bool
    resolved_keys: Mapping[str, str]


@dataclass(frozen=True)
class VertexLibrary:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    S: np.ndarray
    raw_scores: np.ndarray
    processed_scores: np.ndarray

    @property
    def size(self) -> int:
        return int(self.A.shape[0])


def _sym(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _scalar(array: np.ndarray, name: str) -> float:
    values = np.asarray(array, dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    value = float(values[0])
    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return value


def _resolve_key(
    archive: Mapping[str, np.ndarray],
    candidates: Sequence[str],
    quantity: str,
    *,
    required: bool = True,
) -> Optional[str]:
    for candidate in candidates:
        if candidate in archive:
            return candidate
    if required:
        raise KeyError(
            f"Missing {quantity}. Tried keys: {', '.join(candidates)}"
        )
    return None


def _solution_key_candidates(prefix: str, quantity: str) -> Tuple[str, ...]:
    clean = prefix.strip("_")
    return (
        f"{clean}_{quantity}",
        f"sol_{clean}_{quantity}",
        f"{clean}_sol_{quantity}",
    )


def load_saved_solution(
    path: Path,
    prefix: str,
    *,
    default_beta: Optional[float] = None,
) -> Tuple[SavedSolution, List[str]]:
    """Load one saved controller, preferring the saved decision variable Y."""

    with np.load(path, allow_pickle=False) as archive:
        available = list(archive.files)
        resolved: Dict[str, str] = {}
        for quantity in ("Q", "K", "gamma2", "mu", "decay_rate"):
            key = _resolve_key(
                archive,
                _solution_key_candidates(prefix, quantity),
                quantity,
            )
            assert key is not None
            resolved[quantity] = key

        beta_key = _resolve_key(
            archive,
            _solution_key_candidates(prefix, "beta"),
            "beta",
            required=False,
        )
        if beta_key is None:
            if default_beta is None:
                raise KeyError(f"Missing beta for solution prefix {prefix}")
            beta = float(default_beta)
            resolved["beta"] = f"fixed at {beta:g}"
        else:
            beta = _scalar(archive[beta_key], beta_key)
            resolved["beta"] = beta_key

        y_key = _resolve_key(
            archive,
            _solution_key_candidates(prefix, "Y"),
            "Y",
            required=False,
        )
        Q = _sym(np.asarray(archive[resolved["Q"]], dtype=float))
        K = np.asarray(archive[resolved["K"]], dtype=float)
        if y_key is None:
            Y = K @ Q
            used_saved_y = False
            resolved["Y"] = "reconstructed as K@Q"
        else:
            Y = np.asarray(archive[y_key], dtype=float)
            used_saved_y = True
            resolved["Y"] = y_key

        solution = SavedSolution(
            Q=Q,
            Y=Y,
            K=K,
            beta=beta,
            gamma2=_scalar(archive[resolved["gamma2"]], resolved["gamma2"]),
            mu=_scalar(archive[resolved["mu"]], resolved["mu"]),
            decay_rate=_scalar(
                archive[resolved["decay_rate"]], resolved["decay_rate"]
            ),
            used_saved_y=used_saved_y,
            resolved_keys=resolved,
        )

    _validate_solution_dimensions(solution)
    return solution, available


def _validate_solution_dimensions(solution: SavedSolution) -> None:
    if solution.Q.shape != (16, 16):
        raise ValueError(f"Q has shape {solution.Q.shape}, expected (16, 16)")
    if solution.Y.shape != (4, 16):
        raise ValueError(f"Y has shape {solution.Y.shape}, expected (4, 16)")
    if solution.K.shape != (4, 16):
        raise ValueError(f"K has shape {solution.K.shape}, expected (4, 16)")
    if solution.beta < 0.0:
        raise ValueError("beta must be nonnegative")
    if solution.gamma2 <= 0.0:
        raise ValueError("gamma2 must be positive")
    if solution.mu <= 0.0:
        raise ValueError("mu must be positive")
    if not 0.0 < solution.decay_rate < 1.0:
        raise ValueError("decay_rate must lie in (0, 1)")


def load_vertex_library(path: Path) -> Tuple[VertexLibrary, List[str]]:
    with np.load(path, allow_pickle=False) as archive:
        available = list(archive.files)
        required = (
            "vertex_delta",
            "vertex_disturbance_injection",
            "raw_scores",
            "processed_scores",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"Missing diagnostic keys: {', '.join(missing)}")
        theta = np.asarray(archive["vertex_delta"], dtype=float)
        injections = np.asarray(
            archive["vertex_disturbance_injection"], dtype=float
        )
        raw_scores = np.asarray(archive["raw_scores"], dtype=float).reshape(-1)
        processed_scores = np.asarray(
            archive["processed_scores"], dtype=float
        ).reshape(-1)

    if theta.ndim != 3 or theta.shape[1:] != (32, 20):
        raise ValueError(
            f"vertex_delta has shape {theta.shape}, expected (N, 32, 20)"
        )
    count = int(theta.shape[0])
    if injections.shape != (count, 16, 6):
        raise ValueError(
            "vertex_disturbance_injection has shape "
            f"{injections.shape}, expected ({count}, 16, 6)"
        )
    if raw_scores.shape != (count,) or processed_scores.shape != (count,):
        raise ValueError("score arrays must have shape (N,)")
    if not all(
        np.all(np.isfinite(array))
        for array in (theta, injections, raw_scores, processed_scores)
    ):
        raise ValueError("vertex diagnostics contain nonfinite values")

    return (
        VertexLibrary(
            A=theta[:, :16, :16],
            B=theta[:, :16, 16:20],
            C=theta[:, 16:32, :16],
            D=theta[:, 16:32, 16:20],
            S=injections,
            raw_scores=raw_scores,
            processed_scores=processed_scores,
        ),
        available,
    )


def build_vertex_lmi(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    Q: np.ndarray,
    Y: np.ndarray,
    gamma2: float,
    decay_rate: float,
    *,
    beta: float = 0.0,
    score: float = 0.0,
) -> np.ndarray:
    """Construct the 54-by-54 signed bounded-real LMI matrix."""

    AQ_BY = A @ Q + B @ Y
    CQ_DY = C @ Q + D @ Y
    zero_16_6 = np.zeros((16, 6))
    zero_16 = np.zeros((16, 16))
    matrix = np.block(
        [
            [
                -decay_rate * Q - beta * score * np.eye(16),
                zero_16_6,
                AQ_BY.T,
                CQ_DY.T,
            ],
            [
                zero_16_6.T,
                -gamma2 * np.eye(6),
                S.T,
                np.zeros((6, 16)),
            ],
            [AQ_BY, S, -Q, zero_16],
            [CQ_DY, np.zeros((16, 6)), zero_16, -np.eye(16)],
        ]
    )
    if matrix.shape != (54, 54):
        raise ValueError(f"assembled vertex LMI has shape {matrix.shape}")
    return _sym(matrix)


def _residual_pair(matrix: np.ndarray) -> Tuple[float, float]:
    eigenvalues = la.eigvalsh(_sym(matrix))
    largest = float(eigenvalues[-1])
    norm_two = max(abs(float(eigenvalues[0])), abs(largest))
    return largest, largest / max(1.0, norm_two)


def largest_eigenvalue(matrix: np.ndarray) -> float:
    size = matrix.shape[0]
    return float(
        la.eigvalsh(_sym(matrix), subset_by_index=[size - 1, size - 1])[0]
    )


def vertex_residuals(
    library: VertexLibrary,
    solution: SavedSolution,
    *,
    decay_rate: float,
    score_weighted: bool,
    indices: Optional[Iterable[int]] = None,
    include_scaled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    selected = (
        np.arange(library.size, dtype=int)
        if indices is None
        else np.asarray(list(indices), dtype=int)
    )
    residuals = np.empty(selected.size, dtype=float)
    scaled = np.empty(selected.size, dtype=float) if include_scaled else None
    for position, index in enumerate(selected):
        matrix = build_vertex_lmi(
            library.A[index],
            library.B[index],
            library.C[index],
            library.D[index],
            library.S[index],
            solution.Q,
            solution.Y,
            solution.gamma2,
            decay_rate,
            beta=solution.beta if score_weighted else 0.0,
            score=(
                float(library.processed_scores[index])
                if score_weighted
                else 0.0
            ),
        )
        if include_scaled:
            residuals[position], scaled[position] = _residual_pair(matrix)
        else:
            residuals[position] = largest_eigenvalue(matrix)
    return selected, residuals, scaled


def _worst_standard_residual(
    library: VertexLibrary,
    solution: SavedSolution,
    indices: np.ndarray,
    coefficient: float,
) -> float:
    _, residuals, _ = vertex_residuals(
        library,
        solution,
        decay_rate=coefficient,
        score_weighted=False,
        indices=indices,
    )
    return float(np.max(residuals))


def find_minimum_common_coefficient(
    library: VertexLibrary,
    solution: SavedSolution,
    verification_indices: np.ndarray,
    *,
    strict_tol: float,
    bar_lambda_tol: float,
    max_iterations: int,
) -> Dict[str, object]:
    """Find the smallest common coefficient passing the selected vertices.

    The worst standard-LMI residual is nonincreasing in the coefficient because
    increasing the coefficient subtracts a positive-semidefinite block
    proportional to Q. Bisection therefore returns the passing endpoint.
    """

    lower_initial = float(solution.decay_rate)
    upper_initial = float(np.nextafter(1.0, 0.0))
    indices = np.asarray(verification_indices, dtype=int)
    if indices.size == 0:
        return {
            "found": False,
            "status": "empty verification set",
            "bar_lambda": None,
            "search_interval_initial": [lower_initial, upper_initial],
            "search_interval_final": None,
            "lower_residual_final": None,
            "upper_residual_final": None,
            "iterations": 0,
            "converged": False,
        }

    lower_residual = _worst_standard_residual(
        library, solution, indices, lower_initial
    )
    if lower_residual <= -strict_tol:
        return {
            "found": True,
            "status": "strict common coefficient found",
            "bar_lambda": lower_initial,
            "search_interval_initial": [lower_initial, upper_initial],
            "search_interval_final": [lower_initial, lower_initial],
            "lower_residual_final": lower_residual,
            "upper_residual_final": lower_residual,
            "iterations": 0,
            "converged": True,
        }

    upper_residual = _worst_standard_residual(
        library, solution, indices, upper_initial
    )
    if upper_residual > -strict_tol:
        return {
            "found": False,
            "status": (
                "no common coefficient below one gives the requested "
                "strict residual"
            ),
            "bar_lambda": None,
            "search_interval_initial": [lower_initial, upper_initial],
            "search_interval_final": [lower_initial, upper_initial],
            "lower_residual_final": float(lower_residual),
            "upper_residual_final": float(upper_residual),
            "iterations": 0,
            "converged": False,
        }

    lower = lower_initial
    upper = upper_initial
    iterations = 0
    while upper - lower > bar_lambda_tol and iterations < max_iterations:
        midpoint = 0.5 * (lower + upper)
        midpoint_residual = _worst_standard_residual(
            library, solution, indices, midpoint
        )
        if midpoint_residual <= -strict_tol:
            upper = midpoint
            upper_residual = midpoint_residual
        else:
            lower = midpoint
            lower_residual = midpoint_residual
        iterations += 1

    if upper_residual > -strict_tol:
        upper_residual = _worst_standard_residual(
            library, solution, indices, upper
        )
    return {
        "found": True,
        "status": "strict common coefficient found",
        "bar_lambda": float(upper),
        "search_interval_initial": [lower_initial, upper_initial],
        "search_interval_final": [float(lower), float(upper)],
        "lower_residual_final": float(lower_residual),
        "upper_residual_final": float(upper_residual),
        "iterations": iterations,
        "converged": bool(upper - lower <= bar_lambda_tol),
    }


def effective_decay_coefficients(
    decay_rate: float,
    beta: float,
    processed_scores: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, float]:
    p_max = float(la.eigvalsh(_sym(P))[-1])
    coefficients = (
        float(decay_rate)
        + float(beta) * np.asarray(processed_scores, dtype=float) * p_max
    )
    return coefficients, p_max


def auxiliary_margins(
    library: VertexLibrary, solution: SavedSolution
) -> Tuple[float, float, float]:
    q_min = float(la.eigvalsh(solution.Q)[0])
    increment = _sym(
        np.block(
            [
                [solution.Q, solution.Y.T],
                [solution.Y, np.eye(solution.Y.shape[0])],
            ]
        )
    )
    input_margin = float(la.eigvalsh(increment)[0])

    output_margin = math.inf
    for index in range(library.size):
        H = library.C[index] @ solution.Q + library.D[index] @ solution.Y
        block = _sym(
            np.block(
                [
                    [solution.Q, H.T],
                    [H, solution.mu * np.eye(H.shape[0])],
                ]
            )
        )
        output_margin = min(output_margin, float(la.eigvalsh(block)[0]))
    return q_min, input_margin, output_margin


def c_infinity(bar_lambda: float, gamma2: float, *, wbar: float = 1.0) -> float:
    """Persistent-disturbance level for a specified coefficient and amplitude."""

    if not 0.0 <= bar_lambda < 1.0:
        raise ValueError("bar_lambda must lie in [0, 1)")
    if gamma2 <= 0.0:
        raise ValueError("gamma2 must be positive")
    if wbar < 0.0:
        raise ValueError("wbar must be nonnegative")
    return float(gamma2 * wbar**2 / (1.0 - bar_lambda))


def matrix_rank_diagnostics(
    matrix: np.ndarray, *, tolerance: Optional[float] = None
) -> Dict[str, object]:
    """Compute SVD diagnostics using an explicit, reported rank tolerance."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix_rank_diagnostics expects a matrix")
    singular_values = la.svdvals(values)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    if tolerance is None:
        tolerance = (
            max(values.shape) * np.finfo(float).eps * sigma_max
            if sigma_max > 0.0
            else 0.0
        )
    if tolerance < 0.0:
        raise ValueError("rank tolerance must be nonnegative")
    rank = int(np.sum(singular_values > tolerance))
    sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
    condition_number = (
        float(sigma_max / sigma_min) if sigma_min > 0.0 else math.inf
    )
    return {
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "rank": rank,
        "rank_tolerance": float(tolerance),
        "singular_values": [float(value) for value in singular_values],
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition_number_2": (
            float(condition_number) if np.isfinite(condition_number) else None
        ),
        "full_row_rank": bool(rank == values.shape[0]),
        "full_column_rank": bool(rank == values.shape[1]),
    }


def raw_qmi_score(delta: np.ndarray, psi: np.ndarray) -> float:
    """Evaluate the signed successor-state raw margin for one candidate."""

    delta = np.asarray(delta, dtype=float)
    psi = np.asarray(psi, dtype=float)
    if delta.shape != (32, 20):
        raise ValueError(f"delta has shape {delta.shape}, expected (32, 20)")
    if psi.shape != (52, 52):
        raise ValueError(f"psi has shape {psi.shape}, expected (52, 52)")
    return dynamics_qmi_raw_score(delta, psi, successor_dim=16)


def _tuple_vectors(library: VertexLibrary) -> np.ndarray:
    return np.hstack(
        tuple(
            array.reshape(library.size, -1)
            for array in (library.A, library.B, library.C, library.D, library.S)
        )
    )


def affine_rank_diagnostics(library: VertexLibrary) -> Dict[str, object]:
    """Report the affine rank of the adopted normalized vertex tuples."""

    vectors = _tuple_vectors(library)
    centered = vectors[1:] - vectors[0]
    diagnostics = matrix_rank_diagnostics(centered)
    diagnostics.pop("singular_values", None)
    diagnostics["tuple_vector_dimension"] = int(vectors.shape[1])
    diagnostics["reference_vertex_index_zero_based"] = 0
    return diagnostics


def _affine_hull_relative_residual(
    vertex_vectors: np.ndarray, target_vector: np.ndarray
) -> float:
    """Distance to the affine hull, without making a convex-containment claim."""

    vertices = np.asarray(vertex_vectors, dtype=float)
    target = np.asarray(target_vector, dtype=float).reshape(-1)
    basis = (vertices[1:] - vertices[0]).T
    coefficients, _, _, _ = la.lstsq(
        basis, target - vertices[0], cond=None, lapack_driver="gelsy"
    )
    fitted = vertices[0] + basis @ coefficients
    return float(
        la.norm(target - fitted) / max(1.0, la.norm(target))
    )


def load_fixed_batch_diagnostics(
    diagnostic_path: Path,
    solution_path: Path,
    library: VertexLibrary,
) -> Dict[str, object]:
    """Read excitation and score diagnostics from the existing saved batch."""

    required = (
        "batch_X_t",
        "batch_U_t",
        "batch_A",
        "batch_B",
        "batch_S_c",
        "batch_Cc",
        "batch_Dc",
        "Psi",
        "parameter_keys",
        "generator_parameters",
        "vertex_parameters",
    )
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            return {
                "available": False,
                "missing_keys": missing,
            }
        X_t = np.asarray(archive["batch_X_t"], dtype=float)
        U_t = np.asarray(archive["batch_U_t"], dtype=float)
        psi = np.asarray(archive["Psi"], dtype=float)
        parameter_keys = [
            str(value) for value in np.asarray(archive["parameter_keys"]).tolist()
        ]
        generator_parameters = np.asarray(
            archive["generator_parameters"], dtype=float
        )
        vertex_parameters = np.asarray(
            archive["vertex_parameters"], dtype=float
        )
        generator_delta = np.block(
            [
                [
                    np.asarray(archive["batch_A"], dtype=float),
                    np.asarray(archive["batch_B"], dtype=float),
                ],
                [
                    np.asarray(archive["batch_Cc"], dtype=float),
                    np.asarray(archive["batch_Dc"], dtype=float),
                ],
            ]
        )
        generator_S = np.asarray(archive["batch_S_c"], dtype=float)
        stored_tau = (
            _scalar(archive["score_tau_effective"], "score_tau_effective")
            if "score_tau_effective" in archive
            else math.nan
        )

    with np.load(solution_path, allow_pickle=False) as archive:
        Ts = _scalar(archive["Ts"], "Ts") if "Ts" in archive else 0.1

    regressor = np.vstack((X_t, U_t))
    regressor_diagnostics = matrix_rank_diagnostics(regressor)
    affine_diagnostics = affine_rank_diagnostics(library)
    generator_raw_score = raw_qmi_score(generator_delta, psi)
    generator_full_qmi_score = full_qmi_raw_score(generator_delta, psi)

    center_values = 0.5 * (
        np.min(vertex_parameters, axis=0)
        + np.max(vertex_parameters, axis=0)
    )
    center_parameters = dict(zip(parameter_keys, center_values))
    try:
        from normalized_coordinates import (
            build_normalized_augmented_matrices,
        )

        center_A, center_B, center_S = build_normalized_augmented_matrices(
            center_parameters, Ts, 9.81
        )
        center_delta = np.block(
            [
                [center_A, center_B],
                [generator_delta[16:32, :16], generator_delta[16:32, 16:20]],
            ]
        )
        center_raw_score = raw_qmi_score(center_delta, psi)
        center_full_qmi_score = full_qmi_raw_score(center_delta, psi)
    except (ImportError, KeyError, ValueError):
        center_S = None
        center_raw_score = math.nan
        center_full_qmi_score = math.nan

    endpoint_scores = np.asarray(library.raw_scores, dtype=float)
    generator_percentile = float(
        100.0 * np.mean(endpoint_scores <= generator_raw_score)
    )
    vertex_vectors = _tuple_vectors(library)
    generator_vector = np.hstack(
        (
            generator_delta[:16, :16].reshape(-1),
            generator_delta[:16, 16:20].reshape(-1),
            generator_delta[16:32, :16].reshape(-1),
            generator_delta[16:32, 16:20].reshape(-1),
            generator_S.reshape(-1),
        )
    )
    generator_affine_residual = _affine_hull_relative_residual(
        vertex_vectors, generator_vector
    )

    return {
        "available": True,
        "batch_length": int(X_t.shape[1]),
        "regressor_dimension": int(regressor.shape[0]),
        "regressor": regressor_diagnostics,
        "affine_tuple_hull": affine_diagnostics,
        "generator_raw_score": generator_raw_score,
        "generator_full_qmi_largest_eigenvalue": generator_full_qmi_score,
        "center_raw_score": _json_number(center_raw_score),
        "center_full_qmi_largest_eigenvalue": _json_number(
            center_full_qmi_score
        ),
        "generator_endpoint_percentile": generator_percentile,
        "generator_affine_hull_relative_residual": generator_affine_residual,
        "score_tau_effective_saved": _json_number(stored_tau),
        "center_disturbance_matrix_reconstructed": bool(center_S is not None),
    }


def set_membership_statistics(
    lambda_eff: np.ndarray,
    residuals: np.ndarray,
    scores: np.ndarray,
    *,
    bar_lambda: float,
    strict_tol: float,
) -> Dict[str, object]:
    """Construct score, stability, direct, and overlap masks and counts."""

    coefficients = np.asarray(lambda_eff, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    scores = np.asarray(scores, dtype=float)
    score_mask = coefficients <= bar_lambda
    stability_mask = coefficients < 1.0
    direct_mask = residuals <= -strict_tol
    overlap_mask = score_mask & direct_mask
    positive_mask = scores > 0.0
    return {
        "score_mask": score_mask,
        "stability_mask": stability_mask,
        "direct_mask": direct_mask,
        "overlap_mask": overlap_mask,
        "N_score": int(np.sum(score_mask)),
        "N_stab": int(np.sum(stability_mask)),
        "N_vertexwise": int(np.sum(stability_mask)) - int(np.sum(score_mask)),
        "N_dir": int(np.sum(direct_mask)),
        "N_overlap": int(np.sum(overlap_mask)),
        "N_dir_minus_score": int(np.sum(direct_mask & ~score_mask)),
        "N_score_minus_dir": int(np.sum(score_mask & ~direct_mask)),
        "positive_score_N_score": int(np.sum(positive_mask & score_mask)),
        "positive_score_N_stab": int(np.sum(positive_mask & stability_mask)),
        "positive_score_N_dir": int(np.sum(positive_mask & direct_mask)),
    }


def _json_number(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def verify_solution(
    role: str,
    scenario: str,
    solution_path: Path,
    solution: SavedSolution,
    solution_keys: Sequence[str],
    diagnostic_path: Path,
    library: VertexLibrary,
    diagnostic_keys: Sequence[str],
    *,
    strict_tol: float,
    bar_lambda_tol: float,
    max_iterations: int,
    batch_diagnostics: Optional[Mapping[str, object]] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Verify one saved solution and return its summary and vertex rows."""

    q_eigenvalues = la.eigvalsh(solution.Q)
    if q_eigenvalues[0] <= 0.0:
        raise ValueError(f"{role}: saved Q is not positive definite")
    P = _sym(la.solve(solution.Q, np.eye(16), assume_a="pos"))
    lambda_eff, p_max = effective_decay_coefficients(
        solution.decay_rate,
        solution.beta,
        library.processed_scores,
        P,
    )
    anchor_mask = library.processed_scores == 0.0
    raw_mask = library.raw_scores <= 0.0
    if role == "proposed":
        selection_indices = np.flatnonzero(anchor_mask)
        selection_rule = "all zero-slack anchors"
    elif role == "baseline":
        selection_indices = np.arange(library.size, dtype=int)
        selection_rule = "all adopted vertices"
    else:
        raise ValueError(f"unknown verification role: {role}")

    search = find_minimum_common_coefficient(
        library,
        solution,
        selection_indices,
        strict_tol=strict_tol,
        bar_lambda_tol=bar_lambda_tol,
        max_iterations=max_iterations,
    )
    if not search["found"]:
        raise RuntimeError(str(search["status"]))
    bar_lambda = float(search["bar_lambda"])

    all_indices, selected_residuals, selected_scaled = vertex_residuals(
        library,
        solution,
        decay_rate=bar_lambda,
        score_weighted=False,
        include_scaled=True,
    )
    _, original_standard, original_scaled = vertex_residuals(
        library,
        solution,
        decay_rate=solution.decay_rate,
        score_weighted=False,
        include_scaled=True,
    )
    _, original_score_weighted, _ = vertex_residuals(
        library,
        solution,
        decay_rate=solution.decay_rate,
        score_weighted=(role == "proposed"),
    )
    assert selected_scaled is not None
    assert original_scaled is not None

    if solution.beta > 0.0:
        s_stab = (
            (1.0 - solution.decay_rate) / (solution.beta * p_max)
        )
        s_bar = (
            (bar_lambda - solution.decay_rate) / (solution.beta * p_max)
        )
    else:
        s_stab = math.inf
        s_bar = math.inf
    memberships = set_membership_statistics(
        lambda_eff,
        selected_residuals,
        library.processed_scores,
        bar_lambda=bar_lambda,
        strict_tol=strict_tol,
    )
    score_mask = np.asarray(memberships.pop("score_mask"), dtype=bool)
    stability_mask = np.asarray(
        memberships.pop("stability_mask"), dtype=bool
    )
    direct_mask = np.asarray(memberships.pop("direct_mask"), dtype=bool)
    overlap_mask = np.asarray(memberships.pop("overlap_mask"), dtype=bool)

    relative_y_error = float(
        la.norm(solution.Y - solution.K @ solution.Q, ord="fro")
        / max(1.0, la.norm(solution.Y, ord="fro"))
    )
    q_min, input_margin, output_margin = auxiliary_margins(library, solution)
    overlap_count = int(memberships["N_overlap"])
    direct_count = int(memberships["N_dir"])
    score_count = int(memberships["N_score"])
    positive_overlap = int(
        np.sum(overlap_mask & (library.processed_scores > 0.0))
    )
    worst_overlap = (
        float(np.max(selected_residuals[overlap_mask]))
        if overlap_count
        else math.nan
    )
    worst_overlap_scaled = (
        float(np.max(selected_scaled[overlap_mask]))
        if overlap_count
        else math.nan
    )
    worst_score = (
        float(np.max(selected_residuals[score_mask]))
        if score_count
        else math.nan
    )
    worst_score_scaled = (
        float(np.max(selected_scaled[score_mask]))
        if score_count
        else math.nan
    )
    worst_direct = (
        float(np.max(selected_residuals[direct_mask]))
        if direct_count
        else math.nan
    )
    worst_direct_scaled = (
        float(np.max(selected_scaled[direct_mask]))
        if direct_count
        else math.nan
    )
    anchor_original_worst = (
        float(np.max(original_standard[anchor_mask]))
        if np.any(anchor_mask)
        else math.nan
    )
    unit_c_infinity = c_infinity(
        bar_lambda, solution.gamma2, wbar=1.0
    )
    unit_ellipsoid_condition = bool(
        solution.gamma2 <= 1.0 - bar_lambda
    )
    controller_type = (
        "Score-weighted proposed controller"
        if role == "proposed"
        else BASELINE_DISPLAY_NAME
    )

    summary: Dict[str, object] = {
        "role": role,
        "controller_type": controller_type,
        "scenario": scenario,
        "solution_file": _portable_source_path(solution_path),
        "diagnostic_file": _portable_source_path(diagnostic_path),
        "solution_sha256": _sha256(solution_path),
        "diagnostic_sha256": _sha256(diagnostic_path),
        "solution_available_keys": list(solution_keys),
        "diagnostic_available_keys": list(diagnostic_keys),
        "resolved_solution_keys": dict(solution.resolved_keys),
        "used_saved_Y": solution.used_saved_y,
        "N": library.size,
        "N_raw": int(np.sum(raw_mask)),
        "N_anchor": int(np.sum(anchor_mask)),
        "raw_consistent_count": int(np.sum(raw_mask)),
        "anchor_count": int(np.sum(anchor_mask)),
        "coefficient_selection_set": selection_rule,
        "coefficient_selection_count": int(selection_indices.size),
        "beta": solution.beta,
        "mu": solution.mu,
        "lambda": solution.decay_rate,
        "lambda_max_P": p_max,
        "minimum_Q_eigenvalue": q_min,
        "condition_number_Q": float(np.linalg.cond(solution.Q)),
        "relative_Y_minus_KQ_error": relative_y_error,
        "gamma_opt": math.sqrt(solution.gamma2),
        "gamma_ver": math.sqrt(solution.gamma2),
        "gamma2_ver": solution.gamma2,
        "bar_lambda": bar_lambda,
        "strict_tolerance": strict_tol,
        "bar_lambda_tolerance": bar_lambda_tol,
        "maximum_bisection_iterations": max_iterations,
        "bar_lambda_search": search,
        "stability_score_threshold": _json_number(s_stab),
        "common_hull_score_threshold": _json_number(s_bar),
        "lambda_eff_min": float(np.min(lambda_eff)),
        "lambda_eff_median": float(np.median(lambda_eff)),
        "lambda_eff_max": float(np.max(lambda_eff)),
        **memberships,
        "theoretical_count": score_count,
        "direct_count": direct_count,
        "verified_intersection_count": overlap_count,
        "positive_score_verified_count": positive_overlap,
        "score_predicted_vertex_fraction": score_count / float(library.size),
        "directly_checked_vertex_fraction": direct_count / float(library.size),
        "overlap_vertex_fraction": overlap_count / float(library.size),
        "verified_vertex_fraction": direct_count / float(library.size),
        "all_anchors_pass_direct_check": bool(
            np.all(direct_mask[anchor_mask])
        ),
        "bar_lambda_less_than_one": bool(bar_lambda < 1.0),
        "worst_standard_residual_original_lambda": float(
            np.max(original_standard)
        ),
        "worst_standard_scaled_residual_original_lambda": float(
            np.max(original_scaled)
        ),
        "score_weighted_full_library_residual_original_lambda": float(
            np.max(original_score_weighted)
        ),
        "anchor_standard_residual_original_lambda": _json_number(
            anchor_original_worst
        ),
        "worst_standard_residual_at_bar_lambda_all_vertices": float(
            np.max(selected_residuals)
        ),
        "worst_standard_scaled_residual_at_bar_lambda_all_vertices": float(
            np.max(selected_scaled)
        ),
        "worst_score_predicted_residual_at_bar_lambda": _json_number(
            worst_score
        ),
        "worst_score_predicted_scaled_residual_at_bar_lambda": _json_number(
            worst_score_scaled
        ),
        "worst_direct_residual_at_bar_lambda": _json_number(worst_direct),
        "worst_direct_scaled_residual_at_bar_lambda": _json_number(
            worst_direct_scaled
        ),
        "worst_overlap_residual_at_bar_lambda": _json_number(worst_overlap),
        "worst_overlap_scaled_residual_at_bar_lambda": _json_number(
            worst_overlap_scaled
        ),
        "worst_verified_residual": _json_number(worst_overlap),
        "worst_verified_scaled_residual": _json_number(worst_overlap_scaled),
        "input_LMI_margin": input_margin,
        "output_LMI_margin": output_margin,
        "c_infinity_bar_lambda_wbar_1": unit_c_infinity,
        "unit_disturbance_c_infinity": unit_c_infinity,
        "unit_ellipsoid_invariance_condition_satisfied": (
            unit_ellipsoid_condition
        ),
        "fixed_K_polishing_used": False,
        "full_synthesis_rerun": False,
        "resynthesis_used": False,
    }
    if batch_diagnostics and bool(batch_diagnostics.get("available", False)):
        regressor = batch_diagnostics["regressor"]
        affine = batch_diagnostics["affine_tuple_hull"]
        assert isinstance(regressor, Mapping)
        assert isinstance(affine, Mapping)
        summary.update(
            {
                "affine_rank": affine["rank"],
                "affine_rank_tolerance": affine["rank_tolerance"],
                "generator_hull_residual": batch_diagnostics[
                    "generator_affine_hull_relative_residual"
                ],
                "generator_hull_residual_type": (
                    "relative distance to the affine hull; "
                    "not a convex-containment certificate"
                ),
                "regressor_rank": regressor["rank"],
                "regressor_rank_tolerance": regressor["rank_tolerance"],
                "regressor_singular_values": regressor["singular_values"],
                "regressor_sigma_min": regressor["sigma_min"],
                "regressor_sigma_max": regressor["sigma_max"],
                "regressor_condition_number": regressor[
                    "condition_number_2"
                ],
                "batch_length": batch_diagnostics["batch_length"],
                "regressor_dimension": batch_diagnostics[
                    "regressor_dimension"
                ],
                "generator_raw_score": batch_diagnostics[
                    "generator_raw_score"
                ],
                "center_raw_score": batch_diagnostics["center_raw_score"],
                "generator_endpoint_percentile": batch_diagnostics[
                    "generator_endpoint_percentile"
                ],
                "fixed_batch_diagnostics": dict(batch_diagnostics),
            }
        )

    rows: List[Dict[str, object]] = []
    for index in all_indices:
        rows.append(
            {
                "vertex_index_zero_based": int(index),
                "vertex_index_one_based": int(index) + 1,
                "raw_score": float(library.raw_scores[index]),
                "processed_score": float(library.processed_scores[index]),
                "raw_consistent": bool(raw_mask[index]),
                "zero_slack_anchor": bool(anchor_mask[index]),
                "lambda_eff": float(lambda_eff[index]),
                "in_score_predicted_set": bool(score_mask[index]),
                "in_stability_set": bool(stability_mask[index]),
                "in_direct_set": bool(direct_mask[index]),
                "in_overlap_set": bool(overlap_mask[index]),
                "in_theoretical_set": bool(score_mask[index]),
                "in_verified_intersection": bool(overlap_mask[index]),
                "score_weighted_residual_original_lambda": float(
                    original_score_weighted[index]
                ),
                "standard_residual_original_lambda": float(
                    original_standard[index]
                ),
                "standard_scaled_residual_original_lambda": float(
                    original_scaled[index]
                ),
                "standard_residual_at_bar_lambda": float(
                    selected_residuals[index]
                ),
                "standard_scaled_residual_at_bar_lambda": float(
                    selected_scaled[index]
                ),
            }
        )
    return summary, rows


def _ensure_writable(paths: Iterable[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"{path} exists; use --force to replace it")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_tex(value: object, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if not np.isfinite(number):
        return "--"
    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude < 1.0e-3 or magnitude >= 1.0e4):
        mantissa, exponent = f"{number:.2e}".split("e")
        return (
            r"\ensuremath{"
            + mantissa
            + r"\times 10^{"
            + str(int(exponent))
            + "}}"
        )
    return f"{number:.{digits}f}"


def _scenario_tex_lines(
    proposed: Mapping[str, object],
    baseline: Mapping[str, object],
) -> List[str]:
    return [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Diagnostic & Proposed & Matched-active ablation \\",
        r"\midrule",
        f"$\\bar\\lambda$ & {_format_tex(proposed['bar_lambda'], 7)} & "
        f"{_format_tex(baseline['bar_lambda'], 7)} \\\\",
        f"$N_{{\\rm score}}$ & {proposed['N_score']} & "
        f"{baseline['N_score']} \\\\",
        f"$N_{{\\rm stab}}$ & {proposed['N_stab']} & "
        f"{baseline['N_stab']} \\\\",
        f"$N_{{\\rm dir}}$ & {proposed['N_dir']} & "
        f"{baseline['N_dir']} \\\\",
        f"$N_{{\\rm overlap}}$ & {proposed['N_overlap']} & "
        f"{baseline['N_overlap']} \\\\",
        f"$\\gamma_{{\\rm opt}}=\\gamma_{{\\rm ver}}$ & "
        f"{_format_tex(proposed['gamma_ver'], 3)} & "
        f"{_format_tex(baseline['gamma_ver'], 3)} \\\\",
        f"$\\lambda_{{\\min}}(Q)$ & "
        f"{_format_tex(proposed['minimum_Q_eigenvalue'])} & "
        f"{_format_tex(baseline['minimum_Q_eigenvalue'])} \\\\",
        f"$\\kappa_2(Q)$ & {_format_tex(proposed['condition_number_Q'], 2)} & "
        f"{_format_tex(baseline['condition_number_Q'], 2)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]


def write_case_outputs(
    output_dir: Path,
    proposed: Mapping[str, object],
    proposed_rows: Sequence[Mapping[str, object]],
    baseline: Mapping[str, object],
    baseline_rows: Sequence[Mapping[str, object]],
    tolerance_rows: Sequence[Mapping[str, object]],
    batch_diagnostics: Mapping[str, object],
    *,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "proposed_json": output_dir / "proposed_summary.json",
        "proposed_summary_csv": output_dir / "proposed_summary.csv",
        "proposed_csv": output_dir / "proposed_vertex_residuals.csv",
        "baseline_json": output_dir / "baseline_summary.json",
        "baseline_summary_csv": output_dir / "baseline_summary.csv",
        "baseline_csv": output_dir / "baseline_vertex_residuals.csv",
        "combined_json": output_dir / "posthoc_certificate_summary.json",
        "tolerance_csv": output_dir / "tolerance_comparison.csv",
        "regressor_json": output_dir / "regressor_diagnostics.json",
        "tex": output_dir / "posthoc_certificate_summary.tex",
    }
    _ensure_writable(paths.values(), force)
    _write_json(paths["proposed_json"], proposed)
    _write_csv(
        paths["proposed_summary_csv"],
        [
            {
                key: value
                for key, value in proposed.items()
                if isinstance(value, (str, bool, int, float)) or value is None
            }
        ],
    )
    _write_csv(paths["proposed_csv"], proposed_rows)
    _write_json(paths["baseline_json"], baseline)
    _write_csv(
        paths["baseline_summary_csv"],
        [
            {
                key: value
                for key, value in baseline.items()
                if isinstance(value, (str, bool, int, float)) or value is None
            }
        ],
    )
    _write_csv(paths["baseline_csv"], baseline_rows)
    _write_csv(paths["tolerance_csv"], tolerance_rows)
    _write_json(paths["regressor_json"], batch_diagnostics)
    _write_json(
        paths["combined_json"],
        {"proposed": proposed, "baseline": baseline},
    )
    paths["tex"].write_text(
        "\n".join(_scenario_tex_lines(proposed, baseline)),
        encoding="utf-8",
    )


def _load_case_summaries(parent: Path) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    for directory in sorted(path for path in parent.iterdir() if path.is_dir()):
        proposed_path = directory / "proposed_summary.json"
        baseline_path = directory / "baseline_summary.json"
        if proposed_path.exists() and baseline_path.exists():
            proposed = json.loads(
                proposed_path.read_text(encoding="utf-8")
            )
            baseline = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
            required = {"N_raw", "N_score", "N_stab", "N_dir", "N_overlap"}
            if not required.issubset(proposed) or not required.issubset(
                baseline
            ):
                continue
            cases.append(
                {
                    "scenario": proposed["scenario"],
                    "proposed": proposed,
                    "baseline": baseline,
                }
            )
    order = {"Standard": 0, "Expanded": 1}
    cases.sort(key=lambda item: order.get(str(item["scenario"]), 99))
    return cases


def _combined_tex(cases: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Certificate-domain statistics obtained by post-processing the "
        r"saved controller solutions.}",
        r"\label{tab:posthoc_certificate_sets}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        r"Scenario & Controller & $N$ & $N_{\rm raw}$ & $N_{\rm anc}$ & "
        r"$\beta$ & $\lambda$ & $\bar\lambda$ & $N_{\rm score}$ & "
        r"$N_{\rm stab}$ & $N_{\rm dir}$ & $N_{\rm overlap}$ \\",
        r"\midrule",
    ]
    for case in cases:
        for key, label in (
            ("proposed", "Proposed"),
            ("baseline", "Matched-active-set ablation"),
        ):
            item = case[key]
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & "
                "{} \\\\".format(
                    item["scenario"],
                    label,
                    item["N"],
                    item["N_raw"],
                    item["N_anchor"],
                    _format_tex(item["beta"], 3),
                    _format_tex(item["lambda"], 3),
                    _format_tex(item["bar_lambda"], 7),
                    item["N_score"],
                    item["N_stab"],
                    item["N_dir"],
                    item["N_overlap"],
                )
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{tabular}{llrrrrrrrrrr}",
            r"\toprule",
            r"Scenario & Controller & $N_{\rm vtx}$ & "
            r"$N_{\rm dir\setminus score}$ & $N_{\rm score\setminus dir}$ & "
            r"$N_{\rm score}^{+}$ & $N_{\rm stab}^{+}$ & "
            r"$\lambda_{\rm eff}^{\min}$ & "
            r"$\operatorname{med}(\lambda_{\rm eff})$ & "
            r"$\lambda_{\rm eff}^{\max}$ & Direct fraction & Affine rank \\",
            r"\midrule",
        ]
    )
    for case in cases:
        for key, label in (
            ("proposed", "Proposed"),
            ("baseline", "Matched-active-set ablation"),
        ):
            item = case[key]
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & "
                "{} \\\\".format(
                    item["scenario"],
                    label,
                    item["N_vertexwise"],
                    item["N_dir_minus_score"],
                    item["N_score_minus_dir"],
                    item["positive_score_N_score"],
                    item["positive_score_N_stab"],
                    _format_tex(item["lambda_eff_min"], 4),
                    _format_tex(item["lambda_eff_median"], 4),
                    _format_tex(item["lambda_eff_max"], 4),
                    _format_tex(item["directly_checked_vertex_fraction"], 4),
                    item.get("affine_rank", "--"),
                )
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\scriptsize $N_{\rm score}$ counts the score-predicted sufficient "
            r"set, $N_{\rm dir}$ counts the negative-residual directly checked "
            r"set, and $N_{\rm overlap}$ counts their intersection. "
            r"$N_{\rm vtx}=N_{\rm stab}-N_{\rm score}$ counts vertices for which "
            r"Proposition~1 gives only a vertexwise strict-decay condition. "
            r"The superscript $+$ denotes positive processed scores. Direct "
            r"fractions are vertex-count fractions, not geometric volumes. "
            r"The matched-active-set ablation reuses the proposed active set "
            r"and is then evaluated over the full vertex library.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Numerical standard-LMI diagnostics from the fixed saved "
            r"controller solutions.}",
            r"\label{tab:posthoc_controller_diagnostics}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{2.4pt}",
            r"\begin{tabular}{llrrrrrrrr}",
            r"\toprule",
            r"Scenario & Controller & $\bar\lambda$ & $\lambda_{\min}(Q)$ & "
            r"$\kappa_2(Q)$ & $\mu$ & $\gamma_{\rm opt}$ & "
            r"$r_{\lambda}^{\rm all}$ & "
            r"$(r_{\lambda}^{\rm all})^{\rm sc}$ & $e_Y$ \\",
            r"\midrule",
        ]
    )
    for case in cases:
        for key, label in (
            ("proposed", "Proposed"),
            ("baseline", "Matched-active-set ablation"),
        ):
            item = case[key]
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                    item["scenario"],
                    label,
                    _format_tex(item["bar_lambda"], 7),
                    _format_tex(item["minimum_Q_eigenvalue"]),
                    _format_tex(item["condition_number_Q"], 2),
                    _format_tex(item["mu"], 3),
                    _format_tex(item["gamma_ver"], 3),
                    _format_tex(
                        item["worst_standard_residual_original_lambda"]
                    ),
                    _format_tex(
                        item[
                            "worst_standard_scaled_residual_original_lambda"
                        ]
                    ),
                    _format_tex(item["relative_Y_minus_KQ_error"]),
                )
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{tabular}{llrrrrrrrr}",
            r"\toprule",
            r"Scenario & Controller & $r_{\bar\lambda}^{\rm all}$ & "
            r"$(r_{\bar\lambda}^{\rm all})^{\rm sc}$ & "
            r"$r_{\bar\lambda}^{\rm score}$ & "
            r"$r_{\bar\lambda}^{\rm dir}$ & $m_u$ & $m_z$ & "
            r"$c_\infty(\bar\lambda;\bar w=1)$ & $\varepsilon_{\rm strict}$ \\",
            r"\midrule",
        ]
    )
    for case in cases:
        for key, label in (
            ("proposed", "Proposed"),
            ("baseline", "Matched-active-set ablation"),
        ):
            item = case[key]
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                    item["scenario"],
                    label,
                    _format_tex(
                        item[
                            "worst_standard_residual_at_bar_lambda_all_vertices"
                        ]
                    ),
                    _format_tex(
                        item[
                            "worst_standard_scaled_residual_at_bar_lambda_all_vertices"
                        ]
                    ),
                    _format_tex(
                        item["worst_score_predicted_residual_at_bar_lambda"]
                    ),
                    _format_tex(item["worst_direct_residual_at_bar_lambda"]),
                    _format_tex(item["input_LMI_margin"]),
                    _format_tex(item["output_LMI_margin"]),
                    _format_tex(item["c_infinity_bar_lambda_wbar_1"], 2),
                    _format_tex(item["strict_tolerance"]),
                )
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\scriptsize The superscript ``all'' denotes maximization over all "
            r"$N$ adopted vertices. The scaled residual uses "
            r"$\lambda_{\max}(\mathcal M_i)/\max\{1,\|\mathcal M_i\|_2\}$. "
            r"The direct-set values are double-precision negative-residual "
            r"checks and do not replace exact semidefinite feasibility in the "
            r"theorems. Here $\gamma_{\rm opt}=\gamma_{\rm ver}$ because the "
            r"saved decision variable is checked without fixed-gain polishing. "
            r"The unit-level invariance condition is "
            r"$\gamma_{\rm ver}^2\le1-\bar\lambda$.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_combined_outputs(parent: Path, *, force: bool) -> None:
    cases = _load_case_summaries(parent)
    if not cases:
        return
    json_path = parent / "posthoc_certificate_summary.json"
    tex_path = parent / "posthoc_certificate_summary.tex"
    sets_path = parent / "posthoc_certificate_sets_table.tex"
    diagnostics_path = parent / "posthoc_controller_diagnostics_table.tex"
    _ensure_writable(
        (json_path, tex_path, sets_path, diagnostics_path),
        force,
    )
    _write_json(json_path, {"scenarios": cases})
    combined = _combined_tex(cases)
    tex_path.write_text(combined, encoding="utf-8")
    split_marker = "\\begin{table*}[t]"
    parts = combined.split(split_marker)
    if len(parts) == 3:
        sets_path.write_text(
            parts[0] + split_marker + parts[1] + "\n",
            encoding="utf-8",
        )
        diagnostics_path.write_text(
            split_marker + parts[2],
            encoding="utf-8",
        )
    else:
        sets_path.write_text(combined, encoding="utf-8")
        diagnostics_path.write_text(
            "% See posthoc_certificate_summary.tex\n",
            encoding="utf-8",
        )


def _tolerance_row(summary: Mapping[str, object]) -> Dict[str, object]:
    return {
        "scenario": summary["scenario"],
        "controller_type": summary["controller_type"],
        "role": summary["role"],
        "strict_tolerance": summary["strict_tolerance"],
        "bar_lambda": summary["bar_lambda"],
        "bar_lambda_less_than_one": summary["bar_lambda_less_than_one"],
        "N_score": summary["N_score"],
        "N_stab": summary["N_stab"],
        "N_dir": summary["N_dir"],
        "N_overlap": summary["N_overlap"],
        "N_dir_minus_score": summary["N_dir_minus_score"],
        "N_score_minus_dir": summary["N_score_minus_dir"],
        "worst_direct_residual": summary[
            "worst_direct_residual_at_bar_lambda"
        ],
        "worst_direct_scaled_residual": summary[
            "worst_direct_scaled_residual_at_bar_lambda"
        ],
        "worst_all_vertex_residual": summary[
            "worst_standard_residual_at_bar_lambda_all_vertices"
        ],
        "worst_all_vertex_scaled_residual": summary[
            "worst_standard_scaled_residual_at_bar_lambda_all_vertices"
        ],
        "all_anchors_pass": summary["all_anchors_pass_direct_check"],
        "condition_number_Q": summary["condition_number_Q"],
    }


def run(args: argparse.Namespace) -> Mapping[str, object]:
    proposed_path = args.solution_proposed.resolve()
    baseline_path = args.solution_baseline.resolve()
    diagnostic_path = args.diagnostics.resolve()
    output_dir = args.output_dir.resolve()
    for path in (proposed_path, baseline_path, diagnostic_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    proposed, proposed_keys = load_saved_solution(
        proposed_path,
        args.proposed_prefix,
    )
    baseline, baseline_keys = load_saved_solution(
        baseline_path,
        args.baseline_prefix,
        default_beta=0.0,
    )
    library, diagnostic_keys = load_vertex_library(diagnostic_path)
    batch_diagnostics = load_fixed_batch_diagnostics(
        diagnostic_path, proposed_path, library
    )

    proposed_summary, proposed_rows = verify_solution(
        "proposed",
        args.scenario,
        proposed_path,
        proposed,
        proposed_keys,
        diagnostic_path,
        library,
        diagnostic_keys,
        strict_tol=args.strict_tol,
        bar_lambda_tol=args.bar_lambda_tol,
        max_iterations=args.max_iterations,
        batch_diagnostics=batch_diagnostics,
    )
    baseline_summary, baseline_rows = verify_solution(
        "baseline",
        args.scenario,
        baseline_path,
        baseline,
        baseline_keys,
        diagnostic_path,
        library,
        diagnostic_keys,
        strict_tol=args.strict_tol,
        bar_lambda_tol=args.bar_lambda_tol,
        max_iterations=args.max_iterations,
        batch_diagnostics=batch_diagnostics,
    )
    tolerance_grid = list(
        dict.fromkeys(
            float(value)
            for value in getattr(
                args, "strict_tol_grid", DEFAULT_STRICT_TOL_GRID
            )
        )
    )
    if args.strict_tol not in tolerance_grid:
        tolerance_grid.append(float(args.strict_tol))
        tolerance_grid.sort()
    tolerance_rows: List[Dict[str, object]] = []
    for tolerance in tolerance_grid:
        for role, solution, solution_keys, primary in (
            ("proposed", proposed, proposed_keys, proposed_summary),
            ("baseline", baseline, baseline_keys, baseline_summary),
        ):
            if tolerance == args.strict_tol:
                candidate = primary
            else:
                candidate, _ = verify_solution(
                    role,
                    args.scenario,
                    proposed_path if role == "proposed" else baseline_path,
                    solution,
                    solution_keys,
                    diagnostic_path,
                    library,
                    diagnostic_keys,
                    strict_tol=tolerance,
                    bar_lambda_tol=args.bar_lambda_tol,
                    max_iterations=args.max_iterations,
                )
            tolerance_rows.append(_tolerance_row(candidate))
    write_case_outputs(
        output_dir,
        proposed_summary,
        proposed_rows,
        baseline_summary,
        baseline_rows,
        tolerance_rows,
        batch_diagnostics,
        force=args.force,
    )
    write_combined_outputs(output_dir.parent, force=args.force)
    return {"proposed": proposed_summary, "baseline": baseline_summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-proposed", type=Path, required=True)
    parser.add_argument("--solution-baseline", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--proposed-prefix", default="proposed")
    parser.add_argument("--baseline-prefix", default="baselineB")
    parser.add_argument("--strict-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--strict-tol-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_STRICT_TOL_GRID),
        metavar="TOL",
        help="strict residual tolerances included in the comparison CSV",
    )
    parser.add_argument("--bar-lambda-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing outputs in the selected output tree",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.strict_tol <= 0.0:
        parser.error("--strict-tol must be positive")
    if any(value <= 0.0 for value in args.strict_tol_grid):
        parser.error("--strict-tol-grid values must be positive")
    if args.bar_lambda_tol <= 0.0:
        parser.error("--bar-lambda-tol must be positive")
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    try:
        summaries = run(args)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        parser.error(str(error))
    for role in ("proposed", "baseline"):
        summary = summaries[role]
        print(
            f"{summary['scenario']} {role}: "
            f"bar_lambda={summary['bar_lambda']:.10f}, "
            f"N_score/N_stab/N_dir/N_overlap="
            f"{summary['N_score']}/"
            f"{summary['N_stab']}/"
            f"{summary['N_dir']}/"
            f"{summary['N_overlap']}, "
            f"worst direct="
            f"{summary['worst_direct_residual_at_bar_lambda']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
