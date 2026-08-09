"""Residual matrices and signed margins for the data-consistency QMI."""

from __future__ import annotations

import numpy as np


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a real square matrix."""

    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    return 0.5 * (value + value.T)


def full_qmi_residual(delta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Evaluate the complete composite QMI residual for one candidate."""

    candidate = np.asarray(delta, dtype=float)
    multiplier = np.asarray(psi, dtype=float)
    if candidate.ndim != 2:
        raise ValueError("delta must be a matrix")
    output_dim, regressor_dim = candidate.shape
    expected = regressor_dim + output_dim
    if multiplier.shape != (expected, expected):
        raise ValueError(
            f"psi must have shape {(expected, expected)}, got {multiplier.shape}"
        )
    selector = np.hstack((candidate, np.eye(output_dim)))
    return symmetrize(selector @ multiplier @ selector.T)


def dynamics_qmi_residual(
    delta: np.ndarray,
    psi: np.ndarray,
    successor_dim: int = 16,
) -> np.ndarray:
    """Return the successor-state principal block of the complete residual."""

    full = full_qmi_residual(delta, psi)
    if not 0 < successor_dim <= full.shape[0]:
        raise ValueError("successor_dim is incompatible with the QMI residual")
    return symmetrize(full[:successor_dim, :successor_dim])


def dynamics_residual_matrix(
    state_input_matrix: np.ndarray,
    regressor: np.ndarray,
    successor: np.ndarray,
    successor_bound: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct R_x and sym(R_x R_x^T - R_x_bound) directly."""

    model = np.asarray(state_input_matrix, dtype=float)
    data = np.asarray(regressor, dtype=float)
    next_state = np.asarray(successor, dtype=float)
    bound = np.asarray(successor_bound, dtype=float)
    if model.ndim != 2 or data.ndim != 2 or next_state.ndim != 2:
        raise ValueError("model, regressor, and successor must be matrices")
    if model.shape[1] != data.shape[0]:
        raise ValueError("state_input_matrix and regressor dimensions do not match")
    if model.shape[0] != next_state.shape[0] or data.shape[1] != next_state.shape[1]:
        raise ValueError("successor dimensions do not match the candidate prediction")
    if bound.shape != (next_state.shape[0], next_state.shape[0]):
        raise ValueError("successor_bound has an incompatible shape")
    residual = next_state - model @ data
    matrix = symmetrize(residual @ residual.T - bound)
    return residual, matrix


def aggregate_residual_bound(
    disturbance_bound: np.ndarray,
    recording_bound: np.ndarray,
    batch_length: int,
) -> np.ndarray:
    """Construct the complete aggregate residual bound used in the QMI."""

    w_c = np.asarray(disturbance_bound, dtype=float)
    w_e = np.asarray(recording_bound, dtype=float)
    if w_c.ndim != 2 or w_c.shape[0] != w_c.shape[1]:
        raise ValueError("disturbance_bound must be square")
    state_dim = w_c.shape[0]
    if w_e.shape != (2 * state_dim, 2 * state_dim):
        raise ValueError("recording_bound has an incompatible shape")
    if batch_length < 0:
        raise ValueError("batch_length must be nonnegative")
    disturbance = np.block(
        [
            [batch_length * w_c, np.zeros((state_dim, state_dim))],
            [np.zeros((state_dim, state_dim)), np.zeros((state_dim, state_dim))],
        ]
    )
    return symmetrize(2.0 * disturbance + 2.0 * w_e)


def largest_eigenvalue(matrix: np.ndarray) -> float:
    """Return the largest eigenvalue after explicit symmetrization."""

    return float(np.linalg.eigvalsh(symmetrize(matrix))[-1])


def full_qmi_raw_score(delta: np.ndarray, psi: np.ndarray) -> float:
    """Largest eigenvalue of the complete composite QMI residual."""

    return largest_eigenvalue(full_qmi_residual(delta, psi))


def dynamics_qmi_raw_score(
    delta: np.ndarray,
    psi: np.ndarray,
    successor_dim: int = 16,
) -> float:
    """Signed raw mismatch margin from the successor-state residual block."""

    return largest_eigenvalue(
        dynamics_qmi_residual(delta, psi, successor_dim=successor_dim)
    )
