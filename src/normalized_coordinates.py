"""Dimensionally consistent coordinates shared by all manuscript simulations.

The physical plant remains in SI-compatible units.  Synthesis, data scoring,
and mixed-channel norm constraints use fixed dimensionless coordinates.  The
scales are prescribed from operating ranges and actuator/disturbance limits;
they are never estimated from the batch used for scoring.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la


EPS_PASS = 1e-4
EPS_NEAR = 2e-4


@dataclass(frozen=True)
class ScaleConfig:
    """Fixed physical scales used to define dimensionless coordinates."""

    # x = [position, velocity, Euler angle, body rate]
    state: Tuple[float, ...] = (
        1.0, 1.0, 1.0,       # m
        1.0, 1.0, 1.0,       # m/s
        0.25, 0.25, 0.25,    # rad
        1.0, 1.0, 1.0,       # rad/s
    )
    # Previous applied input stored in the augmented state: N, N m, N m, N m.
    input_memory: Tuple[float, ...] = (6.0, 4.0, 4.0, 4.0)
    # Applied input increment: N, N m, N m, N m.
    increment: Tuple[float, ...] = (3.5, 3.5, 3.5, 3.5)
    # Translational acceleration (first three) and angular acceleration.
    disturbance: Tuple[float, ...] = (2.4, 2.4, 2.4, 2.4, 2.4, 2.4)
    # Per-channel physical residual scales in m, m/s, rad, and rad/s.
    residual_cap: Tuple[float, ...] = (
        3e-3, 3e-3, 3e-3,
        3e-3, 3e-3, 3e-3,
        3e-3, 3e-3, 3e-3,
        3e-3, 3e-3, 3e-3,
    )
    residual_std: Tuple[float, ...] = (
        5e-4, 5e-4, 5e-4,
        5e-4, 5e-4, 5e-4,
        5e-4, 5e-4, 5e-4,
        5e-4, 5e-4, 5e-4,
    )

    def __post_init__(self) -> None:
        lengths = {
            "state": (self.state, 12),
            "input_memory": (self.input_memory, 4),
            "increment": (self.increment, 4),
            "disturbance": (self.disturbance, 6),
            "residual_cap": (self.residual_cap, 12),
            "residual_std": (self.residual_std, 12),
        }
        for name, (values, expected) in lengths.items():
            if len(values) != expected:
                raise ValueError(f"{name} must contain {expected} scales")
            if np.any(np.asarray(values, dtype=float) <= 0.0):
                raise ValueError(f"{name} scales must be strictly positive")

    @property
    def T_x(self) -> np.ndarray:
        return np.diag(np.asarray(self.state, dtype=float))

    @property
    def T_u(self) -> np.ndarray:
        return np.diag(np.asarray(self.input_memory, dtype=float))

    @property
    def T_du(self) -> np.ndarray:
        return np.diag(np.asarray(self.increment, dtype=float))

    @property
    def T_d(self) -> np.ndarray:
        return np.diag(np.asarray(self.disturbance, dtype=float))

    @property
    def T_res(self) -> np.ndarray:
        return np.diag(np.asarray(self.residual_cap, dtype=float))

    @property
    def T_xc(self) -> np.ndarray:
        return la.block_diag(self.T_x, self.T_u)

    def as_dict(self) -> Dict[str, Sequence[float]]:
        return {
            "state": list(self.state),
            "input_memory": list(self.input_memory),
            "increment": list(self.increment),
            "disturbance": list(self.disturbance),
            "residual_cap": list(self.residual_cap),
            "residual_std": list(self.residual_std),
        }


DEFAULT_SCALES = ScaleConfig()


def E(k: float, Ts: float) -> float:
    return float(np.exp(-k * Ts))


def S(k: float, Ts: float) -> float:
    return float((1.0 - np.exp(-k * Ts)) / k)


def eta(k: float, Ts: float) -> float:
    return float(Ts / k - (1.0 - np.exp(-k * Ts)) / (k ** 2))


def build_physical_augmented_matrices(
    p: Dict[str, float], Ts: float, g: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 16-state augmented model in physical coordinates."""

    sigma_t = p["sigma_t"]
    Jx, Jy, Jz = p["Jx"], p["Jy"], p["Jz"]
    kx, ky, kz = p["kx"], p["ky"], p["kz"]
    kp, kq, kr = p["kp"], p["kq"], p["kr"]

    E_v = np.diag([E(kx, Ts), E(ky, Ts), E(kz, Ts)])
    S_v = np.diag([S(kx, Ts), S(ky, Ts), S(kz, Ts)])
    eta_v = np.diag([eta(kx, Ts), eta(ky, Ts), eta(kz, Ts)])
    E_om = np.diag([E(kp, Ts), E(kq, Ts), E(kr, Ts)])
    S_om = np.diag([S(kp, Ts), S(kq, Ts), S(kr, Ts)])
    eta_om = np.diag([eta(kp, Ts), eta(kq, Ts), eta(kr, Ts)])

    G_eta = np.array([
        [0.0, g * eta(kx, Ts), 0.0],
        [-g * eta(ky, Ts), 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    G_S = np.array([
        [0.0, g * S(kx, Ts), 0.0],
        [-g * S(ky, Ts), 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    Phi_eta_T = np.zeros((3, 4))
    Phi_eta_T[2, 0] = sigma_t * eta(kz, Ts)
    Phi_S_T = np.zeros((3, 4))
    Phi_S_T[2, 0] = sigma_t * S(kz, Ts)

    Jinv = np.diag([1.0 / Jx, 1.0 / Jy, 1.0 / Jz])
    Psi_eta_tau = np.zeros((3, 4))
    Psi_eta_tau[:, 1:] = Jinv @ eta_om
    Psi_S_tau = np.zeros((3, 4))
    Psi_S_tau[:, 1:] = Jinv @ S_om

    S_d = np.zeros((12, 6))
    S_d[0:3, 0:3] = eta_v
    S_d[3:6, 0:3] = S_v
    S_d[6:9, 3:6] = eta_om
    S_d[9:12, 3:6] = S_om

    A = np.zeros((16, 16))
    A[0:3, 0:3] = np.eye(3)
    A[0:3, 3:6] = S_v
    A[0:3, 6:9] = G_eta
    A[0:3, 12:16] = Phi_eta_T
    A[3:6, 3:6] = E_v
    A[3:6, 6:9] = G_S
    A[3:6, 12:16] = Phi_S_T
    A[6:9, 6:9] = np.eye(3)
    A[6:9, 9:12] = S_om
    A[6:9, 12:16] = Psi_eta_tau
    A[9:12, 9:12] = E_om
    A[9:12, 12:16] = Psi_S_tau
    A[12:16, 12:16] = np.eye(4)

    B = np.zeros((16, 4))
    B[0:3, :] = Phi_eta_T
    B[3:6, :] = Phi_S_T
    B[6:9, :] = Psi_eta_tau
    B[9:12, :] = Psi_S_tau
    B[12:16, :] = np.eye(4)

    S_c = np.zeros((16, 6))
    S_c[0:12, :] = S_d
    return A, B, S_c


def normalize_augmented_matrices(
    A: np.ndarray,
    B: np.ndarray,
    S_c: np.ndarray,
    scales: ScaleConfig = DEFAULT_SCALES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inv_Txc = np.diag(1.0 / np.diag(scales.T_xc))
    return (
        inv_Txc @ A @ scales.T_xc,
        inv_Txc @ B @ scales.T_du,
        inv_Txc @ S_c @ scales.T_d,
    )


def build_normalized_augmented_matrices(
    p: Dict[str, float],
    Ts: float,
    g: float,
    scales: ScaleConfig = DEFAULT_SCALES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return normalize_augmented_matrices(
        *build_physical_augmented_matrices(p, Ts, g), scales=scales
    )


def build_physical_performance_matrices(
    q_diag: Sequence[float], r_diag: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q_diag, dtype=float)
    r = np.asarray(r_diag, dtype=float)
    if q.shape != (12,) or r.shape != (4,):
        raise ValueError("performance diagonals must have lengths 12 and 4")
    C = np.zeros((16, 16))
    C[0:12, 0:12] = np.diag(np.sqrt(q))
    D = np.zeros((16, 4))
    D[12:16, :] = np.diag(np.sqrt(r))
    return C, D


def build_normalized_performance_matrices(
    q_diag: Sequence[float],
    r_diag: Sequence[float],
    scales: ScaleConfig = DEFAULT_SCALES,
) -> Tuple[np.ndarray, np.ndarray]:
    C_phys, D_phys = build_physical_performance_matrices(q_diag, r_diag)
    return C_phys @ scales.T_xc, D_phys @ scales.T_du


def state_to_normalized(
    x: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> np.ndarray:
    return np.diag(1.0 / np.diag(scales.T_xc)) @ np.asarray(x, dtype=float)


def state_to_physical(
    x_bar: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> np.ndarray:
    return scales.T_xc @ np.asarray(x_bar, dtype=float)


def increment_to_normalized(
    u: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> np.ndarray:
    return np.diag(1.0 / np.diag(scales.T_du)) @ np.asarray(u, dtype=float)


def increment_to_physical(
    u_bar: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> np.ndarray:
    return scales.T_du @ np.asarray(u_bar, dtype=float)


def disturbance_to_normalized(
    d: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> np.ndarray:
    return np.diag(1.0 / np.diag(scales.T_d)) @ np.asarray(d, dtype=float)


def disturbance_to_physical(
    d_bar: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> np.ndarray:
    return scales.T_d @ np.asarray(d_bar, dtype=float)


def gain_to_physical(
    K_bar: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> np.ndarray:
    inv_Txc = np.diag(1.0 / np.diag(scales.T_xc))
    return scales.T_du @ np.asarray(K_bar, dtype=float) @ inv_Txc


def gain_to_normalized(
    K_phys: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> np.ndarray:
    inv_Tdu = np.diag(1.0 / np.diag(scales.T_du))
    return inv_Tdu @ np.asarray(K_phys, dtype=float) @ scales.T_xc


def project_unit_ball(v: np.ndarray) -> Tuple[np.ndarray, float]:
    arr = np.asarray(v, dtype=float)
    nrm = float(np.linalg.norm(arr))
    factor = 1.0 if nrm <= 1.0 or nrm == 0.0 else 1.0 / nrm
    return arr * factor, factor


def project_increment_physical(
    u_cmd: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> Tuple[np.ndarray, float]:
    u_bar, factor = project_unit_ball(increment_to_normalized(u_cmd, scales))
    return increment_to_physical(u_bar, scales), factor


def project_disturbance_physical(
    d_raw: np.ndarray, scales: ScaleConfig = DEFAULT_SCALES
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Project complete disturbance columns in normalized coordinates."""

    raw = np.asarray(d_raw, dtype=float)
    was_vector = raw.ndim == 1
    if was_vector:
        raw = raw.reshape(6, 1)
    if raw.shape[0] != 6:
        raise ValueError("disturbance array must have six rows")
    inv_Td = np.diag(1.0 / np.diag(scales.T_d))
    raw_bar = inv_Td @ raw
    raw_norm = np.linalg.norm(raw_bar, axis=0)
    factors = np.ones(raw.shape[1])
    mask = raw_norm > 1.0
    factors[mask] = 1.0 / raw_norm[mask]
    applied = raw * factors.reshape(1, -1)
    applied_norm = np.linalg.norm(inv_Td @ applied, axis=0)
    stats: Dict[str, object] = {
        "raw_peak_normalized": float(np.max(raw_norm)) if raw_norm.size else 0.0,
        "applied_peak_normalized": float(np.max(applied_norm)) if applied_norm.size else 0.0,
        "projection_count": int(np.sum(mask)),
        "projection_rate": float(np.mean(mask)) if mask.size else 0.0,
        "average_projection_factor": float(np.mean(factors)) if factors.size else 1.0,
        "raw_component_max_abs": np.max(np.abs(raw), axis=1).tolist(),
        "applied_component_max_abs": np.max(np.abs(applied), axis=1).tolist(),
    }
    return (applied[:, 0] if was_vector else applied), stats


def sample_capped_physical_residual(
    rng: np.random.Generator,
    scales: ScaleConfig = DEFAULT_SCALES,
    residual_std: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float]:
    std = np.asarray(
        scales.residual_std if residual_std is None else residual_std, dtype=float
    )
    if std.ndim == 0:
        std = np.full(12, float(std))
    if std.shape != (12,) or np.any(std < 0.0):
        raise ValueError("residual_std must be a nonnegative scalar or 12-vector")
    raw = rng.standard_normal(12) * std
    inv_cap = np.diag(1.0 / np.asarray(scales.residual_cap, dtype=float))
    normalized, factor = project_unit_ball(inv_cap @ raw)
    return scales.T_res @ normalized, factor


def build_normalized_residual_bound(
    L: int,
    scales: ScaleConfig = DEFAULT_SCALES,
    W_E_z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Deterministic bound for capped residuals in normalized state rows."""

    inv_Tx = np.diag(1.0 / np.diag(scales.T_x))
    residual_map = inv_Tx @ scales.T_res
    W_x = float(L) * residual_map @ residual_map.T
    W_z = np.zeros((16, 16)) if W_E_z is None else np.asarray(W_E_z, dtype=float)
    if W_z.shape != (16, 16):
        raise ValueError("W_E_z must be 16 by 16")
    return la.block_diag(W_x, np.zeros((4, 4)), W_z)


def verify_realized_record_residual(
    batch: Dict[str, np.ndarray], W_E: np.ndarray
) -> Dict[str, float]:
    """Check W_E against the realized batch-generator recording residual."""

    dynamic_residual = (
        batch["X_tp1"]
        - batch["A"] @ batch["X_t"]
        - batch["B"] @ batch["U_t"]
        - batch["S_c"] @ batch["Dbar_t"]
    )
    output_residual = batch["Z_t"] - (
        batch["Cc"] @ batch["X_t"] + batch["Dc"] @ batch["U_t"]
    )
    residual = np.vstack([dynamic_residual, output_residual])
    gap = np.asarray(W_E, dtype=float) - residual @ residual.T
    return {
        "residual_gram_max_eig": float(la.eigvalsh(residual @ residual.T)[-1]),
        "minimum_psd_gap": float(la.eigvalsh(0.5 * (gap + gap.T))[0]),
        "maximum_output_residual": float(np.max(np.abs(output_residual))),
        "maximum_input_memory_residual": float(
            np.max(np.abs(dynamic_residual[12:16, :]))
        ),
    }


def _append_mapping_arrays(
    payload: Dict[str, np.ndarray], prefix: str, values: Mapping[str, Any]
) -> None:
    for key, value in values.items():
        if value is None:
            continue
        name = f"{prefix}_{key}"
        if isinstance(value, str):
            payload[name] = np.asarray([value])
            continue
        arr = np.asarray(value)
        if arr.dtype != object:
            payload[name] = arr.reshape(1) if arr.ndim == 0 else arr


def build_fusion_diagnostics_payload(
    batch: Mapping[str, Any],
    vertices: Sequence[Mapping[str, Any]],
    Psi: np.ndarray,
    W_c: np.ndarray,
    W_E: np.ndarray,
    wc_diagnostics: Mapping[str, Any],
    score_diagnostics: Mapping[str, Any],
    parameter_keys: Sequence[str],
    generator_parameters: Mapping[str, float],
    *,
    model_residual_all: Optional[Mapping[str, Any]] = None,
    model_residual_certified: Optional[Mapping[str, Any]] = None,
    scales: ScaleConfig = DEFAULT_SCALES,
) -> Dict[str, np.ndarray]:
    """Assemble machine-readable data-fusion diagnostics for a clean rerun."""

    keys = list(parameter_keys)
    payload: Dict[str, np.ndarray] = {
        "parameter_keys": np.asarray(keys),
        "generator_parameters": np.asarray(
            [float(generator_parameters[key]) for key in keys], dtype=float
        ),
        "vertex_parameters": np.asarray(
            [[float(vertex["p"][key]) for key in keys] for vertex in vertices],
            dtype=float,
        ),
        "vertex_delta": np.asarray([vertex["Delta"] for vertex in vertices]),
        "vertex_disturbance_injection": np.asarray(
            [vertex["S"] for vertex in vertices]
        ),
        "raw_scores": np.asarray(
            [float(vertex.get("s_raw", vertex["s"])) for vertex in vertices]
        ),
        "processed_scores": np.asarray(
            [float(vertex["s"]) for vertex in vertices]
        ),
        "zero_score_mask": np.asarray(
            [float(vertex["s"]) == 0.0 for vertex in vertices], dtype=bool
        ),
        "Psi": np.asarray(Psi, dtype=float),
        "W_c": np.asarray(W_c, dtype=float),
        "W_E": np.asarray(W_E, dtype=float),
        "state_scales": np.asarray(scales.state, dtype=float),
        "input_memory_scales": np.asarray(scales.input_memory, dtype=float),
        "increment_scales": np.asarray(scales.increment, dtype=float),
        "disturbance_scales": np.asarray(scales.disturbance, dtype=float),
        "residual_cap_scales": np.asarray(scales.residual_cap, dtype=float),
        "residual_standard_deviations": np.asarray(
            scales.residual_std, dtype=float
        ),
    }
    if all("s_full_qmi" in vertex for vertex in vertices):
        payload["full_qmi_raw_scores"] = np.asarray(
            [float(vertex["s_full_qmi"]) for vertex in vertices], dtype=float
        )

    for key in (
        "X_t", "X_tp1", "U_t", "Z_t", "Dbar_t", "X_phys",
        "X_tp1_recorded_phys", "U_phys", "Dbar_phys", "A", "B", "S_c",
        "Cc", "Dc", "K0", "K0_physical",
    ):
        if key in batch:
            payload[f"batch_{key}"] = np.asarray(batch[key])

    _append_mapping_arrays(payload, "wc", wc_diagnostics)
    _append_mapping_arrays(payload, "score", score_diagnostics)
    if "record_residual_check" in batch:
        _append_mapping_arrays(
            payload, "record_residual", batch["record_residual_check"]
        )
    if "disturbance_stats" in batch:
        _append_mapping_arrays(
            payload, "batch_disturbance", batch["disturbance_stats"]
        )
    if "input_saturation_stats" in batch:
        _append_mapping_arrays(
            payload, "batch_input_saturation", batch["input_saturation_stats"]
        )
    if model_residual_all is not None:
        _append_mapping_arrays(payload, "model_residual_all", model_residual_all)
    if model_residual_certified is not None:
        _append_mapping_arrays(
            payload, "model_residual_certified", model_residual_certified
        )
    return payload


def build_synthesis_diagnostics_payload(
    outputs: Mapping[str, Any],
    parameter_keys: Sequence[str],
    scales: ScaleConfig = DEFAULT_SCALES,
) -> Dict[str, np.ndarray]:
    """Collect controller, solver, active-set, and certificate diagnostics."""

    keys = list(parameter_keys)
    payload: Dict[str, np.ndarray] = {
        "parameter_keys": np.asarray(keys),
        "state_scales": np.asarray(scales.state, dtype=float),
        "input_memory_scales": np.asarray(scales.input_memory, dtype=float),
        "increment_scales": np.asarray(scales.increment, dtype=float),
        "disturbance_scales": np.asarray(scales.disturbance, dtype=float),
    }
    for name in ("p_nom", "p_data_source"):
        if name in outputs:
            payload[name] = np.asarray(
                [float(outputs[name][key]) for key in keys], dtype=float
            )

    for name, value in outputs.items():
        if name.startswith("K_") and isinstance(value, np.ndarray):
            payload[name] = np.asarray(value)
        elif name.startswith("gamma_") and np.isscalar(value):
            payload[name] = np.asarray([float(value)])
        elif name.startswith("sol_") and isinstance(value, Mapping):
            _append_mapping_arrays(payload, name, value)

    active = outputs.get("vertices_common")
    if active:
        payload["active_vertex_parameters"] = np.asarray(
            [[float(vertex["p"][key]) for key in keys] for vertex in active]
        )
        payload["active_vertex_scores"] = np.asarray(
            [float(vertex["s"]) for vertex in active]
        )

    vertices = outputs.get("verts_norm")
    if vertices:
        processed = np.asarray([float(vertex["s"]) for vertex in vertices])
        payload["processed_scores"] = processed
        payload["zero_score_count"] = np.asarray([np.sum(processed == 0.0)])
        payload["total_vertex_count"] = np.asarray([len(vertices)])

    cert_results = outputs.get("cert_results")
    if isinstance(cert_results, Mapping):
        for controller, record in cert_results.items():
            if not isinstance(record, Mapping):
                continue
            safe = "".join(ch if ch.isalnum() else "_" for ch in controller).strip("_")
            for key in ("classification", "max_viol", "n_violated", "gamma", "polished", "K"):
                if key in record:
                    _append_mapping_arrays(
                        payload, f"cert_{safe}", {key: record[key]}
                    )
            if isinstance(record.get("sol"), Mapping):
                _append_mapping_arrays(payload, f"cert_{safe}_sol", record["sol"])
    return payload


def append_tracking_simulation_payload(
    payload: Dict[str, np.ndarray],
    prefix: str,
    simulation: Mapping[str, Any],
) -> None:
    """Append the numerical source arrays for one tracking trajectory."""

    for key in (
        "t", "x", "u_c_raw", "u_c", "u_abs", "dbar", "Pref",
        "A_true", "B_true", "S_true", "K_physical", "sat_rate",
        "sat_norm", "sat_abs", "increment_raw_norm_normalized",
        "increment_applied_norm_normalized",
    ):
        if key in simulation:
            value = np.asarray(simulation[key])
            if value.dtype != object:
                payload[f"{prefix}_{key}"] = value
    if "disturbance_stats" in simulation:
        _append_mapping_arrays(
            payload, f"{prefix}_disturbance", simulation["disturbance_stats"]
        )


def evaluate_auxiliary_lmi_residuals(
    vertices: Sequence[Mapping[str, Any]],
    K: np.ndarray,
    Q: np.ndarray,
    mu: float,
    *,
    x0: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return signed PSD residuals for the non-dissipation synthesis LMIs."""

    Q = 0.5 * (np.asarray(Q, dtype=float) + np.asarray(Q, dtype=float).T)
    K = np.asarray(K, dtype=float)
    Y = K @ Q
    output_residuals = []
    for vertex in vertices:
        C = np.asarray(vertex["C"], dtype=float)
        D = np.asarray(vertex["D"], dtype=float)
        H = C @ Q + D @ Y
        block = np.block([[Q, H.T], [H, float(mu) * np.eye(H.shape[0])]])
        output_residuals.append(-float(la.eigvalsh(0.5 * (block + block.T))[0]))

    increment_block = np.block([[Q, Y.T], [Y, np.eye(Y.shape[0])]])
    diagnostics = {
        "output_lmi_signed_violation": (
            float(np.max(output_residuals)) if output_residuals else float("nan")
        ),
        "increment_lmi_signed_violation": -float(
            la.eigvalsh(0.5 * (increment_block + increment_block.T))[0]
        ),
        "Q_psd_signed_violation": -float(la.eigvalsh(Q)[0]),
    }
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        initial_block = np.block(
            [[np.ones((1, 1)), x0.reshape(1, -1)],
             [x0.reshape(-1, 1), Q]]
        )
        diagnostics["initial_lmi_signed_violation"] = -float(
            la.eigvalsh(0.5 * (initial_block + initial_block.T))[0]
        )
    return diagnostics


def compute_inconsistency_quota(
    scores: Sequence[float], K_budget: int, rho_incon: float
) -> int:
    if K_budget < 1:
        raise ValueError("K_budget must be positive")
    if not 0.0 <= rho_incon <= 1.0:
        raise ValueError("rho_incon must lie in [0,1]")
    n_positive = int(np.sum(np.asarray(scores, dtype=float) > 0.0))
    return min(int(np.ceil(rho_incon * K_budget)), n_positive)


def allocate_stratified_budget(
    K_budget: int,
    ratios: Sequence[float],
    tier_sizes: Sequence[int],
) -> Tuple[int, ...]:
    """Allocate a fixed budget across nonempty tiers, including K_budget < 3."""

    sizes = np.asarray(tier_sizes, dtype=int)
    weights = np.asarray(ratios, dtype=float)
    if K_budget < 0 or sizes.ndim != 1 or weights.shape != sizes.shape:
        raise ValueError("invalid stratified-budget arguments")
    if np.any(sizes < 0) or np.any(weights < 0.0):
        raise ValueError("tier sizes and ratios must be nonnegative")
    K = min(int(K_budget), int(np.sum(sizes)))
    if K == 0:
        return tuple(0 for _ in sizes)

    valid = sizes > 0
    weights = np.where(valid, weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = valid.astype(float)
    weights /= float(np.sum(weights))
    targets = K * weights
    allocation = np.minimum(np.floor(targets).astype(int), sizes)

    while int(np.sum(allocation)) < K:
        candidates = np.flatnonzero(allocation < sizes)
        if candidates.size == 0:
            break
        deficit = targets[candidates] - allocation[candidates]
        best = int(candidates[int(np.argmax(deficit))])
        allocation[best] += 1

    return tuple(int(v) for v in allocation)


def classify_surrogate_status(
    sdp_success: bool,
    v_max: float,
    eps_pass: float = EPS_PASS,
    eps_near: float = EPS_NEAR,
) -> str:
    if eps_pass < 0.0 or eps_near < eps_pass:
        raise ValueError("status thresholds must satisfy 0 <= eps_pass <= eps_near")
    if not sdp_success or not np.isfinite(v_max) or v_max > eps_near:
        return "Failed"
    if v_max <= eps_pass:
        return "Surrogate pass"
    return "Near surrogate"
