# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import csv
import itertools
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence, Callable, Any

import numpy as np
from score_processing import nonnegative_quantile_threshold
from raw_qmi_scores import (
    aggregate_residual_bound,
    dynamics_qmi_raw_score,
    dynamics_residual_matrix,
    full_qmi_residual,
    largest_eigenvalue,
)
from numpy.linalg import eigvalsh
import scipy.linalg as la

import matplotlib as mpl
import matplotlib.pyplot as plt

from normalized_coordinates import (
    DEFAULT_SCALES,
    build_normalized_augmented_matrices,
    build_normalized_performance_matrices,
    build_normalized_residual_bound,
    build_fusion_diagnostics_payload,
    build_synthesis_diagnostics_payload,
    build_physical_augmented_matrices,
    build_physical_performance_matrices,
    allocate_stratified_budget,
    classify_surrogate_status,
    compute_inconsistency_quota,
    evaluate_auxiliary_lmi_residuals,
    gain_to_normalized,
    gain_to_physical,
    increment_to_normalized,
    project_disturbance_physical,
    project_increment_physical,
    sample_capped_physical_residual,
    verify_realized_record_residual,
)
from mosek_helpers import (
    fusion_matrix_level,
    mosek_exception_payload,
    mosek_status_payload,
    validate_mosek_solution,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results_revised")


def result_path(*parts: str) -> str:
    path = os.path.join(RESULTS_ROOT, *parts)
    leaf = os.path.basename(path)
    if os.path.splitext(leaf)[1]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)
    return path


# =========================================================
# Plotting utilities
# =========================================================
def set_publication_style(
        *,
        context: str = "paper",
        column: str = "single",
        font_family: Optional[List[str]] = None,
        use_tex: bool = False,
) -> Dict[str, Tuple[float, float]]:
    mm_to_in = 1.0 / 25.4
    fig_w_single = 89.0 * mm_to_in
    fig_w_double = 183.0 * mm_to_in
    fig_h_single = 65.0 * mm_to_in
    fig_h_double = 70.0 * mm_to_in

    sizes = {
        "single": (fig_w_single, fig_h_single),
        "double": (fig_w_double, fig_h_double),
    }

    if font_family is None:
        font_family = ["Arial", "Helvetica", "DejaVu Sans"]

    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]

    base = {
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.25,

        "font.family": "sans-serif",
        "font.sans-serif": font_family,
        "text.usetex": bool(use_tex),
        "mathtext.fontset": "dejavusans",

        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,

        "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    mpl.rcParams.update(base)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=palette)

    _ = column
    return sizes


def save_pub_figure(
        fig: mpl.figure.Figure,
        out_dir: str,
        stem: str,
        *,
        formats: Tuple[str, ...] = ("pdf", "png"),
        dpi_png: int = 600,
        transparent: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        fmt_l = fmt.lower()
        path = os.path.join(out_dir, f"{stem}.{fmt_l}")
        if fmt_l in ("png", "jpg", "jpeg", "tif", "tiff"):
            fig.savefig(path, dpi=dpi_png, transparent=transparent)
        else:
            fig.savefig(path, transparent=transparent)






# =========================================================
# Plotting utilities
# =========================================================
def configure_mosek_license(
        license_path: Optional[str] = None,
        search_dir: str = "",
        verbose: bool = True,
) -> str:
    if license_path and ("@" in license_path):
        os.environ["MOSEKLM_LICENSE_FILE"] = license_path
        if verbose:
            print(f"[MOSEK] Using license server: {license_path}")
        return license_path

    if license_path and os.path.isfile(license_path):
        os.environ["MOSEKLM_LICENSE_FILE"] = license_path
        if verbose:
            print(f"[MOSEK] Using license file: {license_path}")
        return license_path

    candidates = []
    if license_path:
        candidates.extend([license_path, license_path + ".txt"])
    for c in candidates:
        if os.path.isfile(c):
            os.environ["MOSEKLM_LICENSE_FILE"] = c
            if verbose:
                print(f"[MOSEK] Using license file: {c}")
            return c

    found = []
    if search_dir and os.path.isdir(search_dir):
        found = glob.glob(os.path.join(search_dir, "**", "*.lic"), recursive=True)

    if found:
        found.sort()
        lic = found[0]
        os.environ["MOSEKLM_LICENSE_FILE"] = lic
        if verbose:
            print("[MOSEK] Auto-found a license file.")
        return lic

    env_val = os.environ.get("MOSEKLM_LICENSE_FILE", "")
    if env_val and ("@" in env_val or os.path.isfile(env_val)):
        if verbose:
            print("[MOSEK] Using the configured license.")
        return env_val
    raise FileNotFoundError(
        "MOSEK license was not found. Set MOSEKLM_LICENSE_FILE to a license "
        "file path or to a license server such as '27000@server'. "
        f"MOSEKLM_LICENSE_FILE is {'set but invalid' if env_val else 'unset'}."
    )


# =========================================================
# Plotting utilities
# =========================================================






# =========================================================
# Plotting utilities
# =========================================================
@dataclass(frozen=True)
class ParamBounds:
    sigma_t: Tuple[float, float]
    Jx: Tuple[float, float]
    Jy: Tuple[float, float]
    Jz: Tuple[float, float]
    kx: Tuple[float, float]
    ky: Tuple[float, float]
    kz: Tuple[float, float]
    kp: Tuple[float, float]
    kq: Tuple[float, float]
    kr: Tuple[float, float]


@dataclass(frozen=True)
class SynthesisParams:
    Ts: float = 0.1
    g: float = 9.81

    d_max: float = 2.4
    du_max: float = 3.5

    u_abs_min: Tuple[float, float, float, float] = (-6.0, -4.0, -4.0, -4.0)
    u_abs_max: Tuple[float, float, float, float] = (6.0, 4.0, 4.0, 4.0)
    du_max_vec: Optional[Tuple[float, float, float, float]] = None
    sat_tol: float = 0.99

    decay_rate: float = 0.95
    w_gamma: float = 1.0
    w_mu: float = 0.1
    w_beta: float = 1e-3

    Qx_perf: Tuple[float, ...] = (20, 20, 40, 5, 5, 5, 50, 50, 20, 1, 1, 1)
    Rd_perf: Tuple[float, ...] = (5.0, 10.0, 10.0, 10.0)
    Qx_lqr: Tuple[float, ...] = (20, 20, 40, 5, 5, 5, 50, 50, 20, 1, 1, 1, 0.1, 0.1, 0.1, 0.1)
    Ru_lqr: Tuple[float, ...] = (5, 10, 10, 10)

    enforce_perf_all_vertices: bool = True
    seed: int = 26

    fig_context: str = "paper"
    fig_column: str = "double"
    fig_formats: Tuple[str, ...] = ("pdf", "png")
    fig_dpi_png: int = 600
    fig_transparent: bool = False
    fig_show_titles: bool = True
    fig_out_dir: str = result_path("Stage3_Figures_Paper")

    # Gaussian std of the sampled-state residual injected on the
    # 12-state physical block in simulate_batch_data / simulate_tracking_with_*.
    sigma_E_x: float = 5e-4
    # Physical residual cap retained for compatibility with the fixed channel scales.
    nu_E_max: float = 3e-3

# =========================================================
# Plotting utilities
# =========================================================


def build_vertex_matrices(p: Dict[str, float], Ts: float, g: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the synthesis model in fixed dimensionless coordinates."""
    return build_normalized_augmented_matrices(p, Ts, g, DEFAULT_SCALES)


def build_physical_matrices(p: Dict[str, float], Ts: float, g: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the plant model in physical coordinates."""
    return build_physical_augmented_matrices(p, Ts, g)


def build_performance_matrices(syn: SynthesisParams) -> Tuple[np.ndarray, np.ndarray]:
    """Performance map for normalized state and increment coordinates."""
    return build_normalized_performance_matrices(syn.Qx_perf, syn.Rd_perf, DEFAULT_SCALES)


def build_physical_performance(syn: SynthesisParams) -> Tuple[np.ndarray, np.ndarray]:
    return build_physical_performance_matrices(syn.Qx_perf, syn.Rd_perf)


# =========================================================
# Plotting utilities
# =========================================================
def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = la.solve_discrete_are(A, B, Q, R)
    K = -la.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K




def clip_vec(u: np.ndarray, umin: np.ndarray, umax: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(u, umin), umax)


def colored_noise(dim: int, steps: int, alpha: float, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((dim, steps))
    for k in range(1, steps):
        x[:, k] = alpha * x[:, k - 1] + (1 - alpha) * rng.standard_normal(dim) * scale
    return x


def spectral_radius(M: np.ndarray) -> float:
    w = la.eigvals(M)
    return float(np.max(np.abs(w)))




# =========================================================
# Plotting utilities
# =========================================================
def simulate_batch_data(
        p_gen: Dict[str, float],
        syn: SynthesisParams,
        L: int = 1200,
        excite_scale: float = 1.2,
        meas_noise_std: float = 0.0005,
        *,
        K_fb: Optional[np.ndarray] = None,
        p_lqr: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    """
    Ts, g = syn.Ts, syn.g
    rng = np.random.default_rng(syn.seed)

    A_gen, B_gen, S_gen = build_physical_matrices(p_gen, Ts, g)
    Cc, Dc = build_physical_performance(syn)

    Qlqr = np.diag(syn.Qx_lqr)
    Rlqr = np.diag(syn.Ru_lqr)

    if K_fb is not None:
        K0_bar = np.asarray(K_fb, dtype=float)
        K0 = gain_to_physical(K0_bar, DEFAULT_SCALES)
    else:
        p_for_lqr = p_lqr if p_lqr is not None else p_gen
        A_lqr, B_lqr, _ = build_physical_matrices(p_for_lqr, Ts, g)
        K0 = dlqr(A_lqr, B_lqr, Qlqr, Rlqr)
        K0_bar = gain_to_normalized(K0, DEFAULT_SCALES)

    u_dither_bar = colored_noise(
        4, L, alpha=0.92, scale=excite_scale, seed=syn.seed + 10
    )
    u_dither = DEFAULT_SCALES.T_du @ u_dither_bar
    d_col = colored_noise(6, L, alpha=0.97, scale=1.0, seed=syn.seed + 20)
    dbar_raw = DEFAULT_SCALES.T_d @ d_col
    dbar_applied, disturbance_stats = project_disturbance_physical(dbar_raw, DEFAULT_SCALES)

    x = np.zeros((16, L + 1))
    x_next_recorded = np.zeros((16, L))
    u_c = np.zeros((4, L))
    dbar = np.zeros((6, L))
    z = np.zeros((16, L))
    increment_saturation_count = 0
    absolute_saturation_count = 0
    command_peak_normalized = 0.0
    applied_peak_normalized = 0.0

    x0 = np.zeros(16)
    x0[0:3] = np.array([0.2, -0.2, 0.1])
    x0[6:9] = np.deg2rad([2.0, -2.0, 1.0])
    x[:, 0] = x0

    for k in range(L):
        # bounded disturbance
        dbar[:, k] = dbar_applied[:, k]

        # control law uses baseline K0 (nominal K)
        uk_cmd = (K0 @ x[:, k]) + u_dither[:, k]
        command_peak_normalized = max(
            command_peak_normalized,
            float(np.linalg.norm(increment_to_normalized(uk_cmd, DEFAULT_SCALES))),
        )
        uk_limited, increment_flags = apply_increment_limits(uk_cmd, syn)
        uk, u_next, absolute_flags = apply_absolute_actuator_limits(
            x[12:16, k], uk_limited, syn
        )
        increment_saturation_count += int(
            increment_flags["rate_sat"] or increment_flags["norm_sat"]
        )
        absolute_saturation_count += int(absolute_flags["abs_sat"])
        applied_peak_normalized = max(
            applied_peak_normalized,
            float(np.linalg.norm(increment_to_normalized(uk, DEFAULT_SCALES))),
        )

        u_c[:, k] = uk
        z[:, k] = Cc @ x[:, k] + Dc @ uk

        # state update uses TRUE/UNKNOWN plant (A_gen, B_gen, S_gen)
        x[:, k + 1] = A_gen @ x[:, k] + B_gen @ uk + S_gen @ dbar[:, k]
        x[12:16, k + 1] = u_next
        x_next_recorded[:, k] = x[:, k + 1]
        nu, _ = sample_capped_physical_residual(
            rng, DEFAULT_SCALES, residual_std=meas_noise_std
        )
        x_next_recorded[0:12, k] += nu

    inv_Txc = np.diag(1.0 / np.diag(DEFAULT_SCALES.T_xc))
    inv_Tdu = np.diag(1.0 / np.diag(DEFAULT_SCALES.T_du))
    inv_Td = np.diag(1.0 / np.diag(DEFAULT_SCALES.T_d))
    A_bar, B_bar, S_bar = build_vertex_matrices(p_gen, Ts, g)
    C_bar, D_bar = build_performance_matrices(syn)
    return dict(
        X_t=inv_Txc @ x[:, 0:L], X_tp1=inv_Txc @ x_next_recorded,
        U_t=inv_Tdu @ u_c, Z_t=z, Dbar_t=inv_Td @ dbar,
        X_phys=x, X_tp1_recorded_phys=x_next_recorded,
        U_phys=u_c, Dbar_phys=dbar,
        A=A_bar, B=B_bar, S_c=S_bar, Cc=C_bar, Dc=D_bar,
        K0=K0_bar, K0_physical=K0,
        disturbance_stats=disturbance_stats,
        input_saturation_stats=dict(
            increment_saturation_count=increment_saturation_count,
            increment_saturation_rate=increment_saturation_count / float(L),
            absolute_saturation_count=absolute_saturation_count,
            absolute_saturation_rate=absolute_saturation_count / float(L),
            command_peak_normalized=command_peak_normalized,
            applied_peak_normalized=applied_peak_normalized,
        ),
    )
def build_disturbance_psd_bound(
        bounds: "ParamBounds",
        syn: SynthesisParams,
        *,
        batch_generator: Optional[Dict[str, float]] = None,
        n_internal_samples: int = 1000,
        safety_factor: float = 1.05,
        seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build the endpoint-exact isotropic PSD upper bound on the normalized
    disturbance-injection Gram matrix over the parameter box.

    Construction (paper eq. for W_c):
        alpha = safety_factor * max lambda_max(S_c S_c.T)
        W_c   = alpha * I_16

    The disturbance columns have disjoint state support and their norms decrease
    with the corresponding positive damping rates, so the box-wide maximum is
    attained at an adopted endpoint. The optional batch generator and random
    interior samples are evaluated only as numerical diagnostics.

    safety_factor (default 1.05) is the inflation factor `1 + eta_c` from the paper; with
    eta_c > 0 the resulting min vertex PSD gap is strictly positive (verified numerically).

    Returns
    -------
    W_c  : (16, 16) PSD matrix.
    info : dict with keys
        alpha, vertex_max_eig, interior_max_eig, safety_factor,
        min_psd_gap_vertex, n_violate_vertex,
        min_psd_gap_interior, n_violate_interior,
        n_vertices, n_internal_samples.
    """
    keys = list(bounds.__dataclass_fields__.keys())
    n_keys = len(keys)
    n_vert = 2 ** n_keys
    rng = np.random.default_rng(seed)

    # ----- vertex set (formal certificate sites) -----
    vertex_samples: List[Dict[str, float]] = []
    for bits in range(n_vert):
        p = {k: getattr(bounds, k)[(bits >> j) & 1] for j, k in enumerate(keys)}
        vertex_samples.append(p)

    # ----- random interior samples (sanity check only) -----
    interior_samples: List[Dict[str, float]] = []
    for _ in range(int(n_internal_samples)):
        p = {k: float(rng.uniform(*getattr(bounds, k))) for k in keys}
        interior_samples.append(p)

    vertex_max_eig = 0.0
    for p in vertex_samples:
        _, _, S_v = build_vertex_matrices(p, syn.Ts, syn.g)
        ev_top = float(la.eigvalsh(S_v @ S_v.T)[-1])
        if ev_top > vertex_max_eig:
            vertex_max_eig = ev_top

    interior_max_eig = 0.0
    for p in interior_samples:
        _, _, S_v = build_vertex_matrices(p, syn.Ts, syn.g)
        ev_top = float(la.eigvalsh(S_v @ S_v.T)[-1])
        if ev_top > interior_max_eig:
            interior_max_eig = ev_top

    generator_max_eig = 0.0
    if batch_generator is not None:
        _, _, S_generator = build_vertex_matrices(batch_generator, syn.Ts, syn.g)
        generator_max_eig = float(la.eigvalsh(S_generator @ S_generator.T)[-1])

    alpha = float(safety_factor) * vertex_max_eig
    old_alpha = alpha
    W_c = alpha * np.eye(16)

    # ex-post PSD-domination check, separately on the vertex set (formal) and interior set (sanity).
    min_gap_v, n_viol_v = float("inf"), 0
    for p in vertex_samples:
        _, _, S_v = build_vertex_matrices(p, syn.Ts, syn.g)
        gap = float(la.eigvalsh(W_c - S_v @ S_v.T)[0])
        if gap < min_gap_v:
            min_gap_v = gap
        if gap < -1e-10:
            n_viol_v += 1

    min_gap_i, n_viol_i = float("inf"), 0
    for p in interior_samples:
        _, _, S_v = build_vertex_matrices(p, syn.Ts, syn.g)
        gap = float(la.eigvalsh(W_c - S_v @ S_v.T)[0])
        if gap < min_gap_i:
            min_gap_i = gap
        if gap < -1e-10:
            n_viol_i += 1

    generator_gap = float("nan")
    if batch_generator is not None:
        generator_gap = float(la.eigvalsh(W_c - S_generator @ S_generator.T)[0])

    info = dict(
        alpha=alpha,
        vertex_max_eig=vertex_max_eig,
        interior_max_eig=interior_max_eig,
        generator_max_eig=generator_max_eig,
        generator_psd_gap=generator_gap,
        old_vertex_only_alpha=old_alpha,
        generator_changed_bound=False,
        generator_exceeds_vertex_max=bool(
            generator_max_eig > vertex_max_eig + 1e-12
        ),
        safety_factor=float(safety_factor),
        min_psd_gap_vertex=min_gap_v,
        n_violate_vertex=n_viol_v,
        min_psd_gap_interior=min_gap_i,
        n_violate_interior=n_viol_i,
        n_vertices=len(vertex_samples),
        n_internal_samples=len(interior_samples),
    )
    return W_c, info


def verify_data_residual_bound(
        batch: Dict[str, np.ndarray],
        syn: SynthesisParams,
        bounds: "ParamBounds",
        W_c: np.ndarray,
        *,
        score_zero_vertices: Optional[List[Dict[str, float]]] = None,
        verbose: bool = True,
) -> Dict[str, float]:
    """
    MODEL residual diagnostic (NOT a W_E construction routine).

    For each candidate vertex (A_i, B_i, S_{c,i}) we compute the sample residual

        E_t^{(i)} := breve X_{t+1} - Delta^{(i)} hat X_t - hat D_t^{(i)},

    where hat D_t^{(i)} stacks S_{c,i} * Dbar_t on the top 16 rows. We report the
    per-sample operator-norm scalar

        rho_E^{(i)} := lambda_max( (1/L) E_i E_i^T ),
        rho_E       := max_{i in I0} rho_E^{(i)},

    and compare it with normalized disturbance and residual proxies.

    Important semantics (paper Section 3.1):
      * For score-zero vertices i in I0, E_i is dominated by the sampled-state residual
        (i.e., the actually realized W_E term).
      * For inconsistent vertices (s_i > 0), E_i additionally contains model mismatch
        and CANNOT be interpreted as a residual sample. This function therefore should
        be invoked on `score_zero_vertices`; if not supplied, the full vertex set is
        scanned only for visual completeness (the per-vertex worst case will then likely
        come from an inconsistent vertex and must be read as a model-residual diagnostic,
        not as a sampled-state residual estimate).
      * The scalar rho_E is reported as a diagnostic ONLY -- it does NOT feed back into
        the construction of W_E used inside build_psi_data. The formal Loewner bound
        used in the QMI is the structured W_E from build_structured_residual_bound().

    Returns a dict with the per-sample scalar rho_E and bookkeeping info.
    """
    X_t = batch["X_t"]
    X_tp1 = batch["X_tp1"]
    U_t = batch["U_t"]
    Z_t = batch["Z_t"]
    Dbar_t = batch.get("Dbar_t", None)
    L = X_t.shape[1]

    # the caller decides what is scanned and how it should be interpreted.
    #   - score_zero_vertices=None  -> ALL prior vertices: this is the MODEL-MISMATCH
    #     diagnostic and is NOT a check that ||E|| is residual-dominated.
    #   - score_zero_vertices=I0    -> processed score-zero subset I0: on this subset
    #     rho_E is interpreted only as an a-posteriori residual diagnostic. Because I0
    #     may include vertices with s_i^raw > 0 (thresholded by tau_eff in
    #     compute_si_from_vi), this is NOT a proof that the residual is purely sampled
    #     noise; the formal certificate of E_t E_t^T <= W_E comes from
    #     build_structured_residual_bound + the L2-norm saturation in simulate_batch_data.
    if score_zero_vertices is None:
        keys = list(bounds.__dataclass_fields__.keys())
        score_zero_vertices = []
        for bits in range(2 ** len(keys)):
            p = {k: getattr(bounds, k)[(bits >> j) & 1] for j, k in enumerate(keys)}
            score_zero_vertices.append(p)
        scanned_label = "MODEL-MISMATCH diagnostic over ALL prior vertices (NOT a residual check)"
    else:
        scanned_label = "processed score-zero subset I0 (a-posteriori residual diagnostic)"

    Cc, Dc = build_performance_matrices(syn)
    hat_X = np.vstack([X_t, U_t])           # (20, L)
    breve_X = np.vstack([X_tp1, Z_t])       # (32, L)

    rho_E_per_sample = 0.0
    worst_vertex_idx = -1
    for idx, p in enumerate(score_zero_vertices):
        A_i, B_i, S_i = build_vertex_matrices(p, syn.Ts, syn.g)
        Delta_i = np.block([[A_i, B_i], [Cc, Dc]])
        if Dbar_t is not None:
            top = S_i @ Dbar_t
            hatD_i = np.vstack([top, np.zeros((16, L))])
        else:
            hatD_i = np.zeros((32, L))
        E_i = breve_X - Delta_i @ hat_X - hatD_i               # (32, L)
        # Per-sample scalar: lambda_max((1/L) E_i E_i^T)
        rho_i = float(la.eigvalsh((E_i @ E_i.T) / float(L))[-1])
        if rho_i > rho_E_per_sample:
            rho_E_per_sample = rho_i
            worst_vertex_idx = idx

    alpha_c = float(np.max(np.diag(W_c)))
    compare_per_sample_dist = alpha_c
    nu_E2_per_sample = float(
        np.max(np.diag(build_structured_residual_bound(1, syn.nu_E_max)))
    )
    residual_std_bar = np.asarray(DEFAULT_SCALES.residual_std) / np.asarray(DEFAULT_SCALES.state)
    sigma_E2_per_sample = float(np.max(residual_std_bar ** 2))

    info = dict(
        rho_E_per_sample=rho_E_per_sample,
        worst_vertex_index=worst_vertex_idx,
        n_vertices_scanned=len(score_zero_vertices),
        L=L,
        compare_per_sample_dist=compare_per_sample_dist,
        compare_per_sample_nu_E2=nu_E2_per_sample,
        compare_per_sample_sigma_E2=sigma_E2_per_sample,
        scanned_set=scanned_label,
    )
    if verbose:
        print(f"  [Residual diagnostic, {scanned_label}]")
        print(f"    rho_E = max_i lambda_max((1/L) E_i E_i^T) = {rho_E_per_sample:.4e} "
              f"(@ vertex {worst_vertex_idx}/{len(score_zero_vertices)})")
        print(f"    normalized disturbance proxy alpha_c = {compare_per_sample_dist:.4e}")
        print(f"    normalized residual-cap proxy = {nu_E2_per_sample:.4e}")
        print(f"    normalized residual-std proxy = {sigma_E2_per_sample:.4e}")
        print(f"    (scalar diagnostic only; formal W_E in build_psi_data is built "
              f"independently from nu_E_max via build_structured_residual_bound)")
    return info




def build_structured_residual_bound(
        L: int,
        nu_E_max: float,
        *,
        W_E_z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Deterministic residual bound in the normalized recorded coordinates."""
    expected_cap = max(DEFAULT_SCALES.residual_cap)
    if not np.isclose(float(nu_E_max), expected_cap):
        raise ValueError("nu_E_max must agree with the fixed residual scaling")
    return build_normalized_residual_bound(L, DEFAULT_SCALES, W_E_z)


def build_psi_data(batch: Dict[str, np.ndarray], syn: SynthesisParams, d_max_override: Optional[float] = None) -> np.ndarray:
    X_t = batch["X_t"]
    X_tp1 = batch["X_tp1"]
    U_t = batch["U_t"]
    Z_t = batch["Z_t"]
    S_c = batch["S_c"]

    L = X_t.shape[1]
    V = np.vstack([X_t, U_t])
    Xbreve = np.vstack([X_tp1, Z_t])

    if d_max_override is not None and not np.isclose(d_max_override, syn.d_max):
        raise ValueError("the disturbance radius is fixed by DEFAULT_SCALES.T_d")
    # The normalized disturbance radius is one:
    # \tilde R_t = 2 * diag(L W_c, 0_{16}) + 2 * W_E.
    W_c_supplied = batch.get("W_c", None)
    if W_c_supplied is None:
        raise ValueError(
            "build_psi_data requires W_c constructed over all adopted prior "
            "vertices and the batch generator"
        )
    Wc_block = np.asarray(W_c_supplied, dtype=float)
    W_E = batch.get("W_E", None)
    if W_E is None:
        raise ValueError(
            "build_psi_data requires the deterministic capped-recording "
            "residual bound W_E"
        )
    W_E = np.asarray(W_E, dtype=float)

    Rtilde = aggregate_residual_bound(Wc_block, W_E, L)

    Psi = np.block([
        [V @ V.T, -(V @ Xbreve.T)],
        [-(Xbreve @ V.T), (Xbreve @ Xbreve.T - Rtilde)]
    ])
    return 0.5 * (Psi + Psi.T)




# =========================================================
# Plotting utilities
# =========================================================
def enumerate_vertices(bounds: ParamBounds) -> List[Dict[str, float]]:
    keys = ["sigma_t", "Jx", "Jy", "Jz", "kx", "ky", "kz", "kp", "kq", "kr"]
    vals = []
    for k in keys:
        lo, hi = getattr(bounds, k)
        vals.append([lo, hi])

    vertices = []
    for comb in itertools.product(*vals):
        vertices.append({k: float(v) for k, v in zip(keys, comb)})
    return vertices




# All manuscript scripts use mode="lambda_max", the signed successor-state
# margin. The complete-QMI eigenvalue remains available as a diagnostic.
def compute_vertex_score_scalar(
        Delta_i: np.ndarray,
        Psi_data: np.ndarray,
        mode: str = "lambda_max",
) -> float:
    if mode == "lambda_max":
        return dynamics_qmi_raw_score(Delta_i, Psi_data, successor_dim=16)

    S = full_qmi_residual(Delta_i, Psi_data)

    ev = eigvalsh(S)

    if mode == "full_qmi_lambda_max":
        return float(ev[-1])
    elif mode == "trace":
        return float(-np.trace(S))
    elif mode == "pos_eig_sum":
        return float(-np.sum(np.maximum(ev, 0.0)))
    elif mode == "min_eig":
        return float(ev[0])
    elif mode == "mean_eig":
        return float(np.mean(ev))
    else:
        raise ValueError(
            "mode must be one of: lambda_max / full_qmi_lambda_max / trace / "
            "pos_eig_sum / min_eig / mean_eig"
        )


def build_all_vertices_and_scores(
        bounds: ParamBounds,
        syn: SynthesisParams,
        Psi_data: np.ndarray,
        score_mode: str = "lambda_max",
        max_vertices: Optional[int] = None,
        seed: int = 123,
        batch: Optional[Dict[str, np.ndarray]] = None,
) -> List[Dict[str, object]]:
    Ts, g = syn.Ts, syn.g
    Cc, Dc = build_performance_matrices(syn)

    verts = enumerate_vertices(bounds)
    if max_vertices is not None and max_vertices < len(verts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(verts), size=max_vertices, replace=False)
        verts = [verts[i] for i in idx]

    direct_score_context = None
    if score_mode == "lambda_max" and batch is not None:
        regressor = np.vstack((batch["X_t"], batch["U_t"]))
        successor = np.asarray(batch["X_tp1"], dtype=float)
        aggregate_bound = aggregate_residual_bound(
            np.asarray(batch["W_c"], dtype=float),
            np.asarray(batch["W_E"], dtype=float),
            int(successor.shape[1]),
        )
        direct_score_context = (
            regressor,
            successor,
            aggregate_bound[:successor.shape[0], :successor.shape[0]],
        )

    out = []
    for p in verts:
        A, B, S_c = build_vertex_matrices(p, Ts, g)
        Delta = np.block([[A, B],
                          [Cc, Dc]])
        if direct_score_context is None:
            s_i = compute_vertex_score_scalar(Delta, Psi_data, mode=score_mode)
        else:
            regressor, successor, successor_bound = direct_score_context
            _, score_matrix = dynamics_residual_matrix(
                Delta[:successor.shape[0], :],
                regressor,
                successor,
                successor_bound,
            )
            s_i = largest_eigenvalue(score_matrix)
        s_full_qmi = compute_vertex_score_scalar(
            Delta, Psi_data, mode="full_qmi_lambda_max"
        )
        out.append(dict(A=A, B=B, S=S_c, C=Cc, D=Dc, Delta=Delta,
                        s=s_i, s_full_qmi=s_full_qmi, p=p))
    return out


def compute_si_from_vi(
        vertices: List[Dict[str, object]],
        L: int,
        *,
        tau: float = 0.0,
        q_scale: float = 0.9,
        eps: float = 1e-12,
        use_len_norm: bool = True,
        soft_fallback: bool = True,
        rho: float = 0.1,
        hard_ratio_min: float = 0.01,
        sigma_degenerate_thr: float = 1e-10,
) -> Tuple[List[Dict[str, object]], Dict[str, Any]]:
    # hard_ratio_min is the implementation name of the manuscript fallback
    # trigger r_trig; it is not a guaranteed raw-consistency fraction.
    raw_s = np.array([float(v["s"]) for v in vertices], dtype=float)
    n = len(raw_s)
    n_raw_consistent = int(np.sum(raw_s <= tau))

    # --- Step 0: Processed-score threshold fallback detection ---
    hard_consistent_mask = raw_s <= tau
    hard_ratio = float(np.mean(hard_consistent_mask))
    soft_mode = False
    tau_effective = tau

    if soft_fallback and hard_ratio < hard_ratio_min:
        tau_effective = nonnegative_quantile_threshold(raw_s, rho)
        hard_consistent_mask = raw_s <= tau_effective
        hard_ratio = float(np.mean(hard_consistent_mask))
        soft_mode = True
        print(f"  [ScorePipeline] PROCESSED-SCORE FALLBACK: "
              f"hard_ratio(tau=0)={float(np.mean(raw_s <= tau)):.4f} < {hard_ratio_min}")
        print(f"  [ScorePipeline]   tau_effective={tau_effective:.6e} "
              f"(max of zero and the rho={rho} raw-score quantile)")

    # --- Step 1: Processed-score thresholding ---
    p = np.maximum(0.0, raw_s - tau_effective)

    # --- Step 2: Scale normalization by data length ---
    if use_len_norm and L > 0:
        p = p / (float(L) + eps)

    # --- Step 3: Robust scaling (quantile-based, not max) ---
    p_pos = p[p > 0.0]
    if len(p_pos) == 0:
        s_final = np.zeros(n)
        out = []
        for i, vertex in enumerate(vertices):
            updated = dict(vertex)
            updated["s_raw"] = float(vertex["s"])
            updated["s"] = float(s_final[i])
            out.append(updated)
        diag = dict(
            tau_effective=tau_effective, soft_mode=soft_mode,
            hard_ratio=hard_ratio, sigma=0.0, uninformative=True,
            n_raw_consistent=n_raw_consistent,
            n_zero_processed=n,
            n_consistent=n, n_total=n, s_min=0.0, s_max=0.0, s_mean=0.0,
        )
        print("  [ScorePipeline] UNINFORMATIVE: no positive score excess; "
              "all s_i set to 0 (prior-only fallback)")
        return out, diag
    sigma = float(np.quantile(p_pos, q_scale))

    uninformative = sigma < sigma_degenerate_thr

    if uninformative:
        s_final = np.zeros(n)
        print(f"  [ScorePipeline] UNINFORMATIVE: sigma={sigma:.3e} < {sigma_degenerate_thr:.1e}, "
              f"all s_i set to 0 (prior-only fallback)")
    else:
        r = p / (sigma + eps)
        # --- Step 4: Saturating map to [0, 1] ---
        s_final = 1.0 - np.exp(-r)

    # --- Final clamp ---
    s_final = np.clip(s_final, 0.0, 1.0)

    # --- Build output ---
    out = []
    for i, v in enumerate(vertices):
        vv = dict(v)
        vv["s_raw"] = float(v["s"])
        vv["s"] = float(s_final[i])
        out.append(vv)

    diag = dict(
        tau_effective=tau_effective,
        soft_mode=soft_mode,
        hard_ratio=hard_ratio,
        sigma=sigma,
        uninformative=uninformative,
        n_raw_consistent=n_raw_consistent,
        n_zero_processed=int(np.sum(s_final == 0.0)),
        n_consistent=int(np.sum(s_final == 0.0)),
        n_total=n,
        s_min=float(np.min(s_final)),
        s_max=float(np.max(s_final)),
        s_mean=float(np.mean(s_final)),
    )

    print(f"  [ScorePipeline] tau_eff={tau_effective:.3e}, sigma={sigma:.3e}, "
          f"soft_mode={soft_mode}, uninformative={uninformative}")
    print(f"  [ScorePipeline] raw consistent={diag['n_raw_consistent']}/{n}; "
          f"processed zero={diag['n_zero_processed']}/{n}, "
          f"min={diag['s_min']:.4f}, max={diag['s_max']:.4f}, mean={diag['s_mean']:.4f}")

    return out, diag



# =========================================================
# Plotting utilities
# =========================================================
def _param_vec_from_bounds(p: Dict[str, float], bounds: ParamBounds) -> np.ndarray:
    keys = list(bounds.__dataclass_fields__.keys())
    v = []
    for k in keys:
        lo, hi = getattr(bounds, k)
        den = (hi - lo) if (hi - lo) > 1e-12 else 1.0
        v.append((float(p[k]) - lo) / den)
    return np.array(v, dtype=float)
def _delta_flat(v: Dict[str, object]) -> np.ndarray:
    """Flatten Delta to 1D vector."""
    return np.asarray(v["Delta"], dtype=float).reshape(-1)




def select_vertices_farthest_delta(
        vertices: List[Dict[str, object]],
        K: int,
        *,
        initial_index: int = 0,
) -> List[Dict[str, object]]:
    """
    """
    if len(vertices) == 0:
        return []
    if K >= len(vertices):
        return list(vertices)

    # X: (n, m)
    X = np.vstack([_delta_flat(v) for v in vertices])  # (n, m)
    # whiten per-coordinate (affine invertible)
    mu = np.mean(X, axis=0, keepdims=True)
    sig = np.std(X, axis=0, keepdims=True)
    sig = np.where(sig < 1e-12, 1.0, sig)
    Xw = (X - mu) / sig

    n = Xw.shape[0]
    initial_index = int(np.clip(initial_index, 0, n - 1))

    selected = [initial_index]
    dist2 = np.sum((Xw - Xw[initial_index]) ** 2, axis=1)

    for _ in range(1, K):
        j = int(np.argmax(dist2))
        selected.append(j)
        dist2 = np.minimum(dist2, np.sum((Xw - Xw[j]) ** 2, axis=1))

    out, seen = [], set()
    for i in selected:
        if i not in seen:
            out.append(vertices[i])
            seen.add(i)
    return out




def select_vertices_farthest(
        vertices: List[Dict[str, object]],
        bounds: ParamBounds,
        K: int,
        seed: int = 0,
        initial_index: int = 0,
) -> List[Dict[str, object]]:
    if K >= len(vertices):
        return list(vertices)
    if len(vertices) == 0:
        return []

    X = np.vstack([_param_vec_from_bounds(v["p"], bounds) for v in vertices])
    initial_index = int(np.clip(initial_index, 0, len(vertices) - 1))

    selected = [initial_index]
    dist2 = np.sum((X - X[initial_index]) ** 2, axis=1)

    for _ in range(1, K):
        j = int(np.argmax(dist2))
        selected.append(j)
        dist2 = np.minimum(dist2, np.sum((X - X[j]) ** 2, axis=1))

    seen = set()
    out = []
    for i in selected:
        if i not in seen:
            out.append(vertices[i])
            seen.add(i)
    return out


def select_vertices_stratified_farthest(
        vertices: List[Dict[str, object]],
        bounds: ParamBounds,
        K: int = 20,
        *,
        ratios: Tuple[float, float, float] = (0.4, 0.2, 0.4),
        seed: int = 0,
) -> List[Dict[str, object]]:
    if K >= len(vertices):
        return list(vertices)
    if len(vertices) == 0:
        return []

    ratios = np.array(ratios, dtype=float)
    if ratios.size != 3 or np.any(ratios < 0) or float(np.sum(ratios)) <= 1e-12:
        ratios = np.array([0.4, 0.2, 0.4], dtype=float)
    ratios = ratios / float(np.sum(ratios))

    s = np.array([float(v["s"]) for v in vertices], dtype=float)
    order = np.argsort(s, kind="stable")
    verts_sorted = [vertices[i] for i in order]
    n = len(verts_sorted)

    cut1 = int(np.floor(n / 3))
    cut2 = int(np.floor(2 * n / 3))
    low = verts_sorted[:max(1, cut1)]
    mid = verts_sorted[max(1, cut1):max(cut2, cut1 + 1)]
    high = verts_sorted[max(cut2, cut1 + 1):]
    k_low, k_mid, k_high = allocate_stratified_budget(
        K, ratios, (len(low), len(mid), len(high))
    )

    rng = np.random.default_rng(int(seed))

    sel_low = []
    if k_low > 0 and len(low) > 0:
        sel_low = select_vertices_farthest(low, bounds, K=min(k_low, len(low)), seed=int(seed) + 11, initial_index=0)

    sel_mid = []
    if k_mid > 0 and len(mid) > 0:
        init = int(rng.integers(0, len(mid)))
        sel_mid = select_vertices_farthest(mid, bounds, K=min(k_mid, len(mid)), seed=int(seed) + 22, initial_index=init)

    sel_high = []
    if k_high > 0 and len(high) > 0:
        init = int(rng.integers(0, len(high)))
        sel_high = select_vertices_farthest(high, bounds, K=min(k_high, len(high)), seed=int(seed) + 33,
                                            initial_index=init)

    out = sel_low + sel_mid + sel_high

    if len(out) < K:
        seen = {id(v) for v in out}
        rest = [v for v in verts_sorted if id(v) not in seen]
        if len(rest) > 0:
            need = K - len(out)
            extra = select_vertices_farthest(rest, bounds, K=min(need, len(rest)), seed=int(seed) + 44, initial_index=0)
            out.extend(extra)

    return out[:K]


# =========================================================
# Plotting utilities
# =========================================================
def _level_to_float(x) -> float:
    arr = np.array(x).reshape(-1)
    return float(arr[0])


def solve_vertex_fusion_sdp_mosek(
        vertices: List[Dict[str, object]],
        syn: SynthesisParams,
        verbose: bool = True,
        eps_Q: float = 1e-6,
        active_indices: Optional[Sequence[int]] = None,
        num_threads: int = 0,
        max_iters: int = 50,
        rel_gap: float = 1e-4,
        time_limit_sec: Optional[int] = 900,
        beta_lb: float = 0.0,
        decay_rate: Optional[float] = None,
        w_gamma: float = 1.0,
        w_mu: float = 0.1,
        w_beta: float = 1e-3,
        enforce_perf_all_vertices: bool = True,
        x0_feas: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    configure_mosek_license(verbose=True)
    import mosek.fusion as mf

    if decay_rate is None:
        decay_rate = getattr(syn, "decay_rate", 0.98)

    if active_indices is None:
        active_indices = list(range(len(vertices)))
    else:
        active_indices = list(active_indices)

    M = mf.Model("vertex_fusion")
    try:
        return _solve_vertex_fusion_inner(M, vertices, syn, verbose, eps_Q,
            active_indices, num_threads, max_iters, rel_gap, time_limit_sec,
            beta_lb, decay_rate, w_gamma, w_mu, w_beta, enforce_perf_all_vertices, x0_feas)
    finally:
        M.dispose()


def _solve_vertex_fusion_inner(M, vertices, syn, verbose, eps_Q,
        active_indices, num_threads, max_iters, rel_gap, time_limit_sec,
        beta_lb, decay_rate, w_gamma, w_mu, w_beta, enforce_perf_all_vertices, x0_feas):
    import mosek.fusion as mf

    if verbose:
        M.setLogHandler(sys.stdout)

    try:
        if num_threads is None or num_threads <= 0:
            ncpu = os.cpu_count() or 1
            num_threads = int(min(8, max(1, ncpu)))
        M.setSolverParam("numThreads", int(num_threads))
    except Exception:
        pass

    try:
        M.setSolverParam("intpntMaxIterations", int(max_iters))
        M.setSolverParam("intpntCoTolRelGap", float(rel_gap))
        M.setSolverParam("intpntCoTolPfeas", 1e-6)
        M.setSolverParam("intpntCoTolDfeas", 1e-6)
        if time_limit_sec is not None and time_limit_sec > 0:
            M.setSolverParam("optimizerMaxTime", float(time_limit_sec))
    except Exception:
        pass

    Q = M.variable("Q", mf.Domain.inPSDCone(16))
    Y = M.variable("Y", [4, 16], mf.Domain.unbounded())
    _all_zero_s = all(float(v.get("s", 0.0)) < 1e-12 for v in vertices)
    if _all_zero_s:
        if beta_lb > 0.0:
            raise ValueError("beta_lb must be zero when every selected score is zero")
        beta_domain = mf.Domain.equalsTo(0.0)
    else:
        beta_domain = mf.Domain.greaterThan(float(beta_lb))
    beta = M.variable("beta", beta_domain)
    gamma2 = M.variable("gamma2", mf.Domain.greaterThan(0.0))
    mu = M.variable("mu", mf.Domain.greaterThan(1e-9))

    I16d = mf.Matrix.dense(np.eye(16))
    I6d = mf.Matrix.dense(np.eye(6))
    I4d = mf.Matrix.dense(np.eye(4))

    I6e = mf.Expr.constTerm(I6d)
    I4e = mf.Expr.constTerm(I4d)
    epsI16e = mf.Expr.constTerm(mf.Matrix.dense(eps_Q * np.eye(16)))

    _Z_cache: Dict[Tuple[int, int], mf.Expression] = {}

    def Z(r: int, c: int):
        key = (r, c)
        if key not in _Z_cache:
            _Z_cache[key] = mf.Expr.constTerm(mf.Matrix.dense(np.zeros((r, c))))
        return _Z_cache[key]

    M.constraint("Q_pd", mf.Expr.sub(Q, epsI16e), mf.Domain.inPSDCone(16))

    if x0_feas is not None:
        x0_feas = np.asarray(x0_feas, dtype=float).reshape(-1)
        assert x0_feas.shape[0] == 16
        one = mf.Matrix.dense([[1.0]])
        xrow = mf.Matrix.dense(x0_feas.reshape(1, 16))
        xcol = mf.Matrix.dense(x0_feas.reshape(16, 1))
        row1 = mf.Expr.hstack([mf.Expr.constTerm(one), mf.Expr.constTerm(xrow)])
        row2 = mf.Expr.hstack([mf.Expr.constTerm(xcol), Q])
        Feas = mf.Expr.vstack([row1, row2])
        M.constraint("feas_x0", Feas, mf.Domain.inPSDCone(17))

    for j, vidx in enumerate(active_indices):
        v = vertices[vidx]
        Ai = v["A"];
        Bi = v["B"];
        Ci = v["C"];
        Di = v["D"]
        Sci = v["S"]
        s_i = float(v["s"])

        Ai_m = mf.Matrix.dense(Ai)
        Bi_m = mf.Matrix.dense(Bi)
        Ci_m = mf.Matrix.dense(Ci)
        Di_m = mf.Matrix.dense(Di)
        Sci_m = mf.Matrix.dense(Sci)
        Sci_e = mf.Expr.constTerm(Sci_m)

        AQ_BY = mf.Expr.add(mf.Expr.mul(Ai_m, Q), mf.Expr.mul(Bi_m, Y))
        CQ_DY = mf.Expr.add(mf.Expr.mul(Ci_m, Q), mf.Expr.mul(Di_m, Y))

        beta_term = mf.Expr.mul(mf.Matrix.dense((-1.0 * s_i) * np.eye(16)), beta)
        tl = mf.Expr.add(mf.Expr.mul(-float(decay_rate), Q), beta_term)

        row1 = mf.Expr.hstack([tl, Z(16, 6), mf.Expr.transpose(AQ_BY), mf.Expr.transpose(CQ_DY)])
        row3 = mf.Expr.hstack([AQ_BY, Sci_e, mf.Expr.neg(Q), Z(16, 16)])
        I16e = mf.Expr.constTerm(I16d)

        gamW = mf.Expr.mul(I6d, gamma2)
        row2 = mf.Expr.hstack([Z(6, 16), mf.Expr.neg(gamW), mf.Expr.transpose(Sci_e), Z(6, 16)])

        row4 = mf.Expr.hstack([CQ_DY, Z(16, 6), Z(16, 16), mf.Expr.neg(I16e)])

        LMI = mf.Expr.vstack([row1, row2, row3, row4])
        M.constraint(f"vertex_lmi_{j}", mf.Expr.neg(LMI), mf.Domain.inPSDCone(54))

        if enforce_perf_all_vertices:
            muI = mf.Expr.mul(I16d, mu)
            rowp1 = mf.Expr.hstack([Q, mf.Expr.transpose(CQ_DY)])
            rowp2 = mf.Expr.hstack([CQ_DY, muI])
            PERF = mf.Expr.vstack([rowp1, rowp2])
            M.constraint(f"perf_lmi_{j}", PERF, mf.Domain.inPSDCone(32))

    if (not enforce_perf_all_vertices) and (len(active_indices) > 0):
        v0 = vertices[active_indices[0]]
        Ci = v0["C"];
        Di = v0["D"]
        Ci_m = mf.Matrix.dense(Ci)
        Di_m = mf.Matrix.dense(Di)
        CQ_DY = mf.Expr.add(mf.Expr.mul(Ci_m, Q), mf.Expr.mul(Di_m, Y))
        muI = mf.Expr.mul(I16d, mu)
        rowp1 = mf.Expr.hstack([Q, mf.Expr.transpose(CQ_DY)])
        rowp2 = mf.Expr.hstack([CQ_DY, muI])
        PERF = mf.Expr.vstack([rowp1, rowp2])
        M.constraint("perf_lmi_nominal", PERF, mf.Domain.inPSDCone(32))

    rowu1 = mf.Expr.hstack([Q, mf.Expr.transpose(Y)])
    rowu2 = mf.Expr.hstack([Y, I4e])
    UINC = mf.Expr.vstack([rowu1, rowu2])
    M.constraint("input_inc", UINC, mf.Domain.inPSDCone(20))

    term_main = mf.Expr.add(mf.Expr.mul(float(w_gamma), gamma2), mf.Expr.mul(float(w_mu), mu))
    obj = mf.Expr.add(term_main, mf.Expr.mul(float(w_beta), beta))
    M.objective("min_obj", mf.ObjectiveSense.Minimize, obj)
    validation = None
    try:
        M.solve()

        validation = validate_mosek_solution(M, allow_feasible=True)
        if not validation["accepted"]:
            raise RuntimeError(
                "MOSEK result rejected by the shared status policy: "
                + validation["reason"]
            )

        Qv = fusion_matrix_level(Q, 16, 16)
        Yv = fusion_matrix_level(Y, 4, 16)
        Qv = 0.5 * (Qv + Qv.T)

        betav = max(_level_to_float(beta.level()), 0.0)
        g2v = _level_to_float(gamma2.level())
        muv = _level_to_float(mu.level())

        if np.isnan(g2v) or np.isinf(g2v):
            raise ValueError("Solver returned NaN/Inf metrics")

        Kv = (la.solve(Qv.T, Yv.T)).T

        status_payload = mosek_status_payload(validation)
        return dict(
            Q=Qv,
            Y=Yv,
            K=Kv,
            beta=np.array([betav]),
            gamma2=np.array([g2v]),
            mu=np.array([muv]),
            decay_rate=np.array([float(decay_rate)]),
            solver_threads=np.array([num_threads]),
            solver_max_iters=np.array([max_iters]),
            solver_rel_gap=np.array([rel_gap]),
            solver_time_limit=np.array([time_limit_sec]),
            beta_lb=np.array([beta_lb]),
            w_gamma=np.array([w_gamma]),
            w_mu=np.array([w_mu]),
            w_beta=np.array([w_beta]),
            objective_value=np.array([w_gamma * g2v + w_mu * muv + w_beta * betav]),
            **status_payload,
            enforce_perf_all_vertices=np.array([1 if enforce_perf_all_vertices else 0]),
            success=True
        )

    except Exception as e:
        print(f"  Optimization failed: {str(e)}")
        print("  Returning dummy solution")

        status_payload = mosek_exception_payload(e, validation)
        return dict(
            Q=np.eye(16),
            Y=np.zeros((4, 16)),
            K=np.zeros((4, 16)),
            beta=np.array([0.0]),
            gamma2=np.array([1e9]),
            mu=np.array([1e9]),
            decay_rate=np.array([float(decay_rate)]),
            solver_threads=np.array([num_threads]),
            solver_max_iters=np.array([max_iters]),
            solver_rel_gap=np.array([rel_gap]),
            solver_time_limit=np.array([time_limit_sec]),
            beta_lb=np.array([beta_lb]),
            w_gamma=np.array([w_gamma]),
            w_mu=np.array([w_mu]),
            w_beta=np.array([w_beta]),
            objective_value=np.array([float("inf")]),
            **status_payload,
            enforce_perf_all_vertices=np.array([0]),
            success=False
        )


# =========================================================
# Plotting utilities
# =========================================================
def eval_vertex_lmi_violation(
        v: Dict[str, object],
        K: np.ndarray,
        Q: np.ndarray,
        gamma2_val: float,
        beta_val: float,
        decay_rate: float,
) -> float:
    """
    """
    Ai = v["A"]; Bi = v["B"]; Ci = v["C"]; Di = v["D"]
    Sci = v["S"]
    s_i = float(v["s"])

    Y = K @ Q
    AQ_BY = Ai @ Q + Bi @ Y
    CQ_DY = Ci @ Q + Di @ Y

    n = 16; nw = 6
    tl = -decay_rate * Q - beta_val * s_i * np.eye(n)

    LMI = np.zeros((n + nw + n + n, n + nw + n + n))
    # row/col 0:16, 16:22, 22:38, 38:54
    # Block (1,1)
    LMI[0:n, 0:n] = tl
    # Block (1,3)
    LMI[0:n, n+nw:n+nw+n] = AQ_BY.T
    # Block (1,4)
    LMI[0:n, n+nw+n:] = CQ_DY.T
    # Block (2,2)
    LMI[n:n+nw, n:n+nw] = -gamma2_val * np.eye(nw)
    # Block (2,3)
    LMI[n:n+nw, n+nw:n+nw+n] = Sci.T
    # Block (3,1)
    LMI[n+nw:n+nw+n, 0:n] = AQ_BY
    # Block (3,2)
    LMI[n+nw:n+nw+n, n:n+nw] = Sci
    # Block (3,3)
    LMI[n+nw:n+nw+n, n+nw:n+nw+n] = -Q
    # Block (4,1)
    LMI[n+nw+n:, 0:n] = CQ_DY
    # Block (4,4)
    LMI[n+nw+n:, n+nw+n:] = -np.eye(n)

    LMI = 0.5 * (LMI + LMI.T)
    return float(eigvalsh(LMI)[-1])


def check_all_vertex_violations(
        all_vertices: List[Dict[str, object]],
        K: np.ndarray,
        Q: np.ndarray,
        gamma2_val: float,
        beta_val: float,
        decay_rate: float,
) -> np.ndarray:
    """
    """
    violations = np.zeros(len(all_vertices))
    for i, v in enumerate(all_vertices):
        violations[i] = eval_vertex_lmi_violation(
            v, K, Q, gamma2_val, beta_val, decay_rate
        )
    return violations


def attach_full_solution_diagnostics(
        solution: Dict[str, Any],
        all_vertices: List[Dict[str, object]],
        decay_rate: float,
) -> Optional[np.ndarray]:
    if not solution.get("success", False):
        return None
    violations = check_all_vertex_violations(
        all_vertices, solution["K"], solution["Q"],
        float(solution["gamma2"][0]), float(solution["beta"][0]),
        decay_rate,
    )
    certified = np.asarray(
        [float(vertex["s"]) == 0.0 for vertex in all_vertices], dtype=bool
    )
    solution["v_sur"] = np.asarray([float(np.max(violations))])
    solution["v_cert"] = np.asarray([
        float(np.max(violations[certified])) if np.any(certified) else float("nan")
    ])
    auxiliary = evaluate_auxiliary_lmi_residuals(
        all_vertices, solution["K"], solution["Q"], float(solution["mu"][0])
    )
    for name, value in auxiliary.items():
        solution[name] = np.asarray([value])
    return violations


def iterative_constraint_exchange(
        all_vertices: List[Dict[str, object]],
        syn: SynthesisParams,
        K_budget: int = 20,
        max_rounds: int = 8,
        viol_tol: float = 1e-4,
        seed: int = 42,
        decay_rate: Optional[float] = None,
        w_gamma: float = 1.0,
        w_mu: float = 0.1,
        w_beta: float = 1e-3,
        enforce_perf_all_vertices: bool = True,
        verbose: bool = True,
        bounds: Optional[ParamBounds] = None,
        stratified_ratios: Tuple[float, float, float] = (0.4, 0.2, 0.4),
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    """
    """
    if decay_rate is None:
        decay_rate = getattr(syn, "decay_rate", 0.98)

    N = len(all_vertices)
    if N == 0 or K_budget < 1:
        raise ValueError("ICE requires at least one vertex and a positive budget")
    K_budget = min(K_budget, N)

    s_all = np.array([float(v["s"]) for v in all_vertices], dtype=float)

    # --- Tier classification: low(0) / mid(1) / high(2) by s rank ---
    order = np.argsort(s_all, kind="stable")
    tier = np.zeros(N, dtype=int)
    n3 = max(1, N // 3)
    for rank, idx in enumerate(order):
        if rank < n3:
            tier[idx] = 0
        elif rank < 2 * n3:
            tier[idx] = 1
        else:
            tier[idx] = 2

    # when all vertices have s=0 (baseline P0/P1 calls), no s>0 quota is meaningful;
    # set m_incon=0 so the swap-out search is not blocked. Otherwise keep 20% quota.
    m_incon = compute_inconsistency_quota(s_all, K_budget, rho_incon=0.2)

    # --- Initial selection: stratified farthest-point sampling ---
    if bounds is not None:
        sel_verts = select_vertices_stratified_farthest(
            all_vertices, bounds, K_budget,
            ratios=stratified_ratios, seed=seed,
        )
        id_to_idx = {id(v): i for i, v in enumerate(all_vertices)}
        active_idx = set()
        for v in sel_verts:
            vid = id(v)
            if vid in id_to_idx:
                active_idx.add(id_to_idx[vid])
        if len(active_idx) < K_budget:
            remaining = [i for i in range(N) if i not in active_idx]
            for i in remaining[:K_budget - len(active_idx)]:
                active_idx.add(i)
    else:
        sorted_idx = sorted(range(N), key=lambda i: s_all[i])
        active_idx = set(sorted_idx[:K_budget])

    active_positive = sum(s_all[i] > 0.0 for i in active_idx)
    if active_positive < m_incon:
        positive_pool = [i for i in np.argsort(-s_all, kind="stable") if s_all[i] > 0.0 and i not in active_idx]
        removable_zero = [i for i in sorted(active_idx) if s_all[i] == 0.0]
        for add_idx, remove_idx in zip(
                positive_pool[:m_incon - active_positive], removable_zero):
            active_idx.remove(remove_idx)
            active_idx.add(int(add_idx))

    if verbose:
        n_s_pos = sum(1 for i in active_idx if s_all[i] > 0.0)
        print(f"[IterExchange] Init: K={K_budget}, m_incon={m_incon}, "
              f"s>0 in active={n_s_pos}, stratified={'yes' if bounds else 'no'}")

    best_sol = None
    best_metric = (float("inf"), N, float("inf"))
    best_active_list = sorted(active_idx)
    quota_relaxations = 0
    rounds_attempted = 0

    for rnd in range(max_rounds):
        rounds_attempted = rnd + 1
        active_list = sorted(active_idx)
        active_verts = [all_vertices[i] for i in active_list]

        if verbose:
            s_vals = [float(all_vertices[i]["s"]) for i in active_list]
            n_nonzero = sum(1 for sv in s_vals if sv > 0.0)
            print(f"\n[Round {rnd+1}/{max_rounds}] Active: {len(active_list)}, "
                  f"s range: [{min(s_vals):.3e}, {max(s_vals):.3e}], s>0: {n_nonzero}")

        sol = solve_vertex_fusion_sdp_mosek(
            active_verts, syn,
            verbose=False,
            num_threads=0,
            max_iters=50,
            rel_gap=1e-4,
            time_limit_sec=1800,
            beta_lb=0.0,
            decay_rate=decay_rate,
            w_gamma=w_gamma,
            w_mu=w_mu,
            w_beta=w_beta,
            enforce_perf_all_vertices=enforce_perf_all_vertices,
            x0_feas=None,
        )

        if not sol.get("success", False):
            print(f"  [Round {rnd+1}] SDP infeasible, stopping.")
            break

        K_ctrl = sol["K"]
        Q_val = sol["Q"]
        gamma2_val = float(sol["gamma2"][0])
        beta_val = float(sol["beta"][0])
        gamma_val = float(np.sqrt(max(gamma2_val, 0.0)))

        if verbose:
            print(f"  SDP solved: gamma={gamma_val:.4f}, beta={beta_val:.4f}")

        violations = check_all_vertex_violations(
            all_vertices, K_ctrl, Q_val, gamma2_val, beta_val, decay_rate
        )

        max_viol_idx = int(np.argmax(violations))
        max_viol_val = float(violations[max_viol_idx])
        n_violated = int(np.sum(violations > viol_tol))
        n_consistent_violated = sum(
            1 for i in range(N)
            if violations[i] > viol_tol and float(all_vertices[i]["s"]) == 0.0
        )

        if verbose:
            print(f"  Verification: max_violation={max_viol_val:.6f} (vertex {max_viol_idx}), "
                  f"n_violated={n_violated}/{N}, consistent_violated={n_consistent_violated}")

        current_metric = (max_viol_val, n_violated, gamma_val)
        if current_metric < best_metric:
            best_metric = current_metric
            best_sol = sol
            best_active_list = list(active_list)

        if max_viol_val <= viol_tol:
            if verbose:
                print(f"  *** Converged: all vertices satisfy LMI (tol={viol_tol:.1e}) ***")
            best_metric = current_metric
            best_sol = sol
            best_active_list = list(active_list)
            break

        # --- Find swap-in: first violator outside active set ---
        viol_order = np.argsort(-violations, kind="stable")
        swap_in_idx = None
        for cand in viol_order:
            cand = int(cand)
            if violations[cand] <= viol_tol:
                break
            if cand not in active_idx:
                swap_in_idx = cand
                break

        if swap_in_idx is None:
            if verbose:
                print(f"  All violators already in active set, stopping.")
            break

        swap_in_viol = float(violations[swap_in_idx])
        swap_in_tier = int(tier[swap_in_idx])
        will_add_nonzero = s_all[swap_in_idx] > 0.0

        # --- Find swap-out: most redundant, respecting tier & m_incon ---
        active_violations = {i: violations[i] for i in active_idx}
        candidates_remove = sorted(active_violations.keys(), key=lambda i: active_violations[i])

        active_s_nonzero_count = sum(1 for i in active_idx if s_all[i] > 0.0)

        def _can_remove(i: int) -> bool:
            removing_nonzero = s_all[i] > 0.0
            new_count = active_s_nonzero_count - (1 if removing_nonzero else 0) + (1 if will_add_nonzero else 0)
            return new_count >= m_incon

        swap_out_idx = None
        for i in candidates_remove:
            if tier[i] == swap_in_tier and _can_remove(i):
                swap_out_idx = i
                break
        if swap_out_idx is None:
            for i in candidates_remove:
                if _can_remove(i):
                    swap_out_idx = i
                    break

        if swap_out_idx is None:
            swap_out_idx = candidates_remove[0] if candidates_remove else None
            quota_relaxations += int(swap_out_idx is not None)
            if verbose and swap_out_idx is not None:
                print(f"  Quota relaxed for this exchange (m_incon={m_incon}).")
        if swap_out_idx is None:
            break

        if verbose:
            s_new = float(all_vertices[swap_in_idx]["s"])
            s_old = float(all_vertices[swap_out_idx]["s"])
            print(f"  Swap: remove vertex {swap_out_idx} "
                  f"(viol={active_violations[swap_out_idx]:.6f}, s={s_old:.3e}, tier={tier[swap_out_idx]})"
                  f" -> add vertex {swap_in_idx} "
                  f"(viol={swap_in_viol:.6f}, s={s_new:.3e}, tier={swap_in_tier})")

        active_idx.discard(swap_out_idx)
        active_idx.add(swap_in_idx)

    if best_sol is None:
        raise RuntimeError("iterative_constraint_exchange: no feasible solution found")

    final_verts = [all_vertices[i] for i in best_active_list]
    final_violations = attach_full_solution_diagnostics(
        best_sol, all_vertices, decay_rate
    )
    if final_violations is None:
        raise RuntimeError("ICE solution diagnostics require a feasible solution")
    v_sur = float(best_sol["v_sur"][0])
    best_sol["status"] = classify_surrogate_status(True, v_sur)
    best_sol["quota_relaxations"] = np.array([quota_relaxations])
    best_sol["exchange_rounds"] = np.array([rounds_attempted])
    best_sol["active_vertex_indices"] = np.asarray(best_active_list, dtype=int)
    best_gamma = best_metric[2]
    if verbose:
        print(f"\n[IterExchange] Final: {len(final_verts)} vertices, "
              f"gamma={best_gamma:.4f}, max_viol={best_metric[0]:.6f}, n_violated={best_metric[1]}")

    return final_verts, best_sol


# =========================================================
# Plotting utilities
# =========================================================
def ref_square_wave_x(
        t: float,
        amp: float = 0.5,
        period: float = 16.0,  # aligned with paper Table 1 (T_ref = 16 s)
        start_delay: float = 1.0,
) -> np.ndarray:
    if t < start_delay:
        return np.array([0.0, 0.0, 0.0])
    tau = t - start_delay
    phase = (tau % period) / period
    val = amp if phase < 0.5 else -amp
    return np.array([val, 0.0, 0.0])


def build_reference_trajectory(
        Ts: float,
        steps: int,
        ref_fun: Callable[[float], np.ndarray],
) -> np.ndarray:
    t = np.arange(steps + 1) * Ts
    Pref = np.zeros((3, steps + 1))
    for k in range(steps + 1):
        Pref[:, k] = ref_fun(float(t[k]))
    return Pref


def square_wave_switch_times(start_delay: float, period: float, T: float) -> List[float]:
    times = []
    half = period / 2.0
    t = start_delay + half
    while t <= T + 1e-9:
        times.append(float(t))
        t += half
    return times


def pick_times_avoiding(
        rng: np.random.Generator,
        num: int,
        t_min: float,
        t_max: float,
        avoid_times: List[float],
        avoid_window: float = 0.6,
        max_tries: int = 5000,
) -> List[float]:
    out: List[float] = []
    tries = 0
    while len(out) < num and tries < max_tries:
        tries += 1
        t = float(rng.uniform(t_min, t_max))
        ok = True
        for a in avoid_times:
            if abs(t - a) <= avoid_window:
                ok = False
                break
        for a in out:
            if abs(t - a) <= avoid_window:
                ok = False
                break
        if ok:
            out.append(t)
    out.sort()
    return out


def build_multi_gust_profile(
        syn: SynthesisParams,
        seconds: float,
        num_gusts: int = 5,
        t_min: float = 2.0,
        t_max: Optional[float] = None,
        dur_range: Tuple[float, float] = (0.6, 1.4),
        amp_range: Tuple[float, float] = (0.45, 1.0),
        bias_frac: float = 0.20,
        avoid_ref_jumps: bool = True,
        ref_start_delay: float = 1.0,
        ref_period: float = 16.0,  # aligned with paper Table 1 (T_ref = 16 s)
        seed_offset: int = 888,
) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, object]]:
    Ts = syn.Ts
    steps = int(round(seconds / Ts))
    if t_max is None:
        t_max = max(t_min + 0.5, seconds - 1.0)

    rng = np.random.default_rng(syn.seed + seed_offset)

    avoid_times: List[float] = []
    if avoid_ref_jumps:
        avoid_times = square_wave_switch_times(ref_start_delay, ref_period, seconds)

    gust_starts = pick_times_avoiding(
        rng=rng,
        num=num_gusts,
        t_min=t_min,
        t_max=t_max,
        avoid_times=avoid_times,
        avoid_window=0.7,
    )

    gusts: List[Dict[str, float]] = []
    dbar = np.zeros((6, steps), dtype=float)

    if bias_frac > 0:
        v = rng.standard_normal(6)
        v = v / (np.linalg.norm(v) + 1e-12)
        bias = DEFAULT_SCALES.T_d @ (v * bias_frac)
    else:
        bias = np.zeros(6, dtype=float)

    dbar[:, :] = bias.reshape(6, 1)

    for t0 in gust_starts:
        dur = float(rng.uniform(dur_range[0], dur_range[1]))
        t1 = min(seconds, t0 + dur)
        amp = float(rng.uniform(amp_range[0], amp_range[1]))

        v = rng.standard_normal(6)
        v = v / (np.linalg.norm(v) + 1e-12)
        gvec = DEFAULT_SCALES.T_d @ (v * amp)

        k0 = int(round(t0 / Ts))
        k1 = int(round(t1 / Ts))
        k0 = int(np.clip(k0, 0, steps - 1))
        k1 = int(np.clip(k1, k0 + 1, steps))

        dbar[:, k0:k1] += gvec.reshape(6, 1)
        gusts.append(dict(t0=float(t0), t1=float(t1), amp=float(amp)))

    dbar, projection_stats = project_disturbance_physical(dbar, DEFAULT_SCALES)
    return dbar, gusts, projection_stats


# =========================================================
# Plotting utilities
# =========================================================
def apply_increment_limits(
        uc: np.ndarray,
        syn: SynthesisParams,
) -> Tuple[np.ndarray, Dict[str, bool]]:
    flags = dict(rate_sat=False, norm_sat=False)
    uc2 = uc.copy()

    if syn.du_max_vec is not None:
        umax = np.array(syn.du_max_vec, dtype=float).reshape(4)
        before = uc2.copy()
        uc2 = clip_vec(uc2, -umax, umax)
        if np.any(np.abs(uc2 - before) > 1e-12):
            flags["rate_sat"] = True

    uc2, projection_factor = project_increment_physical(uc2, DEFAULT_SCALES)
    flags["norm_sat"] = bool(projection_factor < 1.0 - 1e-12)

    return uc2, flags


def apply_absolute_actuator_limits(
        u_abs: np.ndarray,
        uc: np.ndarray,
        syn: SynthesisParams,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, bool]]:
    flags = dict(abs_sat=False)

    umin = np.array(syn.u_abs_min, dtype=float).reshape(4)
    umax = np.array(syn.u_abs_max, dtype=float).reshape(4)

    u_next_raw = u_abs + uc
    u_next = clip_vec(u_next_raw, umin, umax)
    if np.any(np.abs(u_next - u_next_raw) > 1e-12):
        flags["abs_sat"] = True

    uc_eff = u_next - u_abs
    return uc_eff, u_next, flags


def simulate_tracking_with_disturbance_profile(
        K: np.ndarray,
        p_true: Dict[str, float],
        syn: SynthesisParams,
        ref_fun: Callable[[float], np.ndarray],
        seconds: float,
        dbar_profile: Optional[np.ndarray],
        meas_noise_std: float = 0.0,
        noise_seed: Optional[int] = None,
) -> Dict[str, Any]:
    Ts, g = syn.Ts, syn.g
    # accept an explicit per-trial noise seed; fall back to the default fixed seed
    # when None for backward compatibility (single-trial Stage-3 time-domain study).
    rng = np.random.default_rng(syn.seed + 2026 if noise_seed is None else int(noise_seed))

    steps = int(round(seconds / Ts))
    A, B, S_c = build_physical_matrices(p_true, Ts, g)
    K_phys = gain_to_physical(K, DEFAULT_SCALES)

    Pref = build_reference_trajectory(Ts, steps, ref_fun)

    if dbar_profile is None:
        dbar_profile = np.zeros((6, steps), dtype=float)
    dbar_profile = np.asarray(dbar_profile, dtype=float)
    assert dbar_profile.shape == (6, steps)
    dbar_profile, disturbance_stats = project_disturbance_physical(
        dbar_profile, DEFAULT_SCALES
    )

    x = np.zeros((16, steps + 1))
    u_c_raw = np.zeros((4, steps))
    u_c_eff = np.zeros((4, steps))
    u_abs = np.zeros((4, steps + 1))
    dbar = dbar_profile.copy()

    flags_rate = np.zeros(steps, dtype=int)
    flags_norm = np.zeros(steps, dtype=int)
    flags_abs = np.zeros(steps, dtype=int)

    x0 = np.zeros(16)
    x0[0:3] = np.array([0.0, 0.0, 0.0])
    x0[6:9] = np.deg2rad([2.0, -1.5, 0.0])
    x0[12:16] = np.zeros(4)
    x[:, 0] = x0
    u_abs[:, 0] = x0[12:16].copy()

    for k in range(steps):
        x_ref = np.zeros(16)
        x_ref[0:3] = Pref[:, k]
        x_feedback = x[:, k].copy()
        if meas_noise_std > 0:
            nu, _ = sample_capped_physical_residual(
                rng, DEFAULT_SCALES, residual_std=meas_noise_std
            )
            x_feedback[0:12] += nu
        e = x_feedback - x_ref

        uc = (K_phys @ e).reshape(4)
        u_c_raw[:, k] = uc

        uc_limited, f1 = apply_increment_limits(uc, syn)
        u_curr = x[12:16, k].copy()
        uc_eff_k, u_next, f2 = apply_absolute_actuator_limits(u_curr, uc_limited, syn)

        flags_rate[k] = int(f1["rate_sat"])
        flags_norm[k] = int(f1["norm_sat"])
        flags_abs[k] = int(f2["abs_sat"])

        u_c_eff[:, k] = uc_eff_k
        u_abs[:, k + 1] = u_next

        x[:, k + 1] = A @ x[:, k] + B @ uc_eff_k + S_c @ dbar[:, k]
        x[12:16, k + 1] = u_next

    t = np.arange(steps + 1) * Ts
    rho_lin = spectral_radius(A + B @ K_phys)
    raw_increment_norm = np.linalg.norm(increment_to_normalized(u_c_raw), axis=0)
    applied_increment_norm = np.linalg.norm(increment_to_normalized(u_c_eff), axis=0)

    return dict(
        t=t,
        x=x,
        u_c_raw=u_c_raw,
        u_c=u_c_eff,
        u_abs=u_abs,
        dbar=dbar,
        Pref=Pref,
        p_true=p_true,
        A_true=A,
        B_true=B,
        S_true=S_c,
        K_physical=K_phys,
        rho=np.array([rho_lin]),
        sat_rate=flags_rate,
        sat_norm=flags_norm,
        sat_abs=flags_abs,
        increment_raw_peak_normalized=float(np.max(raw_increment_norm)),
        increment_applied_peak_normalized=float(np.max(applied_increment_norm)),
        disturbance_stats=disturbance_stats,
    )


# =========================================================
# Plotting utilities
# =========================================================


# =========================================================
# Plotting utilities
# =========================================================


# =========================================================
# Plotting utilities
# =========================================================


# =========================================================
# Plotting utilities
# =========================================================




# =========================================================
# Plotting utilities
# =========================================================

# =========================================================
# Plotting utilities
# =========================================================
def synthesize_controllers_for_stage3(
        syn: SynthesisParams,
        bounds: ParamBounds,
        max_vertices_proposed: int = 256,
        direct_vertex_limit: int = 20,
        score_mode: str = "lambda_max",
        enforce_perf_all_vertices: bool = True,

        baseline1_vertices: int = 20,
        baseline1_threads: int = 1,
) -> Dict[str, object]:
    p_nom = {k: 0.5 * (getattr(bounds, k)[0] + getattr(bounds, k)[1])
             for k in bounds.__dataclass_fields__.keys()}
    A_nom_lqr, B_nom_lqr, _ = build_physical_matrices(p_nom, syn.Ts, syn.g)
    Qlqr = np.diag(syn.Qx_lqr)
    Rlqr = np.diag(syn.Ru_lqr)
    K_nom_lqr = gain_to_normalized(
        dlqr(A_nom_lqr, B_nom_lqr, Qlqr, Rlqr), DEFAULT_SCALES
    )
    rng_data = np.random.default_rng(syn.seed + 999)
    p_data_source = {}
    print("\n[Data Generation] Sampling random 'historical' plant parameters within bounds:")
    for k in bounds.__dataclass_fields__.keys():
        low, high = getattr(bounds, k)
        val = float(rng_data.uniform(low, high))
        p_data_source[k] = val

    print(f"  sigma_t: {p_data_source['sigma_t']:.4f} (Nom: {p_nom['sigma_t']:.4f})")

    print(f"  Simulating batch data (L=1200) from random plant...")
    batch = simulate_batch_data(
        p_data_source, syn, L=1200, excite_scale=1.2, meas_noise_std=0.0005,
        K_fb=K_nom_lqr, p_lqr=p_nom
    )
    batch_sat = batch["input_saturation_stats"]
    print("  [Offline input] increment/absolute saturation rates: "
          f"{batch_sat['increment_saturation_rate']:.4f}/"
          f"{batch_sat['absolute_saturation_rate']:.4f}; "
          "normalized command/applied increment peaks: "
          f"{batch_sat['command_peak_normalized']:.4f}/"
          f"{batch_sat['applied_peak_normalized']:.4f}")

    # --- Fixed disturbance bound (paper-consistent: $\bar d_{\max}$ in Table 1) ---
    print(f"  [Stage3] Fixed common disturbance channel scale: d_max = {syn.d_max:.4f}")

    # --- PSD upper bound W_c on the disturbance-injection Gram matrix ---
    # The endpoint maximum is the box-wide maximum for this disturbance
    # structure. Interior and generator values are diagnostic checks.
    W_c, info_Wc = build_disturbance_psd_bound(
        bounds, syn, batch_generator=p_data_source,
        n_internal_samples=1000, safety_factor=1.05,
    )
    print(f"  [W_c] alpha_c = {info_Wc['alpha']:.6e}  "
          f"(safety_factor={info_Wc['safety_factor']:.3f}, "
          f"vertex_max_eig={info_Wc['vertex_max_eig']:.6e}, "
          f"interior_max_eig={info_Wc['interior_max_eig']:.6e})")
    print(f"  [W_c verify] vertex set : min PSD gap = {info_Wc['min_psd_gap_vertex']:.3e}, "
          f"violations = {info_Wc['n_violate_vertex']}/{info_Wc['n_vertices']}  (formal)")
    print(f"  [W_c verify] interior   : min PSD gap = {info_Wc['min_psd_gap_interior']:.3e}, "
          f"violations = {info_Wc['n_violate_interior']}/{info_Wc['n_internal_samples']}  (sanity)")
    print(f"  [W_c generator diagnostic] endpoint alpha={info_Wc['alpha']:.6e}, "
          f"generator max eig={info_Wc['generator_max_eig']:.6e}, "
          f"generator PSD gap={info_Wc['generator_psd_gap']:.3e}")

    # --- structured sampled-state residual upper bound W_E (paper eq:WE_bound) ---
    # W_E is built from syn.sigma_E_x (12-state physical block residual std) and is
    # purely a synthesis-side bound; it is NOT estimated from data to preserve data
    # consistency semantics. See build_structured_residual_bound() for the structure.
    L_batch = int(batch["X_t"].shape[1])
    W_E = build_structured_residual_bound(L_batch, float(syn.nu_E_max))
    print(f"  [W_E] deterministic normalized residual bound: "
          f"||W_E^x||_op = {np.max(np.diag(W_E[:12, :12])):.3e}; "
          f"W_E^z = 0 (Z_t is a deterministic readback)")

    record_residual_check = verify_realized_record_residual(batch, W_E)
    batch["record_residual_check"] = record_residual_check
    print(f"  [W_E realized check] min PSD gap = "
          f"{record_residual_check['minimum_psd_gap']:.3e}; "
          f"max output residual = "
          f"{record_residual_check['maximum_output_residual']:.3e}; "
          f"max input-memory residual = "
          f"{record_residual_check['maximum_input_memory_residual']:.3e}")

    # --- Build Psi using full batch ---
    batch["W_c"] = W_c           # PSD upper bound used by build_psi_data
    batch["W_E"] = W_E           # structured residual upper bound used by build_psi_data
    Psi = build_psi_data(batch, syn)

    # --- model-mismatch diagnostic over ALL prior vertices (NOT a residual check) ---
    # On all prior vertices E_i mixes residual with model mismatch, so this only shows
    # how large the worst-case prior-model deviation is.
    model_residual_all = verify_data_residual_bound(
        batch, syn, bounds, W_c, verbose=True
    )

    verts_all = build_all_vertices_and_scores(
        bounds, syn, Psi,
        score_mode=score_mode,
        max_vertices=1024,
        seed=syn.seed + 1234,
        batch=batch,
    )

    raw_s_values = np.array([v["s"] for v in verts_all], dtype=float)
    print(
        f"\nScore statistics: n={len(raw_s_values)}, "
        f"min={raw_s_values.min():.4e}, max={raw_s_values.max():.4e}, "
        f"mean={raw_s_values.mean():.4e}"
    )

    # ===== Score processing for score-weighted surrogate relaxation =====
    # Fixed d_max regime: Psi_t built on the full batch; L_data = L.
    L_data = batch["X_t"].shape[1]
    verts_norm, score_diag = compute_si_from_vi(
        verts_all,
        L_data,
        tau=0.0,
        q_scale=0.9,
        rho=0.1,
        hard_ratio_min=0.01,
    )

    s_norm = np.array([float(v["s"]) for v in verts_norm], dtype=float)
    print(f"[Vertices-Pool] robust s: min={s_norm.min():.3e}, max={s_norm.max():.3e}, "
          f"n={len(verts_norm)}, n_zero={score_diag['n_consistent']}, "
          f"soft_mode={score_diag['soft_mode']}")

    # --- a-posteriori residual diagnostic on processed score-zero I0.
    # Note that compute_si_from_vi may threshold s_i^raw > 0 to s_i = 0 via tau_eff;
    # therefore I0 here is the *processed* score-zero subset and rho_E on it is NOT a
    # proof that the residual is purely sampled noise.
    I0_vertices = [v["p"] for v in verts_norm if float(v["s"]) == 0.0]
    if I0_vertices:
        print(f"  [I0 residual diagnostic] |I0| (processed s==0) = {len(I0_vertices)}/{len(verts_norm)}, "
              f"tau_eff = {score_diag.get('tau_effective', float('nan')):.3e}, "
              f"soft_mode = {score_diag.get('soft_mode', False)}")
        model_residual_i0 = verify_data_residual_bound(
            batch, syn, bounds, W_c,
            score_zero_vertices=I0_vertices, verbose=True,
        )
    else:
        model_residual_i0 = None
        print(f"  [I0 residual diagnostic] processed score-zero subset I0 is EMPTY; "
              f"residual diagnostic on I0 is skipped (only model-mismatch diagnostic available)")

    diagnostic_payload = build_fusion_diagnostics_payload(
        batch, verts_norm, Psi, W_c, W_E, info_Wc, score_diag,
        list(bounds.__dataclass_fields__.keys()), p_data_source,
        model_residual_all=model_residual_all,
        model_residual_certified=model_residual_i0,
    )
    diagnostic_path = result_path("caches", "fusion_ablation_data_diagnostics.npz")
    np.savez_compressed(diagnostic_path, **diagnostic_payload)
    print(f"Saved data-fusion diagnostics to: {diagnostic_path}")

    K = int(min(int(direct_vertex_limit), int(baseline1_vertices)))
    K = max(1, K)

    # ===== Iterative Constraint Exchange (Cutting-Plane) =====
    print(f"\n{'='*60}")
    print(f"[IterConstraintExchange] Starting with K_budget={K}, max_rounds=4")
    print(f"{'='*60}")

    verts_common, sol_prop = iterative_constraint_exchange(
        all_vertices=verts_norm,
        syn=syn,
        K_budget=K,
        max_rounds=4,
        viol_tol=1e-4,
        seed=syn.seed + 7777,
        decay_rate=syn.decay_rate,
        w_gamma=syn.w_gamma,
        w_mu=syn.w_mu,
        w_beta=syn.w_beta,
        enforce_perf_all_vertices=enforce_perf_all_vertices,
        verbose=True,
        bounds=bounds,
    )

    K_prop = sol_prop["K"]
    gamma_prop = float(np.sqrt(max(float(sol_prop["gamma2"][0]), 0.0)))


    # ===== Matched-active-set ablation: proposed active set, s=0 =====
    verts_baseB = []
    for v in verts_common:
        vv = dict(v)
        vv["s"] = 0.0
        verts_baseB.append(vv)

    print(f"\n[BaselineB] Same {len(verts_baseB)} vertices as Proposed, s_i=0 (no relaxation). threads={baseline1_threads}")

    sol_baseB = solve_vertex_fusion_sdp_mosek(
        verts_baseB, syn,
        verbose=True,
        num_threads=int(baseline1_threads),
        max_iters=50,
        rel_gap=1e-4,
        time_limit_sec=1800,
        beta_lb=0.0,
        decay_rate=syn.decay_rate,
        w_gamma=syn.w_gamma,
        w_mu=syn.w_mu,
        w_beta=syn.w_beta,
        enforce_perf_all_vertices=enforce_perf_all_vertices,
        x0_feas=None,
    )
    attach_full_solution_diagnostics(sol_baseB, verts_norm, syn.decay_rate)

    if sol_baseB.get("success", False):
        K_baseB = sol_baseB["K"]
        gamma_baseB = float(np.sqrt(max(float(sol_baseB["gamma2"][0]), 0.0)))
    else:
        print("  BaselineB optimization failed, using zero gain")
        K_baseB = np.zeros((4, 16))
        gamma_baseB = 9999.9

    K_nom = batch["K0"]

    # ===== P0: Prior-only robust (all vertices s_i=0, ICE) =====
    print(f"\n{'='*60}")
    print(f"[P0] Prior-only robust: {len(verts_norm)} vertices, all s_i=0")
    print(f"{'='*60}")
    verts_P0_all = [dict(v, s=0.0) for v in verts_norm]
    verts_P0_active, sol_P0 = iterative_constraint_exchange(
        all_vertices=verts_P0_all, syn=syn, K_budget=K,
        max_rounds=4, viol_tol=1e-4, seed=syn.seed + 8888,
        decay_rate=syn.decay_rate, w_gamma=syn.w_gamma, w_mu=syn.w_mu,
        w_beta=syn.w_beta,
        enforce_perf_all_vertices=enforce_perf_all_vertices,
        verbose=True, bounds=bounds,
    )
    if sol_P0.get("success", False):
        K_P0 = sol_P0["K"]
        gamma_P0 = float(np.sqrt(max(float(sol_P0["gamma2"][0]), 0.0)))
    else:
        print("  P0 optimization failed, using zero gain")
        K_P0 = np.zeros((4, 16))
        gamma_P0 = 9999.9

    # ===== F1a: thresholded hard selection with percentile fallback =====
    raw_s_arr = np.array([float(v["s"]) for v in verts_all], dtype=float)
    tau_f1a = 0.0
    consistent_idx = [i for i in range(len(verts_all)) if raw_s_arr[i] <= tau_f1a]
    if len(consistent_idx) < 1:
        tau_f1a = float(np.quantile(raw_s_arr, 0.05))
        consistent_idx = [i for i in range(len(verts_all)) if raw_s_arr[i] <= tau_f1a]
        print(f"  [F1a] Adaptive threshold: tau={tau_f1a:.4e} (5th percentile, no s<=0 vertices)")
    n_f1a = min(len(consistent_idx), K)
    print(f"\n{'='*60}")
    print(f"[F1a] Thresholded hard selection with fallback: "
          f"{len(consistent_idx)}/{len(verts_all)} selected (s<={tau_f1a:.4e}), using {n_f1a}")
    print(f"{'='*60}")
    if len(consistent_idx) >= 1:
        verts_F1a_pool = [dict(verts_all[i], s=0.0) for i in consistent_idx]
        if len(verts_F1a_pool) > n_f1a:
            verts_F1a_sel = select_vertices_farthest_delta(verts_F1a_pool, K=n_f1a, initial_index=0)
        else:
            verts_F1a_sel = verts_F1a_pool
        print(f"  Using {len(verts_F1a_sel)} vertices for SDP (budget K={K})")
        sol_F1a = solve_vertex_fusion_sdp_mosek(
            verts_F1a_sel, syn, verbose=True, num_threads=1,
            max_iters=50, rel_gap=1e-4, time_limit_sec=1800, beta_lb=0.0,
            decay_rate=syn.decay_rate, w_gamma=syn.w_gamma, w_mu=syn.w_mu,
            w_beta=syn.w_beta,
            enforce_perf_all_vertices=enforce_perf_all_vertices, x0_feas=None,
        )
        attach_full_solution_diagnostics(sol_F1a, verts_norm, syn.decay_rate)
        if sol_F1a.get("success", False):
            K_F1a = sol_F1a["K"]
            gamma_F1a = float(np.sqrt(max(float(sol_F1a["gamma2"][0]), 0.0)))
        else:
            print("  F1a SDP failed")
            K_F1a = np.zeros((4, 16))
            gamma_F1a = 9999.9
    else:
        print("  No consistent vertices found, F1a infeasible")
        K_F1a = np.zeros((4, 16))
        gamma_F1a = 9999.9
        sol_F1a = dict(success=False, gamma2=np.array([1e9]), K=np.zeros((4, 16)))

    # ===== F1b: top-budget hard selection by raw score =====
    order_by_raw_s = np.argsort(raw_s_arr, kind="stable")
    top_K_idx = order_by_raw_s[:K]
    verts_F1b = [dict(verts_all[int(i)], s=0.0) for i in top_K_idx]
    print(f"\n{'='*60}")
    print(f"[F1b] Top-budget hard selection: top-{K} by raw consistency score")
    print(f"  Score range of selected: [{raw_s_arr[top_K_idx[0]]:.4e}, {raw_s_arr[top_K_idx[-1]]:.4e}]")
    print(f"{'='*60}")
    sol_F1b = solve_vertex_fusion_sdp_mosek(
        verts_F1b, syn, verbose=True, num_threads=1,
        max_iters=50, rel_gap=1e-4, time_limit_sec=1800, beta_lb=0.0,
        decay_rate=syn.decay_rate, w_gamma=syn.w_gamma, w_mu=syn.w_mu,
        w_beta=syn.w_beta,
        enforce_perf_all_vertices=enforce_perf_all_vertices, x0_feas=None,
    )
    attach_full_solution_diagnostics(sol_F1b, verts_norm, syn.decay_rate)
    if sol_F1b.get("success", False):
        K_F1b = sol_F1b["K"]
        gamma_F1b = float(np.sqrt(max(float(sol_F1b["gamma2"][0]), 0.0)))
    else:
        print("  F1b SDP failed")
        K_F1b = np.zeros((4, 16))
        gamma_F1b = 9999.9

    # ===== S0: Data-only (LS identification + LQR) =====
    print(f"\n{'='*60}")
    print(f"[S0] Data-only: LS identification from batch + LQR")
    print(f"{'='*60}")
    X_t_s0 = batch["X_t"]
    U_t_s0 = batch["U_t"]
    X_tp1_s0 = batch["X_tp1"]
    ZZ = np.vstack([X_t_s0, U_t_s0])
    AB_hat = X_tp1_s0 @ np.linalg.pinv(ZZ)
    A_hat_s0 = AB_hat[:, :X_t_s0.shape[0]]
    B_hat_s0 = AB_hat[:, X_t_s0.shape[0]:]
    rho_hat = spectral_radius(A_hat_s0)
    print(f"  LS model: rho(A_hat)={rho_hat:.4f}, dim A={A_hat_s0.shape}, B={B_hat_s0.shape}")
    Qlqr_s0 = DEFAULT_SCALES.T_xc.T @ np.diag(syn.Qx_lqr) @ DEFAULT_SCALES.T_xc
    Rlqr_s0 = DEFAULT_SCALES.T_du.T @ np.diag(syn.Ru_lqr) @ DEFAULT_SCALES.T_du
    try:
        K_S0 = dlqr(A_hat_s0, B_hat_s0, Qlqr_s0, Rlqr_s0)
        rho_cl = spectral_radius(A_hat_s0 + B_hat_s0 @ K_S0)
        print(f"  S0 LQR ok: rho(A+BK)={rho_cl:.4f}")
    except Exception as e:
        print(f"  S0 LQR failed ({e}), falling back to nominal LQR")
        K_S0 = K_nom.copy()

    return dict(
        p_nom=p_nom,
        p_data_source=p_data_source,
        batch=batch,
        Psi=Psi,
        W_c=W_c,
        W_E=W_E,
        wc_diagnostics=info_Wc,
        score_diagnostics=score_diag,
        verts_norm=verts_norm,
        vertices_common=verts_common,
        K_proposed=K_prop,
        gamma_proposed=gamma_prop,
        sol_proposed=sol_prop,
        K_baselineB=K_baseB,
        gamma_baselineB=gamma_baseB,
        sol_baselineB=sol_baseB,
        K_nominal=K_nom,
        K_P0=K_P0,
        gamma_P0=gamma_P0,
        sol_P0=sol_P0,
        K_F1a=K_F1a,
        gamma_F1a=gamma_F1a,
        sol_F1a=sol_F1a,
        K_F1b=K_F1b,
        gamma_F1b=gamma_F1b,
        sol_F1b=sol_F1b,
        K_S0=K_S0,
    )


# =========================================================
# Plotting utilities
# =========================================================


# =========================================================
# Plotting utilities
# =========================================================
def run_monte_carlo_campaign(
        syn: SynthesisParams,
        bounds: ParamBounds,
        controllers: Dict[str, np.ndarray],
        N_mc: int = 2000,
        seconds: float = 30.0,
        ref_amp: float = 0.5,
        ref_period: float = 16.0,
) -> Tuple[Dict, List, Dict, Dict, Dict]:
    print(f"\n{'=' * 80}")
    print(f"[Monte Carlo] Starting All-in-One Campaign: N={N_mc} trials")
    print(f"{'=' * 80}")

    rng_mc = np.random.default_rng(syn.seed + 7777)

    # 'Success' is a per-trial boolean mask; failed trials store NaN in RMSE/Peak/Energy.
    metrics = {k: {'RMSE': [], 'Peak': [], 'Energy': [], 'Success': [], 'SuccessCount': 0,
                   'IncrementSatRate': [], 'AbsoluteSatRate': [],
                   'CommandPeakNormalized': [], 'AppliedPeakNormalized': []}
               for k in controllers.keys()}
    failure_info = {k: [] for k in controllers.keys()}
    deviations = []
    all_params = []
    campaign_disturbance_stats = []

    envelope_data = {k: [] for k in controllers.keys()}

    keys = list(bounds.__dataclass_fields__.keys())
    p_nom_vec = np.array([np.mean(getattr(bounds, k)) for k in keys])
    p_range_vec = np.array([getattr(bounds, k)[1] - getattr(bounds, k)[0] for k in keys])

    ref_fun = lambda tt: ref_square_wave_x(tt, amp=ref_amp, period=ref_period, start_delay=1.0)
    DIVERGENCE_THRESHOLD = 5.0

    for i in range(N_mc):
        p_cand = {}
        p_curr_vec = []
        for k in keys:
            val = float(rng_mc.uniform(*getattr(bounds, k)))
            p_cand[k] = val
            p_curr_vec.append(val)

        p_curr_arr = np.array(p_curr_vec)
        dev = np.linalg.norm((p_curr_arr - p_nom_vec) / (np.abs(p_nom_vec) + 1e-12))
        deviations.append(dev)
        all_params.append(p_curr_arr.copy())

        dbar_profile, _, d_stats = build_multi_gust_profile(
            syn=syn, seconds=seconds, num_gusts=5, bias_frac=0.4, seed_offset=i * 100
        )
        campaign_disturbance_stats.append(d_stats)

        # per-trial noise seed shared across controllers (within-trial fairness preserved,
        # cross-trial independence ensured). All controllers in the same trial i use the same noise.
        trial_noise_seed = syn.seed + 2026 + i * 7919
        for name, K in controllers.items():
            sim = simulate_tracking_with_disturbance_profile(
                K=K, p_true=p_cand, syn=syn, ref_fun=ref_fun,
                seconds=seconds, dbar_profile=dbar_profile, meas_noise_std=5e-4,
                noise_seed=trial_noise_seed,
            )

            e = np.linalg.norm(sim["x"][0:3, :] - sim["Pref"], axis=0)
            u_bar = increment_to_normalized(sim['u_c'])
            u_energy = np.mean(np.sum(u_bar ** 2, axis=0))
            metrics[name]['IncrementSatRate'].append(float(np.mean(sim['sat_norm'] > 0)))
            metrics[name]['AbsoluteSatRate'].append(float(np.mean(sim['sat_abs'] > 0)))
            metrics[name]['CommandPeakNormalized'].append(
                float(sim['increment_raw_peak_normalized'])
            )
            metrics[name]['AppliedPeakNormalized'].append(
                float(sim['increment_applied_peak_normalized'])
            )

            diverged = np.any(np.isnan(e)) or np.max(e) > DIVERGENCE_THRESHOLD
            if diverged:
                # store NaN sentinels so downstream masks via np.isnan are unambiguous.
                metrics[name]['RMSE'].append(float('nan'))
                metrics[name]['Peak'].append(float('nan'))
                metrics[name]['Energy'].append(float('nan'))
                metrics[name]['Success'].append(False)
                failure_info[name].append(i)
                # Preserve a NaN trace so failed trials remain explicit in the
                # released trajectory archive and are excluded from finite summaries.
                envelope_data[name].append((sim['t'], np.full_like(e, np.nan)))
            else:
                metrics[name]['SuccessCount'] += 1
                metrics[name]['RMSE'].append(float(np.sqrt(np.mean(e ** 2))))
                metrics[name]['Peak'].append(float(np.max(e)))
                metrics[name]['Energy'].append(float(u_energy))
                metrics[name]['Success'].append(True)
                # Store the unclipped successful-trial trajectory in the raw archive.
                err_norm = np.linalg.norm(sim["x"][0:3, :] - sim["Pref"], axis=0)
                envelope_data[name].append((sim['t'], err_norm))

        if (i + 1) % 200 == 0:
            print(f"  MC Progress: {i + 1}/{N_mc}...")

    devs_arr = np.array(deviations)
    params_arr = np.array(all_params)
    lower = np.array([getattr(bounds, k)[0] for k in keys])
    upper = np.array([getattr(bounds, k)[1] for k in keys])
    outside_mask = np.any((params_arr < lower) | (params_arr > upper), axis=1)
    parameter_diagnostics = dict(
        keys=keys,
        samples=params_arr,
        minimum=np.min(params_arr, axis=0),
        maximum=np.max(params_arr, axis=0),
        outside_mask=outside_mask,
        outside_before=int(np.sum(outside_mask)),
        outside_after=int(np.sum(outside_mask)),
        disturbance_stats=campaign_disturbance_stats,
    )
    print(f"  Parameter-envelope check: outside prior = "
          f"{parameter_diagnostics['outside_after']}/{N_mc}")

    # ---- Table 1: Summary statistics ----
    target_order = ["NominalLQR", "S0(DataOnly)", "P0(PriorOnly)",
                    "P1(NoRelax)", "F1a(HardStrict)", "F1b(HardBudget)", "Proposed"]
    print("\n" + "=" * 120)
    print(f">>> [Table 1] Monte Carlo Statistics Summary (N={N_mc})")
    print("=" * 120)
    hdr = (f"{'Controller':<22} | {'RMSE Mean':>10} | {'RMSE Med':>10} | {'RMSE Std':>10} | "
           f"{'Peak Mean':>10} | {'Peak Max':>10} | {'Energy Mean':>12} | {'Success':>8}")
    print(hdr)
    print("-" * 120)
    for name in target_order:
        if name not in metrics:
            continue
        m = metrics[name]
        rmse = np.array(m['RMSE'], dtype=float)
        peak = np.array(m['Peak'], dtype=float)
        energy = np.array(m['Energy'], dtype=float)
        # use NaN-based mask (sentinel-free); successful-trial statistics only.
        valid = ~np.isnan(rmse)
        sc = m['SuccessCount']
        if np.sum(valid) > 0:
            print(f"{name:<22} | {np.mean(rmse[valid]):10.4f} | {np.median(rmse[valid]):10.4f} | "
                  f"{np.std(rmse[valid]):10.4f} | {np.mean(peak[valid]):10.4f} | "
                  f"{np.max(peak[valid]):10.4f} | {np.mean(energy[valid]):12.4f} | "
                  f"{sc / N_mc * 100:7.1f}%")
        else:
            print(f"{name:<22} | {'--':>10} | {'--':>10} | {'--':>10} | "
                  f"{'--':>10} | {'--':>10} | {'--':>12} | {sc / N_mc * 100:7.1f}%")
    print("-" * 120)

    # ---- Table 2: Improvement of Proposed vs each baseline (median RMSE) ----
    # replace `< 4.0` sentinel filter with NaN-mask filter.
    if "Proposed" in metrics:
        prop_rmse = np.array(metrics["Proposed"]["RMSE"], dtype=float)
        med_ours_rmse = np.median(prop_rmse[~np.isnan(prop_rmse)])
        print(f"\n>>> [Table 2] Proposed Improvement (Median)")
        print(f"{'Baseline':<22} | {'RMSE Imp%':>10}")
        print("-" * 36)
        for name in target_order:
            if name == "Proposed" or name not in metrics:
                continue
            r = np.array(metrics[name]["RMSE"], dtype=float)
            vm_r = ~np.isnan(r)
            if np.sum(vm_r) > 0:
                med_r = np.median(r[vm_r])
                imp_r = (med_r - med_ours_rmse) / med_r * 100 if med_r > 1e-8 else 0
                print(f"{name:<22} | {imp_r:+10.2f}%")
        print()

    # ---- Table 3: Failure analysis ----
    print(f"{'Controller':<22} | {'#Fail':>6} | {'Fail Dev Mean':>14} | {'Fail Dev Med':>13} | "
          f"{'All Dev Mean':>13} | {'All Dev Med':>12}")
    print("-" * 100)
    all_dev_mean = np.mean(devs_arr)
    all_dev_med = np.median(devs_arr)
    for name in target_order:
        if name not in failure_info:
            continue
        fail_idx = failure_info[name]
        n_fail = len(fail_idx)
        if n_fail > 0:
            fail_devs = devs_arr[fail_idx]
            print(f"{name:<22} | {n_fail:6d} | {np.mean(fail_devs):14.4f} | {np.median(fail_devs):13.4f} | "
                  f"{all_dev_mean:13.4f} | {all_dev_med:12.4f}")
        else:
            print(f"{name:<22} | {0:6d} |           --     |          --     | "
                  f"{all_dev_mean:13.4f} | {all_dev_med:12.4f}")
    print("-" * 100)

    if any(len(v) > 0 for v in failure_info.values()):
        all_fail_idx = sorted(set().union(*[set(v) for v in failure_info.values()]))
        proposed_fail = set(failure_info.get("Proposed", []))
        print(f"\n  Total unique failed trials (any controller): {len(all_fail_idx)}")
        print(f"  Proposed failed trials: {len(proposed_fail)}")
        if len(all_fail_idx) > 0:
            fail_dev_all = devs_arr[all_fail_idx]
            succ_mask = np.ones(N_mc, dtype=bool)
            succ_mask[all_fail_idx] = False
            succ_dev = devs_arr[succ_mask]
            print(f"  Failed trials - dev: mean={np.mean(fail_dev_all):.4f}, "
                  f"median={np.median(fail_dev_all):.4f}, max={np.max(fail_dev_all):.4f}")
            if len(succ_dev) > 0:
                print(f"  Success trials - dev: mean={np.mean(succ_dev):.4f}, "
                      f"median={np.median(succ_dev):.4f}, max={np.max(succ_dev):.4f}")
    print()

    return metrics, deviations, envelope_data, failure_info, parameter_diagnostics


def plot_mc_boxplots(metrics: Dict[str, Dict[str, List[float]]], out_dir: str, syn: SynthesisParams):
    ALL_KEYS = ["NominalLQR", "S0(DataOnly)", "P0(PriorOnly)",
                "P1(NoRelax)", "F1a(HardStrict)", "F1b(HardBudget)", "Proposed"]
    ALL_LABELS = ["Nominal\nLQR", "Data-only\nLQR", "Prior-only\nbaseline",
                  "Matched-active\nablation", "Thresholded\nselection",
                  "Top-budget\nselection", "Proposed"]
    ALL_COLORS = {
        "NominalLQR":      "#999999",
        "S0(DataOnly)":    "#CC79A7",
        "P0(PriorOnly)":   "#E69F00",
        "P1(NoRelax)":     "#D55E00",
        "F1a(HardStrict)": "#56B4E9",
        "F1b(HardBudget)": "#009E73",
        "Proposed":        "#0072B2",
    }

    keys = [k for k in ALL_KEYS if k in metrics]
    labels = [ALL_LABELS[ALL_KEYS.index(k)] for k in keys]

    if len(keys) < 2:
        print("[Warning] Not enough controllers for boxplot.")
        return

    sizes = set_publication_style(context=syn.fig_context, column="double")
    fig, axes = plt.subplots(1, 2, figsize=(sizes["double"][0] * 1.4, sizes["double"][1] * 0.95),
                             constrained_layout=True)

    Y_LIMITS = {"RMSE": (0, 1.0), "Peak": (0, 2.0)}
    metric_types = [("RMSE", "RMSE (m)"), ("Peak", "Peak Error (m)")]

    for ax, (m_key, m_label) in zip(axes, metric_types):
        # filter NaN sentinels before plotting to keep boxplot statistics on successful trials.
        data = [np.asarray(metrics[k][m_key], dtype=float) for k in keys]
        data = [d[~np.isnan(d)] for d in data]

        bplot = ax.boxplot(
            data, patch_artist=True, labels=labels,
            showfliers=True,
            flierprops=dict(marker='x', markerfacecolor='gray', markeredgecolor='gray',
                            markersize=3, alpha=0.5, markeredgewidth=0.5),
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )

        for patch, k in zip(bplot['boxes'], keys):
            patch.set_facecolor(ALL_COLORS.get(k, "#AAAAAA"))
            patch.set_alpha(0.75)

        ax.set_ylabel(m_label)
        ax.set_ylim(*Y_LIMITS[m_key])
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', labelsize=5.5, rotation=0)

        if "Proposed" in keys:
            # median computed only on NaN-filtered successful trials.
            prop_arr = np.asarray(metrics["Proposed"][m_key], dtype=float)
            med_ours = float(np.median(prop_arr[~np.isnan(prop_arr)]))
            lines = []
            for k in keys:
                if k == "Proposed":
                    continue
                k_arr = np.asarray(metrics[k][m_key], dtype=float)
                k_arr = k_arr[~np.isnan(k_arr)]
                if k_arr.size == 0:
                    continue
                med_k = float(np.median(k_arr))
                if med_k > 1e-6:
                    imp = (med_k - med_ours) / med_k * 100
                    short_labels = {
                        "NominalLQR": "Nominal LQR",
                        "S0(DataOnly)": "Data-only",
                        "P0(PriorOnly)": "Prior-only baseline",
                        "P1(NoRelax)": "Matched-active ablation",
                        "F1a(HardStrict)": "Threshold fallback",
                        "F1b(HardBudget)": "Top budget",
                    }
                    short = short_labels.get(k, k.split("(")[0] if "(" in k else k)
                    lines.append(f"vs {short}: {imp:+.1f}%")
            if lines:
                txt = "Relative median reduction\n" + "\n".join(lines)
                ax.text(0.98, 0.97, txt, transform=ax.transAxes,
                        ha='right', va='top', fontsize=4.8, family='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

        n_out = sum(int(np.sum(d > Y_LIMITS[m_key][1])) for d in data)
        if n_out > 0:
            ax.text(0.02, 0.97, f"{n_out} outliers\nabove axis",
                    transform=ax.transAxes, ha='left', va='top', fontsize=5,
                    color='red', alpha=0.7)

    save_pub_figure(fig, out_dir, "Stage4_Boxplot_Stats", formats=syn.fig_formats, dpi_png=600)
    plt.close(fig)
    print(f"[Figures] Saved Monte Carlo boxplots to {out_dir}")








if __name__ == "__main__":
    # ==========================================
    my_bounds = ParamBounds(
        sigma_t=(0.65, 1.10),
        Jx=(0.010, 0.020),
        Jy=(0.010, 0.020),
        Jz=(0.022, 0.038),
        kx=(0.08, 0.35),
        ky=(0.08, 0.35),
        kz=(0.08, 0.35),
        kp=(0.08, 0.45),
        kq=(0.08, 0.45),
        kr=(0.08, 0.45),
    )

    # ==========================================
    my_syn = SynthesisParams(
        Ts=0.1,
        g=9.81,
        d_max=2.4,
        du_max=3.5,
        u_abs_min=(-6.0, -4.0, -4.0, -4.0),
        u_abs_max=(6.0, 4.0, 4.0, 4.0),
        decay_rate=0.95,
        w_gamma=1.0,
        w_mu=0.1,
        w_beta=1e-3,
        Qx_perf=(20, 20, 40, 5, 5, 5, 50, 50, 20, 1, 1, 1),
        Rd_perf=(5.0, 10.0, 10.0, 10.0),
        Qx_lqr=(20, 20, 40, 5, 5, 5, 50, 50, 20, 1, 1, 1, 0.1, 0.1, 0.1, 0.1),
        Ru_lqr=(5, 10, 10, 10),
        fig_context="paper",
        fig_out_dir=result_path("monclo_Result")
    )

    # ==========================================
    print("\n" + "=" * 80)
    print("Phase 1: Controller Synthesis")
    print("=" * 80)

    out = synthesize_controllers_for_stage3(
        syn=my_syn,
        bounds=my_bounds,
        max_vertices_proposed=1024,
        direct_vertex_limit=20,
        baseline1_vertices=20,
        score_mode="lambda_max",
        enforce_perf_all_vertices=my_syn.enforce_perf_all_vertices,
    )

    required_keys = ["K_nominal", "K_baselineB", "K_proposed",
                     "K_P0", "K_F1a", "K_F1b", "K_S0"]
    for key in required_keys:
        if key not in out:
            print(f"[ERROR] Missing key '{key}' in synthesis output")
            sys.exit(1)

    synthesis_path = result_path("caches", "fusion_ablation_synthesis.npz")
    np.savez_compressed(
        synthesis_path,
        **build_synthesis_diagnostics_payload(
            out, list(my_bounds.__dataclass_fields__.keys())
        ),
    )
    print(f"[Cache] Saved synthesis diagnostics to {synthesis_path}")

    controllers = {
        "NominalLQR": out["K_nominal"],
        "P0(PriorOnly)": out["K_P0"],
        "P1(NoRelax)": out["K_baselineB"],
        "F1a(HardStrict)": out["K_F1a"],
        "F1b(HardBudget)": out["K_F1b"],
        "S0(DataOnly)": out["K_S0"],
        "Proposed": out["K_proposed"],
    }

    # ==========================================
    print("\n" + "=" * 80)
    print("Phase 2: Monte Carlo Campaign (N=2000)")
    print("=" * 80)

    N_MC = 2000
    mc_metrics, deviations, envelope_data, failure_info, parameter_diagnostics = run_monte_carlo_campaign(
        syn=my_syn,
        bounds=my_bounds,
        controllers=controllers,
        N_mc=N_MC,
        seconds=30.0,
        ref_amp=0.5,
        ref_period=16.0
    )

    # Persist raw Monte Carlo artifacts before plotting so a plot failure does
    # not invalidate the campaign.
    try:
        mc_payload: Dict[str, np.ndarray] = {
            "N_mc": np.asarray([N_MC]),
            "parameter_keys": np.asarray(parameter_diagnostics["keys"]),
            "parameter_samples": np.asarray(parameter_diagnostics["samples"]),
            "parameter_minimum": np.asarray(parameter_diagnostics["minimum"]),
            "parameter_maximum": np.asarray(parameter_diagnostics["maximum"]),
            "outside_prior_mask": np.asarray(
                parameter_diagnostics["outside_mask"], dtype=bool
            ),
            "outside_prior_before": np.asarray([parameter_diagnostics["outside_before"]]),
            "outside_prior_after": np.asarray([parameter_diagnostics["outside_after"]]),
            "normalized_parameter_deviation": np.asarray(deviations),
            "parameter_rng_seed": np.asarray([my_syn.seed + 7777]),
            "measurement_noise_seeds": np.asarray(
                [my_syn.seed + 2026 + i * 7919 for i in range(N_MC)]
            ),
            "gust_seeds": np.asarray(
                [my_syn.seed + i * 100 for i in range(N_MC)]
            ),
        }
        for controller, record in mc_metrics.items():
            safe = "".join(
                ch if ch.isalnum() else "_" for ch in controller
            ).strip("_")
            for metric_name, values in record.items():
                mc_payload[f"{safe}_{metric_name}"] = np.asarray(values)
            mc_payload[f"{safe}_failure_indices"] = np.asarray(
                failure_info[controller], dtype=int
            )
            traces = envelope_data[controller]
            if traces:
                mc_payload[f"{safe}_time"] = np.asarray(traces[0][0])
                mc_payload[f"{safe}_error_traces"] = np.stack(
                    [np.asarray(item[1]) for item in traces]
                )

        disturbance_records = parameter_diagnostics["disturbance_stats"]
        if disturbance_records:
            for key in disturbance_records[0]:
                values = np.asarray([record[key] for record in disturbance_records])
                if values.dtype != object:
                    mc_payload[f"disturbance_{key}"] = values

        cache_path = result_path("caches", "fusion_ablation_monte_carlo_raw.npz")
        np.savez_compressed(cache_path, **mc_payload)
        print(f"[Cache] Saved raw Monte Carlo artifacts to {cache_path}")

        summary_path = result_path("tables", "fusion_ablation_monte_carlo_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "Controller", "SuccessRatePct", "RMSEMean", "RMSEMedian",
                "RMSEStd", "RMSE95", "PeakMean", "PeakMax", "EnergyMean",
                "IncrementSatRateMean", "AbsoluteSatRateMean",
                "CommandPeakNormalizedMax", "AppliedPeakNormalizedMax",
            ])
            for controller, record in mc_metrics.items():
                rmse = np.asarray(record["RMSE"], dtype=float)
                peak = np.asarray(record["Peak"], dtype=float)
                energy = np.asarray(record["Energy"], dtype=float)
                valid = np.isfinite(rmse)
                def finite_stat(values, fn):
                    arr = np.asarray(values, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    return float(fn(arr)) if arr.size else float("nan")
                writer.writerow([
                    controller,
                    100.0 * float(record["SuccessCount"]) / N_MC,
                    finite_stat(rmse[valid], np.mean),
                    finite_stat(rmse[valid], np.median),
                    finite_stat(rmse[valid], np.std),
                    finite_stat(rmse[valid], lambda x: np.percentile(x, 95)),
                    finite_stat(peak[valid], np.mean),
                    finite_stat(peak[valid], np.max),
                    finite_stat(energy[valid], np.mean),
                    finite_stat(record["IncrementSatRate"], np.mean),
                    finite_stat(record["AbsoluteSatRate"], np.mean),
                    finite_stat(record["CommandPeakNormalized"], np.max),
                    finite_stat(record["AppliedPeakNormalized"], np.max),
                ])
        print(f"[Table] Saved Monte Carlo summary to {summary_path}")
    except Exception as e:
        print(f"[Cache] WARNING: failed to persist MC artifacts: {e}")

    print("\n[Plotting] Generating Monte Carlo boxplots...")
    try:
        plot_mc_boxplots(mc_metrics, my_syn.fig_out_dir, my_syn)
    except Exception as e:
        print(f"[ERROR] Monte Carlo boxplot generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("All simulations completed successfully!")
    print("=" * 80)
