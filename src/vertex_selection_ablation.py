# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import csv
import pickle
import itertools
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence, Callable, Any, Set

import numpy as np
from score_processing import nonnegative_quantile_threshold
from numpy.linalg import eigvalsh
import scipy.linalg as la

import matplotlib as mpl
import matplotlib.pyplot as plt

from normalized_coordinates import (
    DEFAULT_SCALES,
    EPS_NEAR,
    EPS_PASS,
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
VIOL_TOL       = EPS_PASS
POLISH_TRIGGER = EPS_NEAR

S1_MAX_ITERS   = 50
S1_REL_GAP     = 1e-4
S1_PFEAS       = 1e-6
S1_DFEAS       = 1e-6
S1_TIME_LIMIT  = 1500                  # seconds

S2_MAX_ITERS   = 200
S2_REL_GAP     = 1e-7
S2_PFEAS       = 1e-8
S2_DFEAS       = 1e-8
S2_TIME_LIMIT  = 600                   # seconds


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
    """Endpoint-exact box-wide PSD bound for disturbance injection.

    The batch generator and random interior samples are numerical diagnostics
    and do not determine the bound."""
    keys = list(bounds.__dataclass_fields__.keys())
    n_vert = 2 ** len(keys)
    rng = np.random.default_rng(seed)

    vertex_samples: List[Dict[str, float]] = []
    for bits in range(n_vert):
        p = {k: getattr(bounds, k)[(bits >> j) & 1] for j, k in enumerate(keys)}
        vertex_samples.append(p)
    interior_samples: List[Dict[str, float]] = []
    for _ in range(int(n_internal_samples)):
        p = {k: float(rng.uniform(*getattr(bounds, k))) for k in keys}
        interior_samples.append(p)

    vertex_max_eig = 0.0
    for p in vertex_samples:
        _, _, S_v = build_vertex_matrices(p, syn.Ts, syn.g)
        vertex_max_eig = max(vertex_max_eig, float(la.eigvalsh(S_v @ S_v.T)[-1]))
    interior_max_eig = 0.0
    for p in interior_samples:
        _, _, S_v = build_vertex_matrices(p, syn.Ts, syn.g)
        interior_max_eig = max(interior_max_eig, float(la.eigvalsh(S_v @ S_v.T)[-1]))

    generator_max_eig = 0.0
    if batch_generator is not None:
        _, _, S_generator = build_vertex_matrices(batch_generator, syn.Ts, syn.g)
        generator_max_eig = float(la.eigvalsh(S_generator @ S_generator.T)[-1])

    alpha = float(safety_factor) * vertex_max_eig
    old_alpha = alpha
    W_c = alpha * np.eye(16)

    min_gap_v, n_viol_v = float("inf"), 0
    for p in vertex_samples:
        _, _, S_v = build_vertex_matrices(p, syn.Ts, syn.g)
        gap = float(la.eigvalsh(W_c - S_v @ S_v.T)[0])
        min_gap_v = min(min_gap_v, gap)
        if gap < -1e-10:
            n_viol_v += 1
    min_gap_i, n_viol_i = float("inf"), 0
    for p in interior_samples:
        _, _, S_v = build_vertex_matrices(p, syn.Ts, syn.g)
        gap = float(la.eigvalsh(W_c - S_v @ S_v.T)[0])
        min_gap_i = min(min_gap_i, gap)
        if gap < -1e-10:
            n_viol_i += 1

    generator_gap = float("nan")
    if batch_generator is not None:
        generator_gap = float(la.eigvalsh(W_c - S_generator @ S_generator.T)[0])

    info = dict(
        alpha=alpha, vertex_max_eig=vertex_max_eig, interior_max_eig=interior_max_eig,
        generator_max_eig=generator_max_eig, generator_psd_gap=generator_gap,
        old_vertex_only_alpha=old_alpha,
        generator_changed_bound=False,
        generator_exceeds_vertex_max=bool(
            generator_max_eig > vertex_max_eig + 1e-12
        ),
        safety_factor=float(safety_factor),
        min_psd_gap_vertex=min_gap_v, n_violate_vertex=n_viol_v,
        min_psd_gap_interior=min_gap_i, n_violate_interior=n_viol_i,
        n_vertices=len(vertex_samples), n_internal_samples=len(interior_samples),
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
    """scalar diagnostic for the model residual; does NOT feed W_E.

    Caller-controlled scope:
      - score_zero_vertices=None  -> ALL prior vertices: MODEL-MISMATCH diagnostic
        (NOT a residual check; E_i mixes residual and model mismatch).
      - score_zero_vertices=I0    -> processed score-zero subset: a-posteriori residual
        diagnostic. Because compute_si_from_vi may threshold s_i^raw > 0 to s_i=0 via
        tau_eff, I0 may contain vertices with s_i^raw > 0; rho_E on I0 is therefore NOT
        a proof that ||E|| is residual-only. The formal certificate E_t E_t^T <= W_E
        comes from build_structured_residual_bound + L2-norm saturation in
        simulate_batch_data.
    """
    X_t = batch["X_t"]; X_tp1 = batch["X_tp1"]; U_t = batch["U_t"]; Z_t = batch["Z_t"]
    Dbar_t = batch.get("Dbar_t", None); L = X_t.shape[1]
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
    hat_X = np.vstack([X_t, U_t]); breve_X = np.vstack([X_tp1, Z_t])
    rho_E = 0.0; worst_idx = -1
    for idx, p in enumerate(score_zero_vertices):
        A_i, B_i, S_i = build_vertex_matrices(p, syn.Ts, syn.g)
        Delta_i = np.block([[A_i, B_i], [Cc, Dc]])
        if Dbar_t is not None:
            hatD_i = np.vstack([S_i @ Dbar_t, np.zeros((16, L))])
        else:
            hatD_i = np.zeros((32, L))
        E_i = breve_X - Delta_i @ hat_X - hatD_i
        rho_i = float(la.eigvalsh((E_i @ E_i.T) / float(L))[-1])
        if rho_i > rho_E:
            rho_E = rho_i; worst_idx = idx
    alpha_c = float(np.max(np.diag(W_c)))
    cmp_dist = alpha_c
    cmp_nu2 = float(np.max(np.diag(build_structured_residual_bound(1, syn.nu_E_max))))
    residual_std_bar = np.asarray(DEFAULT_SCALES.residual_std) / np.asarray(DEFAULT_SCALES.state)
    cmp_sig2 = float(np.max(residual_std_bar ** 2))
    info = dict(rho_E_per_sample=rho_E, worst_vertex_index=worst_idx,
                n_vertices_scanned=len(score_zero_vertices), L=L,
                compare_per_sample_dist=cmp_dist,
                compare_per_sample_nu_E2=cmp_nu2,
                compare_per_sample_sigma_E2=cmp_sig2,
                scanned_set=scanned_label)
    if verbose:
        print(f"  [Residual diagnostic, {scanned_label}]")
        print(f"    rho_E = max_i lambda_max((1/L) E_i E_i^T) = {rho_E:.4e} "
              f"(@ vertex {worst_idx}/{len(score_zero_vertices)})")
        print(f"    normalized disturbance proxy alpha_c = {cmp_dist:.4e}")
        print(f"    normalized residual-cap proxy = {cmp_nu2:.4e}")
        print(f"    normalized residual-std proxy = {cmp_sig2:.4e}")
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
    # Psi[bot-right] = Xbreve Xbreve^T - \tilde R_t.
    # The normalized disturbance radius is one:
    # \tilde R_t = 2 * diag(L W_c, 0_{16}) + 2 * W_E.
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
    W_c_supplied = batch.get("W_c", None)
    if W_c_supplied is None:
        raise ValueError(
            "build_psi_data requires W_c constructed over all adopted prior "
            "vertices and the batch generator"
        )
    Wc_block = np.asarray(W_c_supplied, dtype=float)
    Dt_top = L * Wc_block
    Dt_block = la.block_diag(Dt_top, np.zeros((16, 16)))

    W_E = batch.get("W_E", None)
    if W_E is None:
        raise ValueError(
            "build_psi_data requires the deterministic capped-recording "
            "residual bound W_E"
        )
    W_E = np.asarray(W_E, dtype=float)

    # \tilde R_t = 2 * Dt_block + 2 * W_E   (paper eq:Rtilde_def, scheme A)
    Rtilde = 2.0 * Dt_block + 2.0 * W_E

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




# All manuscript scripts use mode="lambda_max". Alternative score modes are
# retained for method-comparison checks and are not used for the reported results.
def compute_vertex_score_scalar(
        Delta_i: np.ndarray,
        Psi_data: np.ndarray,
        mode: str = "lambda_max",
) -> float:
    M = np.hstack([Delta_i, np.eye(32)])
    S = M @ Psi_data @ M.T
    S = 0.5 * (S + S.T)

    ev = eigvalsh(S)

    if mode == "trace":
        return float(-np.trace(S))
    elif mode == "lambda_max":
        return float(ev[-1])
    elif mode == "pos_eig_sum":
        return float(-np.sum(np.maximum(ev, 0.0)))
    elif mode == "min_eig":
        return float(ev[0])
    elif mode == "mean_eig":
        return float(np.mean(ev))
    else:
        raise ValueError("mode must be one of: trace / lambda_max / pos_eig_sum / min_eig / mean_eig")


def build_all_vertices_and_scores(
        bounds: ParamBounds,
        syn: SynthesisParams,
        Psi_data: np.ndarray,
        score_mode: str = "lambda_max",
        max_vertices: Optional[int] = None,
        seed: int = 123,
) -> List[Dict[str, object]]:
    Ts, g = syn.Ts, syn.g
    Cc, Dc = build_performance_matrices(syn)

    verts = enumerate_vertices(bounds)
    if max_vertices is not None and max_vertices < len(verts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(verts), size=max_vertices, replace=False)
        verts = [verts[i] for i in idx]

    out = []
    for p in verts:
        A, B, S_c = build_vertex_matrices(p, Ts, g)
        Delta = np.block([[A, B],
                          [Cc, Dc]])
        s_i = compute_vertex_score_scalar(Delta, Psi_data, mode=score_mode)
        out.append(dict(A=A, B=B, S=S_c, C=Cc, D=Dc, Delta=Delta, s=s_i, p=p))
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
        max_iters: int = S1_MAX_ITERS,
        rel_gap: float = S1_REL_GAP,
        pfeas: float = S1_PFEAS,
        dfeas: float = S1_DFEAS,
        time_limit_sec: Optional[int] = S1_TIME_LIMIT,
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
            active_indices, num_threads, max_iters, rel_gap, pfeas, dfeas,
            time_limit_sec,
            beta_lb, decay_rate, w_gamma, w_mu, w_beta, enforce_perf_all_vertices, x0_feas)
    finally:
        M.dispose()


def _solve_vertex_fusion_inner(M, vertices, syn, verbose, eps_Q,
        active_indices, num_threads, max_iters, rel_gap, pfeas, dfeas,
        time_limit_sec,
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
        M.setSolverParam("intpntCoTolPfeas", float(pfeas))
        M.setSolverParam("intpntCoTolDfeas", float(dfeas))
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
        init_indices: Optional[Set[int]] = None,
        enforce_initial_inconsistency_quota: bool = True,
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

    m_incon = compute_inconsistency_quota(s_all, K_budget, rho_incon=0.2)

    # --- Initial selection ---
    if init_indices is not None:
        active_idx = set(init_indices)
        if len(active_idx) < K_budget:
            remaining = [i for i in range(N) if i not in active_idx]
            for i in remaining[:K_budget - len(active_idx)]:
                active_idx.add(i)
        if len(active_idx) > K_budget:
            active_idx = set(sorted(active_idx)[:K_budget])
    elif bounds is not None:
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
    if enforce_initial_inconsistency_quota and active_positive < m_incon:
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
    actual_rounds = 0
    quota_relaxations = 0

    for rnd in range(max_rounds):
        actual_rounds = rnd + 1
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
            max_iters=S1_MAX_ITERS,
            rel_gap=S1_REL_GAP,
            pfeas=S1_PFEAS,
            dfeas=S1_DFEAS,
            time_limit_sec=S1_TIME_LIMIT,
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
    best_sol["active_vertex_indices"] = np.asarray(best_active_list, dtype=int)
    best_gamma = best_metric[2]
    if verbose:
        print(f"\n[IterExchange] Final: {len(final_verts)} vertices, "
              f"gamma={best_gamma:.4f}, max_viol={best_metric[0]:.6f}, n_violated={best_metric[1]}")

    best_sol["actual_rounds"] = np.array([actual_rounds])
    return final_verts, best_sol


# =========================================================
# Plotting utilities
# =========================================================
def classify_synthesis(
        sdp_success: bool,
        max_viol: float,
        n_violated: int,
        tol: float = VIOL_TOL,
) -> str:
    # These labels describe only the full-vertex surrogate LMI check. Hard
    # certification remains restricted to the score-zero subset.
    del n_violated, tol
    status = classify_surrogate_status(sdp_success, max_viol, EPS_PASS, EPS_NEAR)
    return {
        "Surrogate pass": "surrogate_pass",
        "Near surrogate": "near_surrogate",
        "Failed": "failed",
    }[status]


def polish_sdp(
        active_verts: List[Dict[str, object]],
        syn: SynthesisParams,
        decay_rate: float,
        w_gamma: float,
        w_mu: float,
        w_beta: float,
        enforce_perf_all_vertices: bool = True,
) -> Dict[str, np.ndarray]:
    return solve_vertex_fusion_sdp_mosek(
        active_verts, syn,
        verbose=False,
        num_threads=0,
        max_iters=S2_MAX_ITERS,
        rel_gap=S2_REL_GAP,
        pfeas=S2_PFEAS,
        dfeas=S2_DFEAS,
        time_limit_sec=S2_TIME_LIMIT,
        beta_lb=0.0,
        decay_rate=decay_rate,
        w_gamma=w_gamma,
        w_mu=w_mu,
        w_beta=w_beta,
        enforce_perf_all_vertices=enforce_perf_all_vertices,
        x0_feas=None,
    )


def verify_and_classify(
        sol: Dict[str, np.ndarray],
        active_verts: List[Dict[str, object]],
        all_vertices: List[Dict[str, object]],
        syn: SynthesisParams,
        decay_rate: float,
        tol: float = VIOL_TOL,
) -> Tuple[str, float, int, Dict[str, np.ndarray]]:
    sdp_ok = sol.get("success", False)
    if not sdp_ok:
        return "failed", 9999.9, len(all_vertices), sol

    K_ctrl = sol["K"]
    Q_val = sol["Q"]
    gamma2_val = float(sol["gamma2"][0])
    beta_val = float(sol["beta"][0])

    violations = check_all_vertex_violations(
        all_vertices, K_ctrl, Q_val, gamma2_val, beta_val,
        decay_rate,
    )
    max_viol = float(np.max(violations))
    n_viol = int(np.sum(violations > tol))

    cls_s1 = classify_synthesis(True, max_viol, n_viol, tol)

    if cls_s1 == "surrogate_pass":
        return "surrogate_pass", max_viol, n_viol, sol

    if cls_s1 == "near_surrogate":
        sol_p = polish_sdp(
            active_verts, syn, decay_rate,
            w_gamma=float(sol.get("w_gamma", np.array([syn.w_gamma]))[0]),
            w_mu=float(sol.get("w_mu", np.array([syn.w_mu]))[0]),
            w_beta=float(sol.get("w_beta", np.array([syn.w_beta]))[0]),
            enforce_perf_all_vertices=bool(sol.get("enforce_perf_all_vertices", np.array([1]))[0]),
        )
        if not sol_p.get("success", False):
            return "near_surrogate", max_viol, n_viol, sol

        violations_p = check_all_vertex_violations(
            all_vertices, sol_p["K"], sol_p["Q"],
            float(sol_p["gamma2"][0]), float(sol_p["beta"][0]),
            decay_rate,
        )
        max_viol_p = float(np.max(violations_p))
        n_viol_p = int(np.sum(violations_p > tol))
        cls_final = classify_synthesis(True, max_viol_p, n_viol_p, tol)
        return cls_final, max_viol_p, n_viol_p, sol_p

    return "failed", max_viol, n_viol, sol


def print_verification_protocol():
    print("\n" + "=" * 80)
    print("TWO-STAGE SOLVER / SURROGATE-LMI VERIFICATION PROTOCOL")
    print("=" * 80)
    print(f"  Stage-1: T1={S1_TIME_LIMIT}s, max_iters={S1_MAX_ITERS}, "
          f"rel_gap={S1_REL_GAP:.0e}, pfeas={S1_PFEAS:.0e}, dfeas={S1_DFEAS:.0e}")
    print(f"  Stage-2 (polish): T2={S2_TIME_LIMIT}s, max_iters={S2_MAX_ITERS}, "
          f"rel_gap={S2_REL_GAP:.0e}, pfeas={S2_PFEAS:.0e}, dfeas={S2_DFEAS:.0e}")
    print(f"  Verification: full-vertex score-weighted surrogate LMI check")
    print(f"  Polish trigger: max_viol <= eps_near={EPS_NEAR:.0e}")
    print(f"  Classification (numerical surrogate status, not hard certification):")
    print(f"    surrogate_pass : max_viol <= eps_pass={EPS_PASS:.0e}")
    print(f"    near_surrogate : eps_pass < max_viol <= eps_near={EPS_NEAR:.0e}")
    print(f"    failed         : max_viol > eps_near or solver infeasible/timeout")
    print(f"  NOTE: hard certification holds only on the score-zero subpolytope U_cert")
    print(f"        A surrogate pass is not a hard certificate over U_prior.")
    print("=" * 80 + "\n")


def print_final_certification_report(
        cert_results: Dict[str, Dict[str, object]],
        mc_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        N_mc: int = 0,
):
    print("\n" + "=" * 100)
    print("FINAL SURROGATE-LMI VERIFICATION REPORT (Two-Stage Protocol)")
    print("=" * 100)

    hdr = (f"{'Strategy':<22} | {'Class':<16} | {'max_viol':>10} | {'n_violated':>10} | "
           f"{'gamma':>10} | {'polished':>8}")
    if mc_metrics is not None:
        hdr += f" | {'MC success':>10}"
    print(hdr)
    print("-" * len(hdr))

    target_order = ["NominalLQR", "S0(DataOnly)", "P0(PriorOnly)",
                    "P1(NoRelax)", "F1a(HardStrict)", "F1b(HardBudget)", "Proposed"]

    n_surrogate_pass = 0
    n_near = 0
    n_failed = 0
    total = 0

    for name in target_order:
        if name not in cert_results:
            continue
        cr = cert_results[name]
        cls = cr["classification"]
        mv = cr["max_viol"]
        nv = cr["n_violated"]
        gam = cr["gamma"]
        pol = cr["polished"]

        total += 1
        if cls == "surrogate_pass":
            n_surrogate_pass += 1
        elif cls == "near_surrogate":
            n_near += 1
        else:
            n_failed += 1

        mv_s = f"{mv:.2e}" if mv < 9000 else "N/A"
        gam_s = f"{gam:.4f}" if gam < 9000 else "N/A"
        pol_s = "yes" if pol else "no"
        row = f"{name:<22} | {cls:<16} | {mv_s:>10} | {nv:>10} | {gam_s:>10} | {pol_s:>8}"

        if mc_metrics is not None and name in mc_metrics:
            sc = mc_metrics[name].get("SuccessCount", 0)
            rate = sc / N_mc * 100 if N_mc > 0 else 0.0
            row += f" | {rate:>9.1f}%"
        print(row)

    print("-" * len(hdr))
    if total > 0:
        print(f"  surrogate_pass: {n_surrogate_pass}/{total} ({n_surrogate_pass/total*100:.1f}%)  |  "
              f"near_surrogate: {n_near}/{total} ({n_near/total*100:.1f}%)  |  "
              f"failed: {n_failed}/{total} ({n_failed/total*100:.1f}%)")

    if mc_metrics is not None and N_mc > 0:
        print(f"\n  Monte Carlo success rates by surrogate-verification tier (N_mc={N_mc}):")
        for tier_name, tier_label in [("surrogate_pass", "surrogate pass"),
                                       ("near_surrogate", "near surrogate")]:
            tier_strats = [n for n in target_order
                           if n in cert_results and cert_results[n]["classification"] == tier_name
                           and n in mc_metrics]
            if tier_strats:
                for sn in tier_strats:
                    sc = mc_metrics[sn].get("SuccessCount", 0)
                    rate = sc / N_mc * 100
                    print(f"    [{tier_label}] {sn:<22}: MC success = {rate:.1f}%")
        failed_strats = [n for n in target_order
                         if n in cert_results and cert_results[n]["classification"] == "failed"
                         and n in mc_metrics]
        if failed_strats:
            for sn in failed_strats:
                sc = mc_metrics[sn].get("SuccessCount", 0)
                rate = sc / N_mc * 100
                print(f"    [failed]    {sn:<22}: MC success = {rate:.1f}% (no certificate)")

    print("=" * 100 + "\n")


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
    L_batch = int(batch["X_t"].shape[1])
    W_E = build_structured_residual_bound(L_batch, float(syn.nu_E_max))
    print(f"  [W_E] deterministic normalized residual bound: "
          f"||W_E^x||_op = {np.max(np.diag(W_E[:12, :12])):.3e}; W_E^z = 0")

    record_residual_check = verify_realized_record_residual(batch, W_E)
    batch["record_residual_check"] = record_residual_check
    print(f"  [W_E realized check] min PSD gap = "
          f"{record_residual_check['minimum_psd_gap']:.3e}; "
          f"max output residual = "
          f"{record_residual_check['maximum_output_residual']:.3e}; "
          f"max input-memory residual = "
          f"{record_residual_check['maximum_input_memory_residual']:.3e}")

    # --- Build Psi using full batch ---
    batch["W_c"] = W_c
    batch["W_E"] = W_E
    Psi = build_psi_data(batch, syn)

    # --- MODEL residual diagnostic ---
    model_residual_all = verify_data_residual_bound(
        batch, syn, bounds, W_c, verbose=True
    )

    verts_all = build_all_vertices_and_scores(
        bounds, syn, Psi,
        score_mode=score_mode,
        max_vertices=1024,
        seed=syn.seed + 1234
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

    # --- residual diagnostic on processed score-zero subset I0 ---
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
              f"residual diagnostic on I0 is skipped")

    diagnostic_payload = build_fusion_diagnostics_payload(
        batch, verts_norm, Psi, W_c, W_E, info_Wc, score_diag,
        list(bounds.__dataclass_fields__.keys()), p_data_source,
        model_residual_all=model_residual_all,
        model_residual_certified=model_residual_i0,
    )
    diagnostic_path = result_path("caches", "vertex_selection_data_diagnostics.npz")
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
        viol_tol=VIOL_TOL,
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
        max_iters=S1_MAX_ITERS,
        rel_gap=S1_REL_GAP,
        pfeas=S1_PFEAS,
        dfeas=S1_DFEAS,
        time_limit_sec=S1_TIME_LIMIT,
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
        max_rounds=4, viol_tol=VIOL_TOL, seed=syn.seed + 8888,
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
            max_iters=S1_MAX_ITERS, rel_gap=S1_REL_GAP,
            pfeas=S1_PFEAS, dfeas=S1_DFEAS,
            time_limit_sec=S1_TIME_LIMIT, beta_lb=0.0,
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
        max_iters=S1_MAX_ITERS, rel_gap=S1_REL_GAP,
        pfeas=S1_PFEAS, dfeas=S1_DFEAS,
        time_limit_sec=S1_TIME_LIMIT, beta_lb=0.0,
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

    # ===== Two-stage verification & classification for all SDP strategies =====
    print_verification_protocol()

    cert_results: Dict[str, Dict[str, object]] = {}

    sdp_strategies = [
        ("Proposed",        sol_prop,  verts_common, verts_norm),
        ("P1(NoRelax)",     sol_baseB, verts_baseB,  verts_norm),
        ("P0(PriorOnly)",   sol_P0,    verts_P0_active, verts_norm),
        ("F1a(HardStrict)", sol_F1a,   verts_F1a_sel if len(consistent_idx) >= 1 else [], verts_norm),
        ("F1b(HardBudget)", sol_F1b,   verts_F1b,    verts_norm),
    ]

    for strat_name, sol_s, active_v, all_v in sdp_strategies:
        cls_f, mv_f, nv_f, sol_f = verify_and_classify(
            sol_s, active_v, all_v, syn, syn.decay_rate, VIOL_TOL,
        )
        polished = (sol_f is not sol_s)
        gamma_f = float(np.sqrt(max(float(sol_f.get("gamma2", [1e9])[0]), 0.0))) \
            if sol_f.get("success", False) else 9999.9
        K_f = sol_f.get("K", np.zeros((4, 16)))

        cert_results[strat_name] = dict(
            classification=cls_f, max_viol=mv_f, n_violated=nv_f,
            gamma=gamma_f, polished=polished, K=K_f, sol=sol_f,
        )
        pol_tag = " [polished]" if polished else ""
        print(f"  [{strat_name}] {cls_f}{pol_tag}: max_viol={mv_f:.2e}, "
              f"n_violated={nv_f}, gamma={gamma_f:.4f}")

    cert_results["NominalLQR"] = dict(
        classification="N/A (LQR)", max_viol=float("nan"), n_violated=0,
        gamma=float("nan"), polished=False, K=K_nom, sol={},
    )
    cert_results["S0(DataOnly)"] = dict(
        classification="N/A (LQR)", max_viol=float("nan"), n_violated=0,
        gamma=float("nan"), polished=False, K=K_S0, sol={},
    )

    print_final_certification_report(cert_results)

    return dict(
        p_nom=p_nom,
        p_data_source=p_data_source,
        batch=batch,
        Psi=Psi,
        W_c=W_c,
        W_E=W_E,
        wc_diagnostics=info_Wc,
        score_diagnostics=score_diag,
        vertices_common=verts_common,
        verts_norm=verts_norm,
        K_proposed=cert_results["Proposed"]["K"],
        gamma_proposed=cert_results["Proposed"]["gamma"],
        sol_proposed=cert_results["Proposed"]["sol"],
        K_baselineB=cert_results["P1(NoRelax)"]["K"],
        gamma_baselineB=cert_results["P1(NoRelax)"]["gamma"],
        sol_baselineB=cert_results["P1(NoRelax)"]["sol"],
        K_nominal=K_nom,
        K_P0=cert_results["P0(PriorOnly)"]["K"],
        gamma_P0=cert_results["P0(PriorOnly)"]["gamma"],
        sol_P0=cert_results["P0(PriorOnly)"]["sol"],
        K_F1a=cert_results["F1a(HardStrict)"]["K"],
        gamma_F1a=cert_results["F1a(HardStrict)"]["gamma"],
        sol_F1a=cert_results["F1a(HardStrict)"]["sol"],
        K_F1b=cert_results["F1b(HardBudget)"]["K"],
        gamma_F1b=cert_results["F1b(HardBudget)"]["gamma"],
        sol_F1b=cert_results["F1b(HardBudget)"]["sol"],
        K_S0=K_S0,
        cert_results=cert_results,
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

STRATEGY_NAMES = [
    "SFPS+ICE",
    "SFPS-init",
    "Random+ICE",
    "TopK+ICE",
    "FPS-Param+ICE",
    "FPS-Delta+ICE",
]

STRATEGY_PRETTY = {
    "SFPS+ICE":       "SFPS + ICE (Ours)",
    "SFPS-init":      "SFPS Init Only",
    "Random+ICE":     "Random + ICE",
    "TopK+ICE":       "Top-$K$ + ICE",
    "FPS-Param+ICE":  "FPS-Param + ICE",
    "FPS-Delta+ICE":  r"FPS-$\Delta$ + ICE",
}

STRATEGY_COLORS = {
    "SFPS+ICE":       "#0072B2",
    "SFPS-init":      "#56B4E9",
    "Random+ICE":     "#999999",
    "TopK+ICE":       "#E69F00",
    "FPS-Param+ICE":  "#D55E00",
    "FPS-Delta+ICE":  "#009E73",
}

STRATEGY_MARKERS = {
    "SFPS+ICE":       "o",
    "SFPS-init":      "s",
    "Random+ICE":     "v",
    "TopK+ICE":       "^",
    "FPS-Param+ICE":  "D",
    "FPS-Delta+ICE":  "p",
}


def _get_init_indices(
        strategy: str,
        all_vertices: List[Dict[str, object]],
        K: int,
        seed: int,
        bounds: ParamBounds,
) -> Set[int]:
    N = len(all_vertices)
    id_map = {id(v): i for i, v in enumerate(all_vertices)}

    if strategy in ("SFPS+ICE", "SFPS-init"):
        sel = select_vertices_stratified_farthest(
            all_vertices, bounds, K, ratios=(0.4, 0.2, 0.4), seed=seed,
        )
        return {id_map[id(v)] for v in sel if id(v) in id_map}

    elif strategy == "Random+ICE":
        rng = np.random.default_rng(seed)
        return set(rng.choice(N, size=min(K, N), replace=False).tolist())

    elif strategy == "TopK+ICE":
        s_arr = np.array([float(v["s"]) for v in all_vertices])
        return set(np.argsort(s_arr, kind="stable")[:K].tolist())

    elif strategy == "FPS-Param+ICE":
        init_idx = seed % N
        sel = select_vertices_farthest(all_vertices, bounds, K, seed=seed, initial_index=init_idx)
        return {id_map[id(v)] for v in sel if id(v) in id_map}

    elif strategy == "FPS-Delta+ICE":
        init_idx = seed % N
        sel = select_vertices_farthest_delta(all_vertices, K, initial_index=init_idx)
        return {id_map[id(v)] for v in sel if id(v) in id_map}

    raise ValueError(f"Unknown strategy: {strategy}")


def run_vertex_selection_experiment(
        all_vertices: List[Dict[str, object]],
        syn: SynthesisParams,
        bounds: ParamBounds,
        K_values: Tuple[int, ...] = (8, 12, 16, 20),
        n_seeds: int = 5,
        N_mc: int = 500,
        max_rounds_ice: int = 4,
        viol_tol: float = VIOL_TOL,
        seconds: float = 30.0,
        ref_amp: float = 0.5,
        ref_period: float = 16.0,
) -> Dict:
    print(f"\n{'='*80}")
    print(f"[VertexSelectionExperiment] K_values={K_values}, n_seeds={n_seeds}, N_mc={N_mc}")
    print(f"  Protocol: tol={viol_tol:.1e}, polish_trigger={2*viol_tol:.1e}, "
          f"T1={S1_TIME_LIMIT}s, T2={S2_TIME_LIMIT}s")
    print(f"{'='*80}")

    decay_rate = getattr(syn, "decay_rate", 0.98)
    ref_fun = lambda tt: ref_square_wave_x(tt, amp=ref_amp, period=ref_period, start_delay=1.0)
    DIVERGE_THR = 5.0

    keys_b = list(bounds.__dataclass_fields__.keys())
    rng_mc_base = np.random.default_rng(syn.seed + 9999)
    mc_plant_seeds = [int(rng_mc_base.integers(0, 2**31)) for _ in range(N_mc)]
    mc_parameter_samples = np.empty((N_mc, len(keys_b)), dtype=float)
    mc_disturbance_profiles = []
    mc_disturbance_stats = []
    for mc_i, plant_seed in enumerate(mc_plant_seeds):
        rng_trial = np.random.default_rng(plant_seed)
        mc_parameter_samples[mc_i, :] = [
            float(rng_trial.uniform(*getattr(bounds, key))) for key in keys_b
        ]
        profile, _, profile_stats = build_multi_gust_profile(
            syn=syn, seconds=seconds, num_gusts=5,
            bias_frac=0.4, seed_offset=mc_i * 100,
        )
        mc_disturbance_profiles.append(profile)
        mc_disturbance_stats.append(profile_stats)

    ckpt_path = os.path.join(getattr(syn, "fig_out_dir", "."), "vsel_checkpoint.pkl")
    results: Dict = {}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "rb") as _f:
                results = pickle.load(_f)
            n_done = sum(
                len(results.get(_K, {}).get(_s, {}).get("classification", []))
                for _K in results for _s in results[_K]
            )
            print(f"[Checkpoint] Resumed from {ckpt_path} ({n_done} records loaded)")
        except Exception as _exc:
            print(f"[Checkpoint] Failed to load {ckpt_path}: {_exc}, starting fresh")
            results = {}

    def _empty_record():
        return {
            "classification": [],
            "gamma": [],
            "max_viol_stage1": [],
            "max_viol_final": [],
            "n_violated": [],
            "n_rounds": [],
            "polished": [],
            "mc_success_rate": [],
            "mc_rmse_95pct": [],
            "mc_success_mask": [],
            "mc_rmse_samples": [],
            "mc_tier": [],
            "synth_success": [],
            "max_violation": [],
            "v_sur": [],
            "v_cert": [],
            "output_lmi_signed_violation": [],
            "increment_lmi_signed_violation": [],
        }

    def _save_ckpt():
        try:
            with open(ckpt_path, "wb") as _f:
                pickle.dump(results, _f)
        except Exception as _exc:
            print(f"[Checkpoint] Save failed: {_exc}")

    for K in K_values:
        if K not in results:
            results[K] = {}
        for strat in STRATEGY_NAMES:
            if strat not in results[K]:
                results[K][strat] = _empty_record()
            else:
                for field, default in _empty_record().items():
                    results[K][strat].setdefault(field, default)

            done_seeds = len(results[K][strat]["classification"])
            if done_seeds >= n_seeds:
                print(f"  [K={K}] {strat}: skipping ({done_seeds} seeds already done from checkpoint)")
                continue

            for s_idx in range(done_seeds, n_seeds):
                seed_val = syn.seed + s_idx * 1000 + K * 7
                init_idx = _get_init_indices(strat, all_vertices, K, seed_val, bounds)

                mr = 1 if strat == "SFPS-init" else max_rounds_ice

                try:
                    active_verts, sol = iterative_constraint_exchange(
                        all_vertices=all_vertices, syn=syn, K_budget=K,
                        max_rounds=mr, viol_tol=viol_tol, seed=seed_val,
                        decay_rate=decay_rate,
                        w_gamma=syn.w_gamma, w_mu=syn.w_mu, w_beta=syn.w_beta,
                        enforce_perf_all_vertices=True,
                        verbose=False, bounds=bounds,
                        init_indices=init_idx,
                        enforce_initial_inconsistency_quota=(strat != "TopK+ICE"),
                    )
                except Exception as exc:
                    results[K][strat]["classification"].append("failed")
                    results[K][strat]["gamma"].append(9999.9)
                    results[K][strat]["max_viol_stage1"].append(9999.9)
                    results[K][strat]["max_viol_final"].append(9999.9)
                    results[K][strat]["n_violated"].append(len(all_vertices))
                    results[K][strat]["n_rounds"].append(0)
                    results[K][strat]["polished"].append(False)
                    results[K][strat]["mc_success_rate"].append(0.0)
                    results[K][strat]["mc_rmse_95pct"].append(9999.9)
                    results[K][strat]["mc_success_mask"].append(
                        np.zeros(N_mc, dtype=bool)
                    )
                    results[K][strat]["mc_rmse_samples"].append(
                        np.full(N_mc, np.nan)
                    )
                    results[K][strat]["mc_tier"].append("failed")
                    results[K][strat]["synth_success"].append(False)
                    results[K][strat]["max_violation"].append(9999.9)
                    results[K][strat]["v_sur"].append(9999.9)
                    results[K][strat]["v_cert"].append(9999.9)
                    results[K][strat]["output_lmi_signed_violation"].append(9999.9)
                    results[K][strat]["increment_lmi_signed_violation"].append(9999.9)
                    print(f"  [K={K}] {strat} seed={s_idx}: failed during synthesis ({exc})")
                    continue

                n_rounds_val = int(sol.get("actual_rounds", [mr])[0])

                cls_final, max_viol_final, n_viol_final, sol_final = \
                    verify_and_classify(sol, active_verts, all_vertices, syn, decay_rate, viol_tol)

                sdp_ok_s1 = sol.get("success", False)
                if sdp_ok_s1 and sol.get("Q") is not None:
                    viol_s1 = check_all_vertex_violations(
                        all_vertices, sol["K"], sol["Q"],
                        float(sol["gamma2"][0]), float(sol["beta"][0]),
                        decay_rate,
                    )
                    max_viol_s1 = float(np.max(viol_s1))
                else:
                    max_viol_s1 = 9999.9

                polished = (sol_final is not sol)

                gamma_val = float(np.sqrt(max(float(sol_final.get("gamma2", [1e9])[0]), 0.0))) \
                    if sol_final.get("success", False) else 9999.9
                K_ctrl = sol_final.get("K", np.zeros((4, 16)))

                results[K][strat]["classification"].append(cls_final)
                results[K][strat]["gamma"].append(gamma_val)
                results[K][strat]["max_viol_stage1"].append(max_viol_s1)
                results[K][strat]["max_viol_final"].append(max_viol_final)
                results[K][strat]["n_violated"].append(n_viol_final)
                results[K][strat]["n_rounds"].append(n_rounds_val)
                results[K][strat]["polished"].append(polished)
                results[K][strat]["synth_success"].append(cls_final != "failed")
                results[K][strat]["max_violation"].append(max_viol_final)
                if sol_final.get("success", False):
                    final_viol = check_all_vertex_violations(
                        all_vertices, sol_final["K"], sol_final["Q"],
                        float(sol_final["gamma2"][0]), float(sol_final["beta"][0]),
                        decay_rate,
                    )
                    cert_idx = [
                        i for i, vertex in enumerate(all_vertices)
                        if float(vertex["s"]) == 0.0
                    ]
                    v_sur = float(np.max(final_viol))
                    v_cert = float(np.max(final_viol[cert_idx])) if cert_idx else float("nan")
                    auxiliary = evaluate_auxiliary_lmi_residuals(
                        all_vertices, sol_final["K"], sol_final["Q"],
                        float(sol_final["mu"][0]),
                    )
                else:
                    v_sur, v_cert = 9999.9, 9999.9
                    auxiliary = {
                        "output_lmi_signed_violation": 9999.9,
                        "increment_lmi_signed_violation": 9999.9,
                    }
                results[K][strat]["v_sur"].append(v_sur)
                results[K][strat]["v_cert"].append(v_cert)
                results[K][strat]["output_lmi_signed_violation"].append(
                    auxiliary["output_lmi_signed_violation"]
                )
                results[K][strat]["increment_lmi_signed_violation"].append(
                    auxiliary["increment_lmi_signed_violation"]
                )

                if cls_final == "failed":
                    results[K][strat]["mc_success_rate"].append(0.0)
                    results[K][strat]["mc_rmse_95pct"].append(9999.9)
                    results[K][strat]["mc_success_mask"].append(
                        np.zeros(N_mc, dtype=bool)
                    )
                    results[K][strat]["mc_rmse_samples"].append(
                        np.full(N_mc, np.nan)
                    )
                    results[K][strat]["mc_tier"].append("failed")
                    print(f"  [K={K}] {strat} seed={s_idx}: {cls_final} "
                          f"(gamma={gamma_val:.2f}, viol_s1={max_viol_s1:.2e}, viol_final={max_viol_final:.2e})")
                    _save_ckpt()
                    continue

                mc_ok = 0
                rmse_list = []
                mc_success_mask = np.zeros(N_mc, dtype=bool)
                mc_rmse_samples = np.full(N_mc, np.nan)
                for mc_i in range(N_mc):
                    p_cand = dict(zip(keys_b, mc_parameter_samples[mc_i, :]))
                    dbar_profile = mc_disturbance_profiles[mc_i]
                    # per-trial noise seed for vsel ablation MC (independent across mc_i,
                    # shared across K/strat/seed within a trial for fairness).
                    sim = simulate_tracking_with_disturbance_profile(
                        K=K_ctrl, p_true=p_cand, syn=syn, ref_fun=ref_fun,
                        seconds=seconds, dbar_profile=dbar_profile, meas_noise_std=5e-4,
                        noise_seed=int(mc_plant_seeds[mc_i]) + 2026,
                    )
                    e = np.linalg.norm(sim["x"][0:3, :] - sim["Pref"], axis=0)
                    diverged = np.any(np.isnan(e)) or np.max(e) > DIVERGE_THR
                    if not diverged:
                        mc_ok += 1
                        rmse_value = float(np.sqrt(np.mean(e ** 2)))
                        rmse_list.append(rmse_value)
                        mc_success_mask[mc_i] = True
                        mc_rmse_samples[mc_i] = rmse_value

                mc_rate = mc_ok / N_mc * 100.0
                rmse_95 = float(np.percentile(rmse_list, 95)) if len(rmse_list) > 0 else 9999.9
                results[K][strat]["mc_success_rate"].append(mc_rate)
                results[K][strat]["mc_rmse_95pct"].append(rmse_95)
                results[K][strat]["mc_success_mask"].append(mc_success_mask)
                results[K][strat]["mc_rmse_samples"].append(mc_rmse_samples)
                results[K][strat]["mc_tier"].append(cls_final)
                _save_ckpt()

                pol_tag = " [polished]" if polished else ""
                print(f"  [K={K}] {strat} seed={s_idx}: {cls_final}{pol_tag}, gamma={gamma_val:.4f}, "
                      f"viol={max_viol_final:.2e}, mc_success={mc_rate:.1f}%, rmse95={rmse_95:.4f}")

            _save_ckpt()

    raw_payload: Dict[str, np.ndarray] = {
        "K_values": np.asarray(K_values, dtype=int),
        "strategies": np.asarray(STRATEGY_NAMES),
        "n_seeds": np.asarray([n_seeds]),
        "N_mc": np.asarray([N_mc]),
        "eps_pass": np.asarray([EPS_PASS]),
        "eps_near": np.asarray([EPS_NEAR]),
        "parameter_keys": np.asarray(keys_b),
        "mc_plant_seeds": np.asarray(mc_plant_seeds, dtype=np.int64),
        "mc_noise_seeds": np.asarray(mc_plant_seeds, dtype=np.int64) + 2026,
        "mc_parameter_samples": mc_parameter_samples,
        "mc_outside_prior_mask": np.any(
            (mc_parameter_samples < np.asarray([
                getattr(bounds, key)[0] for key in keys_b
            ]))
            | (mc_parameter_samples > np.asarray([
                getattr(bounds, key)[1] for key in keys_b
            ])),
            axis=1,
        ),
        "mc_disturbance_profiles": np.stack(mc_disturbance_profiles),
    }
    for key in mc_disturbance_stats[0] if mc_disturbance_stats else ():
        values = np.asarray([record[key] for record in mc_disturbance_stats])
        if values.dtype != object:
            raw_payload[f"mc_disturbance_{key}"] = values

    for K in K_values:
        for strategy in STRATEGY_NAMES:
            safe = "".join(
                ch if ch.isalnum() else "_" for ch in strategy
            ).strip("_")
            prefix = f"K{K}_{safe}"
            for field, values in results[K][strategy].items():
                arr = np.asarray(values)
                if arr.dtype != object:
                    raw_payload[f"{prefix}_{field}"] = arr

    raw_path = result_path("caches", "vertex_selection_ablation_raw.npz")
    np.savez_compressed(raw_path, **raw_payload)
    print(f"[Cache] Saved vertex-selection raw results to {raw_path}")

    csv_path = result_path("tables", "vertex_selection_ablation_by_seed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "KBudget", "Strategy", "SeedIndex", "SynthesisSeed",
            "Classification", "SynthesisSuccess", "Gamma", "SDPRounds",
            "MaxViolationStage1", "MaxViolationFinal", "VSur", "VCert",
            "OutputLMISignedViolation", "IncrementLMISignedViolation",
            "NViolated", "Polished", "MonteCarloSuccessPct", "RMSE95",
        ])
        for K in K_values:
            for strategy in STRATEGY_NAMES:
                record = results[K][strategy]
                for seed_index, classification in enumerate(record["classification"]):
                    writer.writerow([
                        K, strategy, seed_index,
                        syn.seed + seed_index * 1000 + K * 7,
                        classification, record["synth_success"][seed_index],
                        record["gamma"][seed_index], record["n_rounds"][seed_index],
                        record["max_viol_stage1"][seed_index],
                        record["max_viol_final"][seed_index],
                        record["v_sur"][seed_index], record["v_cert"][seed_index],
                        record["output_lmi_signed_violation"][seed_index],
                        record["increment_lmi_signed_violation"][seed_index],
                        record["n_violated"][seed_index],
                        record["polished"][seed_index],
                        record["mc_success_rate"][seed_index],
                        record["mc_rmse_95pct"][seed_index],
                    ])
    print(f"[Table] Saved by-seed vertex-selection results to {csv_path}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return results


def plot_vertex_selection_results(
        results: Dict,
        out_dir: str,
        syn: SynthesisParams,
):
    K_values = sorted(results.keys())
    strategies = STRATEGY_NAMES

    sizes = set_publication_style(context=syn.fig_context, column="double")
    fig, axes = plt.subplots(1, 2, figsize=(sizes["double"][0], sizes["double"][1] * 0.8),
                             constrained_layout=True)

    ax_synth, ax_mc = axes

    for strat in strategies:
        synth_means, synth_cis = [], []
        mc_means, mc_cis = [], []

        for K in K_values:
            ss = results[K][strat]["synth_success"]
            sr = [100.0 if s else 0.0 for s in ss]
            synth_means.append(np.mean(sr))
            synth_cis.append(1.96 * np.std(sr) / max(np.sqrt(len(sr)), 1))

            mc = results[K][strat]["mc_success_rate"]
            mc_means.append(np.mean(mc))
            mc_cis.append(1.96 * np.std(mc) / max(np.sqrt(len(mc)), 1))

        color = STRATEGY_COLORS[strat]
        marker = STRATEGY_MARKERS[strat]
        label = STRATEGY_PRETTY[strat]

        ax_synth.errorbar(K_values, synth_means, yerr=synth_cis,
                          color=color, marker=marker, markersize=5,
                          linewidth=1.2, capsize=3, label=label)
        ax_mc.errorbar(K_values, mc_means, yerr=mc_cis,
                       color=color, marker=marker, markersize=5,
                       linewidth=1.2, capsize=3, label=label)

    for ax, title, ylabel in [
        (ax_synth, "Synthesis Success Rate", "Synthesis Success (%)"),
        (ax_mc, "Closed-loop Success Rate", "MC Success (%)"),
    ]:
        ax.set_xlabel("Vertex Budget $K$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(K_values)
        ax.set_ylim(-5, 110)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    ax_mc.legend(fontsize=5.5, loc='lower right', frameon=True, framealpha=0.9,
                 ncol=2, handlelength=1.5, columnspacing=0.8)

    save_pub_figure(fig, out_dir, "Stage5_VertexSelection_SuccessVsK",
                    formats=syn.fig_formats, dpi_png=600)
    plt.close(fig)
    print(f"[Figures] Saved vertex selection comparison to {out_dir}")

    # --- Gamma vs K plot ---
    fig2, ax2 = plt.subplots(figsize=(sizes["double"][0] * 0.55, sizes["double"][1] * 0.75),
                              constrained_layout=True)
    for strat in strategies:
        gamma_means = []
        gamma_cis = []
        for K in K_values:
            gvals = [g for g, s in zip(results[K][strat]["gamma"],
                                        results[K][strat]["synth_success"]) if s]
            if len(gvals) > 0:
                gamma_means.append(np.mean(gvals))
                gamma_cis.append(1.96 * np.std(gvals) / max(np.sqrt(len(gvals)), 1))
            else:
                gamma_means.append(np.nan)
                gamma_cis.append(0)

        finite = np.isfinite(gamma_means)
        if not np.any(finite):
            continue

        K_plot = np.asarray(K_values)[finite]
        gamma_plot = np.asarray(gamma_means)[finite]
        gamma_ci_plot = np.asarray(gamma_cis)[finite]
        ax2.errorbar(K_plot, gamma_plot, yerr=gamma_ci_plot,
                     color=STRATEGY_COLORS[strat], marker=STRATEGY_MARKERS[strat],
                     markersize=5, linewidth=1.2, capsize=3,
                     label=STRATEGY_PRETTY[strat])

    ax2.set_xlabel("Vertex Budget $K$")
    ax2.set_ylabel(r"$\gamma$ (H$_\infty$ bound)")
    ax2.set_xticks(K_values)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.3, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=5.5, loc='best', frameon=True, framealpha=0.9,
               ncol=2, handlelength=1.5, columnspacing=0.8)

    save_pub_figure(fig2, out_dir, "Stage5_VertexSelection_GammaVsK",
                    formats=syn.fig_formats, dpi_png=600)
    plt.close(fig2)
    print(f"[Figures] Saved gamma vs K plot to {out_dir}")


def export_vertex_selection_tables(
        results: Dict,
        out_dir: str,
):
    K_values = sorted(results.keys())
    strategies = STRATEGY_NAMES

    lines = []
    lines.append("% ============================================================")
    lines.append("% Table: Vertex Selection Strategy Comparison (fixed K)")
    lines.append("% ============================================================")

    for K in K_values:
        lines.append(f"% --- K = {K} ---")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{Vertex selection strategy comparison ($K={K}$, "
                      f"$N_\\mathrm{{seed}}={len(results[K][strategies[0]]['synth_success'])}$).}}")
        lines.append(f"\\label{{tab:vsel_K{K}}}")
        lines.append(r"\begin{tabular}{l c c c c c c}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Strategy} & \textbf{Synth.\ (\%)} & \textbf{$\bar\gamma$} "
                      r"& \textbf{SDP rounds} & \textbf{MaxViol} "
                      r"& \textbf{MC Succ.\ (\%)} & \textbf{RMSE$_{95}$} \\")
        lines.append(r"\midrule")

        for strat in strategies:
            d = results[K][strat]
            synth_pct = np.mean([100.0 if s else 0.0 for s in d["synth_success"]])

            ok_gammas = [g for g, s in zip(d["gamma"], d["synth_success"]) if s]
            gamma_str = f"{np.mean(ok_gammas):.2f}" if len(ok_gammas) > 0 else "---"

            ok_rounds = [r for r, s in zip(d["n_rounds"], d["synth_success"]) if s]
            rounds_str = f"{np.mean(ok_rounds):.1f}" if len(ok_rounds) > 0 else "---"

            ok_viols = [v for v, s in zip(d["max_violation"], d["synth_success"]) if s and v < 9000]
            viol_str = f"{np.max(ok_viols):.1e}" if len(ok_viols) > 0 else "---"

            mc_vals = d["mc_success_rate"]
            mc_mean = np.mean(mc_vals) if len(mc_vals) > 0 else 0.0
            mc_std = np.std(mc_vals) if len(mc_vals) > 1 else 0.0

            rmse_vals = [r for r in d["mc_rmse_95pct"] if r < 9000]
            rmse_str = f"{np.mean(rmse_vals):.3f}" if len(rmse_vals) > 0 else "---"

            pretty = STRATEGY_PRETTY[strat].replace("$", "").replace(r"\Delta", r"$\Delta$")
            if strat == "SFPS+ICE":
                pretty = r"\textbf{" + pretty + "}"

            mc_cell = f"${mc_mean:.1f} \\pm {mc_std:.1f}$"

            lines.append(f"{pretty} & {synth_pct:.0f} & {gamma_str} & {rounds_str} & {viol_str} & {mc_cell} & {rmse_str} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # --- Table: K sweep (SFPS-init vs SFPS+ICE) ---
    lines.append("% ============================================================")
    lines.append("% ============================================================")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect of ICE iterations: SFPS-init only vs.\ SFPS + ICE across vertex budgets $K$.}")
    lines.append(r"\label{tab:vsel_ksweep}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"$K$ & \multicolumn{2}{c}{\textbf{SFPS Init Only}} & \multicolumn{2}{c}{\textbf{SFPS + ICE}} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r" & Synth.\ (\%) & MC Succ.\ (\%) & Synth.\ (\%) & MC Succ.\ (\%) \\")
    lines.append(r"\midrule")

    for K in K_values:
        d_init = results[K]["SFPS-init"]
        d_ice = results[K]["SFPS+ICE"]

        s_init = np.mean([100.0 if s else 0.0 for s in d_init["synth_success"]])
        m_init = np.mean(d_init["mc_success_rate"])
        s_ice = np.mean([100.0 if s else 0.0 for s in d_ice["synth_success"]])
        m_ice = np.mean(d_ice["mc_success_rate"])

        lines.append(f"{K} & {s_init:.0f} & {m_init:.1f} & {s_ice:.0f} & {m_ice:.1f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = os.path.join(out_dir, "u-vertex_selection_tables.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Export] Saved vertex selection LaTeX tables to {out_path}")

    summary_path = result_path("tables", "vertex_selection_table8_summary.csv")
    rows_path = result_path("tables", "vertex_selection_table8_rows.tex")

    def scientific_tex(value: float) -> str:
        if not np.isfinite(value):
            return "---"
        mantissa, exponent = f"{value:.1e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"

    with open(summary_path, "w", newline="", encoding="utf-8") as csv_file, \
            open(rows_path, "w", encoding="utf-8") as tex_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "KBudget", "Strategy", "SurrogatePassPct", "NearSurrogatePct",
            "FailPct", "GammaMeanSuccessful", "SDPRoundsMeanSuccessful",
            "MaxViolationSuccessful", "MonteCarloSuccessMeanPct",
            "MonteCarloSuccessStdPct", "RMSE95MeanSuccessful",
        ])
        for K in K_values:
            for strategy in strategies:
                record = results[K][strategy]
                classes = np.asarray(record["classification"])
                total = max(classes.size, 1)
                pass_pct = 100.0 * np.sum(classes == "surrogate_pass") / total
                near_pct = 100.0 * np.sum(classes == "near_surrogate") / total
                fail_pct = 100.0 * np.sum(classes == "failed") / total
                success = np.asarray(record["synth_success"], dtype=bool)

                def successful_mean(field):
                    values = np.asarray(record[field], dtype=float)
                    valid = success & np.isfinite(values) & (values < 9000.0)
                    return float(np.mean(values[valid])) if np.any(valid) else float("nan")

                gamma_mean = successful_mean("gamma")
                rounds_mean = successful_mean("n_rounds")
                rmse_mean = successful_mean("mc_rmse_95pct")
                violations = np.asarray(record["max_violation"], dtype=float)
                valid_violation = success & np.isfinite(violations) & (violations < 9000.0)
                max_violation = (
                    float(np.max(violations[valid_violation]))
                    if np.any(valid_violation) else float("nan")
                )
                mc = np.asarray(record["mc_success_rate"], dtype=float)
                mc_mean = float(np.mean(mc)) if mc.size else 0.0
                mc_std = float(np.std(mc)) if mc.size else 0.0
                writer.writerow([
                    K, strategy, pass_pct, near_pct, fail_pct, gamma_mean,
                    rounds_mean, max_violation, mc_mean, mc_std, rmse_mean,
                ])

                pretty = STRATEGY_PRETTY[strategy]
                if strategy == "SFPS+ICE":
                    pretty = r"\textbf{SFPS + ICE (Ours)}"
                gamma_tex = "---" if not np.isfinite(gamma_mean) else f"{gamma_mean:.2f}"
                rounds_tex = "---" if not np.isfinite(rounds_mean) else f"{rounds_mean:.1f}"
                violation_tex = scientific_tex(max_violation)
                rmse_tex = "---" if not np.isfinite(rmse_mean) else f"{rmse_mean:.3f}"
                tex_file.write(
                    f"& {pretty} & {pass_pct:.0f} & {near_pct:.0f} & {fail_pct:.0f} "
                    f"& {gamma_tex} & {rounds_tex} & {violation_tex} "
                    f"& ${mc_mean:.1f}\\pm{mc_std:.1f}$ & {rmse_tex} \\\\\n"
                )
            if K != K_values[-1]:
                tex_file.write(r"\midrule" + "\n")
    print(f"[Export] Saved Table 8 summary to {summary_path}")
    print(f"[Export] Saved Table 8 row fragment to {rows_path}")


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
        fig_out_dir=result_path("monclo_pointsResult")
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

    synthesis_path = result_path("caches", "vertex_selection_synthesis.npz")
    np.savez_compressed(
        synthesis_path,
        **build_synthesis_diagnostics_payload(
            out, list(my_bounds.__dataclass_fields__.keys())
        ),
    )
    print(f"[Cache] Saved synthesis diagnostics to {synthesis_path}")

    print("\n" + "=" * 80)
    print("Phase 3: Vertex Selection Strategy Comparison")
    print("=" * 80)

    verts_norm_all = out.get("verts_norm", None)
    if verts_norm_all is None or len(verts_norm_all) == 0:
        print("[ERROR] verts_norm not available from synthesis output, skipping Phase 3")
    else:
        try:
            vsel_results = run_vertex_selection_experiment(
                all_vertices=verts_norm_all,
                syn=my_syn,
                bounds=my_bounds,
                K_values=(8, 12, 16, 20),
                n_seeds=5,
                N_mc=500,
                max_rounds_ice=4,
                viol_tol=1e-4,
                seconds=30.0,
                ref_amp=0.5,
                ref_period=16.0,
            )

            print("\n[Plotting] Generating vertex selection comparison plots...")
            plot_vertex_selection_results(vsel_results, my_syn.fig_out_dir, my_syn)

            print("\n[Export] Generating vertex selection LaTeX tables...")
            export_vertex_selection_tables(vsel_results, my_syn.fig_out_dir)

        except Exception as e:
            print(f"[ERROR] Vertex selection experiment failed: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All simulations completed successfully!")
    print("=" * 80)
