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
from matplotlib.ticker import AutoMinorLocator

from normalized_coordinates import (
    DEFAULT_SCALES,
    EPS_NEAR,
    EPS_PASS,
    build_normalized_augmented_matrices,
    build_normalized_performance_matrices,
    build_normalized_residual_bound,
    build_fusion_diagnostics_payload,
    build_physical_augmented_matrices,
    build_physical_performance_matrices,
    append_tracking_simulation_payload,
    allocate_stratified_budget,
    classify_surrogate_status,
    compute_inconsistency_quota,
    disturbance_to_normalized,
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


def _setup_axes(ax: plt.Axes) -> None:
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(top=False, right=False, which="both")
    ax.grid(True, which="major", axis="y", linewidth=0.3, alpha=0.25)


def _add_gust_shading(ax: plt.Axes, gusts: List[Dict[str, float]]) -> None:
    for g in gusts:
        ax.axvspan(g["t0"], g["t1"], alpha=0.10, linewidth=0)


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

    # Gaussian std of the sampled-state residual.
    sigma_E_x: float = 5e-4
    # Physical cap used by the fixed channel-scaled residual ellipsoid.
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

#
def discrete_time_sum(y: np.ndarray, Ts: float) -> float:
    """Rectangular discrete-time approximation ``Ts * sum(y)``."""
    return float(np.sum(y) * Ts)


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

        dbar[:, k] = dbar_applied[:, k]


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


        x[:, k + 1] = A_gen @ x[:, k] + B_gen @ uk + S_gen @ dbar[:, k]
        x[12:16, k + 1] = u_next
        x_next_recorded[:, k] = x[:, k + 1]
        # Project the recording perturbation into the fixed residual ellipsoid.
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
    Endpoint-exact isotropic PSD upper bound W_c on the normalized
    disturbance-injection Gram matrix over the parameter box:
        alpha = safety_factor * max lambda_max(S_c S_c.T)
        W_c   = alpha * I_16
    The optional batch generator and random interior samples are evaluated as
    numerical diagnostics. They do not determine the bound.
    safety_factor=1+eta_c=1.05.
    """
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
    """Return a scalar model-residual diagnostic that does not define W_E."""
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
        print(f"    rho_E = {rho_E:.4e} (@ vertex {worst_idx}/{len(score_zero_vertices)})")
        print(f"    normalized alpha_c = {cmp_dist:.4e} | residual-cap proxy = {cmp_nu2:.4e} | "
              f"normalized residual-std proxy = {cmp_sig2:.4e}")
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

    p = np.maximum(0.0, raw_s - tau_effective)

    if use_len_norm and L > 0:
        p = p / (float(L) + eps)

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
        s_final = 1.0 - np.exp(-r)

    s_final = np.clip(s_final, 0.0, 1.0)

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
def compute_metrics_excess_multi_gust(
        sim_dist: Dict[str, Any],
        sim_nodist: Dict[str, Any],
        syn: SynthesisParams,
        gusts: List[Dict[str, float]],
        eps_excess: float = 0.05,
        hold_time: float = 0.5,
        recovery_ref: str = "gust_end",
        post_window: float = 3.0,
) -> Dict[str, float]:
    Ts = syn.Ts

    e0 = np.linalg.norm(sim_nodist["x"][0:3, :] - sim_nodist["Pref"], axis=0)
    e1 = np.linalg.norm(sim_dist["x"][0:3, :] - sim_dist["Pref"], axis=0)
    N = min(e0.size, e1.size)
    e0 = e0[:N]
    e1 = e1[:N]
    e_ex = np.maximum(0.0, e1 - e0)

    rmse_pos = float(np.sqrt(np.mean(e1 ** 2)))
    rmse_ex = float(np.sqrt(np.mean(e_ex ** 2)))

    iae_pos = discrete_time_sum(np.abs(e1), Ts)
    iae_ex = discrete_time_sum(np.abs(e_ex), Ts)

    peak_pos = float(np.max(e1))
    peak_ex = float(np.max(e_ex))

    hold_N = max(1, int(round(hold_time / Ts)))
    win_N = max(1, int(round(post_window / Ts)))

    settle_times: List[float] = []
    iae_ex_post_list: List[float] = []
    peak_ex_post_list: List[float] = []

    for g in gusts:
        t_ref = g["t1"] if recovery_ref == "gust_end" else g["t0"]
        k_ref = int(round(t_ref / Ts))
        k_ref = int(np.clip(k_ref, 0, N - 1))

        settle_idx = None
        for k in range(k_ref, N - hold_N):
            if np.all(e_ex[k:k + hold_N] <= eps_excess):
                settle_idx = k
                break
        if settle_idx is None:
            settle_times.append(float("nan"))
        else:
            settle_times.append(float((settle_idx - k_ref) * Ts))

        k1 = k_ref
        k2 = min(N, k1 + win_N)
        iae_ex_post_list.append(discrete_time_sum(e_ex[k1:k2], Ts))
        peak_ex_post_list.append(float(np.max(e_ex[k1:k2])) if k2 > k1 else float("nan"))

    settle_arr = np.array(settle_times, dtype=float)
    settle_valid = settle_arr[np.isfinite(settle_arr)]
    settle_mean = float(np.mean(settle_valid)) if settle_valid.size > 0 else float("nan")
    settle_worst = float(np.max(settle_valid)) if settle_valid.size > 0 else float("nan")

    iae_ex_post_avg = float(np.mean(iae_ex_post_list)) if len(iae_ex_post_list) > 0 else float("nan")
    peak_ex_post_avg = float(np.mean(peak_ex_post_list)) if len(peak_ex_post_list) > 0 else float("nan")

    uc = sim_dist["u_c"]
    uabs = sim_dist["u_abs"]
    uc_norm = np.linalg.norm(increment_to_normalized(uc), axis=0)
    inv_Tu = np.diag(1.0 / np.diag(DEFAULT_SCALES.T_u))
    uabs_norm = np.linalg.norm(inv_Tu @ uabs, axis=0)

    energy_uc = discrete_time_sum(uc_norm ** 2, Ts)
    energy_uabs = discrete_time_sum(uabs_norm ** 2, Ts)

    du_peak = float(np.max(uc_norm)) if uc_norm.size > 0 else float("nan")
    uabs_peak = float(np.max(uabs_norm)) if uabs_norm.size > 0 else float("nan")

    sat_tol = float(syn.sat_tol)
    duty_du = float(np.mean(uc_norm >= sat_tol)) if uc_norm.size > 0 else float("nan")
    duty_abs = float(np.mean(sim_dist["sat_abs"] > 0)) if "sat_abs" in sim_dist else float("nan")
    duty_norm = float(np.mean(sim_dist["sat_norm"] > 0)) if "sat_norm" in sim_dist else float("nan")
    duty_rate = float(np.mean(sim_dist["sat_rate"] > 0)) if "sat_rate" in sim_dist else float("nan")

    return dict(
        RMSE_pos=rmse_pos,
        RMSE_excess=rmse_ex,
        IAE_pos=iae_pos,
        IAE_excess=iae_ex,
        peak_pos=peak_pos,
        peak_excess=peak_ex,
        settle_mean_excess=settle_mean,
        settle_worst_excess=settle_worst,
        IAE_excess_postgust_avg=iae_ex_post_avg,
        peak_excess_postgust_avg=peak_ex_post_avg,
        du_peak=du_peak,
        uabs_peak=uabs_peak,
        energy_uc=energy_uc,
        energy_uabs=energy_uabs,
        duty_du=duty_du,
        duty_abs=duty_abs,
        duty_normsat=duty_norm,
        duty_ratesat=duty_rate,
        du_cmd_peak=float(sim_dist["increment_raw_peak_normalized"]),
        du_act_peak=float(sim_dist["increment_applied_peak_normalized"]),
        disturbance_projection_rate=float(sim_dist["disturbance_stats"]["projection_rate"]),
        rho=float(sim_dist["rho"][0]),
    )


# =========================================================
# Plotting utilities
# =========================================================
def plot_stage3_A_C_publication(
        sims_dist: Dict[str, Dict[str, Any]],
        sims_nodist: Dict[str, Dict[str, Any]],
        syn: SynthesisParams,
        gusts: List[Dict[str, float]],
) -> None:
    sizes = set_publication_style(context=syn.fig_context, column=syn.fig_column)
    figsize = sizes["double"] if syn.fig_column == "double" else sizes["single"]
    out_dir = syn.fig_out_dir

    display_name = {
        "Proposed": "Proposed",
        "BaselineB(NoRelax)": "Matched-active ablation",
        "NominalLQR": "Nominal LQR",
    }
    linestyle = {
        "Proposed": "-",
        "BaselineB(NoRelax)": "--",
        "NominalLQR": ":",
    }

    any_key = list(sims_dist.keys())[0]
    t = sims_dist[any_key]["t"]
    Pref = sims_dist[any_key]["Pref"]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax)
    _add_gust_shading(ax, gusts)
    ax.plot(t, Pref[0, :], linestyle=":", linewidth=1.2, color="black", label="Reference")
    for name, sim in sims_dist.items():
        ax.plot(t, sim["x"][0, :], linestyle=linestyle.get(name, "-"), label=display_name.get(name, name))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$x$ position (m)")
    if syn.fig_show_titles:
        ax.set_title("Position tracking along $x$ under multi-gust disturbance")
    ax.legend(frameon=False, loc="best", ncols=1)
    save_pub_figure(fig, out_dir, "FigS3_PosX", formats=syn.fig_formats, dpi_png=syn.fig_dpi_png,
                    transparent=syn.fig_transparent)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax)
    _add_gust_shading(ax, gusts)
    ax.plot(t, Pref[1, :], linestyle=":", linewidth=0.8, color="black", label="Reference")
    for name, sim in sims_dist.items():
        ax.plot(t, sim["x"][1, :], linestyle=linestyle.get(name, "-"), label=display_name.get(name, name))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$y$ position (m)")
    if syn.fig_show_titles:
        ax.set_title("Position tracking along $y$ under multi-gust disturbance")
    ax.legend(frameon=False, loc="best", ncols=1)
    save_pub_figure(fig, out_dir, "FigS3_PosY", formats=syn.fig_formats, dpi_png=syn.fig_dpi_png,
                    transparent=syn.fig_transparent)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax)
    _add_gust_shading(ax, gusts)
    ax.plot(t, Pref[2, :], linestyle=":", linewidth=0.8, color="black", label="Reference")
    for name, sim in sims_dist.items():
        ax.plot(t, sim["x"][2, :], linestyle=linestyle.get(name, "-"), label=display_name.get(name, name))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$z$ position (m)")
    if syn.fig_show_titles:
        ax.set_title("Position tracking along $z$ under multi-gust disturbance")
    ax.legend(frameon=False, loc="best", ncols=1)
    save_pub_figure(fig, out_dir, "FigS3_PosZ", formats=syn.fig_formats, dpi_png=syn.fig_dpi_png,
                    transparent=syn.fig_transparent)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax)
    _add_gust_shading(ax, gusts)
    for name, sim in sims_dist.items():
        e = np.linalg.norm(sim["x"][0:3, :] - sim["Pref"], axis=0)
        ax.plot(t, e, linestyle=linestyle.get(name, "-"), label=display_name.get(name, name))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|p - p_{\mathrm{ref}}\|_2$ (m)")
    if syn.fig_show_titles:
        ax.set_title("Position tracking error norm (disturbed)")
    ax.legend(frameon=False, loc="best", ncols=1)
    save_pub_figure(fig, out_dir, "FigS3_ErrNorm", formats=syn.fig_formats, dpi_png=syn.fig_dpi_png,
                    transparent=syn.fig_transparent)
    plt.close(fig)

    if len(gusts) > 0:
        gmax = max(gusts, key=lambda gg: float(gg.get("amp", 0.0)))
        t0 = float(gmax["t0"])
        t1 = float(gmax["t1"])
        zoom_l = max(0.0, t0 - 1.0)
        zoom_r = min(float(t[-1]), t1 + 4.0)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        _setup_axes(ax)
        _add_gust_shading(ax, gusts)
        for name, sim in sims_dist.items():
            e = np.linalg.norm(sim["x"][0:3, :] - sim["Pref"], axis=0)
            ax.plot(t, e, linestyle=linestyle.get(name, "-"), label=display_name.get(name, name))
        ax.set_xlim(zoom_l, zoom_r)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\|p - p_{\mathrm{ref}}\|_2$ (m)")
        if syn.fig_show_titles:
            ax.set_title("Zoomed error around strongest gust (disturbed)")
        ax.legend(frameon=False, loc="best", ncols=1)
        save_pub_figure(fig, out_dir, "FigS3_ErrNorm_ZoomStrongestGust", formats=syn.fig_formats,
                        dpi_png=syn.fig_dpi_png, transparent=syn.fig_transparent)
        plt.close(fig)

    if ("Proposed" in sims_dist) and ("BaselineB(NoRelax)" in sims_dist):
        eP = np.linalg.norm(sims_dist["Proposed"]["x"][0:3, :] - sims_dist["Proposed"]["Pref"], axis=0)
        eB = np.linalg.norm(sims_dist["BaselineB(NoRelax)"]["x"][0:3, :] - sims_dist["BaselineB(NoRelax)"]["Pref"], axis=0)
        N = min(eP.size, eB.size)
        delta = eP[:N] - eB[:N]

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        _setup_axes(ax)
        _add_gust_shading(ax, gusts)
        ax.plot(t[:N], delta, linewidth=1.2, label="Proposed $-$ matched-active ablation")
        ax.axhline(0.0, linestyle=":", linewidth=1.0, color="black")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\Delta \|e_p\|_2$ (m)")
        if syn.fig_show_titles:
            ax.set_title("Error-norm difference (Proposed minus matched-active ablation)")
        ax.legend(frameon=False, loc="best")
        save_pub_figure(fig, out_dir, "FigS3_ErrNorm_Delta_PropMinusBaseA", formats=syn.fig_formats,
                        dpi_png=syn.fig_dpi_png, transparent=syn.fig_transparent)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax)
    _add_gust_shading(ax, gusts)
    for name, sim in sims_dist.items():
        du = np.linalg.norm(increment_to_normalized(sim["u_c"]), axis=0)
        ax.plot(t[:-1], du, linestyle=linestyle.get(name, "-"), label=display_name.get(name, name))
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="black",
               label="synthesis unit-norm threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|\bar u_c^{\mathrm{act}}\|_2$")
    if syn.fig_show_titles:
        ax.set_title("Normalized applied control increment")
    ax.legend(frameon=False, loc="best", ncols=1)
    save_pub_figure(fig, out_dir, "FigS3_DuNorm", formats=syn.fig_formats, dpi_png=syn.fig_dpi_png,
                    transparent=syn.fig_transparent)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax)
    _add_gust_shading(ax, gusts)
    for name, sim in sims_dist.items():
        inv_Tu = np.diag(1.0 / np.diag(DEFAULT_SCALES.T_u))
        ua = np.linalg.norm(inv_Tu @ sim["u_abs"], axis=0)
        ax.plot(t, ua, linestyle=linestyle.get(name, "-"), label=display_name.get(name, name))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|T_u^{-1}u\|_2$")
    if syn.fig_show_titles:
        ax.set_title("Absolute actuator command norm (with clipping)")
    ax.legend(frameon=False, loc="best", ncols=1)
    save_pub_figure(fig, out_dir, "FigS3_UabsNorm", formats=syn.fig_formats, dpi_png=syn.fig_dpi_png,
                    transparent=syn.fig_transparent)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax)
    _add_gust_shading(ax, gusts)
    for name in sims_dist.keys():
        sim1 = sims_dist[name]
        sim0 = sims_nodist[name]
        e1 = np.linalg.norm(sim1["x"][0:3, :] - sim1["Pref"], axis=0)
        e0 = np.linalg.norm(sim0["x"][0:3, :] - sim0["Pref"], axis=0)
        N = min(e0.size, e1.size)
        e_excess = np.maximum(0.0, e1[:N] - e0[:N])
        ax.plot(t[:N], e_excess, linestyle=linestyle.get(name, "-"), label=display_name.get(name, name))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Excess error (m)")
    if syn.fig_show_titles:
        ax.set_title("Disturbance-induced excess tracking error")
    ax.legend(frameon=False, loc="best", ncols=1)
    save_pub_figure(fig, out_dir, "FigS3_ExcessErr", formats=syn.fig_formats, dpi_png=syn.fig_dpi_png,
                    transparent=syn.fig_transparent)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax)
    _add_gust_shading(ax, gusts)
    dnorm = np.linalg.norm(disturbance_to_normalized(sims_dist[any_key]["dbar"]), axis=0)
    ax.plot(t[:-1], dnorm, linewidth=1.2, color="black", label=r"$\|\bar d\|_2$")
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="black", label="normalized bound")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|\bar d\|_2$")
    if syn.fig_show_titles:
        ax.set_title("Multi-gust disturbance profile (norm)")
    ax.legend(frameon=False, loc="best", ncols=1)
    save_pub_figure(fig, out_dir, "FigS3_Disturbance", formats=syn.fig_formats, dpi_png=syn.fig_dpi_png,
                    transparent=syn.fig_transparent)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax)
    _add_gust_shading(ax, gusts)
    win = max(1, int(round(0.5 / syn.Ts)))
    kernel = np.ones(win) / win
    for name, sim in sims_dist.items():
        sat_abs = sim["sat_abs"].astype(float)
        sat_norm = sim["sat_norm"].astype(float)
        sat_abs_ma = np.convolve(sat_abs, kernel, mode="same")
        sat_norm_ma = np.convolve(sat_norm, kernel, mode="same")
        ax.plot(t[:-1], sat_abs_ma, linestyle=linestyle.get(name, "-"),
                label=f"{display_name.get(name, name)}: abs clip")
        ax.plot(t[:-1], sat_norm_ma, linestyle=":", linewidth=1.0, label=f"{display_name.get(name, name)}: norm sat")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Saturation duty (moving avg.)")
    if syn.fig_show_titles:
        ax.set_title("Actuator saturation activity (0.5 s moving average)")
    ax.legend(frameon=False, loc="upper right", ncols=1)
    save_pub_figure(fig, out_dir, "FigS3_Saturation", formats=syn.fig_formats, dpi_png=syn.fig_dpi_png,
                    transparent=syn.fig_transparent)
    plt.close(fig)

    print(f"\n[Figures] Publication-quality figures exported to: ./{out_dir}/")


# =========================================================
# Plotting utilities
# =========================================================
def plot_stage3_story_figure(
        sims_dist: Dict[str, Dict[str, Any]],
        sims_nodist: Dict[str, Dict[str, Any]],
        syn: SynthesisParams,
        gusts: List[Dict[str, float]],
        out_dir: str,
        stem: str = "FigExp1_Story",
) -> None:
    ORDER = ["Proposed", "BaselineB(NoRelax)", "NominalLQR"]
    PRETTY = {
        "Proposed": "Proposed",
        "BaselineB(NoRelax)": "Matched-active ablation",
        "NominalLQR": "Nominal LQR",
    }
    LS = {
        "Proposed": "-",
        "BaselineB(NoRelax)": "--",
        "NominalLQR": ":",
    }

    names = [n for n in ORDER if n in sims_dist]
    any_key = names[0]
    t = sims_dist[any_key]["t"]

    sizes = set_publication_style(context=syn.fig_context, column="double")
    fig_w = sizes["double"][0]
    fig_h = sizes["double"][1] * 3.4

    fig, axes = plt.subplots(4, 1, figsize=(fig_w, fig_h), constrained_layout=True,
                             sharex=False)
    ax_err, ax_excess, ax_du, ax_delta = axes

    # --- subplot 1: error norm ||p - p_ref||_2 (disturbed) ---
    _setup_axes(ax_err)
    _add_gust_shading(ax_err, gusts)
    for name in names:
        sim = sims_dist[name]
        e = np.linalg.norm(sim["x"][0:3, :] - sim["Pref"], axis=0)
        ax_err.plot(t, e, linestyle=LS.get(name, "-"),
                    label=PRETTY.get(name, name))
    ax_err.set_ylabel(r"$\|p - p_{\mathrm{ref}}\|_2$ (m)")
    ax_err.legend(frameon=False, loc="best", ncols=2, fontsize=6)

    # --- subplot 2: excess error max(0, ||e_dist|| - ||e_nodist||) ---
    _setup_axes(ax_excess)
    _add_gust_shading(ax_excess, gusts)
    for name in names:
        sim1 = sims_dist[name]
        sim0 = sims_nodist[name]
        e1 = np.linalg.norm(sim1["x"][0:3, :] - sim1["Pref"], axis=0)
        e0 = np.linalg.norm(sim0["x"][0:3, :] - sim0["Pref"], axis=0)
        N = min(e0.size, e1.size)
        e_ex = np.maximum(0.0, e1[:N] - e0[:N])
        ax_excess.plot(t[:N], e_ex, linestyle=LS.get(name, "-"),
                       label=PRETTY.get(name, name))
    ax_excess.set_ylabel("Excess error (m)")
    ax_excess.legend(frameon=False, loc="best", ncols=2, fontsize=6)

    _setup_axes(ax_du)
    _add_gust_shading(ax_du, gusts)
    for name in names:
        sim = sims_dist[name]
        du = np.linalg.norm(increment_to_normalized(sim["u_c"]), axis=0)
        ax_du.plot(t[:-1], du, linestyle=LS.get(name, "-"),
                   label=PRETTY.get(name, name))
    ax_du.axhline(1.0, linestyle=":", linewidth=1.0, color="black",
                  label="synthesis unit-norm threshold")
    ax_du.set_ylabel(r"$\|\bar u_c^{\mathrm{act}}\|_2$")
    ax_du.legend(frameon=False, loc="best", ncols=2, fontsize=6)

    _setup_axes(ax_delta)
    has_pair = ("Proposed" in sims_dist) and ("BaselineB(NoRelax)" in sims_dist)
    if has_pair:
        eP = np.linalg.norm(sims_dist["Proposed"]["x"][0:3, :] - sims_dist["Proposed"]["Pref"], axis=0)
        eB = np.linalg.norm(sims_dist["BaselineB(NoRelax)"]["x"][0:3, :] - sims_dist["BaselineB(NoRelax)"]["Pref"], axis=0)
        Nd = min(eP.size, eB.size)
        delta_e = eP[:Nd] - eB[:Nd]
        _add_gust_shading(ax_delta, gusts)
        ax_delta.plot(t[:Nd], delta_e, linewidth=1.2,
                      label=r"Proposed $-$ matched-active ablation")
        ax_delta.axhline(0.0, linestyle=":", linewidth=1.0, color="black")
        ax_delta.set_ylabel(r"$\Delta \|e_p\|_2$ (m)")
        ax_delta.legend(frameon=False, loc="best", fontsize=6)
        if len(gusts) > 0:
            gmax = max(gusts, key=lambda gg: float(gg.get("amp", 0.0)))
            t0g = float(gmax["t0"])
            t1g = float(gmax["t1"])
            zoom_l = max(0.0, t0g - 1.0)
            zoom_r = min(float(t[-1]), t1g + 4.0)
            ax_delta.set_xlim(zoom_l, zoom_r)
    ax_delta.set_xlabel("Time (s)")

    save_pub_figure(fig, out_dir, stem, formats=syn.fig_formats,
                    dpi_png=syn.fig_dpi_png, transparent=syn.fig_transparent)
    plt.close(fig)


# =========================================================
# Plotting utilities
# =========================================================
def export_exp1_metrics_table(
        metrics_all: Dict[str, Dict[str, float]],
        out_dir: str,
        gamma_map: Optional[Dict[str, float]] = None,
        order: Optional[List[str]] = None,
        pretty_names: Optional[Dict[str, str]] = None,
        stem: str = "Exp1_metrics_table",
) -> None:
    if order is None:
        order = ["Proposed", "BaselineB(NoRelax)", "NominalLQR"]
    if pretty_names is None:
        pretty_names = {
            "Proposed": "Proposed",
            "BaselineB(NoRelax)": "Matched-active ablation",
            "NominalLQR": "Nominal LQR",
        }

    columns = [
        "gamma",
        "RMSE_pos", "RMSE_excess",
        "IAE_pos", "IAE_excess",
        "peak_pos", "peak_excess",
        "settle_mean_excess", "settle_worst_excess",
        "du_peak", "uabs_peak",
        "energy_uc", "energy_uabs",
        "duty_abs", "duty_normsat", "duty_ratesat",
        "rho",
    ]

    rows = [n for n in order if n in metrics_all]

    os.makedirs(out_dir, exist_ok=True)

    # --- CSV ---
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Controller"] + columns)
        for name in rows:
            met = metrics_all[name]
            vals: List[str] = []
            for c in columns:
                if c == "gamma" and gamma_map is not None and name in gamma_map:
                    vals.append(f"{gamma_map[name]:.4f}")
                elif c in met:
                    vals.append(f"{met[c]:.4g}")
                else:
                    vals.append("")
            writer.writerow([pretty_names.get(name, name)] + vals)

    # --- LaTeX ---
    def _fmt(v: float) -> str:
        if not np.isfinite(v):
            return "--"
        if abs(v) >= 1000:
            return f"{v:.2e}"
        if abs(v) < 0.001 and v != 0.0:
            return f"{v:.3e}"
        return f"{v:.4g}"

    tex_path = os.path.join(out_dir, f"{stem}.tex")
    n_cols = len(columns)
    col_spec = "l" + "r" * n_cols
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")
        header = " & ".join(["Controller"] + [c.replace("_", r"\_") for c in columns])
        f.write(header + " \\\\\n")
        f.write("\\midrule\n")
        for name in rows:
            met = metrics_all[name]
            cells: List[str] = [pretty_names.get(name, name)]
            for c in columns:
                if c == "gamma" and gamma_map is not None and name in gamma_map:
                    cells.append(_fmt(gamma_map[name]))
                elif c in met:
                    cells.append(_fmt(met[c]))
                else:
                    cells.append("--")
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")




# =========================================================
# Fusion Mechanism Visualization










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
        print(f"  [I0 residual diagnostic] processed score-zero subset I0 is EMPTY; skipping residual diagnostic on I0")

    diagnostic_payload = build_fusion_diagnostics_payload(
        batch, verts_norm, Psi, W_c, W_E, info_Wc, score_diag,
        list(bounds.__dataclass_fields__.keys()), p_data_source,
        model_residual_all=model_residual_all,
        model_residual_certified=model_residual_i0,
    )
    diagnostic_path = result_path("caches", "standard_data_fusion_diagnostics.npz")
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
    )


# =========================================================
# Plotting utilities
# =========================================================
def main_stage3_time_domain_validation_A_C(
        syn: SynthesisParams,
        bounds: ParamBounds,

        ref_type: str = "square",
        seconds: float = 20.0,
        num_gusts: int = 5,
        bias_frac: float = 0.20,
        gust_dur_range: Tuple[float, float] = (0.6, 1.4),
        gust_amp_range: Tuple[float, float] = (0.45, 1.0),
        meas_noise_std: float = 5e-4,

        max_vertices_proposed: int = 128,
        direct_vertex_limit: int = 20,
        baseline1_vertices: int = 20,
        score_mode: str = "lambda_max",
        eps_excess: float = 0.05,
        hold_time: float = 0.5,
        post_window: float = 3.0,
        ref_amp: float = 0.5,
        ref_period: float = 16.0,
        ref_start_delay: float = 1.0,
) -> None:
    out = synthesize_controllers_for_stage3(
        syn=syn, bounds=bounds,
        max_vertices_proposed=max_vertices_proposed,
        direct_vertex_limit=direct_vertex_limit,
        baseline1_vertices=baseline1_vertices,
        score_mode=score_mode,
        enforce_perf_all_vertices=syn.enforce_perf_all_vertices,
        baseline1_threads=1,
    )

    print(f"\nGamma comparison:")
    print(f"  Proposed:              {out['gamma_proposed']:.6g}")
    print(f"  BaselineB (no-relax):  {out['gamma_baselineB']:.6g}\n")

    imp_B = (out['gamma_baselineB'] - out['gamma_proposed']) / out['gamma_baselineB'] * 100
    print(f"  Improvement vs BaselineB: {imp_B:.2f}%")

    rng = np.random.default_rng(syn.seed + 555)
    print("\nUsing data source parameters with 5% noise for validation")
    p_true = out['p_data_source']
    p_true_raw = {k: v * rng.uniform(0.95, 1.05) for k, v in p_true.items()}
    outside_raw = [
        k for k, value in p_true_raw.items()
        if not (getattr(bounds, k)[0] <= value <= getattr(bounds, k)[1])
    ]
    # clip to prior bounds for strict near-source validation within prior box
    p_true = {k: float(np.clip(v_raw, getattr(bounds, k)[0], getattr(bounds, k)[1]))
              for k, v_raw in p_true_raw.items()}
    print(f"  Within-prior validation: raw local sample outside on "
          f"{len(outside_raw)}/{len(p_true_raw)} parameter channels; clipped channels={outside_raw}")

    print("\n" + "=" * 40)
    print(" True plant parameters:")
    print("=" * 40)
    for k in sorted(p_true.keys()):
        val = p_true[k]
        low, high = getattr(bounds, k)
        print(f"  {k:<8}: {val:.6f}  (range: {low:.3f}-{high:.3f})")
    print("=" * 40 + "\n")

    if ref_type == "square":
        ref_fun = lambda tt: ref_square_wave_x(
            tt,
            amp=float(ref_amp),
            period=float(ref_period),
            start_delay=float(ref_start_delay),
        )
        print(f"Reference: square wave on x (amp={ref_amp}, period={ref_period}s, hold={ref_period / 2:.2f}s)")
    else:
        raise ValueError("ref_type must be 'square'")

    dbar0 = np.zeros((6, int(round(seconds / syn.Ts))), dtype=float)
    dbar1, gusts, gust_projection_stats = build_multi_gust_profile(
        syn=syn,
        seconds=seconds,
        num_gusts=num_gusts,
        t_min=2.0,
        t_max=seconds - 1.0,
        dur_range=gust_dur_range,
        amp_range=gust_amp_range,
        bias_frac=bias_frac,
        avoid_ref_jumps=True,
        ref_start_delay=float(ref_start_delay),
        ref_period=float(ref_period),
        seed_offset=888,
    )
    print(f"Multi-gust disturbance: {len(gusts)} gusts, bias={bias_frac:.3g}")
    print(f"  normalized disturbance raw/applied peak = "
          f"{gust_projection_stats['raw_peak_normalized']:.3f}/"
          f"{gust_projection_stats['applied_peak_normalized']:.3f}; "
          f"projection rate = {gust_projection_stats['projection_rate']:.3f}, "
          f"mean factor = {gust_projection_stats['average_projection_factor']:.3f}")
    for i, g in enumerate(gusts[:5]):
        print(f"  gust {i}: t={g['t0']:.2f}-{g['t1']:.2f}s, amp={g['amp']:.2f}")

    sims_dist: Dict[str, Dict[str, Any]] = {}
    sims_nodist: Dict[str, Dict[str, Any]] = {}

    for name, K in [
        ("Proposed", out["K_proposed"]),
        ("BaselineB(NoRelax)", out["K_baselineB"]),
        ("NominalLQR", out["K_nominal"]),
    ]:
        sim0 = simulate_tracking_with_disturbance_profile(
            K=K,
            p_true=p_true,
            syn=syn,
            ref_fun=ref_fun,
            seconds=seconds,
            dbar_profile=dbar0,
            meas_noise_std=meas_noise_std,
        )
        sim1 = simulate_tracking_with_disturbance_profile(
            K=K,
            p_true=p_true,
            syn=syn,
            ref_fun=ref_fun,
            seconds=seconds,
            dbar_profile=dbar1,
            meas_noise_std=meas_noise_std,
        )
        sims_nodist[name] = sim0
        sims_dist[name] = sim1

    print("\nStage 3 metrics (multi-gust + saturation):")
    metrics_all: Dict[str, Dict[str, float]] = {}
    for name in sims_dist.keys():
        met = compute_metrics_excess_multi_gust(
            sim_dist=sims_dist[name],
            sim_nodist=sims_nodist[name],
            syn=syn,
            gusts=gusts,
            eps_excess=eps_excess,
            hold_time=hold_time,
            recovery_ref="gust_end",
            post_window=post_window,
        )
        metrics_all[name] = met

        print(f"\n[{name}]")
        print(f"  RMSE_pos                    = {met['RMSE_pos']:.6g}")
        print(f"  RMSE_excess                 = {met['RMSE_excess']:.6g}")
        print(f"  IAE_pos                     = {met['IAE_pos']:.6g}")
        print(f"  IAE_excess                  = {met['IAE_excess']:.6g}")
        print(f"  peak_pos                    = {met['peak_pos']:.6g}")
        print(f"  peak_excess                 = {met['peak_excess']:.6g}")
        print(
            f"  settle_mean_excess          = {met['settle_mean_excess']:.6g} s  (eps={eps_excess}, hold={hold_time}s)")
        print(f"  settle_worst_excess         = {met['settle_worst_excess']:.6g} s")
        print(f"  IAE_excess_postgust_avg      = {met['IAE_excess_postgust_avg']:.6g}  (win={post_window}s)")
        print(f"  peak_excess_postgust_avg     = {met['peak_excess_postgust_avg']:.6g}  (win={post_window}s)")
        print(f"  normalized du_peak          = {met['du_peak']:.6g}   (limit=1)")
        print(
            f"  uabs_peak                   = {met['uabs_peak']:.6g}   (u_abs bounds={syn.u_abs_min}..{syn.u_abs_max})")
        print(f"  duty_du (normalized increment near limit) = {met['duty_du']:.3f}")
        print(f"  duty_abs (abs clamp)         = {met['duty_abs']:.3f}")
        print(f"  duty_normsat (norm sat)      = {met['duty_normsat']:.3f}")
        print(f"  duty_ratesat (rate sat)      = {met['duty_ratesat']:.3f}")
        print(f"  max normalized command/applied increment = "
              f"{met['du_cmd_peak']:.3f}/{met['du_act_peak']:.3f}")
        print(f"  rho(A+BK) (linear proxy)     = {met['rho']:.6g}")
        if met["rho"] >= 1.0:
            print("  Note: rho(A+BK) >= 1, linear stability may be marginal")

    plot_stage3_A_C_publication(sims_dist, sims_nodist, syn, gusts)

    plot_stage3_story_figure(
        sims_dist, sims_nodist, syn, gusts,
        out_dir=syn.fig_out_dir, stem="FigExp1_Story",
    )

    gamma_map = {
        "Proposed": out["gamma_proposed"],
        "BaselineB(NoRelax)": out["gamma_baselineB"],
        "NominalLQR": float("nan"),
    }
    export_exp1_metrics_table(
        metrics_all, out_dir=syn.fig_out_dir,
        gamma_map=gamma_map, stem="Exp1_metrics_table",
    )

    parameter_keys = list(bounds.__dataclass_fields__.keys())
    save_payload: Dict[str, Any] = dict(
        parameter_keys=np.asarray(parameter_keys),
        p_true=np.asarray([p_true[k] for k in parameter_keys], dtype=float),
        p_true_raw=np.asarray([p_true_raw[k] for k in parameter_keys], dtype=float),
        p_true_raw_outside_channel_count=np.array([len(outside_raw)]),
        p_true_raw_outside_plant_count=np.array([int(bool(outside_raw))]),
        p_true_applied_outside_plant_count=np.array([0]),
        gamma_proposed=np.array([out["gamma_proposed"]]),
        gamma_baselineB=np.array([out["gamma_baselineB"]]),
        Ts=np.array([syn.Ts]),
        du_max=np.array([syn.du_max]),
        d_max=np.array([syn.d_max]),
        u_abs_min=np.array(syn.u_abs_min, dtype=float),
        u_abs_max=np.array(syn.u_abs_max, dtype=float),
        seconds=np.array([seconds]),
        num_gusts=np.array([len(gusts)]),
        bias_frac=np.array([bias_frac]),
        disturbance_raw_peak_normalized=np.array([gust_projection_stats["raw_peak_normalized"]]),
        disturbance_applied_peak_normalized=np.array([gust_projection_stats["applied_peak_normalized"]]),
        disturbance_projection_count=np.array([gust_projection_stats["projection_count"]]),
        disturbance_projection_rate=np.array([gust_projection_stats["projection_rate"]]),
        disturbance_average_projection_factor=np.array([gust_projection_stats["average_projection_factor"]]),
        state_scales=np.asarray(DEFAULT_SCALES.state),
        input_memory_scales=np.asarray(DEFAULT_SCALES.input_memory),
        increment_scales=np.asarray(DEFAULT_SCALES.increment),
        disturbance_scales=np.asarray(DEFAULT_SCALES.disturbance),
        residual_cap_scales=np.asarray(DEFAULT_SCALES.residual_cap),
        residual_standard_deviations=np.asarray(DEFAULT_SCALES.residual_std),
        active_vertex_parameters=np.asarray(
            [[v["p"][k] for k in parameter_keys] for v in out["vertices_common"]],
            dtype=float,
        ),
        active_vertex_scores=np.asarray([v["s"] for v in out["vertices_common"]]),
        zero_score_count=np.array([
            sum(float(v["s"]) == 0.0 for v in out["verts_norm"])
        ]),
        total_vertex_count=np.array([len(out["verts_norm"])]),
        eps_pass=np.array([EPS_PASS]),
        eps_near=np.array([EPS_NEAR]),
        proposed_solver_success=np.array([bool(out["sol_proposed"].get("success", False))]),
        proposed_surrogate_status=np.asarray([
            str(out["sol_proposed"].get("status", "not_checked"))
        ]),
        baseline_solver_success=np.array([bool(out["sol_baselineB"].get("success", False))]),
        proposed_objective=np.array([
            syn.w_gamma * float(out["sol_proposed"]["gamma2"][0])
            + syn.w_mu * float(out["sol_proposed"]["mu"][0])
            + syn.w_beta * float(out["sol_proposed"]["beta"][0])
        ]),
        baseline_objective=np.array([
            syn.w_gamma * float(out["sol_baselineB"]["gamma2"][0])
            + syn.w_mu * float(out["sol_baselineB"]["mu"][0])
            + syn.w_beta * float(out["sol_baselineB"]["beta"][0])
        ]),
        eps_excess=np.array([eps_excess]),
        hold_time=np.array([hold_time]),
        post_window=np.array([post_window]),
        meas_noise_std=np.array([meas_noise_std]),
        ref_amp=np.array([float(ref_amp)]),
        ref_period=np.array([float(ref_period)]),
        ref_start_delay=np.array([float(ref_start_delay)]),
    )
    for i, g in enumerate(gusts):
        save_payload[f"gust_{i}_t0"] = np.array([g["t0"]], dtype=float)
        save_payload[f"gust_{i}_t1"] = np.array([g["t1"]], dtype=float)
        save_payload[f"gust_{i}_amp"] = np.array([g["amp"]], dtype=float)

    for name, met in metrics_all.items():
        prefix = name.replace("(", "_").replace(")", "_").replace(" ", "_")
        for k, v in met.items():
            save_payload[f"{prefix}_{k}"] = np.array([float(v)])

    for k, v in out["sol_proposed"].items():
        if isinstance(v, np.ndarray):
            save_payload[f"proposed_{k}"] = v
    for k, v in out["sol_baselineB"].items():
        if isinstance(v, np.ndarray):
            save_payload[f"baselineB_{k}"] = v

    for controller, simulation in sims_dist.items():
        safe = "".join(ch if ch.isalnum() else "_" for ch in controller).strip("_")
        append_tracking_simulation_payload(
            save_payload, f"disturbed_{safe}", simulation
        )
    for controller, simulation in sims_nodist.items():
        safe = "".join(ch if ch.isalnum() else "_" for ch in controller).strip("_")
        append_tracking_simulation_payload(
            save_payload, f"undisturbed_{safe}", simulation
        )

    save_payload["dbar_profile"] = dbar1

    results_path = result_path("caches", "stage3_time_domain_results.npz")
    np.savez(results_path, **save_payload)
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
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
        fig_out_dir=result_path("time-new")
    )

    main_stage3_time_domain_validation_A_C(
        syn=my_syn,
        bounds=my_bounds,
        ref_type="square",
        seconds=30.0,
        num_gusts=5,
        bias_frac=0.40,
        gust_dur_range=(0.6, 1.4),
        gust_amp_range=(0.45, 1.0),
        meas_noise_std=5e-4,
        max_vertices_proposed=1024,
        direct_vertex_limit=20,
        baseline1_vertices=20,
        score_mode="lambda_max",
        ref_amp=0.5,
        ref_period=16.0,
        ref_start_delay=1.0,
    )
