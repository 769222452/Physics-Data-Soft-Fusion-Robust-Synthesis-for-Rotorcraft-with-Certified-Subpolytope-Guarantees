# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import csv
import pickle
import itertools
import glob
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Optional, Sequence, Callable, Any, Set
from scipy.optimize import linprog

import numpy as np
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
    build_synthesis_diagnostics_payload,
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
    state_to_normalized,
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
            print(f"[MOSEK] Auto-found license file: {lic}")
        return lic

    env_val = os.environ.get("MOSEKLM_LICENSE_FILE", "")
    if env_val and ("@" in env_val or os.path.isfile(env_val)):
        if verbose:
            print(f"[MOSEK] Using configured license: {env_val}")
        return env_val
    raise FileNotFoundError(
        "MOSEK license was not found. Set MOSEKLM_LICENSE_FILE to a license "
        "file path or to a license server such as '27000@server'. "
        f"Current MOSEKLM_LICENSE_FILE='{env_val}'."
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


# =========================================================
# Single source of truth for the 8 geometric pairs shown in the paper.
# File names are `Exp1_Geometry_{x}_vs_{y}.pdf` and must match the LaTeX
# includegraphics paths in cas-sc-template.tex.
# =========================================================
FIG4_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("Jx", "Jy"),
    ("Jx", "kp"),
    ("Jy", "kq"),
    ("Jz", "kr"),
    ("sigma_t", "kz"),
    ("kx", "ky"),
    ("kp", "kq"),
    ("ky", "kr"),
)




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
    """PSD upper bound at every adopted vertex and the batch generator.

    Random interior samples are evaluated separately as a numerical sanity check only."""
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

    old_alpha = float(safety_factor) * vertex_max_eig
    alpha = float(safety_factor) * max(vertex_max_eig, generator_max_eig)
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
        generator_changed_bound=bool(generator_max_eig > vertex_max_eig),
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


def farthest_point_select_vertices(
        all_vertices: List[Dict[str, object]],
        K_budget: int,
        bounds: ParamBounds,
        seed: int = 0,
) -> List[int]:
    """
    """
    keys = ["sigma_t", "Jx", "Jy", "Jz", "kx", "ky", "kz", "kp", "kq", "kr"]
    scales = []
    for k in keys:
        lo, hi = getattr(bounds, k)
        scales.append((lo, hi - lo if hi > lo else 1.0))

    N = len(all_vertices)
    coords = np.zeros((N, len(keys)))
    for i, v in enumerate(all_vertices):
        for j, k in enumerate(keys):
            coords[i, j] = (float(v["p"][k]) - scales[j][0]) / scales[j][1]

    rng = np.random.default_rng(seed)
    selected = [rng.integers(N)]
    min_dist = np.full(N, np.inf)

    for _ in range(K_budget - 1):
        last = coords[selected[-1]]
        d = np.sum((coords - last) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[selected] = -1.0
        selected.append(int(np.argmax(min_dist)))

    return sorted(selected)


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
def selfcheck_print_param_sraw(
        p: Dict[str, float],
        syn: SynthesisParams,
        Psi_data: np.ndarray,
        *,
        label: str = "p_true",
        score_mode: str = "lambda_max",
        compare_scores: Optional[np.ndarray] = None,
) -> float:
    """
    """
    Ts, g = syn.Ts, syn.g
    A, B, _ = build_vertex_matrices(p, Ts, g)
    Cc, Dc = build_performance_matrices(syn)
    Delta = np.block([[A, B],
                      [Cc, Dc]])

    s_raw = compute_vertex_score_scalar(Delta, Psi_data, mode=score_mode)

    msg = f"[SelfCheck] {label}: s_raw={s_raw:.6e} (mode={score_mode})"

    if compare_scores is not None and len(compare_scores) > 0:
        cs = np.asarray(compare_scores, dtype=float)
        cs_min = float(np.min(cs))
        cs_max = float(np.max(cs))
        rank_le = int(np.sum(cs <= s_raw))
        pct = 100.0 * rank_le / float(len(cs))
        msg += (f" | among pool: min={cs_min:.6e}, max={cs_max:.6e}, "
                f"percentile={pct:.2f}% (<=count {rank_le}/{len(cs)}), "
                f"gap_to_min={s_raw - cs_min:.3e}")

    print(msg)
    return float(s_raw)


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
    raw_s = np.array([float(v["s"]) for v in vertices], dtype=float)
    n = len(raw_s)
    n_raw_consistent = int(np.sum(raw_s <= tau))

    # --- Step 0: Processed-score threshold fallback detection ---
    hard_consistent_mask = raw_s <= tau
    hard_ratio = float(np.mean(hard_consistent_mask))
    soft_mode = False
    tau_effective = tau

    if soft_fallback and hard_ratio < hard_ratio_min:
        tau_effective = float(np.quantile(raw_s, rho))
        hard_consistent_mask = raw_s <= tau_effective
        hard_ratio = float(np.mean(hard_consistent_mask))
        soft_mode = True
        print(f"  [ScorePipeline] PROCESSED-SCORE FALLBACK: "
              f"hard_ratio(tau=0)={float(np.mean(raw_s <= tau)):.4f} < {hard_ratio_min}")
        print(f"  [ScorePipeline]   tau_effective={tau_effective:.6e} "
              f"(rho={rho} quantile of raw scores)")

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


def sample_consistent_models(
        bounds: ParamBounds,
        syn: SynthesisParams,
        Psi_data: np.ndarray,
        *,
        tau: float,
        score_mode: str = "lambda_max",
        n_samples: int = 8000,
        seed: int = 0,
) -> List[Dict[str, object]]:
    Ts, g = syn.Ts, syn.g
    Cc, Dc = build_performance_matrices(syn)
    keys = list(bounds.__dataclass_fields__.keys())
    rng = np.random.default_rng(int(seed))

    out = []
    for _ in range(int(n_samples)):
        p = {}
        for k in keys:
            lo, hi = getattr(bounds, k)
            p[k] = float(rng.uniform(lo, hi))

        A, B, S_c = build_vertex_matrices(p, Ts, g)
        Delta = np.block([[A, B],
                          [Cc, Dc]])
        s_raw = float(compute_vertex_score_scalar(Delta, Psi_data, mode=score_mode))
        if s_raw <= float(tau):
            out.append(dict(A=A, B=B, S=S_c, C=Cc, D=Dc, Delta=Delta, s=s_raw, p=p))
    return out

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


def _affine_whiten_from_columns(V: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    mu = np.mean(V, axis=1, keepdims=True)
    sig = np.std(V, axis=1, keepdims=True)
    sig = np.where(sig < eps, 1.0, sig)
    Vw = (V - mu) / sig
    return Vw, mu, sig


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

def select_vertices_support_delta(
        vertices: List[Dict[str, object]],
        K: int,
        *,
        n_dirs: int = 300,
        seed: int = 0,
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

    mu = np.mean(X, axis=0, keepdims=True)
    sig = np.std(X, axis=0, keepdims=True)
    sig = np.where(sig < 1e-12, 1.0, sig)
    Xw = (X - mu) / sig

    rng = np.random.default_rng(int(seed))
    m = Xw.shape[1]
    dirs = rng.standard_normal((int(n_dirs), m))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    idxs = set()
    for d in dirs:
        proj = Xw @ d
        idxs.add(int(np.argmax(proj)))
        idxs.add(int(np.argmin(proj)))
        if len(idxs) >= 5 * K:
            break

    cand = [vertices[i] for i in sorted(idxs)]

    if len(cand) > K:
        cand = select_vertices_farthest_delta(cand, K=K, initial_index=int(initial_index))
    elif len(cand) < K:
        rest = [vertices[i] for i in range(len(vertices)) if i not in idxs]
        if len(rest) > 0:
            extra = select_vertices_farthest_delta(rest, K=min(K - len(cand), len(rest)), initial_index=0)
            cand = cand + extra

    return cand[:K]

def select_vertices_stratified_farthest_delta(
        vertices: List[Dict[str, object]],
        K: int,
        *,
        ratios: Tuple[float, float, float] = (0.4, 0.2, 0.4),
        seed: int = 0,
) -> List[Dict[str, object]]:
    """
    """
    if len(vertices) == 0:
        return []
    if K >= len(vertices):
        return list(vertices)

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
        sel_low = select_vertices_farthest_delta(low, K=min(k_low, len(low)), initial_index=0)

    sel_mid = []
    if k_mid > 0 and len(mid) > 0:
        init = int(rng.integers(0, len(mid)))
        sel_mid = select_vertices_farthest_delta(mid, K=min(k_mid, len(mid)), initial_index=init)

    sel_high = []
    if k_high > 0 and len(high) > 0:
        init = int(rng.integers(0, len(high)))
        sel_high = select_vertices_farthest_delta(high, K=min(k_high, len(high)), initial_index=init)

    out = sel_low + sel_mid + sel_high

    if len(out) < K:
        seen = {id(v) for v in out}
        rest = [v for v in verts_sorted if id(v) not in seen]
        if len(rest) > 0:
            need = K - len(out)
            extra = select_vertices_farthest_delta(rest, K=min(need, len(rest)), initial_index=0)
            out.extend(extra)

    return out[:K]


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
        n_s_pos = sum(1 for i in active_idx if s_all[i] > 1e-12)
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
            n_nonzero = sum(1 for sv in s_vals if sv > 1e-12)
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
            if violations[i] > viol_tol and float(all_vertices[i]["s"]) < 1e-6
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
        will_add_nonzero = s_all[swap_in_idx] > 1e-12

        # --- Find swap-out: most redundant, respecting tier & m_incon ---
        active_violations = {i: violations[i] for i in active_idx}
        candidates_remove = sorted(active_violations.keys(), key=lambda i: active_violations[i])

        active_s_nonzero_count = sum(1 for i in active_idx if s_all[i] > 1e-12)

        def _can_remove(i: int) -> bool:
            removing_nonzero = s_all[i] > 1e-12
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
        "BaselineB(NoRelax)": "Robust (No Relax.)",
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
        ax.plot(t[:N], delta, linewidth=1.2, label=r"Proposed $-$ Robust (No Relax.)")
        ax.axhline(0.0, linestyle=":", linewidth=1.0, color="black")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\Delta \|e_p\|_2$ (m)")
        if syn.fig_show_titles:
            ax.set_title("Error-norm difference (Proposed minus Robust No Relax.)")
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
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="black", label="normalized bound")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|\bar u_c\|_2$")
    if syn.fig_show_titles:
        ax.set_title("Increment command norm (rate constraint activity)")
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
        "BaselineB(NoRelax)": "Robust (No Relax.)",
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
                  label="normalized bound")
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
        ax_delta.plot(t[:Nd], delta_e, linewidth=1.2, label=r"Proposed $-$ Robust (No Relax.)")
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
            "BaselineB(NoRelax)": "Robust (No Relax.)",
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


def plot_gamma_comparison_nature(
        gamma_baseB: float,
        gamma_prop: float,
        out_dir: str,
        syn: SynthesisParams,
        stem: str = "FigS3_Gamma_Comparison",
) -> None:
    sizes = set_publication_style(context=syn.fig_context, column="single")
    fig_w = sizes["single"][0] * 0.85
    fig_h = fig_w * 1.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    values = [gamma_baseB, gamma_prop]
    labels = ["Robust\n(No Relax.)", "Proposed"]
    colors = ["#0072B2", "#D55E00"]

    bars = ax.bar(labels, values, color=colors, width=0.45,
                  edgecolor="black", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02 * max(values),
                f"{val:.3f}",
                ha='center', va='bottom', fontsize=7, fontweight='bold', color="black")

    xB = bars[0].get_x() + bars[0].get_width() / 2.0
    xP = bars[1].get_x() + bars[1].get_width() / 2.0
    y_max = max(values)
    y_line = y_max * 1.15
    y_text = y_line + 0.02 * y_max

    ax.plot([xB, xB, xP, xP], [y_max * 1.05, y_line, y_line, gamma_prop * 1.05 + (y_max - gamma_prop)],
            color="black", linewidth=0.6)

    imp = (gamma_baseB - gamma_prop) / gamma_baseB * 100
    ax.text((xB + xP) / 2.0, y_text, f"$-${imp:.1f}\\% Conservatism",
            ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel(r"$H_\infty$ Performance Level $\gamma$", fontsize=8)
    ax.set_ylim(0, y_line * 1.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    save_pub_figure(fig, out_dir, stem, formats=syn.fig_formats, dpi_png=600)
    plt.close(fig)


# =========================================================
# Plotting utilities
# =========================================================
def plot_fusion_mechanism_and_histogram(
        bounds: ParamBounds,
        syn: SynthesisParams,
        Psi: np.ndarray,
        p_center: Dict[str, float],
        p_true_source: Dict[str, float],
        vertices_all: List[Dict[str, object]],
        out_dir: str,
        param_pair: Tuple[str, str] = ("Jx", "sigma_t"),
        suffix: str = ""
):
    import matplotlib.patches as patches

    x_name, y_name = param_pair
    print(f"    Plotting 2D slice: {x_name} vs {y_name} ...")

    label_map = {
        "sigma_t": r"Thrust Coeff. $\sigma_t$",
        "Jx": r"Inertia $J_x$ (kg$\cdot$m$^2$)",
        "Jy": r"Inertia $J_y$ (kg$\cdot$m$^2$)",
        "Jz": r"Inertia $J_z$ (kg$\cdot$m$^2$)",
        "kx": r"Drag Coeff. $k_x$",
        "ky": r"Drag Coeff. $k_y$",
        "kz": r"Drag Coeff. $k_z$",
        "kp": r"Gain/Damp $k_p$",
        "kq": r"Gain/Damp $k_q$",
        "kr": r"Gain/Damp $k_r$",
    }

    x_range = getattr(bounds, x_name)
    y_range = getattr(bounds, y_name)

    margin_x = (x_range[1] - x_range[0]) * 0.2
    margin_y = (y_range[1] - y_range[0]) * 0.2

    if margin_x < 1e-9: margin_x = 0.01
    if margin_y < 1e-9: margin_y = 0.01

    x_grid = np.linspace(x_range[0] - margin_x, x_range[1] + margin_x, 30)
    y_grid = np.linspace(y_range[0] - margin_y, y_range[1] + margin_y, 30)
    X, Y = np.meshgrid(x_grid, y_grid)

    Z_score = np.zeros_like(X)

    Cc, Dc = build_performance_matrices(syn)
    Ts, g = syn.Ts, syn.g

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p_temp = p_center.copy()
            p_temp[x_name] = X[i, j]
            p_temp[y_name] = Y[i, j]

            A, B, S_c = build_vertex_matrices(p_temp, Ts, g)
            Delta = np.block([[A, B], [Cc, Dc]])

            s_raw = compute_vertex_score_scalar(Delta, Psi, mode="lambda_max")
            Z_score[i, j] = s_raw
    sizes = set_publication_style(context="paper", column="single")
    fig1, ax1 = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)

    cmap = plt.cm.RdBu_r

    levels = np.linspace(np.min(Z_score), np.max(Z_score), 20)
    cf = ax1.contourf(X, Y, Z_score, levels=levels, cmap=cmap, alpha=0.8, extend='both')

    if np.min(Z_score) < 0 < np.max(Z_score):
        ax1.contour(X, Y, Z_score, levels=[0], colors='green', linewidths=2.0, linestyles='--')

    rect = patches.Rectangle(
        (x_range[0], y_range[0]),
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        linewidth=2, edgecolor='black', facecolor='none', linestyle='-', label=r'$\mathcal{U}_{\mathrm{prior}}$'
    )
    ax1.add_patch(rect)

    ax1.scatter(p_true_source[x_name], p_true_source[y_name], c='yellow', edgecolors='black', s=100, marker='*',
                zorder=10, label=r'True Plant')

    ax1.set_xlabel(label_map.get(x_name, x_name))
    ax1.set_ylabel(label_map.get(y_name, y_name))
    ax1.set_title(f"Fusion Geometry: {x_name} vs {y_name}")

    cbar = fig1.colorbar(cf, ax=ax1)
    # this field is the RAW data-inconsistency score s_i^raw (not the processed s_i).
    cbar.set_label(r"Raw inconsistency score $s_i^{\mathrm{raw}}$")

    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    fname = f"Exp1_Geometry_{x_name}_vs_{y_name}{suffix}"
    save_pub_figure(fig1, out_dir, fname, formats=("pdf", "png"))
    plt.close(fig1)

def check_selected_polytope_covers_near_ellipsoid(
        verts_sel,
        bounds,
        syn,
        Psi_data,
        *,
        tau,
        score_mode="lambda_max",
        n_samples=3000,
        max_checked=200,
        lp_tol=1e-6,
        seed=0,
):
    """


    """
    rng = np.random.default_rng(int(seed))

    D_sel = np.vstack([np.asarray(v["Delta"], dtype=float).reshape(1, -1) for v in verts_sel])  # (K, m)
    V = D_sel.T  # (m, K)
    m, K = V.shape

    Vw, mu, sig = _affine_whiten_from_columns(V)

    keys = list(bounds.__dataclass_fields__.keys())

    def sample_param_dict():
        p = {}
        for k in keys:
            lo, hi = getattr(bounds, k)
            p[k] = float(rng.uniform(lo, hi))
        return p

    def sraw_and_delta(p):
        A, B, _ = build_vertex_matrices(p, syn.Ts, syn.g)
        Cc, Dc = build_performance_matrices(syn)
        Delta = np.block([[A, B],
                          [Cc, Dc]])
        s_raw = float(compute_vertex_score_scalar(Delta, Psi_data, mode=score_mode))
        return s_raw, Delta

    X_cons = []
    for _ in range(int(n_samples)):
        p = sample_param_dict()
        s_raw, Delta = sraw_and_delta(p)
        if s_raw <= float(tau):
            x = np.asarray(Delta, dtype=float).reshape(-1, 1)  # (m,1)
            xw = (x - mu) / sig
            X_cons.append(xw.reshape(-1))

    n_cons = len(X_cons)
    if n_cons == 0:
        print(f"[HullCheck] No samples found with s_raw<=tau ({tau:.3e}). "
              f"Try increasing n_samples or tau.")
        return

    idx = np.arange(n_cons)
    rng.shuffle(idx)
    idx = idx[:min(int(max_checked), n_cons)]

    def inside_convex_hull_lp(xw: np.ndarray) -> Tuple[bool, float]:
        """
        min t  s.t.  ||Vw a - xw||_inf <= t, sum(a)=1, a>=0, t>=0
        """
        xw = np.asarray(xw, dtype=float).reshape(m)

        c = np.zeros(K + 1)
        c[-1] = 1.0  # min t

        A_ub = np.zeros((2 * m, K + 1))
        b_ub = np.zeros(2 * m)

        # Vw a - t <= xw
        A_ub[0:m, 0:K] = Vw
        A_ub[0:m, K] = -1.0
        b_ub[0:m] = xw

        # -Vw a - t <= -xw
        A_ub[m:2 * m, 0:K] = -Vw
        A_ub[m:2 * m, K] = -1.0
        b_ub[m:2 * m] = -xw

        A_eq = np.zeros((1, K + 1))
        A_eq[0, 0:K] = 1.0
        b_eq = np.array([1.0])

        bounds_lp = [(0.0, None)] * K + [(0.0, None)]

        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds_lp, method="highs"
        )
        if res.status != 0 or res.x is None:
            return False, float("inf")
        t = float(res.x[-1])
        return (t <= float(lp_tol)), t

    inside = 0
    t_list = []
    for j in idx:
        ok, t = inside_convex_hull_lp(X_cons[j])
        t_list.append(t)
        if ok:
            inside += 1

    t_list = np.asarray(t_list, dtype=float)
    cover_ratio = inside / float(len(idx))

    print(
        f"[HullCheck] (Delta-space) tau={tau:.3e}, sampled={n_samples}, consistent={n_cons} "
        f"(~{100.0*n_cons/float(n_samples):.2f}%), checked={len(idx)} | "
        f"inside={inside} ({100.0*cover_ratio:.1f}%), "
        f"t_inf: median={np.median(t_list):.2e}, max={np.max(t_list):.2e}, tol={lp_tol:.1e}"
    )

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
    print(f"  [W_c generator] old alpha={info_Wc['old_vertex_only_alpha']:.6e}, "
          f"new alpha={info_Wc['alpha']:.6e}, changed={info_Wc['generator_changed_bound']}, "
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

    # use FIG4_PAIRS as the single source of truth for plot output names.
    plot_pairs = list(FIG4_PAIRS)

    print(f"\nGenerating {len(plot_pairs)} geometric fusion plots...")

    verts_all = build_all_vertices_and_scores(
        bounds, syn, Psi,
        score_mode=score_mode,
        max_vertices=1024,
        seed=syn.seed + 1234
    )

    for pair in plot_pairs:
        plot_fusion_mechanism_and_histogram(
            bounds=bounds,
            syn=syn,
            Psi=Psi,
            p_center=p_nom,
            p_true_source=p_data_source,
            vertices_all=verts_all,
            out_dir=syn.fig_out_dir,
            param_pair=pair,
            suffix=""
        )

    sizes = set_publication_style(context="paper", column="single")
    s_values = np.array([v["s"] for v in verts_all], dtype=float)
    s_true = selfcheck_print_param_sraw(
        p_data_source, syn, Psi,
        label="(hist) p_data_source",
        score_mode=score_mode,
    )

    fig_hist, ax_hist = plt.subplots(figsize=sizes["single"], constrained_layout=True)
    _setup_axes(ax_hist)

    s_pos = s_values[s_values > 0]
    use_log = False
    if len(s_pos) > 0 and float(np.max(s_pos)) > 10 * float(np.min(s_pos) + 1e-12):
        log_lo = np.log10(max(float(np.min(s_pos)), 1e-6))
        log_hi = np.log10(float(np.max(s_pos)) * 1.05)
        if log_hi > log_lo:
            bins_arr = np.logspace(log_lo, log_hi, 35)
            ax_hist.set_xscale("log")
            use_log = True
        else:
            bins_arr = 30
    else:
        bins_arr = 30

    n_h, bins_h, patches_h = ax_hist.hist(
        s_values, bins=bins_arr, color='#0072B2', edgecolor='black',
        linewidth=0.5, alpha=0.8,
    )

    pct25 = float(np.percentile(s_values, 25))
    ax_hist.axvline(pct25, color='#009E73', linestyle='--', linewidth=1.2,
                    label=f'25th percentile ($s={pct25:.1f}$)')

    if np.isfinite(s_true):
        ax_hist.axvline(s_true, color='#D55E00', linestyle='-', linewidth=1.5,
                        label=f'True plant ($s={s_true:.2f}$)')

    ax_hist.set_xlabel(r"Raw consistency score $s_i$" + (" (log scale)" if use_log else ""))
    ax_hist.set_ylabel("Number of vertices")
    ax_hist.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=6.5, loc='upper right')

    save_pub_figure(fig_hist, syn.fig_out_dir, "Exp2_Score_Histogram_Global", formats=("pdf", "png"))
    plt.close(fig_hist)

    print("  Plots generated")

    raw_s_values = np.array([v["s"] for v in verts_all])
    print(
        f"\nScore statistics: n={len(raw_s_values)}, min={raw_s_values.min():.4e}, max={raw_s_values.max():.4e}, mean={raw_s_values.mean():.4e}")
    hist, bin_edges = np.histogram(raw_s_values, bins=10)
    print(f"  Histogram: {hist}\n")
    # ===== Self-check: true plant / nominal plant raw score position =====
    selfcheck_print_param_sraw(
        p_data_source, syn, Psi,
        label="p_data_source (data-generating plant)",
        score_mode=score_mode,
        compare_scores=raw_s_values
    )

    selfcheck_print_param_sraw(
        p_nom, syn, Psi,
        label="p_nom (nominal center)",
        score_mode=score_mode,
        compare_scores=raw_s_values
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


    # ===== Baseline B (NoRelax): same vertices as Proposed, s=0 =====
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

    print("\nGenerating gamma comparison chart...")
    plot_gamma_comparison_nature(
        gamma_baseB=float(out['gamma_baselineB']),
        gamma_prop=float(out['gamma_proposed']),
        out_dir=syn.fig_out_dir,
        syn=syn,
        stem="FigS3_Gamma_Comparison"
    )

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

    metrics = {k: {'RMSE': [], 'Peak': [], 'Energy': [], 'SuccessCount': 0,
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
            # Failed trials use NaN metrics and an explicit success flag.
            if diverged:
                metrics[name]['RMSE'].append(np.nan)
                metrics[name]['Peak'].append(np.nan)
                metrics[name]['Energy'].append(np.nan)
                metrics[name].setdefault('Success', []).append(False)
                failure_info[name].append(i)
                # NaN-tagged trajectory so plot statistics exclude diverged trials.
                envelope_data[name].append((sim['t'], np.full_like(e, np.nan)))
            else:
                metrics[name]['SuccessCount'] += 1
                metrics[name]['RMSE'].append(float(np.sqrt(np.mean(e ** 2))))
                metrics[name]['Peak'].append(float(np.max(e)))
                metrics[name]['Energy'].append(float(u_energy))
                metrics[name].setdefault('Success', []).append(True)
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
        # replace `< 4.0` sentinel filter with NaN-mask filter (success-only stats).
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

    # ---- Table 2: Improvement of Proposed vs each baseline (median RMSE, success-only) ----
    if "Proposed" in metrics:
        # NaN-mask success-only median for both Proposed and each baseline.
        prop_rmse = np.array(metrics["Proposed"]["RMSE"], dtype=float)
        med_ours_rmse = np.median(prop_rmse[~np.isnan(prop_rmse)])
        print(f"\n>>> [Table 2] Proposed Improvement (Median, success-only)")
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
    ALL_LABELS = ["Nominal\nLQR", "Data-only\nLQR", "Prior-only\nRobust",
                  "Robust\n(No Relax.)", "Thresholded\nselection", "Top-budget\nselection", "Proposed"]
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
        # filter NaN sentinels before plotting; boxplot statistics on successful trials only.
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
            # success-only median for the on-figure improvement annotation.
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
                        "P0(PriorOnly)": "Prior-only",
                        "P1(NoRelax)": "No Relax.",
                        "F1a(HardStrict)": "Threshold fallback",
                        "F1b(HardBudget)": "Top budget",
                    }
                    short = short_labels.get(k, k.split("(")[0] if "(" in k else k)
                    lines.append(f"vs {short}: {imp:+.1f}%")
            if lines:
                txt = "Proposed improvement\n" + "\n".join(lines)
                ax.text(0.98, 0.97, txt, transform=ax.transAxes,
                        ha='right', va='top', fontsize=4.8, family='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

        n_out = sum(1 for d in data for v in d if v > Y_LIMITS[m_key][1])
        if n_out > 0:
            ax.text(0.02, 0.97, f"{n_out} outliers\nabove axis",
                    transform=ax.transAxes, ha='left', va='top', fontsize=5,
                    color='red', alpha=0.7)

    save_pub_figure(fig, out_dir, "Stage4_Boxplot_Stats", formats=syn.fig_formats, dpi_png=600)
    plt.close(fig)
    print(f"[Figures] Saved Monte Carlo boxplots to {out_dir}")


def plot_mc_sensitivity_scatter(
        metrics: Dict[str, Dict[str, List[float]]],
        deviations: List[float],
        out_dir: str,
        syn: SynthesisParams
):
    if "Proposed" not in metrics:
        print("[Warning] 'Proposed' not in metrics. Skipping sensitivity scatter.")
        return

    SCATTER_SERIES = [
        ("P0(PriorOnly)",   "#E69F00", "o", "Prior-only Robust"),
        ("P1(NoRelax)",     "#D55E00", "s", "Robust (No Relax.)"),
        ("F1a(HardStrict)", "#56B4E9", "D", "Thresholded hard selection with fallback"),
        ("F1b(HardBudget)", "#009E73", "v", "Top-budget hard selection"),
        ("S0(DataOnly)",    "#CC79A7", "p", "Data-only LQR"),
        ("Proposed",        "#0072B2", "^", "Proposed"),
    ]

    sizes = set_publication_style(context=syn.fig_context, column="double")
    fig, ax = plt.subplots(figsize=(sizes["double"][0], sizes["double"][1] * 0.7), constrained_layout=True)

    devs_all = np.array(deviations, dtype=float)

    for name, color, marker, label in SCATTER_SERIES:
        if name not in metrics or "RMSE" not in metrics[name]:
            continue
        rmse = np.array(metrics[name]["RMSE"], dtype=float)
        # NaN-mask successful trials only (no sentinel `< 4.9`).
        valid = (~np.isnan(rmse)) & np.isfinite(devs_all)
        d_v = devs_all[valid]
        r_v = rmse[valid]
        if len(d_v) < 2:
            continue
        ax.scatter(d_v, r_v, c=color, s=12, alpha=0.45, marker=marker, edgecolors='none')
        z = np.polyfit(d_v, r_v, 1)
        x_fit = np.linspace(d_v.min(), d_v.max(), 80)
        ax.plot(x_fit, np.poly1d(z)(x_fit), c=color, linestyle='--', linewidth=1.4, alpha=0.85, label=label)

    ax.set_xlabel(r"Normalized Parameter Deviation $\|\delta \mathbf{p}\|$")
    ax.set_ylabel("RMSE (m)")
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    ax.grid(True, linestyle=':', alpha=0.4)

    save_pub_figure(fig, out_dir, "Stage4_Sensitivity_Scatter", formats=syn.fig_formats, dpi_png=600)
    plt.close(fig)
    print(f"[Figures] Saved Sensitivity Scatter Plot to {out_dir}")


def plot_spaghetti_from_data(envelope_data, out_dir, syn):
    print(f"[Figures] Generating Spaghetti Plot from collected data...")

    if not envelope_data:
        print("[Warning] No trajectory data to plot (envelope_data is empty)")
        return

    Y_CLIP = 2.0

    sizes = set_publication_style(context=syn.fig_context, column="double")
    fig, ax = plt.subplots(figsize=(sizes["double"][0] * 1.15, sizes["double"][1] * 0.65),
                           constrained_layout=True)

    colors = {
        'Proposed': '#0072B2', 'P1(NoRelax)': '#D55E00', 'P0(PriorOnly)': '#E69F00',
        'F1a(HardStrict)': '#56B4E9', 'F1b(HardBudget)': '#009E73',
        'S0(DataOnly)': '#CC79A7', 'NominalLQR': '#999999',
    }

    PLOT_ORDER = ["NominalLQR", "S0(DataOnly)", "P0(PriorOnly)",
                  "P1(NoRelax)", "F1a(HardStrict)", "F1b(HardBudget)", "Proposed"]
    SHORT = {"NominalLQR": "Nom. LQR", "S0(DataOnly)": "Data-only", "P0(PriorOnly)": "Prior-only",
             "P1(NoRelax)": "No Relax.", "F1a(HardStrict)": "Threshold",
             "F1b(HardBudget)": "Top budget", "Proposed": "Proposed"}

    N = None
    plotted_any = False

    for name in PLOT_ORDER:
        if name not in envelope_data:
            continue
        runs = envelope_data[name]
        c = colors.get(name, 'black')

        # Success/failure status is determined uniformly by the NaN tag attached
        # in run_monte_carlo_campaign (NaN trace iff trial diverged). Do NOT introduce a
        # second filter such as `np.max(x) > Y_CLIP*3`: that would silently exclude
        # large-but-successful trials and contradict the divergence-threshold definition.
        all_traces = []
        time_grid = None
        for item in runs:
            try:
                t, x = item
                if np.any(np.isnan(x)):
                    continue  # drop NaN-tagged (=diverged) trials
                all_traces.append(x)
                if time_grid is None:
                    time_grid = t
            except (TypeError, ValueError):
                continue

        if not all_traces or time_grid is None:
            continue
        if len(time_grid) == 0:
            continue

        # mean and 95th-percentile envelopes are computed on the UNCLIPPED
        # successful traces. Visual clipping happens only when drawing the faint
        # individual traces below, and is purely a display-axis decision (controlled
        # by ax.set_ylim(0, Y_CLIP)).
        data = np.asarray(all_traces)
        mean_trace = np.nanmean(data, axis=0)
        pct95_trace = np.nanpercentile(data, 95, axis=0)

        if N is None:
            N = len(all_traces)

        step = max(1, len(time_grid) // 250)
        for trace in all_traces[:min(50, len(all_traces))]:
            tr_disp = np.clip(trace, 0, Y_CLIP)  # display-only clip
            ax.plot(time_grid[::step], tr_disp[::step], color=c, alpha=0.06, linewidth=0.3)

        sn = SHORT.get(name, name)
        ax.plot(time_grid, mean_trace, color=c, linestyle='-', linewidth=1.6,
                label=f"{sn} mean", alpha=0.9)
        ax.plot(time_grid, pct95_trace, color=c, linestyle='--', linewidth=0.9,
                label=f"{sn} 95%", alpha=0.7)

        plotted_any = True

    if not plotted_any or N is None:
        print("[Warning] No valid trajectory data to plot")
        plt.close(fig)
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Position Error $\|e_p\|$ (m)")
    ax.legend(loc='upper right', ncol=len([k for k in PLOT_ORDER if k in envelope_data]),
              fontsize=5.5, handlelength=1.5, columnspacing=0.8, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, Y_CLIP)

    save_pub_figure(fig, out_dir, "Stage4_Spaghetti_Plot", formats=syn.fig_formats, dpi_png=600)
    plt.close()
    print(f"[Figures] Saved Spaghetti Plot to {out_dir}")


def plot_mc_success_rate(
        metrics: Dict[str, Dict[str, Any]],
        N_mc: int,
        out_dir: str,
        syn: SynthesisParams,
):
    target_order = ["NominalLQR", "S0(DataOnly)", "P0(PriorOnly)",
                    "P1(NoRelax)", "F1a(HardStrict)", "F1b(HardBudget)", "Proposed"]
    NAMES = {
        "NominalLQR": "Nominal LQR", "S0(DataOnly)": "Data-only LQR",
        "P0(PriorOnly)": "Prior-only Robust", "P1(NoRelax)": "Robust (No Relax.)",
        "F1a(HardStrict)": "Thresholded hard selection with fallback",
        "F1b(HardBudget)": "Top-budget hard selection",
        "Proposed": "Proposed",
    }
    COLORS = {
        "NominalLQR": "#999999", "S0(DataOnly)": "#CC79A7",
        "P0(PriorOnly)": "#E69F00", "P1(NoRelax)": "#D55E00",
        "F1a(HardStrict)": "#56B4E9", "F1b(HardBudget)": "#009E73",
        "Proposed": "#0072B2",
    }

    keys = [k for k in target_order if k in metrics]
    if len(keys) < 2:
        print("[Warning] Not enough controllers for success rate plot.")
        return

    labels = [NAMES.get(k, k) for k in keys]
    rates = [metrics[k]['SuccessCount'] / N_mc * 100 for k in keys]
    bar_colors = [COLORS.get(k, "#AAAAAA") for k in keys]

    sizes = set_publication_style(context=syn.fig_context, column="double")
    fig, ax = plt.subplots(figsize=(sizes["double"][0], sizes["double"][1] * 0.85),
                           constrained_layout=True)

    x = np.arange(len(keys))
    bars = ax.bar(x, rates, color=bar_colors, width=0.6,
                  edgecolor="black", linewidth=0.8, zorder=3)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.8,
                f"{rate:.1f}%",
                ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=15, ha='right')
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    save_pub_figure(fig, out_dir, "Stage4_Success_Rate", formats=syn.fig_formats, dpi_png=600)
    plt.close(fig)
    print(f"[Figures] Saved success rate comparison to {out_dir}")


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

    # # ==========================================
    # # ==========================================
    # controllers = {
    #     "NominalLQR": out["K_nominal"],
    #     "P0(PriorOnly)": out["K_P0"],
    #     "P1(NoRelax)": out["K_baselineB"],
    #     "F1a(HardStrict)": out["K_F1a"],
    #     "F1b(HardBudget)": out["K_F1b"],
    #     "S0(DataOnly)": out["K_S0"],
    #     "Proposed": out["K_proposed"],
    # }
    # N_MC = 2000
    # try:
    #     mc_metrics, deviations, envelope_data, failure_info = run_monte_carlo_campaign(
    #         syn=my_syn,
    #         bounds=my_bounds,
    #         controllers=controllers,
    #         N_mc=N_MC,
    #         seconds=30.0,
    #         ref_amp=0.5,
    #         ref_period=16.0
    #     )
    #     plot_mc_boxplots(mc_metrics, my_syn.fig_out_dir, my_syn)
    #     plot_mc_sensitivity_scatter(mc_metrics, deviations, my_syn.fig_out_dir, my_syn)
    #     plot_spaghetti_from_data(envelope_data, my_syn.fig_out_dir, my_syn)
    #     plot_mc_success_rate(mc_metrics, N_MC, my_syn.fig_out_dir, my_syn)
    #     print_final_certification_report(out["cert_results"], mc_metrics, N_MC)
    # except Exception as e:
    #     print(f"[ERROR] Monte Carlo analysis failed: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)

    # ==========================================
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
