"""Regenerate manuscript time-domain figures from saved trajectories only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


MM_TO_IN = 1.0 / 25.4
FIGURE_SIZE = (183.0 * MM_TO_IN, 70.0 * MM_TO_IN)
CONTROLLERS = (
    ("Proposed", "Proposed", "-"),
    ("BaselineB_NoRelax", "Matched-active ablation", "--"),
    ("NominalLQR", "Nominal LQR", ":"),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
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
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "text.usetex": False,
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
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
    )


def setup_axis(axis: plt.Axes) -> None:
    axis.minorticks_on()
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(top=False, right=False, which="both")
    axis.grid(True, which="major", axis="y", linewidth=0.3, alpha=0.25)


def gust_intervals(archive: Mapping[str, np.ndarray]) -> Iterable[tuple[float, float]]:
    count = int(np.asarray(archive["num_gusts"]).reshape(-1)[0])
    for index in range(count):
        yield (
            float(np.asarray(archive[f"gust_{index}_t0"]).reshape(-1)[0]),
            float(np.asarray(archive[f"gust_{index}_t1"]).reshape(-1)[0]),
        )


def shade_gusts(axis: plt.Axes, intervals: Iterable[tuple[float, float]]) -> None:
    for start, end in intervals:
        axis.axvspan(start, end, alpha=0.10, linewidth=0)


def array(archive: Mapping[str, np.ndarray], state: str, key: str, field: str) -> np.ndarray:
    return np.asarray(archive[f"{state}_{key}_{field}"], dtype=float)


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=False)
    plt.close(fig)


def plot_archive(input_path: Path, output_dir: Path, suffix: str) -> None:
    with np.load(input_path, allow_pickle=False) as archive:
        intervals = tuple(gust_intervals(archive))
        t = array(archive, "disturbed", "Proposed", "t")
        reference = array(archive, "disturbed", "Proposed", "Pref")
        increment_scales = np.asarray(archive["increment_scales"], dtype=float)

        fig, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
        setup_axis(axis)
        shade_gusts(axis, intervals)
        axis.plot(t, reference[0], linestyle=":", linewidth=1.2, color="black", label="Reference")
        for key, label, linestyle in CONTROLLERS:
            state = array(archive, "disturbed", key, "x")
            axis.plot(t, state[0], linestyle=linestyle, label=label)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(r"$x$ position (m)")
        axis.set_title(r"Position tracking along $x$ under multi-gust disturbance")
        axis.legend(frameon=False, loc="best", ncols=1)
        save(fig, output_dir / f"figure3_{suffix}.pdf")

        fig, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
        setup_axis(axis)
        shade_gusts(axis, intervals)
        for key, label, linestyle in CONTROLLERS:
            state = array(archive, "disturbed", key, "x")
            pref = array(archive, "disturbed", key, "Pref")
            error = np.linalg.norm(state[:3] - pref, axis=0)
            axis.plot(t, error, linestyle=linestyle, label=label)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(r"$\|p-p_{\mathrm{ref}}\|_2$ (m)")
        axis.set_title("Position tracking error norm")
        axis.legend(frameon=False, loc="best", ncols=1)
        save(fig, output_dir / f"figure4_{suffix}.pdf")

        fig, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
        setup_axis(axis)
        shade_gusts(axis, intervals)
        for key, label, linestyle in CONTROLLERS:
            applied = array(archive, "disturbed", key, "u_c")
            normalized = applied / increment_scales[:, None]
            axis.plot(t[:-1], np.linalg.norm(normalized, axis=0), linestyle=linestyle, label=label)
        axis.axhline(
            1.0,
            linestyle=":",
            linewidth=1.0,
            color="black",
            label="implemented unit increment limit",
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(r"$\|\bar u_c^{\mathrm{act}}\|_2$")
        axis.set_title("Normalized applied control increment")
        axis.legend(frameon=False, loc="best", ncols=1)
        save(fig, output_dir / f"figure5_{suffix}.pdf")

        fig, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
        setup_axis(axis)
        shade_gusts(axis, intervals)
        for key, label, linestyle in CONTROLLERS:
            disturbed_state = array(archive, "disturbed", key, "x")
            disturbed_ref = array(archive, "disturbed", key, "Pref")
            undisturbed_state = array(archive, "undisturbed", key, "x")
            undisturbed_ref = array(archive, "undisturbed", key, "Pref")
            disturbed_error = np.linalg.norm(disturbed_state[:3] - disturbed_ref, axis=0)
            undisturbed_error = np.linalg.norm(undisturbed_state[:3] - undisturbed_ref, axis=0)
            excess = np.maximum(0.0, disturbed_error - undisturbed_error)
            axis.plot(t, excess, linestyle=linestyle, label=label)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Excess error (m)")
        axis.set_title("Disturbance-induced excess tracking error")
        axis.legend(frameon=False, loc="best", ncols=1)
        save(fig, output_dir / f"figure6_{suffix}.pdf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    plot_archive(args.standard, args.output_dir, "standard")
    plot_archive(args.expanded, args.output_dir, "expanded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
