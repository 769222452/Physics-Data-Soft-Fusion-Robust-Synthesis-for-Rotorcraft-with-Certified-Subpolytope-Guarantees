"""Post-process saved Monte Carlo trials without running any simulation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from scipy.stats import binomtest


METHODS: Mapping[str, str] = {
    "NominalLQR": "Nominal LQR",
    "P0_PriorOnly": "Prior-only robust baseline",
    "P1_NoRelax": "Matched-active-set unrelaxed ablation",
    "F1a_HardStrict": "Thresholded selection",
    "F1b_HardBudget": "Top-budget selection",
    "S0_DataOnly": "Data-only LQR",
    "Proposed": "Proposed",
}


def latex_scientific(value: float, digits: int = 3) -> str:
    """Format a positive scalar in LaTeX scientific notation."""

    if value <= 0.0:
        return f"{value:.{digits}g}"
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{max(0, digits - 1)}f}\times10^{{{exponent}}}"


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def paired_bootstrap_mean_interval(
    differences: np.ndarray,
    *,
    replications: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a nonempty vector")
    rng = np.random.default_rng(seed)
    chunk_size = min(1000, replications)
    means = np.empty(replications, dtype=float)
    offset = 0
    while offset < replications:
        count = min(chunk_size, replications - offset)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[offset : offset + count] = np.mean(values[indices], axis=1)
        offset += count
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def summarize(
    archive_path: Path,
    *,
    baseline_key: str,
    bootstrap_replications: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    with np.load(archive_path, allow_pickle=False) as archive:
        trials = int(np.asarray(archive["N_mc"]).reshape(-1)[0])
        method_rows = []
        for key, label in METHODS.items():
            success = np.asarray(archive[f"{key}_Success"], dtype=bool)
            count = int(np.sum(success))
            lower, upper = wilson_interval(count, trials)
            method_rows.append(
                {
                    "method_key": key,
                    "method": label,
                    "trials": trials,
                    "successes": count,
                    "success_rate": count / trials,
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                }
            )

        proposed_success = np.asarray(
            archive["Proposed_Success"], dtype=bool
        )
        baseline_success = np.asarray(
            archive[f"{baseline_key}_Success"], dtype=bool
        )
        proposed_only = int(np.sum(proposed_success & ~baseline_success))
        baseline_only = int(np.sum(~proposed_success & baseline_success))
        discordant = proposed_only + baseline_only
        mcnemar_p = (
            float(
                binomtest(
                    min(proposed_only, baseline_only),
                    discordant,
                    0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else 1.0
        )

        proposed_rmse = np.asarray(archive["Proposed_RMSE"], dtype=float)
        baseline_rmse = np.asarray(
            archive[f"{baseline_key}_RMSE"], dtype=float
        )
        paired_mask = (
            proposed_success
            & baseline_success
            & np.isfinite(proposed_rmse)
            & np.isfinite(baseline_rmse)
        )
        paired_differences = proposed_rmse[paired_mask] - baseline_rmse[paired_mask]
        bootstrap_lower, bootstrap_upper = paired_bootstrap_mean_interval(
            paired_differences,
            replications=bootstrap_replications,
            seed=bootstrap_seed,
        )

    return {
        "source_file": archive_path.as_posix(),
        "monte_carlo_rerun": False,
        "success_intervals": method_rows,
        "paired_success_comparison": {
            "proposed_method": METHODS["Proposed"],
            "baseline_method": METHODS[baseline_key],
            "proposed_only_successes": proposed_only,
            "baseline_only_successes": baseline_only,
            "discordant_pairs": discordant,
            "exact_mcnemar_p_value": mcnemar_p,
        },
        "paired_successful_trial_rmse": {
            "conditioning": (
                "Only trials successful for both controllers are included; "
                "the result is survivor-conditioned."
            ),
            "paired_trial_count": int(np.sum(paired_mask)),
            "difference_definition": "Proposed RMSE minus baseline RMSE",
            "mean_difference": float(np.mean(paired_differences)),
            "median_difference": float(np.median(paired_differences)),
            "bootstrap_95_lower": bootstrap_lower,
            "bootstrap_95_upper": bootstrap_upper,
            "bootstrap_replications": bootstrap_replications,
            "bootstrap_seed": bootstrap_seed,
        },
    }


def write_outputs(summary: Mapping[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "monte_carlo_postprocessing.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = summary["success_intervals"]
    with (output_dir / "monte_carlo_success_intervals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    paired = summary["paired_successful_trial_rmse"]
    success = summary["paired_success_comparison"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Post-processing of the saved paired Monte Carlo trials.}",
        r"\label{tab:mc_postprocessing}",
        r"\footnotesize",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Success rate & Wilson 95\% interval \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {:.1f}\\% & [{:.1f}, {:.1f}]\\% \\\\".format(
                row["method"],
                100.0 * row["success_rate"],
                100.0 * row["wilson_95_lower"],
                100.0 * row["wilson_95_upper"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            (
                r"\begin{minipage}{0.96\linewidth}\scriptsize "
                f"Exact McNemar comparison of Proposed and "
                f"{success['baseline_method']}: "
                f"{success['proposed_only_successes']} proposed-only and "
                f"{success['baseline_only_successes']} baseline-only successes, "
                f"$p={latex_scientific(success['exact_mcnemar_p_value'])}$. "
                f"Across {paired['paired_trial_count']} trials successful for "
                f"both, the mean paired RMSE difference (Proposed minus baseline) "
                f"is {paired['mean_difference']:.4f} m with a percentile "
                f"bootstrap 95\\% interval "
                f"[{paired['bootstrap_95_lower']:.4f}, "
                f"{paired['bootstrap_95_upper']:.4f}] m. "
                r"The RMSE comparison is survivor-conditioned."
                r"\end{minipage}"
            ),
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "monte_carlo_postprocessing_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-key", choices=tuple(METHODS), default="P1_NoRelax")
    parser.add_argument("--bootstrap-replications", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    args = parser.parse_args()
    if args.bootstrap_replications <= 0:
        parser.error("--bootstrap-replications must be positive")
    summary = summarize(
        args.input,
        baseline_key=args.baseline_key,
        bootstrap_replications=args.bootstrap_replications,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_outputs(summary, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
