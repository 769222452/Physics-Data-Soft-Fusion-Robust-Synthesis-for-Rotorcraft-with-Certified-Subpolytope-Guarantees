"""Minimal post-hoc check of the score-bounded certificate.

The script reads saved controller variables and adopted vertex tuples. It
does not solve an optimization problem, run SFPS or ICE, regenerate data, or
execute Monte Carlo or time-domain simulations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.linalg as la

from posthoc_certificate_verification import (
    effective_decay_coefficients,
    find_minimum_common_coefficient,
    load_saved_solution,
    load_vertex_library,
    vertex_residuals,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "results" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "score_hull_check"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    solution_path: Path
    library_path: Path


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return 0.5 * (values + values.T)


def saved_solution_health(Q: np.ndarray, Y: np.ndarray, K: np.ndarray) -> Dict[str, float]:
    eigenvalues = la.eigvalsh(symmetrize(Q), check_finite=False)
    q_min = float(eigenvalues[0])
    if q_min <= 0.0:
        raise ValueError("saved Q is not positive definite")
    relative_y_kq = float(
        la.norm(np.asarray(Y) - np.asarray(K) @ np.asarray(Q), ord="fro")
        / max(1.0, la.norm(np.asarray(Y), ord="fro"))
    )
    identity = np.eye(Q.shape[0])
    P = symmetrize(la.solve(Q, identity, assume_a="pos"))
    return {
        "lambda_min_Q": q_min,
        "condition_number_Q": float(eigenvalues[-1] / q_min),
        "relative_Y_minus_KQ": relative_y_kq,
        "lambda_max_P": float(la.eigvalsh(P, check_finite=False)[-1]),
    }


def check_scenario(
    spec: ScenarioSpec,
    *,
    strict_tolerance: float,
    surrogate_tolerance: float,
    coefficient_tolerance: float,
    maximum_iterations: int,
) -> Dict[str, object]:
    solution, _ = load_saved_solution(spec.solution_path, "proposed")
    library, _ = load_vertex_library(spec.library_path)
    health = saved_solution_health(solution.Q, solution.Y, solution.K)

    all_indices, surrogate_residuals, surrogate_scaled = vertex_residuals(
        library,
        solution,
        decay_rate=solution.decay_rate,
        score_weighted=True,
        include_scaled=True,
    )
    assert surrogate_scaled is not None
    if all_indices.size != library.size:
        raise RuntimeError("the full-library surrogate check is incomplete")

    anchor_indices = np.flatnonzero(library.processed_scores == 0.0)
    if anchor_indices.size == 0:
        raise ValueError(f"{spec.name}: no zero-slack anchors were found")
    coefficient_result = find_minimum_common_coefficient(
        library,
        solution,
        anchor_indices,
        strict_tol=strict_tolerance,
        bar_lambda_tol=coefficient_tolerance,
        max_iterations=maximum_iterations,
    )
    bar_lambda = float(coefficient_result["bar_lambda"])

    identity = np.eye(solution.Q.shape[0])
    P = symmetrize(la.solve(solution.Q, identity, assume_a="pos"))
    effective_coefficients, _ = effective_decay_coefficients(
        solution.decay_rate,
        solution.beta,
        library.processed_scores,
        P,
    )
    score_indices = np.flatnonzero(effective_coefficients <= bar_lambda)
    if score_indices.size == 0:
        raise RuntimeError(f"{spec.name}: the score-bounded set is empty")

    _, selected_residuals, selected_scaled = vertex_residuals(
        library,
        solution,
        decay_rate=bar_lambda,
        score_weighted=False,
        indices=score_indices,
        include_scaled=True,
    )
    assert selected_scaled is not None

    worst_selected = float(np.max(selected_residuals))
    if worst_selected > -strict_tolerance:
        raise RuntimeError(
            f"{spec.name}: selected score set fails the requested "
            f"negative margin ({worst_selected:.6e})"
        )

    return {
        "scenario": spec.name,
        "N": library.size,
        "N_anchor": int(anchor_indices.size),
        "bar_lambda_anchor": bar_lambda,
        "N_score": int(score_indices.size),
        "N_positive_score_selected": int(
            np.count_nonzero(library.processed_scores[score_indices] > 0.0)
        ),
        "worst_selected_set_BRL_residual": worst_selected,
        "worst_selected_set_BRL_scaled_residual": float(
            np.max(selected_scaled)
        ),
        "strict_tolerance": strict_tolerance,
        "worst_full_library_surrogate_residual": float(
            np.max(surrogate_residuals)
        ),
        "worst_full_library_surrogate_scaled_residual": float(
            np.max(surrogate_scaled)
        ),
        "surrogate_reporting_tolerance": surrogate_tolerance,
        "full_library_surrogate_within_reporting_tolerance": bool(
            np.max(surrogate_residuals) <= surrogate_tolerance
        ),
        **health,
    }


def write_outputs(rows: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "score_hull_check.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump({"scenarios": rows}, stream, indent=2)

    fieldnames = list(rows[0].keys())
    with (output_dir / "score_hull_check.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Minimal post-hoc check of the score-bounded "
            r"certificate.}"
        ),
        r"\label{tab:minimal_score_hull_check}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Scenario & $N$ & $N_{\rm anc}$ & "
            r"$\bar\lambda_{\rm anc}$ & $N_{\rm score}$ & "
            r"$N_{\rm score}^{+}$ & $r_{\rm score}^{\max}$ \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['scenario']} & {row['N']} & {row['N_anchor']} & "
            f"{row['bar_lambda_anchor']:.6f} & {row['N_score']} & "
            f"{row['N_positive_score_selected']} & "
            f"${row['worst_selected_set_BRL_residual']:.2e}$ \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "score_hull_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    metadata = {
        "script": "src/posthoc_score_hull_check.py",
        "synthesis_rerun": False,
        "sfps_ice_rerun": False,
        "time_domain_rerun": False,
        "monte_carlo_rerun": False,
        "offline_batch_regenerated": False,
        "controller_modified": False,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the saved score-bounded certificate without solving or "
            "simulating."
        )
    )
    parser.add_argument(
        "--standard-solution",
        type=Path,
        default=DEFAULT_RAW / "stage3_time_domain_results.npz",
    )
    parser.add_argument(
        "--expanded-solution",
        type=Path,
        default=DEFAULT_RAW / "stage3_time_domain_results-ex.npz",
    )
    parser.add_argument(
        "--standard-library",
        type=Path,
        default=DEFAULT_RAW / "standard_data_fusion_diagnostics.npz",
    )
    parser.add_argument(
        "--expanded-library",
        type=Path,
        default=DEFAULT_RAW / "expanded_data_fusion_diagnostics.npz",
    )
    parser.add_argument("--strict-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--surrogate-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--coefficient-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--maximum-iterations", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    specs = [
        ScenarioSpec(
            "Standard", args.standard_solution, args.standard_library
        ),
        ScenarioSpec(
            "Expanded", args.expanded_solution, args.expanded_library
        ),
    ]
    rows = [
        check_scenario(
            spec,
            strict_tolerance=args.strict_tolerance,
            surrogate_tolerance=args.surrogate_tolerance,
            coefficient_tolerance=args.coefficient_tolerance,
            maximum_iterations=args.maximum_iterations,
        )
        for spec in specs
    ]
    write_outputs(rows, args.output_dir)
    for row in rows:
        print(
            f"{row['scenario']}: N_score={row['N_score']}, "
            f"bar_lambda={row['bar_lambda_anchor']:.8f}, "
            f"worst_selected_BRL="
            f"{row['worst_selected_set_BRL_residual']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
