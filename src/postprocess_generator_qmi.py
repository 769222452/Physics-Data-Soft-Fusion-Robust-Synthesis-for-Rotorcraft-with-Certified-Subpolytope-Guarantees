"""Evaluate saved batch-generator raw-QMI scores without regenerating data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


REPORTING_TOLERANCE = 1.0e-8


def sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def score_matrix(delta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    selector = np.hstack((delta, np.eye(delta.shape[0])))
    return sym(selector @ psi @ selector.T)


def evaluate(name: str, source: Path) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as archive:
        psi = np.asarray(archive["Psi"], dtype=float)
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
        endpoint_delta = np.asarray(archive["vertex_delta"], dtype=float)
        archived_scores = np.asarray(archive["raw_scores"], dtype=float)
        zero_score_mask = np.asarray(archive["zero_score_mask"], dtype=bool)

    matrix = score_matrix(generator_delta, psi)
    # Match the eigensolver used by the released score-construction code.
    eigenvalues = np.linalg.eigvalsh(matrix)
    score = float(eigenvalues[-1])
    norm_2 = float(np.max(np.abs(eigenvalues)))
    reproduced = np.asarray(
        [
            np.linalg.eigvalsh(score_matrix(delta, psi))[-1]
            for delta in endpoint_delta
        ],
        dtype=float,
    )

    if score <= 0.0:
        status = "Raw-QMI consistent"
    elif score <= REPORTING_TOLERANCE:
        status = "Numerically near boundary (positive)"
    else:
        status = "Not raw-QMI consistent"

    return {
        "scenario": name,
        "source_file": source.as_posix(),
        "generator_raw_qmi_score": score,
        "status": status,
        "passes_exact_nonpositive_test": bool(score <= 0.0),
        "n_raw": int(np.sum(archived_scores <= 0.0)),
        "n_anchor": int(np.sum(zero_score_mask)),
        "qmi_matrix_norm_2": norm_2,
        "relative_positive_residual": float(max(score, 0.0) / max(norm_2, 1.0)),
        "positive_eigenvalue_count": int(np.sum(eigenvalues > 0.0)),
        "positive_eigenvalue_count_above_reporting_tolerance": int(
            np.sum(eigenvalues > REPORTING_TOLERANCE)
        ),
        "endpoint_score_max_abs_reproduction_error": float(
            np.max(np.abs(reproduced - archived_scores))
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "scenario",
        "source_file",
        "generator_raw_qmi_score",
        "status",
        "passes_exact_nonpositive_test",
        "n_raw",
        "n_anchor",
        "qmi_matrix_norm_2",
        "relative_positive_residual",
        "positive_eigenvalue_count",
        "positive_eigenvalue_count_above_reporting_tolerance",
        "endpoint_score_max_abs_reproduction_error",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        evaluate("Standard", args.standard),
        evaluate("Expanded", args.expanded),
    ]
    report = {
        "analysis": "generator raw-QMI score from saved artifacts",
        "formula": (
            "lambda_max(sym([Theta_gen I_32] Psi [Theta_gen^T; I_32]))"
        ),
        "exact_raw_qmi_test": "score <= 0",
        "near_boundary_reporting_tolerance": REPORTING_TOLERANCE,
        "batch_regenerated": False,
        "solver_called": False,
        "simulation_rerun": False,
        "scenarios": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.csv_output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
