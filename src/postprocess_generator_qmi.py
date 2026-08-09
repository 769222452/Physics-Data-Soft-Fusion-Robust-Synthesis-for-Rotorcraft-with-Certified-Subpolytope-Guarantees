"""Evaluate saved batch-generator raw-QMI scores without regenerating data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from raw_qmi_scores import (
    aggregate_residual_bound,
    dynamics_qmi_residual,
    dynamics_residual_matrix,
    full_qmi_residual,
    largest_eigenvalue,
)


REPORTING_TOLERANCE = 1.0e-8


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
        X_t = np.asarray(archive["batch_X_t"], dtype=float)
        U_t = np.asarray(archive["batch_U_t"], dtype=float)
        X_tp1 = np.asarray(archive["batch_X_tp1"], dtype=float)
        W_c = np.asarray(archive["W_c"], dtype=float)
        W_E = np.asarray(archive["W_E"], dtype=float)

    regressor = np.vstack((X_t, U_t))
    residual_bound = aggregate_residual_bound(W_c, W_E, X_t.shape[1])
    successor_bound = residual_bound[:16, :16]
    full_matrix = full_qmi_residual(generator_delta, psi)
    full_eigenvalues = np.linalg.eigvalsh(full_matrix)
    _, dynamics_matrix = dynamics_residual_matrix(
        generator_delta[:16, :], regressor, X_tp1, successor_bound
    )
    dynamics_eigenvalues = np.linalg.eigvalsh(dynamics_matrix)
    dynamics_qmi_matrix = dynamics_qmi_residual(generator_delta, psi)
    full_score = float(full_eigenvalues[-1])
    score = float(dynamics_eigenvalues[-1])
    norm_2 = float(np.max(np.abs(dynamics_eigenvalues)))
    reproduced_full = np.asarray(
        [
            largest_eigenvalue(full_qmi_residual(delta, psi))
            for delta in endpoint_delta
        ],
        dtype=float,
    )
    reproduced_dynamics = np.asarray(
        [
            largest_eigenvalue(dynamics_qmi_residual(delta, psi))
            for delta in endpoint_delta
        ],
        dtype=float,
    )

    if score <= 0.0:
        status = "Negative dynamics-block margin"
    elif score <= REPORTING_TOLERANCE:
        status = "Numerically near boundary (positive)"
    else:
        status = "Not raw-QMI consistent"

    return {
        "scenario": name,
        "source_file": source.as_posix(),
        "generator_raw_qmi_score": score,
        "generator_full_qmi_largest_eigenvalue": full_score,
        "generator_dynamics_qmi_block_largest_eigenvalue": largest_eigenvalue(
            dynamics_qmi_matrix
        ),
        "generator_dynamics_residual_minimum_eigenvalue": float(
            dynamics_eigenvalues[0]
        ),
        "generator_dynamics_residual_maximum_eigenvalue": float(
            dynamics_eigenvalues[-1]
        ),
        "status": status,
        "dynamics_margin_nonpositive": bool(score <= 0.0),
        "n_raw": int(np.sum(reproduced_dynamics <= 0.0)),
        "n_anchor": int(np.sum(zero_score_mask)),
        "qmi_matrix_norm_2": norm_2,
        "relative_positive_residual": float(max(score, 0.0) / max(norm_2, 1.0)),
        "positive_eigenvalue_count": int(np.sum(dynamics_eigenvalues > 0.0)),
        "positive_eigenvalue_count_above_reporting_tolerance": int(
            np.sum(dynamics_eigenvalues > REPORTING_TOLERANCE)
        ),
        "endpoint_score_max_abs_reproduction_error": float(
            np.max(np.abs(reproduced_dynamics - archived_scores))
        ),
        "endpoint_full_qmi_max_abs_reproduction_error": float(
            np.max(np.abs(reproduced_full - archived_scores))
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "scenario",
        "source_file",
        "generator_raw_qmi_score",
        "generator_full_qmi_largest_eigenvalue",
        "generator_dynamics_qmi_block_largest_eigenvalue",
        "generator_dynamics_residual_minimum_eigenvalue",
        "generator_dynamics_residual_maximum_eigenvalue",
        "status",
        "dynamics_margin_nonpositive",
        "n_raw",
        "n_anchor",
        "qmi_matrix_norm_2",
        "relative_positive_residual",
        "positive_eigenvalue_count",
        "positive_eigenvalue_count_above_reporting_tolerance",
        "endpoint_score_max_abs_reproduction_error",
        "endpoint_full_qmi_max_abs_reproduction_error",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
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
        "analysis": "generator dynamics-block raw-QMI score from saved artifacts",
        "formula": (
            "lambda_max(sym(R_x,gen R_x,gen^T - R_tilde_x,t))"
        ),
        "full_qmi_retained_as_diagnostic": True,
        "numerical_raw_margin_test": "dynamics-block score <= 0",
        "near_boundary_reporting_tolerance": REPORTING_TOLERANCE,
        "batch_regenerated": False,
        "solver_called": False,
        "simulation_rerun": False,
        "scenarios": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    write_csv(args.csv_output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
