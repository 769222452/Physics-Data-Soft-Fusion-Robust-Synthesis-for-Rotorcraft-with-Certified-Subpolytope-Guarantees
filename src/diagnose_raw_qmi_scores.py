"""Compare legacy full-QMI scores with signed successor-state margins."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import time_domain_standard as model
from raw_qmi_scores import (
    aggregate_residual_bound,
    dynamics_qmi_residual,
    dynamics_residual_matrix,
    full_qmi_residual,
    largest_eigenvalue,
)


SUCCESSOR_DIM = 16
RAW_ABS_TOL = 1.0e-8
RAW_REL_TOL = 1.0e-12
PROCESSED_ABS_TOL = 1.0e-12
PROCESSED_REL_TOL = 1.0e-12
STRUCTURE_ABS_TOL = 1.0e-8
RELATIVE_FLOOR = 1.0e-12

TAU = 0.0
Q_SCALE = 0.9
Q_TAU = 0.1
R_TRIGGER = 0.01
SCORE_DENOMINATOR_EPS = 1.0e-12
SIGMA_DEGENERACY_TOL = 1.0e-10

SFPS_BUDGET = 20
SFPS_RATIOS = (0.4, 0.2, 0.4)
SFPS_SEED = 26 + 7777


def process_scores(raw_scores: np.ndarray, batch_length: int) -> dict[str, Any]:
    """Reproduce the score mapping used by the synthesis scripts."""

    raw = np.asarray(raw_scores, dtype=float)
    raw_consistent = raw <= TAU
    raw_ratio = float(np.mean(raw_consistent))
    tau_effective = TAU
    fallback = raw_ratio < R_TRIGGER
    if fallback:
        tau_effective = max(0.0, float(np.quantile(raw, Q_TAU)))

    excess = np.maximum(0.0, raw - tau_effective)
    excess /= float(batch_length) + SCORE_DENOMINATOR_EPS
    positive = excess[excess > 0.0]
    if positive.size == 0:
        sigma = 0.0
        processed = np.zeros_like(raw)
        uninformative = True
    else:
        sigma = float(np.quantile(positive, Q_SCALE))
        uninformative = sigma < SIGMA_DEGENERACY_TOL
        if uninformative:
            processed = np.zeros_like(raw)
        else:
            processed = 1.0 - np.exp(
                -excess / (sigma + SCORE_DENOMINATOR_EPS)
            )
            processed = np.clip(processed, 0.0, 1.0)

    return {
        "processed": processed,
        "tau_effective": tau_effective,
        "sigma": sigma,
        "fallback": fallback,
        "uninformative": uninformative,
        "raw_consistent_mask": raw_consistent,
        "anchor_mask": processed == 0.0,
    }


def tier_membership(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stable score order and the low/middle/high SFPS tier labels."""

    values = np.asarray(scores, dtype=float)
    order = np.argsort(values, kind="stable")
    tiers = np.zeros(values.size, dtype=int)
    one_third = max(1, values.size // 3)
    for rank, index in enumerate(order):
        if rank < one_third:
            tiers[index] = 0
        elif rank < 2 * one_third:
            tiers[index] = 1
        else:
            tiers[index] = 2
    return order, tiers


def sfps_initial_indices(
    parameter_keys: np.ndarray,
    parameters: np.ndarray,
    processed_scores: np.ndarray,
) -> list[int]:
    """Run only the deterministic SFPS initializer used before any SDP solve."""

    keys = [str(value) for value in parameter_keys.tolist()]
    bounds = model.ParamBounds(
        **{
            key: (
                float(np.min(parameters[:, column])),
                float(np.max(parameters[:, column])),
            )
            for column, key in enumerate(keys)
        }
    )
    vertices = []
    for index, row in enumerate(parameters):
        vertices.append(
            {
                "index": index,
                "s": float(processed_scores[index]),
                "p": {key: float(row[column]) for column, key in enumerate(keys)},
            }
        )
    selected = model.select_vertices_stratified_farthest(
        vertices,
        bounds,
        K=SFPS_BUDGET,
        ratios=SFPS_RATIOS,
        seed=SFPS_SEED,
    )
    return [int(vertex["index"]) for vertex in selected]


def relative_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    denominator = np.maximum.reduce(
        (
            np.abs(first),
            np.abs(second),
            np.full_like(first, RELATIVE_FLOOR),
        )
    )
    return np.abs(first - second) / denominator


def evaluate_scenario(name: str, source: Path, output_dir: Path) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as archive:
        psi = np.asarray(archive["Psi"], dtype=float)
        deltas = np.asarray(archive["vertex_delta"], dtype=float)
        archived_raw = np.asarray(archive["raw_scores"], dtype=float)
        archived_processed = np.asarray(archive["processed_scores"], dtype=float)
        parameter_keys = np.asarray(archive["parameter_keys"])
        parameters = np.asarray(archive["vertex_parameters"], dtype=float)
        state = np.asarray(archive["batch_X_t"], dtype=float)
        input_data = np.asarray(archive["batch_U_t"], dtype=float)
        successor = np.asarray(archive["batch_X_tp1"], dtype=float)
        output = np.asarray(archive["batch_Z_t"], dtype=float)
        generator_delta = np.block(
            [
                [archive["batch_A"], archive["batch_B"]],
                [archive["batch_Cc"], archive["batch_Dc"]],
            ]
        ).astype(float)
        residual_bound = aggregate_residual_bound(
            np.asarray(archive["W_c"], dtype=float),
            np.asarray(archive["W_E"], dtype=float),
            int(state.shape[1]),
        )

    regressor = np.vstack((state, input_data))
    successor_bound = residual_bound[:SUCCESSOR_DIM, :SUCCESSOR_DIM]
    full_scores = np.empty(deltas.shape[0], dtype=float)
    dynamics_scores = np.empty_like(full_scores)
    dynamics_from_qmi = np.empty_like(full_scores)
    max_structure_entry = 0.0
    max_dynamics_block_entry = 0.0
    max_output_block_norm = 0.0
    max_cross_block_norm = 0.0

    zero_output = np.zeros((16, 16), dtype=float)
    for index, delta in enumerate(deltas):
        full = full_qmi_residual(delta, psi)
        _, dynamics = dynamics_residual_matrix(
            delta[:SUCCESSOR_DIM, :],
            regressor,
            successor,
            successor_bound,
        )
        qmi_dynamics = dynamics_qmi_residual(
            delta, psi, successor_dim=SUCCESSOR_DIM
        )
        block_form = np.block(
            [[dynamics, zero_output], [zero_output, zero_output]]
        )
        full_scores[index] = largest_eigenvalue(full)
        dynamics_scores[index] = largest_eigenvalue(dynamics)
        dynamics_from_qmi[index] = largest_eigenvalue(qmi_dynamics)
        max_structure_entry = max(
            max_structure_entry, float(np.max(np.abs(full - block_form)))
        )
        max_dynamics_block_entry = max(
            max_dynamics_block_entry,
            float(np.max(np.abs(qmi_dynamics - dynamics))),
        )
        max_output_block_norm = max(
            max_output_block_norm,
            float(np.linalg.norm(full[SUCCESSOR_DIM:, SUCCESSOR_DIM:], ord=2)),
        )
        max_cross_block_norm = max(
            max_cross_block_norm,
            float(np.linalg.norm(full[:SUCCESSOR_DIM, SUCCESSOR_DIM:], ord=2)),
        )

    old_mapping = process_scores(full_scores, state.shape[1])
    new_mapping = process_scores(dynamics_scores, state.shape[1])
    old_order = np.argsort(full_scores, kind="stable")
    new_order = np.argsort(dynamics_scores, kind="stable")
    old_processed_order, old_tiers = tier_membership(old_mapping["processed"])
    new_processed_order, new_tiers = tier_membership(new_mapping["processed"])
    old_initial = sfps_initial_indices(
        parameter_keys, parameters, old_mapping["processed"]
    )
    new_initial = sfps_initial_indices(
        parameter_keys, parameters, new_mapping["processed"]
    )

    raw_abs_difference = np.abs(full_scores - dynamics_scores)
    raw_relative_difference = relative_difference(full_scores, dynamics_scores)
    processed_difference = np.abs(
        old_mapping["processed"] - new_mapping["processed"]
    )

    generator_full = full_qmi_residual(generator_delta, psi)
    generator_residual, generator_dynamics = dynamics_residual_matrix(
        generator_delta[:SUCCESSOR_DIM, :],
        regressor,
        successor,
        successor_bound,
    )
    generator_dynamics_qmi = dynamics_qmi_residual(
        generator_delta, psi, successor_dim=SUCCESSOR_DIM
    )
    generator_dynamics_eigenvalues = np.linalg.eigvalsh(generator_dynamics)
    generator_full_eigenvalues = np.linalg.eigvalsh(generator_full)
    generator_output_residual = output - generator_delta[SUCCESSOR_DIM:, :] @ regressor

    endpoint_equivalent = bool(
        np.array_equal(full_scores <= 0.0, dynamics_scores <= 0.0)
    )
    raw_close = bool(
        np.allclose(
            full_scores,
            dynamics_scores,
            atol=RAW_ABS_TOL,
            rtol=RAW_REL_TOL,
        )
    )
    processed_close = bool(
        np.allclose(
            old_mapping["processed"],
            new_mapping["processed"],
            atol=PROCESSED_ABS_TOL,
            rtol=PROCESSED_REL_TOL,
        )
    )
    anchor_equal = bool(
        np.array_equal(old_mapping["anchor_mask"], new_mapping["anchor_mask"])
    )
    raw_order_equal = bool(np.array_equal(old_order, new_order))
    processed_order_equal = bool(
        np.array_equal(old_processed_order, new_processed_order)
    )
    tiers_equal = bool(np.array_equal(old_tiers, new_tiers))
    sfps_initial_equal = old_initial == new_initial

    downstream_unchanged = bool(
        processed_close
        and anchor_equal
        and raw_order_equal
        and processed_order_equal
        and tiers_equal
        and sfps_initial_equal
    )

    csv_path = output_dir / f"{name.lower()}_endpoint_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "vertex_index",
            "old_full_qmi_raw_score",
            "archived_raw_score",
            "new_dynamics_raw_score",
            "dynamics_score_from_qmi_block",
            "absolute_difference",
            "relative_difference",
            "old_processed_score",
            "new_processed_score",
            "old_anchor",
            "new_anchor",
            "old_tier",
            "new_tier",
        ]
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for index in range(deltas.shape[0]):
            writer.writerow(
                {
                    "vertex_index": index,
                    "old_full_qmi_raw_score": full_scores[index],
                    "archived_raw_score": archived_raw[index],
                    "new_dynamics_raw_score": dynamics_scores[index],
                    "dynamics_score_from_qmi_block": dynamics_from_qmi[index],
                    "absolute_difference": raw_abs_difference[index],
                    "relative_difference": raw_relative_difference[index],
                    "old_processed_score": old_mapping["processed"][index],
                    "new_processed_score": new_mapping["processed"][index],
                    "old_anchor": bool(old_mapping["anchor_mask"][index]),
                    "new_anchor": bool(new_mapping["anchor_mask"][index]),
                    "old_tier": int(old_tiers[index]),
                    "new_tier": int(new_tiers[index]),
                }
            )

    return {
        "scenario": name,
        "source_file": source.as_posix(),
        "endpoint_csv": csv_path.as_posix(),
        "endpoint_count": int(deltas.shape[0]),
        "dimensions": {
            "delta": list(deltas.shape[1:]),
            "psi": list(psi.shape),
            "full_residual": [32, 32],
            "dynamics_residual": [16, 16],
            "regressor": list(regressor.shape),
        },
        "old_full_score_reproduction_max_abs_error": float(
            np.max(np.abs(full_scores - archived_raw))
        ),
        "new_qmi_block_vs_direct_max_abs_score_error": float(
            np.max(np.abs(dynamics_from_qmi - dynamics_scores))
        ),
        "old_vs_new_max_absolute_difference": float(np.max(raw_abs_difference)),
        "old_vs_new_max_relative_difference": float(
            np.max(raw_relative_difference)
        ),
        "raw_scores_close_under_tolerance": raw_close,
        "raw_score_order_equal": raw_order_equal,
        "old_raw_consistent_count": int(
            np.sum(old_mapping["raw_consistent_mask"])
        ),
        "new_raw_consistent_count": int(
            np.sum(new_mapping["raw_consistent_mask"])
        ),
        "endpoint_feasibility_classification_equal": endpoint_equivalent,
        "old_tau_effective": float(old_mapping["tau_effective"]),
        "new_tau_effective": float(new_mapping["tau_effective"]),
        "old_anchor_count": int(np.sum(old_mapping["anchor_mask"])),
        "new_anchor_count": int(np.sum(new_mapping["anchor_mask"])),
        "anchor_index_set_equal": anchor_equal,
        "old_sigma_s": float(old_mapping["sigma"]),
        "new_sigma_s": float(new_mapping["sigma"]),
        "processed_score_max_absolute_difference": float(
            np.max(processed_difference)
        ),
        "archived_processed_reproduction_max_abs_error": float(
            np.max(np.abs(old_mapping["processed"] - archived_processed))
        ),
        "processed_scores_close_under_tolerance": processed_close,
        "processed_score_order_equal": processed_order_equal,
        "sfps_tier_membership_equal": tiers_equal,
        "sfps_tier_difference_count": int(np.sum(old_tiers != new_tiers)),
        "sfps_initial_indices_old": old_initial,
        "sfps_initial_indices_new": new_initial,
        "sfps_initialization_equal": sfps_initial_equal,
        "structural_diagnostics": {
            "max_entry_full_minus_exact_block_form": max_structure_entry,
            "max_entry_qmi_dynamics_minus_direct_dynamics": max_dynamics_block_entry,
            "max_output_block_spectral_norm": max_output_block_norm,
            "max_cross_block_spectral_norm": max_cross_block_norm,
            "within_structure_tolerance": bool(
                max_structure_entry <= STRUCTURE_ABS_TOL
            ),
        },
        "generator": {
            "full_qmi_largest_eigenvalue": float(generator_full_eigenvalues[-1]),
            "dynamics_qmi_block_largest_eigenvalue": largest_eigenvalue(
                generator_dynamics_qmi
            ),
            "dynamics_signed_raw_score": float(generator_dynamics_eigenvalues[-1]),
            "dynamics_residual_minimum_eigenvalue": float(
                generator_dynamics_eigenvalues[0]
            ),
            "dynamics_residual_maximum_eigenvalue": float(
                generator_dynamics_eigenvalues[-1]
            ),
            "dynamics_qmi_vs_direct_max_abs_entry": float(
                np.max(np.abs(generator_dynamics_qmi - generator_dynamics))
            ),
            "state_residual_frobenius_norm": float(
                np.linalg.norm(generator_residual, ord="fro")
            ),
            "performance_output_residual_max_abs": float(
                np.max(np.abs(generator_output_residual))
            ),
            "full_qmi_cross_block_spectral_norm": float(
                np.linalg.norm(
                    generator_full[:SUCCESSOR_DIM, SUCCESSOR_DIM:], ord=2
                )
            ),
            "full_qmi_output_block_spectral_norm": float(
                np.linalg.norm(
                    generator_full[SUCCESSOR_DIM:, SUCCESSOR_DIM:], ord=2
                )
            ),
            "full_positive_due_to_zero_block_roundoff": bool(
                generator_full_eigenvalues[-1] > 0.0
                and generator_dynamics_eigenvalues[-1] < 0.0
                and np.linalg.norm(
                    generator_full[SUCCESSOR_DIM:, SUCCESSOR_DIM:], ord=2
                )
                <= STRUCTURE_ABS_TOL
            ),
        },
        "downstream_quantities_unchanged": downstream_unchanged,
    }


def format_number(value: float) -> str:
    return f"{value:.12e}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    scenarios = report["scenarios"]
    lines = [
        "# Raw-QMI score structural diagnostic",
        "",
        "## Mathematical structure",
        "",
        "For every adopted candidate, the performance matrices are fixed and the saved batch satisfies "
        "`Z_t = [C_d^c D_d^c] [X_t; U_t]`. The performance-output rows of the residual and of the aggregate bound are therefore zero. In exact arithmetic, the complete QMI residual is `blockdiag(H_x,i, 0_16)`, where `H_x,i = R_x,i R_x,i^T - R_tilde_x,t`. Hence the complete QMI is negative semidefinite if and only if `H_x,i` is negative semidefinite, but its largest eigenvalue is `max(lambda_max(H_x,i), 0)`. The dynamics-block eigenvalue is used as the signed raw margin; the complete residual is retained for equivalence diagnostics.",
        "",
        "## Endpoint comparison",
        "",
        "| Scenario | Max abs. difference | Max rel. difference | Raw counts old/new | Anchors old/new | Scores ordered identically | Tiers identical | SFPS initialization identical |",
        "|---|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for item in scenarios:
        lines.append(
            "| {scenario} | {abs_diff} | {rel_diff} | {old_raw}/{new_raw} | "
            "{old_anchor}/{new_anchor} | {order} | {tiers} | {sfps} |".format(
                scenario=item["scenario"],
                abs_diff=format_number(item["old_vs_new_max_absolute_difference"]),
                rel_diff=format_number(item["old_vs_new_max_relative_difference"]),
                old_raw=item["old_raw_consistent_count"],
                new_raw=item["new_raw_consistent_count"],
                old_anchor=item["old_anchor_count"],
                new_anchor=item["new_anchor_count"],
                order="yes" if item["raw_score_order_equal"] else "no",
                tiers="yes" if item["sfps_tier_membership_equal"] else "no",
                sfps="yes" if item["sfps_initialization_equal"] else "no",
            )
        )

    lines.extend(["", "### Score-processing details", ""])
    for item in scenarios:
        lines.extend(
            [
                f"**{item['scenario']}**",
                "",
                f"- Old/new `tau_eff`: {format_number(item['old_tau_effective'])} / {format_number(item['new_tau_effective'])}",
                f"- Old/new `sigma_s`: {format_number(item['old_sigma_s'])} / {format_number(item['new_sigma_s'])}",
                f"- Maximum processed-score difference: {format_number(item['processed_score_max_absolute_difference'])}",
                f"- Anchor index sets identical: {item['anchor_index_set_equal']}",
                f"- Processed-score ordering identical: {item['processed_score_order_equal']}",
                f"- Per-vertex data: `{item['endpoint_csv']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Saved generator diagnostic",
            "",
            "| Scenario | Full-QMI largest eigenvalue | Dynamics signed margin | Dynamics eig. min | Dynamics eig. max | Zero-block roundoff diagnosis |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for item in scenarios:
        generator = item["generator"]
        lines.append(
            "| {scenario} | {full} | {dynamic} | {minimum} | {maximum} | {diagnosis} |".format(
                scenario=item["scenario"],
                full=format_number(generator["full_qmi_largest_eigenvalue"]),
                dynamic=format_number(generator["dynamics_signed_raw_score"]),
                minimum=format_number(
                    generator["dynamics_residual_minimum_eigenvalue"]
                ),
                maximum=format_number(
                    generator["dynamics_residual_maximum_eigenvalue"]
                ),
                diagnosis=(
                    "yes"
                    if generator["full_positive_due_to_zero_block_roundoff"]
                    else "no"
                ),
            )
        )

    lines.extend(
        [
            "",
            "The positive complete-matrix values near `1e-10` are dominated by numerical perturbations in the structurally zero performance-output block. They do not describe the signed dynamics margin.",
            "",
            "## Rerun decision",
            "",
            report["rerun_decision"],
            "",
            "No SDP, controller synthesis, Monte Carlo campaign, batch generation, or time-domain simulation was run for this diagnostic.",
            "",
            "## Numerical tolerances",
            "",
            f"- Raw-score comparison: `atol={RAW_ABS_TOL:.1e}`, `rtol={RAW_REL_TOL:.1e}`",
            f"- Processed-score comparison: `atol={PROCESSED_ABS_TOL:.1e}`, `rtol={PROCESSED_REL_TOL:.1e}`",
            f"- Structural zero-block diagnostic: `{STRUCTURE_ABS_TOL:.1e}`",
            f"- Relative-difference denominator floor: `{RELATIVE_FLOOR:.1e}`",
            f"- Score denominator epsilon: `{SCORE_DENOMINATOR_EPS:.1e}`",
            f"- Degenerate-scale threshold: `{SIGMA_DEGENERACY_TOL:.1e}`",
            "",
            "## Code changes",
            "",
            "- `src/raw_qmi_scores.py`: `symmetrize`, `aggregate_residual_bound`, `full_qmi_residual`, `dynamics_qmi_residual`, `dynamics_residual_matrix`, and the two score evaluators provide the shared implementation.",
            "- `src/time_domain_standard.py`, `src/time_domain_expanded.py`, `src/fusion_ablation.py`, and `src/vertex_selection_ablation.py`: `build_psi_data` uses the shared aggregate bound; `build_all_vertices_and_scores` computes the formal score directly from the successor residual and retains the full-QMI score; `compute_vertex_score_scalar` exposes both diagnostic modes.",
            "- `src/normalized_coordinates.py`: `build_fusion_diagnostics_payload` saves full-QMI scores separately in future archives.",
            "- `src/postprocess_generator_qmi.py` and `src/posthoc_certificate_verification.py`: saved-solution diagnostics use the signed dynamics margin and also report the complete-QMI eigenvalue.",
            "- `src/diagnose_raw_qmi_scores.py`: `evaluate_scenario` reproduces both endpoint score definitions, score processing, tier membership, SFPS initialization, and generator diagnostics from saved artifacts.",
            "- `tests/test_algebra.py` and `tests/test_released_artifacts.py`: structural equivalence and released-artifact regression checks.",
        ]
    )
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        evaluate_scenario("Standard", args.standard, args.output_dir),
        evaluate_scenario("Expanded", args.expanded, args.output_dir),
    ]
    unchanged = all(item["downstream_quantities_unchanged"] for item in scenarios)
    if unchanged:
        rerun_decision = (
            "No downstream controller re-synthesis is required because the "
            "quantities entering the SDP and SFPS+ICE are unchanged."
        )
    else:
        changed = []
        for item in scenarios:
            if not item["downstream_quantities_unchanged"]:
                changed.append(item["scenario"])
        rerun_decision = (
            "Downstream quantities changed for: " + ", ".join(changed)
            + ". Stop before any large rerun and identify the affected synthesis outputs."
        )

    report = {
        "analysis": "legacy full-QMI versus dynamics-block signed raw scores",
        "formal_data_consistency_set_changed": False,
        "solver_called": False,
        "batch_regenerated": False,
        "monte_carlo_rerun": False,
        "time_domain_rerun": False,
        "tolerances": {
            "raw_absolute": RAW_ABS_TOL,
            "raw_relative": RAW_REL_TOL,
            "processed_absolute": PROCESSED_ABS_TOL,
            "processed_relative": PROCESSED_REL_TOL,
            "structure_absolute": STRUCTURE_ABS_TOL,
            "relative_denominator_floor": RELATIVE_FLOOR,
            "score_denominator_epsilon": SCORE_DENOMINATOR_EPS,
            "sigma_degeneracy": SIGMA_DEGENERACY_TOL,
        },
        "score_hyperparameters": {
            "tau": TAU,
            "q_tau": Q_TAU,
            "q_sigma": Q_SCALE,
            "r_trigger": R_TRIGGER,
        },
        "sfps_settings_checked": {
            "budget": SFPS_BUDGET,
            "ratios": SFPS_RATIOS,
            "seed": SFPS_SEED,
        },
        "scenarios": scenarios,
        "downstream_rerun_required": not unchanged,
        "rerun_decision": rerun_decision,
    }
    json_path = args.output_dir / "raw_qmi_score_diagnostic.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    write_markdown(args.output_dir / "RAW_QMI_SCORE_DIAGNOSTIC.md", report)
    print(rerun_decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
