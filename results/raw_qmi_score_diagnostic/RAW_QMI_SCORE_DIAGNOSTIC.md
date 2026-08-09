# Raw-QMI score structural diagnostic

## Mathematical structure

For every adopted candidate, the performance matrices are fixed and the saved batch satisfies `Z_t = [C_d^c D_d^c] [X_t; U_t]`. The performance-output rows of the residual and of the aggregate bound are therefore zero. In exact arithmetic, the complete QMI residual is `blockdiag(H_x,i, 0_16)`, where `H_x,i = R_x,i R_x,i^T - R_tilde_x,t`. Hence the complete QMI is negative semidefinite if and only if `H_x,i` is negative semidefinite, but its largest eigenvalue is `max(lambda_max(H_x,i), 0)`. The dynamics-block eigenvalue is used as the signed raw margin; the complete residual is retained for equivalence diagnostics.

## Endpoint comparison

| Scenario | Max abs. difference | Max rel. difference | Raw counts old/new | Anchors old/new | Scores ordered identically | Tiers identical | SFPS initialization identical |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| Standard | 4.683897714131e-11 | 4.964462511626e-14 | 0/0 | 103/103 | yes | yes | yes |
| Expanded | 7.230482879095e-11 | 2.290324321420e-14 | 0/0 | 103/103 | yes | yes | yes |

### Score-processing details

**Standard**

- Old/new `tau_eff`: 3.795227961298e+02 / 3.795227961298e+02
- Old/new `sigma_s`: 2.778508815124e+00 / 2.778508815124e+00
- Maximum processed-score difference: 3.996802888651e-15
- Anchor index sets identical: True
- Processed-score ordering identical: True
- Per-vertex data: `results/raw_qmi_score_diagnostic/standard_endpoint_scores.csv`

**Expanded**

- Old/new `tau_eff`: 3.037874766639e+03 / 3.037874766639e+03
- Old/new `sigma_s`: 8.147221524025e+00 / 8.147221524025e+00
- Maximum processed-score difference: 8.881784197001e-16
- Anchor index sets identical: True
- Processed-score ordering identical: True
- Per-vertex data: `results/raw_qmi_score_diagnostic/expanded_endpoint_scores.csv`

## Saved generator diagnostic

| Scenario | Full-QMI largest eigenvalue | Dynamics signed margin | Dynamics eig. min | Dynamics eig. max | Zero-block roundoff diagnosis |
|---|---:|---:|---:|---:|:---:|
| Standard | 9.667930374295e-11 | -1.482923900254e+02 | -1.501046345573e+02 | -1.482923900254e+02 | yes |
| Expanded | 4.668459817884e-11 | -1.487469805129e+02 | -1.505483545040e+02 | -1.487469805129e+02 | yes |

The positive complete-matrix values near `1e-10` are dominated by numerical perturbations in the structurally zero performance-output block. They do not describe the signed dynamics margin.

## Rerun decision

No downstream controller re-synthesis is required because the quantities entering the SDP and SFPS+ICE are unchanged.

No SDP, controller synthesis, Monte Carlo campaign, batch generation, or time-domain simulation was run for this diagnostic.

## Numerical tolerances

- Raw-score comparison: `atol=1.0e-08`, `rtol=1.0e-12`
- Processed-score comparison: `atol=1.0e-12`, `rtol=1.0e-12`
- Structural zero-block diagnostic: `1.0e-08`
- Relative-difference denominator floor: `1.0e-12`
- Score denominator epsilon: `1.0e-12`
- Degenerate-scale threshold: `1.0e-10`

## Code changes

- `src/raw_qmi_scores.py`: `symmetrize`, `aggregate_residual_bound`, `full_qmi_residual`, `dynamics_qmi_residual`, `dynamics_residual_matrix`, and the two score evaluators provide the shared implementation.
- `src/time_domain_standard.py`, `src/time_domain_expanded.py`, `src/fusion_ablation.py`, and `src/vertex_selection_ablation.py`: `build_psi_data` uses the shared aggregate bound; `build_all_vertices_and_scores` computes the formal score directly from the successor residual and retains the full-QMI score; `compute_vertex_score_scalar` exposes both diagnostic modes.
- `src/normalized_coordinates.py`: `build_fusion_diagnostics_payload` saves full-QMI scores separately in future archives.
- `src/postprocess_generator_qmi.py` and `src/posthoc_certificate_verification.py`: saved-solution diagnostics use the signed dynamics margin and also report the complete-QMI eigenvalue.
- `src/diagnose_raw_qmi_scores.py`: `evaluate_scenario` reproduces both endpoint score definitions, score processing, tier membership, SFPS initialization, and generator diagnostics from saved artifacts.
- `tests/test_algebra.py` and `tests/test_released_artifacts.py`: structural equivalence and released-artifact regression checks.
