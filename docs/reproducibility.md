# Reproducibility Notes

This document summarizes how to reproduce the numerical studies in the
manuscript. The repository contains source scripts, selected reference logs,
curated figures, and curated tables.

## Environment

Use Python 3.8 or newer. Install the package requirements from the repository
root:

```bash
python -m pip install -r requirements.txt
```

The SDP steps require MOSEK and a valid license. Configure the license before
running the scripts:

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

On Windows PowerShell:

```powershell
$env:MOSEKLM_LICENSE_FILE = "C:\path\to\mosek.lic"
```

## Script Map

| Script | Purpose | Curated outputs |
| --- | --- | --- |
| `src/time_domain_standard.py` | Standard time-domain tracking study | `results/figures/time_domain_standard/`, `results/tables/time_domain_standard_metrics.csv` |
| `src/time_domain_expanded.py` | Expanded time-domain tracking study | `results/figures/time_domain_expanded/`, `results/tables/time_domain_expanded_metrics.csv` |
| `src/fusion_ablation.py` | Monte Carlo soft fusion ablation | `results/figures/monte_carlo/` |
| `src/vertex_selection_ablation.py` | Vertex-selection ablation | `results/tables/vertex_selection_table8_summary.csv`, `results/tables/vertex_selection_ablation_by_seed.csv` |

## Running the Studies

From the repository root, run:

```bash
python src/time_domain_standard.py
python src/time_domain_expanded.py
python src/fusion_ablation.py
python src/vertex_selection_ablation.py
```

The scripts create local output directories under `results_revised/` at
runtime:

| Script | Runtime output directory |
| --- | --- |
| `src/time_domain_standard.py` | `results_revised/time-new/` |
| `src/time_domain_expanded.py` | `results_revised/time-new-ex/` |
| `src/fusion_ablation.py` | `results_revised/monclo_Result/` |
| `src/vertex_selection_ablation.py` | `results_revised/monclo_pointsResult/` |

Shared raw arrays and solver records are written to
`results_revised/caches/`, while processed CSV and LaTeX table fragments are
written to `results_revised/tables/`. Delete `results_revised/` before a clean
campaign; the scripts do not read the earlier curated `results/` files.

## Coordinate Scaling

The simulator propagates the physical model. All synthesis and scoring norms
use the fixed diagonal scales defined in `src/normalized_coordinates.py`:

| Quantity | Diagonal scales |
| --- | --- |
| Physical state | `(1,1,1,1,1,1,0.25,0.25,0.25,1,1,1)` |
| Input memory | `(6,4,4,4)` |
| Input increment | `(3.5,3.5,3.5,3.5)` |
| Disturbance | `(2.4,2.4,2.4,2.4,2.4,2.4)` |
| State-record residual cap | `3e-3` on each physical state channel |
| State-record residual standard deviation | `5e-4` on each channel |

These scales are prescribed before data generation and are not estimated from
the batch used for consistency scoring.

Existing curated outputs are retained under `results/` so that manuscript
figures and tables can be checked without rerunning all SDP computations. Full
SDP and Monte Carlo reruns can take substantial time depending on hardware and
MOSEK configuration.

The `results/raw/` directory contains the NPZ artifacts from the reported clean
rerun, including batch data, vertices, processed scores, controller variables,
solver diagnostics, trajectories, Monte Carlo seeds, and per-seed ablation
outcomes. Load these files with `numpy.load(..., allow_pickle=False)`.

The archived data-fusion NPZ files retain the legacy scalar field
`score_n_consistent`, which equals the number of zero processed scores. For an
unambiguous audit, compute the raw count as `sum(raw_scores <= 0)` and the
processed count as `sum(processed_scores == 0)`. Current source runs also emit
the explicit fields `score_n_raw_consistent` and `score_n_zero_processed`.

## Random Seeds and Numerical Solvers

The base random seed is `26`; derived seeds are stored with the generated raw
arrays. The numerical full-vertex thresholds are `1e-4` for Surrogate pass and
`2e-4` for Near surrogate. These are implementation tolerances only; the
theoretical LMIs are exact. Small solver-dependent differences can occur across
MOSEK versions, BLAS/LAPACK backends, thread schedules, or operating systems.
The reference logs in `logs/` record solver parameters, status, objectives,
and numerical residuals. Repository and license paths in these logs are
sanitized placeholders; all numerical lines are unchanged from the clean run.

### MOSEK acceptance and matrix extraction

All experiment scripts use `src/mosek_helpers.py`. A returned solution is
accepted only when the problem status is `PrimalAndDualFeasible` and both the
primal and dual solution statuses are `Optimal` or `Feasible`. The latter
category permits a numerically feasible time-limited solution, but every
accepted controller must still pass the separate a posteriori LMI residual
checks. Unknown, infeasible, certificate, ill-posed, and undefined statuses are
rejected before decision-variable levels are read.

Fusion matrix levels are reconstructed in fixed row-major (`order="C"`) order.
This convention is verified against indexed Fusion entries using a
non-symmetric rectangular test matrix; no matrix order is inferred from the
symmetry of `Q`.

The archived clean-rerun logs use the earlier message
`SOFT-CONSISTENCY mode` for the quantile fallback. This label means only that
the processed-score threshold fallback was activated; it does not assert raw
QMI consistency. New runs print `PROCESSED-SCORE FALLBACK`.

## Scope of the Released Data

The released artifacts are simulation data and generated numerical outputs.
No physical flight-test or hardware experimental data are included.
