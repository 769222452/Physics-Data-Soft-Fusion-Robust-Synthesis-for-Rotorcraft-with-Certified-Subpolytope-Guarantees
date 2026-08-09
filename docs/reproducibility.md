# Reproducibility Notes

This document records the numerical workflow associated with the manuscript.
The repository includes source code, reference logs, saved machine-readable
outputs, manuscript figures, and numerical tables.

## Environment

Use Python 3.8 or newer and install the requirements from the repository root:

```bash
python -m pip install -r requirements.txt
```

The SDP stages require MOSEK and a valid license. Saved-result processing uses
NumPy and SciPy only. The exact campaign environment and solver settings are
listed in `rerun_environment.txt`.

## Script and Output Map

| Script | Purpose | Released output |
| --- | --- | --- |
| `src/time_domain_standard.py` | Standard-range tracking study | `results/figures/time_domain_standard/`, `results/tables/time_domain_standard_metrics.csv` |
| `src/time_domain_expanded.py` | Expanded-range tracking study | `results/figures/time_domain_expanded/`, `results/tables/time_domain_expanded_metrics.csv` |
| `src/fusion_ablation.py` | Monte Carlo fusion ablation | `results/figures/monte_carlo/`, `results/tables/fusion_ablation_monte_carlo_summary.csv` |
| `src/vertex_selection_ablation.py` | Vertex-selection ablation | `results/tables/vertex_selection_*` |
| `src/posthoc_score_hull_check.py` | Saved score-bounded certificate check | `results/score_hull_check/` |
| `src/postprocess_generator_qmi.py` | Saved batch-generator raw-QMI evaluation | `results/generator_qmi/` |
| `src/postprocess_monte_carlo_statistics.py` | Statistics from paired saved trials | `results/posthoc_statistics/` |
| `src/plot_saved_time_domain_figures.py` | Figure export from saved trajectories | `results/figures/time_domain_*` |

`src/posthoc_certificate_verification.py` contains the fixed-solution matrix
evaluation routines used by `posthoc_score_hull_check.py` and exposes a
standalone command-line interface for scenario-specific checks.

## Full Numerical Campaign

From the repository root, run:

```bash
python src/time_domain_standard.py
python src/time_domain_expanded.py
python src/fusion_ablation.py
python src/vertex_selection_ablation.py
```

Runtime outputs are written below `results_revised/`:

| Script | Runtime directory |
| --- | --- |
| `src/time_domain_standard.py` | `results_revised/time-new/` |
| `src/time_domain_expanded.py` | `results_revised/time-new-ex/` |
| `src/fusion_ablation.py` | `results_revised/monclo_Result/` |
| `src/vertex_selection_ablation.py` | `results_revised/monclo_pointsResult/` |

Raw arrays and solver records are written to `results_revised/caches/`, while
processed table files are written to `results_revised/tables/`. The released
`results/` directory is not used as an input by the full campaign.

## Saved-Result Checks

The saved-result tools do not regenerate the offline batch, synthesize a
controller, run SFPS+ICE, execute a time-domain simulation, or repeat Monte
Carlo trials. Their exact commands are listed in the main README.

The score-hull check reports positive definiteness of the saved `Q`, the
`Y-KQ` consistency residual, the full-library score-weighted surrogate
residual, the common coefficient, and the worst standard-LMI residual over the
selected score-bounded vertex set. Residual matrices are symmetrized before
eigenvalue evaluation. Numerical tolerances classify floating-point output;
they do not replace the exact semidefinite assumptions in the theoretical
results.

The Monte Carlo post-processor computes Wilson confidence intervals, an exact
McNemar comparison for paired success outcomes, and a seeded paired bootstrap
for trials in which both compared controllers succeed.

## Coordinate Scaling

The simulator propagates the physical model. Synthesis and scoring use the
fixed diagonal scales defined in `src/normalized_coordinates.py`:

| Quantity | Diagonal scales |
| --- | --- |
| Physical state | `(1,1,1,1,1,1,0.25,0.25,0.25,1,1,1)` |
| Input memory | `(6,4,4,4)` |
| Input increment | `(3.5,3.5,3.5,3.5)` |
| Disturbance | `(2.4,2.4,2.4,2.4,2.4,2.4)` |
| State-record residual cap | `3e-3` on each physical state channel |
| State-record residual standard deviation | `5e-4` on each channel |

The first three disturbance scales have units of `m/s^2`; the final three
have units of `rad/s^2`. The six normalized disturbance components are jointly
projected onto one Euclidean unit ball. These scales are fixed before data
generation and are not estimated from the scoring batch.

## Raw Files

The files under `results/raw/` contain the reported batches, adopted vertices,
scores, controller variables, solver diagnostics, trajectories, Monte Carlo
seeds, and per-seed ablation outcomes. They can be read with:

```python
import numpy as np

data = np.load("results/raw/stage3_time_domain_results.npz",
               allow_pickle=False)
```

The archived data-fusion files retain the legacy field
`score_n_consistent`, which stores the number of zero processed scores. The
unambiguous raw and processed counts are `sum(raw_scores <= 0)` and
`sum(processed_scores == 0)`. Current full runs also emit
`score_n_raw_consistent` and `score_n_zero_processed`.

## Seeds and Solver Acceptance

The base random seed is `26`. Derived seeds are stored in the raw files. The
fusion ablation contains 2000 paired trials. The vertex-selection study uses
five synthesis seeds and 500 Monte Carlo plants per run.

All experiment scripts use `src/mosek_helpers.py`. A returned solution is
accepted only when the problem status is `PrimalAndDualFeasible` and the
primal and dual solution statuses are `Optimal` or `Feasible`. Every accepted
controller is then subjected to the separate numerical LMI residual checks.
Other statuses are rejected before decision-variable levels are read.

Fusion matrix levels are reconstructed in fixed row-major (`order="C"`) order.
The convention is tested with a non-symmetric rectangular matrix.

Numerical full-vertex classification uses `1e-4` for Surrogate pass and
`2e-4` for Near surrogate. Small differences can occur across MOSEK versions,
BLAS/LAPACK backends, thread schedules, or operating systems. Reference logs
record the solver settings, statuses, objectives, and numerical residuals.

## Data Scope

All released data and outputs are generated by simulation. The repository
contains no physical flight-test records, hardware measurements, or personally
identifiable data.
