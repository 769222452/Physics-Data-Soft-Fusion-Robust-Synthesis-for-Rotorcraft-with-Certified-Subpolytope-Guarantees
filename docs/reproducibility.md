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
| `src/fusion_ablation.py` | Monte Carlo soft-fusion ablation | `results/figures/monte_carlo/` |
| `src/vertex_selection_ablation.py` | Vertex-selection ablation | `results/tables/vertex_selection_tables.tex` |

## Running the Studies

From the repository root, run:

```bash
python src/time_domain_standard.py
python src/time_domain_expanded.py
python src/fusion_ablation.py
python src/vertex_selection_ablation.py
```

The scripts create local output directories at runtime. Existing curated
outputs are retained under `results/` so that manuscript figures and tables can
be checked without rerunning all SDP computations.

## Random Seeds and Numerical Solvers

Random seeds used for the reported simulation batches are defined in the
scripts. Small solver-dependent differences can occur across MOSEK versions,
BLAS/LAPACK backends, or operating systems. The reference logs in `logs/`
record the reported run status and numerical diagnostics.

## Scope of the Released Data

The released artifacts are simulation data and generated numerical outputs.
No physical flight-test or hardware experimental data are included.
