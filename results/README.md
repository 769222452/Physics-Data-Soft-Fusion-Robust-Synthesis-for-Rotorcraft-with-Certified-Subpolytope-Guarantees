# Released Numerical Artifacts

This directory contains the curated output of the simulation campaign
completed on 2026-07-12 and the deterministic post-processing applied to the
saved files. The software and solver environment is recorded in
`../rerun_environment.txt`.

## Directory Map

- `figures/time_domain_standard/`: four standard-range time-domain figures.
- `figures/time_domain_expanded/`: four expanded-range time-domain figures.
- `figures/monte_carlo/`: the survivor-conditioned Monte Carlo boxplot used in
  the manuscript.
- `tables/`: time-domain metrics, the fusion-ablation summary, all per-seed
  vertex-selection records, and the aggregated Table 8 rows.
- `raw/`: batches, adopted vertices, scores, controller variables,
  trajectories, random seeds, solver diagnostics, and Monte Carlo outcomes.
- `score_hull_check/`: the manuscript-facing saved-solution certificate check.
- `generator_qmi/`: the raw-QMI evaluation of the saved batch-generating
  models.
- `raw_qmi_score_diagnostic/`: per-vertex comparison of the legacy complete-QMI
  eigenvalue and the signed successor-state margin, including the downstream
  rerun decision.
- `posthoc_statistics/`: confidence intervals and paired comparisons computed
  from the saved Monte Carlo trials.

All NPZ files can be loaded without pickle:

```python
import numpy as np

data = np.load("results/raw/stage3_time_domain_results.npz",
               allow_pickle=False)
```

In the archived data-fusion files, `score_n_consistent` is a legacy name for
the zero processed-score count. The raw and processed counts are obtained from
`sum(raw_scores <= 0)` and `sum(processed_scores == 0)`, respectively. Current
full runs also store `score_n_raw_consistent` and
`score_n_zero_processed`.

The post-processing directories were generated only from the released NPZ
files. They do not represent a new controller synthesis, offline-data
generation, time-domain simulation, or Monte Carlo campaign. The repository
contains no flight-test, hardware, or personally identifiable data.

SHA-256 checksums for every released result and reference log are listed in
`manifest.sha256`.
