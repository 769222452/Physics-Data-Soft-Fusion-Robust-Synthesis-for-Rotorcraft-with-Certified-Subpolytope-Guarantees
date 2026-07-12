# Released numerical artifacts

The files in this directory are the curated outputs of the clean simulation
campaign completed on 2026-07-12. They correspond to the values reported in
the manuscript. The exact software and solver environment is recorded in
`../rerun_environment.txt`.

## Directory map

- `figures/time_domain_standard/`: five standard-range manuscript panels.
- `figures/time_domain_expanded/`: five expanded-range manuscript panels.
- `figures/monte_carlo/`: the two Monte Carlo figures used in the manuscript.
- `tables/time_domain_*_metrics.csv`: time-domain metrics before display
  rounding.
- `tables/fusion_ablation_monte_carlo_summary.csv`: the 2000-trial fusion
  ablation summary.
- `tables/vertex_selection_ablation_by_seed.csv`: all 120 per-seed
  vertex-selection records.
- `tables/vertex_selection_table8_summary.csv`: the 24 aggregated Table 8
  rows before display rounding.
- `tables/vertex_selection_table8_rows.tex`: the corresponding LaTeX row
  fragment.
- `raw/`: machine-readable batches, vertices, scores, controller variables,
  trajectories, seeds, solver diagnostics, and Monte Carlo outcomes.

All NPZ files can be loaded without pickle:

```python
import numpy as np

data = np.load("results/raw/stage3_time_domain_results.npz",
               allow_pickle=False)
```

In the archived data-fusion files, `score_n_consistent` is a legacy name for
the zero processed score count. The two unambiguous counts are obtained from
`sum(raw_scores <= 0)` and `sum(processed_scores == 0)`, respectively. Current
source runs additionally store these as `score_n_raw_consistent` and
`score_n_zero_processed`.

The simulation campaign contains no flight-test, hardware, or personally
identifiable data. SHA-256 checksums for all released results and reference
logs are listed in `manifest.sha256`.
