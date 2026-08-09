# Physics- and Data-Guided Rotorcraft Synthesis

This repository contains the simulation code and reproducibility artifacts for
the manuscript:

**Robust Rotorcraft Synthesis Guided by Physics and Data with
Score-Dependent Dissipation Guarantees**.

The implementation covers the reported time-domain tracking studies, Monte
Carlo fusion ablation, vertex-selection ablation, and saved-solution
certificate checks. All data are generated in simulation; no flight-test or
hardware data are included.

## Repository Layout

```text
src/                         Simulation and post-processing scripts
tests/                       Algebra and implementation regression tests
results/raw/                 Saved batches, solutions, trials, and trajectories
results/figures/             Figures used in the manuscript
results/tables/              Numerical tables and LaTeX table rows
results/score_hull_check/    Score-bounded certificate results
results/generator_qmi/       Batch-generator raw-QMI evaluation
results/posthoc_statistics/  Statistics computed from saved Monte Carlo trials
results/manifest.sha256      Checksums for released results and logs
logs/                        Reference run logs
docs/reproducibility.md      Detailed reproduction instructions
rerun_environment.txt        Reported software and solver environment
```

## Requirements

The scripts support Python 3.8 or newer and use NumPy, SciPy, Matplotlib, and
MOSEK. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Controller synthesis requires a valid MOSEK license. Configure it using
MOSEK's standard `MOSEKLM_LICENSE_FILE` environment variable. The saved-result
post-processing commands below do not require MOSEK.

Run the test suite before a numerical campaign:

```bash
python -m unittest discover -s tests -v
```

## Main Numerical Studies

Run the four study scripts from the repository root:

```bash
python src/time_domain_standard.py
python src/time_domain_expanded.py
python src/fusion_ablation.py
python src/vertex_selection_ablation.py
```

Full synthesis and Monte Carlo runs can take many hours. The scripts write new
outputs under the ignored `results_revised/` directory and do not overwrite the
released artifacts under `results/`.

## Saved-Result Processing

The following commands reproduce manuscript-facing checks from the released
NPZ files without solving an SDP or running a simulation:

```bash
python src/posthoc_score_hull_check.py

python src/postprocess_generator_qmi.py \
  --standard results/raw/standard_data_fusion_diagnostics.npz \
  --expanded results/raw/expanded_data_fusion_diagnostics.npz \
  --json-output results/generator_qmi/generator_qmi.json \
  --csv-output results/generator_qmi/generator_qmi.csv

python src/postprocess_monte_carlo_statistics.py \
  --input results/raw/fusion_ablation_monte_carlo_raw.npz \
  --output-dir results/posthoc_statistics

python src/plot_saved_time_domain_figures.py \
  --standard results/raw/stage3_time_domain_results.npz \
  --expanded results/raw/stage3_time_domain_results-ex.npz \
  --output-dir results/figures
```

`src/posthoc_certificate_verification.py` provides the shared fixed-solution
matrix checks used by the score-hull tool and can also be run directly with
explicit solution, diagnostics, scenario, and output arguments.

The released simulation campaign was completed on 2026-07-12. Subsequent
saved-result processing did not alter the controllers, trajectories, Monte
Carlo outcomes, or simulation settings. Use `results/manifest.sha256` to verify
the released files byte for byte.

The physical rotorcraft model is simulated in physical units. Synthesis,
data-consistency scoring, and mixed-channel bounds use the fixed coordinate
scales in `src/normalized_coordinates.py`; recovered gains are converted to
physical coordinates before simulation.

See [docs/reproducibility.md](docs/reproducibility.md) for the output map,
solver acceptance rules, random seeds, and coordinate scales.

## Citation

Please cite the associated manuscript and this repository. Citation metadata
are provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
