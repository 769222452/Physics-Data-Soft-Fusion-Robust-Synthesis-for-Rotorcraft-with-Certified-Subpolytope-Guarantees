# Physics-Data Soft Fusion Robust Synthesis for Rotorcraft

This repository contains the simulation code and selected reproducibility
artifacts for the manuscript:

**Physics-Data Soft Fusion Robust Synthesis for Rotorcraft with Certified
Subpolytope Guarantees**.

The code implements the numerical studies reported in the manuscript,
including time-domain tracking simulations, Monte Carlo fusion ablations,
and vertex-selection studies for the adopted physics-data prior representation.

## Repository Layout

```text
src/                         Simulation scripts
results/figures/             Curated figures used for manuscript validation
results/tables/              Curated numerical tables
logs/                        Reference run logs
docs/reproducibility.md      Detailed reproduction notes
LICENSE                      Apache-2.0 license
CITATION.cff                 Citation metadata
requirements.txt             Python dependencies
```

## Requirements

The scripts are written for Python 3.8 or newer and require NumPy, SciPy,
Matplotlib, and MOSEK. A valid MOSEK license is required for the SDP solves.
Set the license path or license server through the standard environment
variable before running the experiments:

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

On Windows PowerShell:

```powershell
$env:MOSEKLM_LICENSE_FILE = "C:\path\to\mosek.lic"
```

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Reproducing the Main Numerical Studies

The main scripts are:

```bash
python src/time_domain_standard.py
python src/time_domain_expanded.py
python src/fusion_ablation.py
python src/vertex_selection_ablation.py
```

The scripts write regenerated outputs to local result folders created at
runtime under `simulation_results/`. The `results/` directory stores curated
output files corresponding to the manuscript figures and tables, and `logs/`
stores reference console logs from the reported runs.

More detailed reproduction notes are provided in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Data Availability

This repository provides the simulation scripts, selected parameter settings,
random seeds embedded in the scripts, reference logs, and curated output files
needed to reproduce the reported numerical evidence. No hardware experimental
data are included.

## Citation

If this repository is useful for your research, please cite the associated
manuscript and this software repository. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff).

## License

This project is released under the Apache License 2.0. See
[`LICENSE`](LICENSE) for details.
