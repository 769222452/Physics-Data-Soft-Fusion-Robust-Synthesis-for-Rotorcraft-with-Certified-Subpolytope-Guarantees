# Soft Fusion of Physics Based Priors and Data Evidence for Robust Rotorcraft Synthesis

This repository contains the simulation code and selected reproducibility
artifacts for the manuscript:

**Soft Fusion of Physics Based Priors and Data Evidence for Robust Rotorcraft
Synthesis with Certified Subpolytope Guarantees**.

The code implements the numerical studies reported in the manuscript,
including time-domain tracking simulations, Monte Carlo fusion ablations,
and vertex-selection studies for the adopted physics-data prior representation.

## Repository Layout

```text
src/                         Simulation scripts and shared normalization code
tests/                       Algebra and implementation regression tests
results/figures/             Curated figures used for manuscript validation
results/tables/              Curated numerical tables
results/raw/                 Machine-readable NPZ rerun artifacts
results/README.md            Artifact map and field notes
results/manifest.sha256      SHA-256 checksums for released results and logs
logs/                        Reference run logs
results_revised/             Local full-rerun outputs (generated)
docs/reproducibility.md      Detailed reproduction notes
rerun_environment.txt        Reported campaign environment and solver settings
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

Run the regression tests before a full numerical campaign:

```bash
python -m unittest discover -s tests -v
```

## Reproducing the Main Numerical Studies

The main scripts are:

```bash
python src/time_domain_standard.py
python src/time_domain_expanded.py
python src/fusion_ablation.py
python src/vertex_selection_ablation.py
```

The scripts write regenerated outputs under `results_revised/`. The four
study-specific folders are `time-new/`, `time-new-ex/`, `monclo_Result/`, and
`monclo_pointsResult/`; machine-readable arrays are stored in `caches/` and
processed table files in `tables/`. The `results/` directory stores the
curated files corresponding to the final manuscript. Machine-readable batch,
controller, trajectory, Monte Carlo, and solver diagnostics are retained in
`results/raw/`, and `logs/` stores the reference console logs from the reported
rerun.

The released artifacts correspond to the clean campaign completed on
2026-07-12. Earlier simulation figures are not included in this repository.
Use `results/manifest.sha256` to verify the released files byte for byte.

The physical rotorcraft simulation remains in physical units. Synthesis,
data-consistency scoring, disturbance and residual bounds, and mixed-channel
norms use the fixed scales in `src/normalized_coordinates.py`. Controller
gains are converted back to physical coordinates before simulation.

More detailed reproduction notes are provided in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Data Availability

This repository provides the simulation scripts, fixed parameter settings,
random seeds, solver diagnostics, reference logs, and curated output files
needed to reproduce the reported numerical evidence. The results are generated
from simulation; no hardware or flight-test data are included.

## Citation

If this repository is useful for your research, please cite the associated
manuscript and this software repository. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff).

## License

This project is released under the Apache License 2.0. See
[`LICENSE`](LICENSE) for details.
