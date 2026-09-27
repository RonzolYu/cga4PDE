# CGA for nonlinear PDEs: manuscript and reproduction package

This repository contains the manuscript source together with the code, numerical inputs,
saved states, result tables, figures, and supplementary records needed to locate the
reported calculations.

Repository: <https://github.com/RonzolYu/cga4PDE>
Release tag: `cga-v3c-20260927`

## Manuscript compilation

The root `main.tex` is the manuscript entry point. Compile from the repository root with
XeLaTeX:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The paper source is under `tex/`; `chapter5_theory.tex` is the standalone Chapter 5 source.
All figures used by the manuscript, including the supplementary figures, are under
`tex/figures/`. The Markdown supplements S1--S4 and the machine-readable indexes are
kept in `supplement/` and `manifest/`.

## Reproduction package

All paths below are relative to this repository root:

- `code/`: generation, audit, reproduction, continuation, and baseline scripts;
- `config/`: environment, experiment, protocol, and path specifications;
- `data/raw/`: accepted states, candidate/reference pools, checkpoints, and source inputs;
- `data/derived/`: derived CSV/JSON records used by the paper;
- `result/`: frozen continuation states, diagnostics, baseline outputs, and generated figures;
- `tex/figures/supplementary/`: supplementary figures cited in the manuscript;
- `manifest/`: experiment indexes, figure/table maps, inventories, and SHA-256 records.

Install the environment listed in `config/environment.yml`, then run from the repository root:

```bash
python code/build_experiment_index.py
python code/reproduce.py --mode artifacts --package-root . --output-root /tmp/cga4pde_artifacts
python code/reproduce.py --mode diagnostics --package-root . --output-root /tmp/cga4pde_diagnostics
python code/verify_revision_prefix.py
```

The artifact mode rebuilds tables and figures from archived numerical records. The diagnostic
mode recomputes finite-trajectory, projection, quartic-identity, and frozen-prefix quantities
from saved states. These commands do not retrain every PDE solver. The continuation from
`N=32` to `N=64` is a frozen-prefix finite-window diagnostic; it is not an unconditional
asymptotic convergence certificate.

## Integrity and scope

Run

```bash
shasum -a 256 -c manifest/SHA256SUMS
```

to check the distributed files. `manifest/package_inventory.csv` lists the package contents
and `manifest/experiment_index_v2.csv` is the authoritative experiment index; the legacy
`experiment_index.csv` is retained for compatibility. The repository provides provenance and
reproduction records for the claims made in the manuscript; it does not claim a full retraining
of every solver or a continuous-dictionary certification.
