# Chebyshev Greedy Algorithm for Nonlinear PDEs

This repository contains the SISC manuscript and the computational records for the
Chebyshev greedy algorithm (CGA) study. The public tree has two top-level folders:

- `paper/sisc_cga/`: the SIAM Journal on Scientific Computing source package and the
  compiled manuscript. Compile with `latexmk -pdf -interaction=nonstopmode -halt-on-error -cd main.tex`.
- `code/`: numerical programs, input data, configurations, generated results, logs,
  manifests, and audit reports. Run the documented checks from this directory.

The manuscript reports finite-range conditional estimates. The numerical records do
not establish an unconditional asymptotic rate. The code and data are released under
this repository so that the reported figures, tables, and finite-range checks can be
located from the manifest files.

## Reproduction entry points

```bash
cd code
python scripts/build_experiment_index.py
python scripts/reproduce.py --mode artifacts --package-root . --output-root /tmp/cga4pde_artifacts
python scripts/reproduce.py --mode diagnostics --package-root . --output-root /tmp/cga4pde_diagnostics
python scripts/check_reproduction.py
```

The default `code/tex` link points to `../paper/sisc_cga`, so the reproduction scripts
use exactly the paper source shipped in this repository. The environment is specified
in `code/config/environment.yml`. The audit reports in `code/review/` record the final
page count, source graph, citations, and logical-chain checks.
