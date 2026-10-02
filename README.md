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

## Consistency revision, 2026-10-01

The clean manuscript synchronizes its metric definitions and comparison scope
with the code. The primary comparisons use the fully archived
`review_replay_20261001` batch: 320 RFM, 32 refitted CGA-prefix, and 114 FEM states.
Historical comparison inputs remain separate. Each primary metric uses a shared
solve/sign/quadrature policy; prescribed RFM denominators remain ten. See
`paper/sisc_cga/supplement/S3_failures_and_statistics.md` and `S4_reproduction.md`
for sample definitions, provenance, and the distinct reproduction modes.

The manuscript and generated comparison tables use descriptive problem names.
Internal directory names and case keys remain in the computational files and
reproduction instructions. P1, P2 and P3 denote the polynomial degrees of the
finite-element approximations. The optional
`python code/scripts/check_reproduction.py --compile-paper` check verifies clean
compilation and the absence of experimental case identifiers in the rendered PDF.
