# S4. Reproduction

The repository root is the package root. The complete numerical scripts, saved states, audit
inputs, results, and supplementary figures are included here; no sibling workspace or external
Material directory is required. Install the environment listed in `config/environment.yml`.

Run from the repository root:

```bash
python code/reproduce.py --mode artifacts \
  --package-root . --output-root /tmp/cga4pde_artifacts
python code/reproduce.py --mode diagnostics \
  --package-root . --output-root /tmp/cga4pde_diagnostics
python code/check_R5_reproduction.py
```

Each invocation builds an isolated temporary copy. Outputs are published only after validation;
input files remain unchanged when a distinct output root is used. Artifact mode rebuilds figures
and generated tables from numerical CSV inputs, recomputes the RFM summary, and checks the
manuscript dependencies. Diagnostics recomputes finite-trajectory source, projections, quartic
identities, and the frozen-prefix N=64 analysis at quadrature orders 16 and 20. It also checks
the unified 617-state denominators described in S3.

The legacy `--mode solver` entry point is unsupported by this package and exits nonzero. The
fixed-pool continuation is documented in `code/revision_experiments.py`; the frozen prepare phase
must not be rerun to change the calibrated constants. Missing inputs, invalid validation flags,
mismatched hashes, failed computations, or missing targets give a nonzero exit code and a FAIL
report.

The repository is publicly available at
<https://github.com/RonzolYu/cga4PDE/tree/cga-v3c-20260927>. The distributed release is fixed
by `manifest/SHA256SUMS`; the continuation remains a finite-window diagnostic and does not
certify an unconditional asymptotic rate.
