# S4. Reproduction

The complete numerical scripts, saved states, evaluation inputs, and generated records are
available in the companion repository <https://github.com/RonzolYu/cga4PDE>. The commands below
are run from the repository root after installing the environment specified by
`config/environment.yml`:

```bash
python code/reproduce.py --mode artifacts \
  --package-root . --output-root reproduction_output
python code/reproduce.py --mode diagnostics \
  --package-root . --output-root reproduction_output
```

Each invocation builds an isolated temporary copy. Outputs are published only after validation;
input files remain unchanged when a distinct output root is used. The artifact mode rebuilds the
figures and generated tables from the numerical CSV inputs, recomputes the RFM summary, and
checks the dependencies recursively included by `main.tex`. Diagnostics recomputes the
finite-trajectory source, projections, quartic identities, and the frozen-prefix ($N=64$)
analysis at quadrature orders 16 and 20. It also checks the unified 617-state denominators
described in S3.

The fixed-pool continuation is documented in
`code/continuation_experiments.py`; the frozen preparation phase must not be rerun to change the
calibrated constants. Missing inputs, invalid validation flags, mismatched hashes, failed
computations, or missing targets give a nonzero exit code and a `FAIL` report.

The repository manifests identify the source state and the files required by the numerical
claims. The finite-range checks in S2 and the independent evaluations in S3 are evidence for the
reported computations; they do not establish an unconditional asymptotic theorem.
