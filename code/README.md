# Numerical programs and records

Run the following from this directory after installing `config/environment.yml`:

```bash
python scripts/build_experiment_index.py
python scripts/verify_continuation_prefix.py
python scripts/reproduce.py --mode artifacts --package-root . --output-root /tmp/cga4pde_artifacts
python scripts/reproduce.py --mode diagnostics --package-root . --output-root /tmp/cga4pde_diagnostics
python tools/rebuild_baseline_evidence.py
```

`data/` contains raw and derived inputs; `result/` contains numerical outputs;
`config/` records parameters; `scripts/` and `tools/` contain the programs;
`logs/`, `reports/`, and `review/` contain execution and verification records.
The `tex` link points to `../paper/sisc_cga`.

Paths in computational CSV and JSON files are relative to this directory.
The frozen raw records retain their original content and recorded hashes; a
historical output path in a raw solver configuration describes the original run.
The public experiment index provides the current location of each case.
The continuation protocol and the underlying saved states are unchanged.

For the linear, cubic, and sinh cases, `p` is not an operator parameter.  The
historical raw JSON may retain the numeric placeholder used by the solver, while
the public experiment index leaves the mathematical parameter blank.  Use
`cga_refactor.config.frozen_case_config` when a paper case (including its
explicit (p) and ReLU settings) must be materialized directly.
