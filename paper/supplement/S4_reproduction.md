# S4. Reproduction

The code and numerical records are at <https://github.com/RonzolYu/cga4PDE>.
The data include 320 RFM, 32 CGA-prefix, and 114 FEM coefficient models. The model files, SHA256 hashes, numerical protocol, and evaluation records are stored in the companion repository at the paths used by the commands below.

Install `code/config/environment.yml`, which includes scikit-fem and pypdf.
From the repository root, rebuild figures/tables or recompute the separate
finite-trajectory diagnostics:

```bash
python code/scripts/reproduce.py --mode artifacts \
  --package-root code --output-root reproduction_output
python code/scripts/reproduce.py --mode diagnostics \
  --package-root code --output-root reproduction_output
python code/tools/rebuild_baseline_evidence.py
python code/scripts/check_reproduction.py
```

Each reproduction invocation uses an isolated temporary directory, verifies model paths and hashes, and recalculates the metric-specific summaries. Artifact mode generates figures and tables from the stored evaluations. Diagnostics mode evaluates the source, projection, and quartic quantities at quadrature orders 16 and 20, using the fixed calibration constants for continuation.

To reevaluate the saved comparison models, without changing their coefficients:

```bash
python code/tools/replay_baseline_models.py --refresh-evaluations
python code/tools/adopt_review_replay.py
```

Refresh writes separate evaluation-update files; adoption validates the complete
batch, merges these updates, and replaces the primary evaluation CSVs. Rebuild
artifacts afterward to reflect any changed evaluations. To perform new
coefficient fits under the same frozen protocol, use a distinct output folder:

```bash
python code/tools/replay_baseline_models.py \
  --output-dir data/raw/review_replay_20261001_fresh
```

The coefficient-fitting command optimizes CGA coefficients on fixed greedy prefixes, fits RFM coefficients for each prescribed width and seed, and fits FEM coefficients for each mesh and degree. The runtime configuration is recorded with the numerical data. The verification commands check model integrity, metric flags, statistical summaries, and figure targets; the reports provide the results of these checks.

The seed records identify the Hessian storage backend used for each fit. Matrix-free and cached-matrix actions differentiate the same quadrature energy with the same coordinate transform and stopping thresholds. Directional derivatives and manufactured coefficient minimizers verify their agreement. Stored coefficient models determine the reported evaluations.
