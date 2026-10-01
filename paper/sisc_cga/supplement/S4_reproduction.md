# S4. Reproduction

The code and numerical records are at <https://github.com/RonzolYu/cga4PDE>.
The comparison uses the explicit `review_replay_20261001` batch. All
320 RFM, 32 refitted CGA-prefix, and 114 FEM comparison states have saved models
under `code/data/raw/review_replay_20261001/models/`, with SHA256 hashes in the
raw CSVs. The frozen protocol, evaluator hashes, and archive verification are
stored beside those models. Historical inputs are retained under
`code/data/raw/historical_baseline_202609/`; their incomplete model archive is
not claimed to reproduce the comparison results.

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

Each reproduction invocation builds an isolated temporary copy. It verifies
all comparison model paths and hashes, recalculates metric-specific RFM
summaries, and checks the recursively loaded manuscript inputs. Artifact mode
rebuilds figures and tables from recorded evaluations; it does not rerun PDE
fits. Diagnostics recomputes the original auxiliary source/projection/quartic
identities and frozen-prefix continuation at quadrature orders 16 and 20.
The original calibration constants and preparation phase must not be changed
when checking that continuation.

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

CGA replay refits coefficients on archived selected prefixes; it does not replay
the greedy selection trajectory. RFM replay fits every configured width and
seed, and FEM replay fits every specified mesh and degree. Runtime comparisons
are not claimed. The validated runtime and library versions are
recorded in `code/data/raw/review_replay_20261001/runtime.json`.

S3's ledger reconstruction is a check of stored metrics and plotting/statistical
provenance, not an independent PDE solve. Fresh model loading and reevaluation
checks are separately recorded in `code/review/review_replay_reevaluation.json`.
Missing inputs, changed model hashes, invalid flags, mismatched summaries, or
missing targets produce a nonzero exit code and a FAIL report. Floating-point
quadrature agreement and finite-range checks provide numerical evidence, not
rigorous enclosures or unconditional asymptotic theorems.

The seed ledger also identifies the Hessian storage backend used in a feature
fit. Matrix-free actions and cached-matrix actions differentiate the same
quadrature energy, with the same coordinate transform and stopping thresholds.
Their agreement is checked against directional derivatives and manufactured
coefficient minimizers. The earlier matrix-free implementation is saved under
`code/data/raw/review_replay_20261001/source_code/`. Archived coefficients define
the reported evaluations; fresh fits record their own numerical solver results.
