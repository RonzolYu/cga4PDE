# Numeric source check

This audit checks the package-local validation records, statistical counts, interpolation brackets,
claim scope, and source-language constraints used by the paper.

| Check | Status | Source |
|---|---|---|
| experiment data validation | PASS | `data/derived/experiments/experiment_validation.json` |
| RFM raw row count | PASS | `data/derived/experiments/rfm_multiseed_raw.csv` |
| RFM prescribed seed set | PASS | `data/derived/experiments/experiment_validation.json` |
| RFM count conservation | PASS | `data/derived/experiments/rfm_multiseed_summary.csv` |
| C5 endpoint keeps prescribed denominator | PASS | `data/derived/experiments/rfm_multiseed_summary.csv` |
| metric-specific summary counts | PASS | `data/derived/experiments/rfm_multiseed_summary.csv` |
| ID17 low-frequency profile | PASS | `config/problems.json` |
| finite-window diagnostic validation | PASS | `data/derived/window_diagnostics/window_certificate_validation.json` |
| finite-window claim scope | PASS | `data/derived/window_diagnostics/window_certificate_validation.json` |
| finite-window transfer rows | PASS | `data/derived/window_diagnostics/transfer_budget.csv` |
| artifact validation | PASS | `data/derived/baselines/artifact_validation.json` |
| figure count metadata | PASS | `data/derived/baselines/artifact_validation.json` |
| common-grid interpolation brackets | PASS | `data/derived/baselines/baseline_common_grid.csv` |
| baseline plot scope | PASS | `data/derived/baselines/artifact_validation.json` |
| aggregate ranking uses complete endpoints | PASS | `data/derived/baselines/artifact_validation.json` |
| quadrature sensitivity rows | PASS | `data/derived/experiments/quadrature_sensitivity.csv` |
| paper source contains no internal-process language | PASS | `main.tex, supplement.tex, sections/*.tex, generated/*.tex` |

Result: **PASS** (17/17 checks).
