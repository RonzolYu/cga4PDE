# Supplementary material map

The main SISC manuscript is 26 pages. The files in this directory preserve
details that are useful for verification but are not needed to state the main
contribution.

- `S1_experiment_index.md`: case and figure/table index.
- `S2_pool_generation.md` and `S2_window_certificate.md`: pool construction and
  finite-range checks.
- `S4_theory_details.tex`: full theory chapter, including the longer
  derivations.
- `S4_background_details/A_source_proof_full.tex`: the one-dimensional
  source-class proof omitted from the main input.
- `S4_background_details/`: full versions of the setting/algorithm,
  inexact-descent, natural-geometry, and energy-family sections.
- `S3_failures_and_statistics.md`: RFM sample definitions and failure
  accounting for the baseline plots.

The generated dyadic tables, protocol records, and epsilon-scan data remain in
`paper/sisc_cga/generated/`, `code/result/`, and `code/review/`. They are
not loaded by `main.tex` in the SISC main text. The public code/data
repository is identified in the main-text code-availability section.
