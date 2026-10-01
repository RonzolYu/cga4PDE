> Historical audit of the pre-2026-10-01 package. Its counts, paths, and conclusions are not current revision evidence.

# Logic-chain and compression audit

Date: 2026-09-30

## Scope

The audit covers every source loaded by `main.tex`, the generated table input,
the figures referenced by the main text, and the supplementary files cited from
the manuscript.

## Repairs made after the compression pass

- Replaced generic proof sketches in the projection/source lemmas, finite-pool
  score lemma, discrete-to-analysis energy lemma, natural-distance results, and
  local-Hessian propositions with statements matching the displayed claims.
- Restored the finite-pool correction inequality and the zero-score convention
  for pool coverage in the descent proposition.
- Restored the quantitative score-error and correction-error absorption
  assumptions needed by the conditional residual-comparison theorem.
- Defined (e_m), (a_*), (P_m), the finite dictionary, the innovation
  norm, (	heta_m), and the approximate-source representation before the
  finite-trajectory source theorem.
- Defined the one-dimensional quartic energy and frozen Hessian before the
  quartic identity, and defined (gamma_m) and the exact projection drop
  before the power-type estimate.
- Added the cited S1, S2, and S4 supplementary files to the package and moved
  the one-dimensional source proof to
  `supplement/S4_background_details/A_source_proof_full.tex`.

## Checks

The current PDF has 26 pages. The source graph contains no missing inputs, the
figure graph contains no missing figures, and the loaded sources have no
undefined or duplicate labels. The 19 cited bibliography keys all occur in
`References/references.bib`. The PDF log has no TeX errors, undefined
references, or overfull boxes.

The compiled text contains no `Route A`, `Route B`, revision-history,
workflow, pipeline, campaign, or version-history wording. Terms such as
“computed state” and “iteration index” occur only as mathematical descriptions
of the algorithm or trajectory.
