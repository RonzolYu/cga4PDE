# Current source and availability check

Date: 2026-10-02

The numerical-source scan checks the forbidden editorial-process phrases in
`scripts/audit_numeric_values.py`. The manuscript uses descriptive names for
the linear, cubic, hyperbolic-sine, and pure quartic problems. The abstract,
comparison discussion, figure captions, table captions, conclusion and code
availability statement use descriptions of approximations and coefficient
optimization instead of internal batch and archive terminology.

The rendered 26-page PDF contains no experimental case identifiers C1--C5,
`review_replay_20261001`, or the phrase `P2 states`. P1, P2 and P3 are explicitly
defined as piecewise linear, quadratic and cubic Lagrange elements. Generated
comparison and denominator tables use descriptive problem names; the reproduction
acceptance check verifies both regenerated tables and extracted PDF text.
Exact directory names and computational case keys remain in the reproduction
instructions, configurations and data files so the published results remain locatable.

The manuscript and regenerated tables contain no revision-highlighting commands;
regenerated plots have white backgrounds. The acceptance check covers clean
artifacts and compilation after regeneration.

The existing title, authors, affiliations, and bibliography database are
unchanged. The repository URL is an actual annotation in the compiled PDF.
Missing inputs, labels, citations, and graphics are checked in
`structure_check_sisc.json`. The historical language audit is retained in
`historical_20260930/` and is not current evidence.
