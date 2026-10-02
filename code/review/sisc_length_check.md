# Final clean manuscript build

Date: 2026-10-02

`paper/sisc_cga/main.tex` compiles to 26 pages with the existing SIAM class,
font sizes, and margins. The current abstract word count and counting method are
recorded in `structure_check_sisc.json`. The final TeX log has no errors, overfull boxes, undefined
references/citations, or duplicate labels. The source dependency check passes:
13 loaded files, 88 labels, 66 references, 30 citation occurrences, and 9 graphics.

The public repository URL occurs as an actual PDF link annotation. The source,
generated tables, and figure backgrounds use clean formatting without revision
highlights. All 26 pages were rendered and inspected in contact sheets; pages
1, 22, 23, and 25 were also inspected individually at 145 dpi. Compilation is recorded in
`structure_check_sisc.json`; it does not certify the scientific claims.

The manuscript uses descriptive problem names in its abstract, comparisons,
captions, conclusion and code availability statement. The rendered PDF contains
no C1--C5 experimental identifiers, internal batch name, or `P2 states` wording.
P1, P2 and P3 are explicitly defined as Lagrange-element polynomial degrees.
Exact computational directory names remain in the reproduction instructions.

The acceptance run passes 15 checks, including clean output formatting,
descriptive names in regenerated comparison tables and the rendered PDF, and a
fresh manuscript build after regenerating missing figures and tables in a
disposable package. All 61 manifested products and 93 exported artifact-mode
outputs match the adopted files. The 12 baseline vector PDFs match the current
CSV data, and all 466 archived model hashes pass verification.

The public inventory and `SHA256SUMS` are refreshed after the PDF and audit
records. `python code/scripts/build_public_inventory.py --check` checks current
package files, artifact outputs and inputs, and PDF sidecar manifests.

The pre-revision length/compression audit remains in `historical_20260930/`.
