# SISC length and compression check

Date: 2026-09-30

## Outcome

The current `paper/sisc_cga_s/main.tex` compiles with the SIAM article class to
26 pages. The previous `paper/sisc_cga` copy compiled to 42 pages. The package
therefore stays at the SISC hard length screen of 26 journal pages, while
remaining close to the journal's recommended maximum of 20 pages. The official
instructions are available at
<https://epubs.siam.org/journal/sisc/instructions-for-authors>; the editorial
policy is at <https://epubs.siam.org/sisc/editorial-policy>.

No font-size or margin reduction was used. Figures remain embedded in the main
text and the abstract remains within the journal limit.

## Main-text changes

1. The duplicated inexact-descent section was removed from the input sequence.
   Its essential statement is included in Section 4.1 of the condensed theory.
2. The theory chapter was reduced to the statements and proof skeletons needed
   for the main contribution: finite-pool descent, residual comparison,
   energy--natural-distance conversion, local Hessian transfer, approximate
   source, and the quartic consequence. The full derivations are retained in
   `supplement/S4_theory_details.tex`.
3. Long background proofs and the Hessian classification table were shortened.
   The original derivations remain in
   `supplement/S4_background_details/`.
4. The numerical methodology keeps the representative case table, the CGA
   figures, and the finite-range interpretation, but removes generated dyadic
   tables and the epsilon-scan figure from the main input.
5. The baseline section keeps the representative Sobolev figures, the natural
   distance figure, and the endpoint table. The long protocol and fixed-
   denominator tables are preserved as numerical records rather than loaded
   into the article.
6. The source-class appendix is supplied as
   `supplement/S4_background_details/A_source_proof_full.tex` and is not counted
   as a main-text appendix.

These edits remove repetition and bookkeeping while leaving the main results,
their assumptions, the proof logic, and the reported representative evidence
in the article itself.

## Verification

The latest build command is:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd main.tex
```

The build exits successfully and reports 26 pages. The structural audit records
no missing inputs or figures, undefined labels or citations, or duplicate
labels in `review/structure_check_sisc.json`; the post-compression logic audit
is recorded in `review/logic_chain_check.md`. The log has no TeX errors or
overfull boxes; only benign underfull vertical-box notices remain. The final
language check in `review/internal_language_check.md` finds no internal
revision labels such as Route A/Route B, no version-history or workflow
language, and no previously-computed-result bookkeeping in the compiled text.
