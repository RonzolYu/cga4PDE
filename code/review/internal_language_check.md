# Internal-language and final manuscript check (SISC-length package)

Date: 2026-09-30

## Scope

The check covers the sources loaded by `main.tex` and the compiled
`main.pdf`. It detects editorial-process language that would look like an
internal revision record, including `Route A`, `Route B`, revision/version
references, workflow or pipeline descriptions, campaign/run bookkeeping, and
references to previously computed results.

## Result

The compiled 26-page manuscript contains no `route`, `Route A`, or `Route B`
wording and no revision-history, version-history, workflow, pipeline, campaign,
run-summary, trial-counter, bookkeeping, or previously-computed language.

The remaining matches from a broad lexical scan are mathematical or
bibliographic statements and are retained:

- “iteration index,” “finite prefix,” “later errors,” and “current space”
  specify indices or parts of a theorem;
- “previous theoretical work” is a literature attribution;
- “fixed pool,” “fixed constants,” and “restarted spaces” define algorithmic
  hypotheses of the finite-trajectory estimates.

The full derivations and numerical records in `supplement/` and the audit
reports are not loaded into the manuscript. Their provenance notes do not
appear in the compiled paper.

## Related final checks

- `main.pdf` compiles with pdfLaTeX to 26 pages.
- The log contains no TeX error, undefined citation/reference, duplicate label,
  or overfull box. It contains only benign underfull vertical-box notices.
- The current structure audit reports no missing input files, missing figures,
  duplicate labels, undefined labels, or undefined citations.
- The title, author order, affiliations, e-mail addresses, and corresponding
  author marker in the PDF agree with `main.tex`.
- The code-and-data paragraph appears as Section 8 at the end of the main text,
  immediately before the references, and points to the public repository without an internal release tag.
