# SISC manuscript package

This directory is the SIAM Journal on Scientific Computing (SISC) package for
the CGA manuscript.  `main.tex` is the compilation entry point and
uses the SIAM class `siamart251216.cls` with the journal review layout.  The
main PDF is 26 pages; the source keeps the SIAM font and page geometry rather
than reducing the type size or margins.

Compile from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd main.tex
```

The main text retains the model setting, the finite-pool descent result, the
natural-distance and local-Hessian transfer statements, the approximate-source
and quartic consequences, representative numerical figures, the endpoint
baseline table, and the discussion.  The duplicated descent chapter, long
derivations, the source-class proof, dyadic diagnostic tables, the epsilon scan,
and detailed finite-trajectory records are not loaded into the main manuscript.
They are preserved in `supplement/` as supporting material so that the main
contribution remains readable while the derivations and numerical records remain
available for inspection.

The numerical code and data used for the reported results are identified in the
main text by the public repository <https://github.com/RonzolYu/cga4PDE>.  The
repository README and the S1--S4 records give the case definitions, parameter
settings, failure statistics, and reproduction information.  The executable
records and audit reports are kept in the sibling `code/` directory of that
repository.

The final numbered section in the main text is
`sections/11_code_data_availability.tex`, placed after the conclusion and before
the references.  The front matter contains no code/data paragraph.

## Supplementary files

- `supplement/S4_theory_details.tex` contains the full theory chapter before
  main-text condensation, and `supplement/S4_background_details/A_source_proof_full.tex`
  contains the one-dimensional source-class proof removed from the main input.
- `supplement/S4_background_details/` contains the longer setting, descent,
  geometry, and energy-family derivations.
- `supplement/S3_failures_and_statistics.md` records the RFM sample definitions
  and failure accounting used to interpret the baseline plots.

The generated tables and figures that are not loaded by `main.tex` are retained
as numerical records; their omission from the main text is recorded in
`../../code/review/sisc_length_check.md`.

## Checks

The current compile and structural audit are recorded in the companion code
directory:

- `../../code/review/sisc_length_check.md`;
- `../../code/review/structure_check_sisc.json`;
- `../../code/review/internal_language_check.md`; and
- `../../code/review/logic_chain_check.md`.

These checks cover page count, missing inputs and figures, labels and citations,
overfull boxes, metadata, and editorial wording.
