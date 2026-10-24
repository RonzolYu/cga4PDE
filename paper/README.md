# CGA manuscript

`main.tex` is the entry point for the SIAM manuscript. The directory contains the LaTeX sources, bibliography, figures, tables, supplementary definitions, and detailed proofs.

## Compile

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For the detailed theory supplement:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd supplement/S4_theory_details.tex
```

## Contents

- `sections/`: variational setting, PDE models, convergence analysis, numerical experiments, comparisons, discussion, and data availability.
- `figures/` and `generated/`: numerical figures and tables.
- `References/references.bib`: bibliography.
- `supplement/S1_experiment_index.md`: experiment and configuration index.
- `supplement/S2_pool_generation.md` and `S2_window_certificate.md`: source construction and finite-trajectory calculations.
- `supplement/S3_failures_and_statistics.md`: statistical sample definitions and solver records.
- `supplement/S4_reproduction.md`: reproduction commands for the companion code repository.
- `supplement/S4_theory_details.tex`: detailed convergence proofs.
- `supplement/S4_background_details/`: background proofs and model details.

Code, configurations, and numerical records are available in the companion repository: <https://github.com/RonzolYu/cga4PDE>.
