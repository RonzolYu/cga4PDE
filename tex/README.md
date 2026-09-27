# CGA submission package

Upload the entire `submit_overleaf.zip` to Overleaf and select the root `main.tex`
with **XeLaTeX**. Do not upload this `tex/` directory alone: the manuscript also uses
the package-root `chapter5_theory.tex`. `tex/main.tex` is the authoritative manuscript;
the root file is only a path-aware entry point.

Local check:

```sh
latexmk -xelatex -g -interaction=nonstopmode -halt-on-error main.tex
```
