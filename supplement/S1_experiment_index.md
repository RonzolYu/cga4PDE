# S1. Experiment index

The supplementary index is `manifest/experiment_index_v2.csv` (the legacy
`experiment_index.csv` is retained for compatibility). Each row has a stable case ID,
source-data path, entry point, dimension, polynomial degree, activation, regularization,
seed, accepted-count target, and output directory. `manifest/figure_table_map.csv` maps
every manuscript figure and table to its producing case or frozen result.

The repository manifest and the associated numerical records are available at
<https://github.com/RonzolYu/cga4PDE>. The index-building and validation commands are
specified in S4 and are run from the repository root. The reported index contains 20
rows; a successful build terminates with `PASS`.
