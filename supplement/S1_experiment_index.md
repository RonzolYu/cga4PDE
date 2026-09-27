# S1. Experiment index

The authoritative machine-readable index is `manifest/experiment_index_v2.csv`; the legacy
`manifest/experiment_index.csv` is retained with the same rows for compatibility. Each row has
the fields `case_id`, `run_id`, `model`, `dim`, `p`, `relu_power`, `epsilon`, `seed`,
`target_accepted`, `config_path`, and `status`. Every `config_path` is relative to the repository
root and resolves inside `data/raw/`. `manifest/figure_table_map.csv` maps manuscript figures
and tables to their producing records.

Run the index check from the repository root:

```bash
python code/build_experiment_index.py
```

The current index contains 20 rows and should finish with `PASS`. The complete code, raw inputs,
saved states, results, and supplementary figures are distributed in this repository; no sibling
workspace or external Material directory is required for path resolution.
