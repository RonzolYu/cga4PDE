# Figure and table source map

| Paper object | Generated object | Data source and rule |
|---|---|---|
| Main semilinear CGA figure | `cga_linear_cubic.pdf` | Dyadic accepted states in `cga_metrics_long.csv`; fixed exponent and dimension-dependent start |
| Main pure-p CGA figure | `cga_p4_representative.pdf` | Dyadic accepted states in `cga_metrics_long.csv`; fixed exponent and dimension-dependent start |
| Low-regularity CGA figure | `cga_p5_boundary.pdf` | Dyadic accepted states in `cga_metrics_long.csv` |
| Finite-window diagnostics | `finite_window_diagnostics.pdf`, `window_diagnostics.tex` | Stepwise diagnostic CSV files at N=8--32 |
| Baseline error figures | `C1--C3_relative_h1_error.pdf`, `C4--C5_relative_w1p_error.pdf` | Actual points in `baseline_actual_points.csv`; RFM median and IQR |
| Baseline V-distance figures | `C4--C5_v_distance.pdf` | Actual points in `baseline_actual_points.csv` |
| Baseline endpoint table | `baseline_terminal.tex` | `baseline_common_grid.csv`, with interpolation brackets recorded |
| FEM P2 supplement | `fem_p2_supplement.pdf`, `fem_pdegree.tex` | Actual FEM states and bracketed endpoint values |
| Complete dyadic tables | `ch8_all_dyadic_supplement.tex` | All available dyadic checkpoints in `cga_metrics_long.csv` |
| Finite-trajectory bounds | `finite_trajectory_bounds.pdf`, `finite_trajectory_conditions.tex`, `finite_trajectory_values.tex` | Entry-only source construction and all accepted states in `section85/*_finite_q20.csv` |
