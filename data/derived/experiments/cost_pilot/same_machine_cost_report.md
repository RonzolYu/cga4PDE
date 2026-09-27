# Cost accounting and timing scope

The selection campaign and the FEM/RFM comparison were executed on the same ARM macOS host family, but they expose different timing boundaries. The selection campaign records end-to-end process wall time; the comparison runner records per-state solver/evaluation time. The combined CSV therefore retains a `timing_scope` field and is descriptive only; it is not a cross-method speed ranking.

## Selection-variant runs

CGA-FP, RD-WOGA, and Random-FC use the formal pilot at accepted width 8. The pilot contains 3 seeds for C1, C2, and C4 and 5 seeds for C3 and C5. Success and failure counts are retained in the CSV. Median and quartile wall times are end-to-end process times.

## FEM/RFM comparison timing

FEM P1/P3 and RFM rows are the actual width-eight or nearest available FEM states from the five-case comparison. Their `wall_time_sec` field is a per-state solver/evaluation measurement, not an end-to-end process measurement. FEM nearest-state rows retain their actual DOF and are not interpolated.

## Instrumentation pilot

The bounded instrumentation pilot contains 12 CGA-only timing repetitions at target width 2 for C1, C2, and C4. Its known selection and correction phases are logged, while setup, pool generation, evaluator internals, serialization, and I/O remain an unattributed remainder. These data validate the logging schema and do not support an efficiency conclusion.

## Reproducibility

Raw package-local pilot records: `data/derived/experiments/cost_pilot/raw/`, with the derived
summary in `data/derived/experiments/cost_pilot/pilot_summary.csv`. The original run provenance is
retained in the copied manifest and source campaign records; those historical records may contain
machine-specific absolute paths and are not required by the package-local timing audit.
