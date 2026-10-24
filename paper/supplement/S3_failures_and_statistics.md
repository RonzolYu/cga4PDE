# S3. Samples, solve failures, and metric-specific statistics

The comparison fits CGA coefficients on fixed greedy prefixes and fits FEM and RFM coefficients under the common sampling and evaluation protocol specified in S4.

The RFM file has 320 observations and 10 recorded solve failures. Every width has ten prescribed and observed seeds, 201–210. Solve success and metric validity are distinct.

An observation enters an energy, Sobolev, or V-distance statistic when the solve succeeds, the metric is finite and nonnegative, and successive evaluation quadratures differ by at most 0.01 in relative value. This policy applies to CGA, FEM, and RFM. Raw energy differences retain their sign. Each metric uses its own valid sample.

C1–C3 use relative H1 errors. C4–C5 use the relative componentwise gradient L4 seminorm and the relative V-distance. Auxiliary L2 and timing records summarize successful solves.

| Case | Width | Prescribed | Solved | Failed | Valid energy | Valid Sobolev | Valid V |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 8 | 10 | 10 | 0 | 10 | 10 | 0 |
| C1 | 16 | 10 | 10 | 0 | 10 | 10 | 0 |
| C1 | 32 | 10 | 10 | 0 | 10 | 10 | 0 |
| C1 | 64 | 10 | 10 | 0 | 10 | 10 | 0 |
| C1 | 128 | 10 | 10 | 0 | 10 | 10 | 0 |
| C1 | 256 | 10 | 10 | 0 | 10 | 10 | 0 |
| C2 | 8 | 10 | 10 | 0 | 10 | 10 | 0 |
| C2 | 16 | 10 | 10 | 0 | 10 | 10 | 0 |
| C2 | 32 | 10 | 10 | 0 | 10 | 10 | 0 |
| C2 | 64 | 10 | 10 | 0 | 10 | 10 | 0 |
| C2 | 128 | 10 | 10 | 0 | 10 | 10 | 0 |
| C2 | 256 | 10 | 10 | 0 | 10 | 10 | 0 |
| C3 | 8 | 10 | 10 | 0 | 10 | 10 | 0 |
| C3 | 16 | 10 | 10 | 0 | 10 | 10 | 0 |
| C3 | 32 | 10 | 10 | 0 | 10 | 10 | 0 |
| C3 | 64 | 10 | 10 | 0 | 10 | 10 | 0 |
| C3 | 128 | 10 | 10 | 0 | 10 | 10 | 0 |
| C3 | 256 | 10 | 10 | 0 | 10 | 10 | 0 |
| C3 | 512 | 10 | 10 | 0 | 9 | 9 | 0 |
| C4 | 8 | 10 | 10 | 0 | 10 | 10 | 10 |
| C4 | 16 | 10 | 10 | 0 | 10 | 10 | 10 |
| C4 | 32 | 10 | 10 | 0 | 10 | 10 | 10 |
| C4 | 64 | 10 | 10 | 0 | 10 | 10 | 10 |
| C4 | 128 | 10 | 10 | 0 | 10 | 10 | 10 |
| C4 | 256 | 10 | 10 | 0 | 1 | 10 | 10 |
| C5 | 8 | 10 | 10 | 0 | 10 | 10 | 10 |
| C5 | 16 | 10 | 10 | 0 | 10 | 10 | 10 |
| C5 | 32 | 10 | 10 | 0 | 10 | 10 | 10 |
| C5 | 64 | 10 | 10 | 0 | 10 | 10 | 10 |
| C5 | 128 | 10 | 9 | 1 | 9 | 9 | 9 |
| C5 | 256 | 10 | 7 | 3 | 6 | 6 | 6 |
| C5 | 512 | 10 | 4 | 6 | 4 | 2 | 4 |

Quartiles use linear interpolation at zero-based positions `(n-1)*p`, with p=0.25, 0.5, 0.75. Each reported valid count uses the ten prescribed seeds as its denominator.

Primary raw CSVs are `code/data/derived/experiments/rfm_multiseed_raw.csv`,
`cga_baseline_raw.csv`, and `fem_baseline_raw.csv`. Every row records a
model path/hash, evaluation rule/hash, metric audits, and exclusion reasons.
There are 320 RFM, 32 CGA-prefix, and 114 FEM coefficient models.
The RFM summary, plotted actual points, endpoints, and interpolation brackets
are in `code/result/experiments/` and `code/result/baselines/`.

`code/tools/rebuild_baseline_evidence.py` verifies the stored statistics and the twelve comparison figures. S4 gives the commands for loading coefficient models, evaluating errors, and performing coefficient fits.
