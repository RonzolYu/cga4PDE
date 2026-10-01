# S3. Samples, solve failures, and metric-specific statistics (2026-10-01)

The revised comparison uses the fully archived `review_replay_20261001` batch,
not the historical comparison inputs. CGA coefficients are refitted on fixed
greedy prefixes; FEM and RFM coefficients are newly fitted under the frozen
`code/config/review_replay.json` protocol. Original CGA training trajectories
and auxiliary finite-range diagnostics remain separate.

The RFM file has 320 observations and 10 recorded solve failures. Every width has ten prescribed and observed seeds, 201–210. Solve success and metric validity are distinct.

An observation contributes to an energy, Sobolev, or V statistic only if its
solve succeeds, its metric is finite and nonnegative, and its successive
evaluation-quadrature relative difference is at most 0.01. This policy is
the same for CGA, FEM, and RFM. Missing or failed audits exclude that metric.
Energy gaps retain their sign; negative gaps are retained raw and excluded
from energy summaries. An energy exclusion does not itself exclude a stable
Sobolev or V observation. Quadrature agreement is not a rigorous enclosure.

C1–C3 use relative H1 errors. C4–C5 use the componentwise gradient L4
seminorm, not the relative full W1,4 norm. The V metric is unavailable for
C1–C3 without affecting their other statistics. Auxiliary L2 and timing
summaries use successful solves; they are not primary audited comparisons.

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

Quartiles use linear interpolation at zero-based positions `(n-1)*p`,
with p=0.25, 0.5, 0.75. Groups with fewer than ten valid observations are
conditional samples; exclusions remain in the prescribed denominator.

Primary raw CSVs are `code/data/derived/experiments/rfm_multiseed_raw.csv`,
`cga_baseline_raw.csv`, and `fem_baseline_raw.csv`. Every row records a
model path/hash, evaluation rule/hash, metric audits, and exclusion reasons.
There are 320 RFM, 32 CGA-prefix, and 114 FEM coefficient models.
The RFM summary, plotted actual points, endpoints, and interpolation brackets
are in `code/result/experiments/` and `code/result/baselines/`.

`code/tools/rebuild_baseline_evidence.py` checks stored statistics and all
twelve vector baseline figures. This is not an independent PDE solve.
Fresh loading and reevaluation checks are described in S4. Historical
inputs remain in `code/data/raw/historical_baseline_202609/` and do not
supply these revised samples.
