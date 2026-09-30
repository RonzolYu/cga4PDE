# S3. Sample definitions, unsuccessful solves, and statistics

The RFM curves and endpoint table use the seed-level records in
`code/result/experiments/rfm_multiseed_raw.csv`. The median and quartile records are
`code/result/experiments/rfm_multiseed_summary.csv`; the plotted values are in
`code/result/baselines/baseline_actual_points.csv`, and the endpoint records are in
`code/result/baselines/baseline_terminal.csv`. These are repository-relative paths in
<https://github.com/RonzolYu/cga4PDE>.

The seed-level file contains 313 states: 304 recorded solver successes and nine
unsuccessful solves. Every width included in the main RFM comparison has ten
prescribed and observed seeds, numbered 201–210. All displayed C1–C4 widths have
ten successful solves. The two incomplete C5 widths are:

| Case | Width | Prescribed and observed | Successful solves | Unsuccessful solves | Median/IQR sample |
|---|---:|---:|---:|---:|---:|
| C5, pure p=4, d=2 | 128 | 10 | 7 | 3 | 7 |
| C5, pure p=4, d=2 | 256 | 10 | 4 | 6 | 4 |

At C5, N=128, the unsuccessful seeds are 203, 207, and 208. At N=256, they are
203, 204, 207, 208, 209, and 210. The recorded solver flag determines inclusion;
finite numerical metric values from an unsuccessful solve are retained in the
seed-level records but do not enter the reported median or quartiles. The raw
solver messages are preserved verbatim in `code/review/rfm_state_audit.csv`.
A solver flag is not an independent bound on the PDE residual or the quadrature
error. In particular, a solver message indicating termination can coexist with
`solver_success=False`; the message alone does not change the sample definition.

Statistics are computed separately for each metric, from successful solves with
a finite nonnegative recorded value for that metric. The V-distance is defined
only for C4 and C5 in these comparisons; its unavailable entries in C1–C3 do not
exclude their energy or H1-error observations. Quartiles use linear interpolation
between adjacent sorted observations at indices `(n-1)*0.25` and `(n-1)*0.75`,
with zero-based indexing. The sample counts are preserved separately from the
prescribed denominator. The C5 endpoint comparison is conditional on its four
successful solves and does not support an aggregate ranking across all seeds.

The record also includes ten C4 states at N=256, outside the displayed C4 range,
and three C5 states at N=512. The latter width does not have the prescribed ten
observations and is excluded from the main RFM curves and endpoint statistics.
Consequently, the displayed widths account for 300 prescribed seed attempts,
with 291 successful solves and nine unsuccessful solves.

`code/tools/rebuild_baseline_evidence.py` reconstructs the state ledger,
medians, quartiles, endpoint ratios, and compact denominator table without
running new numerical experiments. The generated evidence files are in
`code/review/`. The script also checks the median, IQR, and comparison
curves encoded in all twelve baseline PDF figures against their numerical input
records. Its dependencies are Python and `pypdf`.
