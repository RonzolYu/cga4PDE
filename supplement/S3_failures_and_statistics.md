# S3. Failures and statistics

The manuscript's primary comparison uses the unified 2026-09-22 evaluation. It retains
617 stored states: 612 newly generated states and five archived terminal states. All 617
satisfy the solver-specific acceptance rule and the independent coefficient-count check.
The stricter signed energy/metric evaluation admits 611/617 states, the common
$H^1$-dual residual threshold admits 394/617, and the independent two-dimensional Bregman
calculation covers 137/138 states. These are separate filters and retain their own
­denominators.

For the random-feature comparison the prescribed denominator is ten seeds at every configured
width. Missing rank-budget states remain failures in that denominator: three C3 states at
$N=256$, one C5 state at $N=128$, and two C5 states at $N=256$. A separate C5 $N=256$
seed-203 evaluation failure is retained in the numerical records and is not silently converted
into a successful observation. The displayed medians and interquartile ranges use only
available states that pass the metric-specific check.

A legacy seed-level file from the 2026-09-14 campaign is also archived in the repository. It
has 313 attempted rows (304 successful solves and 9 failures, all in C5: three at DOF 128 and
six at DOF 256). Those numbers are not the denominator for the primary manuscript tables; they
are retained only for provenance and historical comparison. The raw files and status messages
are indexed by the repository manifests. The supplementary numerical data are available at
<https://github.com/RonzolYu/cga4PDE>.
