# S2a. Candidate pools, reference pools, and source coefficients

This supplement corresponds to the `base_pure` and `epsilon_01` one-dimensional
quartic models. All paths below are relative to the companion repository
<https://github.com/RonzolYu/cga4PDE> and are resolved from its root. The construction
distinguishes solver normalization from the frozen-Hessian normalization used in Section 5.

## 1. Pool construction and ordering

The manufactured solution is $u^*(x)=\cos(2\pi x)$, with $k=3$ and
$\varepsilon=0$ or $0.1$. The configuration requests 512 candidate atoms and 1024
reference atoms, with seeds 201 and 1210 (201+1009), respectively. The one-dimensional
generator uses stratified breakpoints: the crossover count is `round(0.9 size)`, with
alternating $+1$ and $-1$ perturbations and $b=-wt$. The remaining atoms use breakpoints
$-0.05$ or $1.05$ as the two full-interval anchors.

The generator randomly permutes each array using its corresponding seed, then removes
duplicates by the parameter-quantization key at scale $10^{-12}$, retaining the first
occurrence. The resulting arrays contain 461+2=463 candidates and 922+2=924 reference
atoms. The permuted first-occurrence order is retained; atoms are not reordered by score or
error. `original_indices` stores the pre-deduplication indices and is reconstructed and
matched elementwise against the archived parameters.

For pure and regularized $p$ models, solver `scales` are gradient $L^p$ norms and the
zero-mean correction is stored in `centers`. An atom is usable when its scale is finite and
larger than $10^{-14}+10^{-12}\max(1,\max\mathrm{scales})$. `candidate_consumed` in a
checkpoint contains both accepted and rejected candidates; it is not the accepted set.

## 2. Frozen projection dictionary

The analysis uses all finite candidates with `scales>1e-12` (463 atoms) and the first 128
reference atoms satisfying the same condition, in archived-array order. “Reference
independent” means generated with a separate seed; it does not mean linearly independent.
The numerical matrix has 591 columns. The symmetric dictionary is represented by these
directions and their negatives through absolute scores and signed source coefficients, so
negative columns need not be copied explicitly. Directions shared by different pools are
retained with their original indices.

Define
\[
 a_*(v,w)=\int_0^1(\varepsilon^2+3|(u^*)'|^2)v'w'\,dx.
\]
Each direction is normalized to unit length in this inner product. After projection
$Q_0d=(I-P_8)d$, the remainder is not renormalized. The files
`result/section85/*_source_q20.npz` store `norms`, `candidate_indices`,
`reference_indices`, and `coefficients`, with candidate columns preceding reference columns.
The accepted directions for both models and $N=1,\ldots,64$ occur in the 463-candidate
pool, which is the fixed-dictionary condition used in the Section 5 estimates. Reference
atoms enter the estimates only; they are not selected by the solver.

## 3. Source coefficients and stepwise quantities

At $N=8$, construct $F_0=(I-P_8)F$ and $e_8=(I-P_8)u^*$. Warm-start coordinate descent
with penalties $10^{-2},10^{-4},10^{-6}$ in that order, using at most 2500 sweeps per
penalty and stopping when the maximum coordinate change is below $10^{-10}$. The final
penalty is fixed in advance rather than selected from the terminal error. Every archived
stage reaches the 2500-sweep limit, so no exact LASSO optimum is claimed.

The theory uses the explicit coefficient vector $a$: $B=\sum_i|a_i|$ and
$\sigma=\|e_8-F_0a\|_*$. The $q20$ source vector and its hash are fixed by the continuation
protocol; no source or decrease constant is re-estimated for $N=33,\ldots,64$.

For each state, form $P_N$ with a complete economic QR, set $e_N=(I-P_N)u^*$ and
$r_N^2=\|e_N\|_*^2$. Define
\[
S_N=\max_i|a_*(e_N,d_i)|,\qquad
\theta_N=|a_*(e_N,d_{N+1})|/S_N,\qquad
\ell_N=\|(I-P_N)d_{N+1}\|_*,
\]
and $W_j=\sum_{N=8}^{8+j-1}\theta_N^2/\ell_N^2$. Zero-score and zero-remainder
branches are stated in `tex/sections/05_theory.tex`.

## 4. Breakpoint quadrature and quartic identity

The quadrature breakpoints are the union of all pool breakpoints lying in $(0,1)$ and
$0.5$, together with the endpoints 0 and 1. On each segment use Gauss quadrature of order
16 and 20. The analysis keeps the full active rank; if the smallest QR singular value is
below $10^{-13}$ times the largest, the computation stops with an error.

Let $h=\widehat G_N'-(u^*)'$ and $D_N^2=\|\widehat G_N-P_Nu^*\|_*^2$. The quartic remainder
$R_N=\int[(u^*)'h^3+h^4/4]$ satisfies
$\delta_N=(r_N^2+D_N^2)/2+R_N$. Here $\delta_N$ is the energy gap, not $r_N^2$.
The quantities $q_N=\sqrt{D_N^2/r_N^2}$ and
$\rho_N=|R_N|/(r_N^2+D_N^2)$ depend on the manufactured solution; they are numerical
quantities, not automatic stopping rules. Once a state is fixed, no additional optimization
error is added.

## 5. File locations and verification

- Original pools and $N\le32$ states: the `runs.*.parent` entries in
  `config/revision_protocol.json`.
- Source coefficients and scalars: `result/section85/*_source_q20.npz` and
  `*_sources_q20.json`.
- Recomputed source data: `data/derived/section85/`.
- Continuation states: `data/raw/revision_20260914/{base_pure,epsilon_01}/states/`.
- Integrity verification: `python code/verify_revision_prefix.py` from the repository root.
- Verification output: `result/revision_20260914/prefix_verification.json`.

All successful checks are floating-point reconstruction and file-integrity evidence. They do
not provide continuous-dictionary certification or an asymptotic convergence guarantee.
