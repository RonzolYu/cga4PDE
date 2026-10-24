# S2. Finite-range estimates and continuation

The complete construction and field definitions are in [S2a: pool generation and source coefficients](S2_pool_generation.md). The associated data and configuration are available in the companion repository <https://github.com/RonzolYu/cga4PDE>.

The calibration range is $N=8$--$32$ for the pure and regularized one-dimensional quartic models. Candidate and reference pools, consumed masks, checkpoints, source coefficients, and state files are indexed by `code/config/continuation_protocol.json` and the data manifests. The frozen protocol has SHA-256 `77a43ae6de7b5e2a2addc5d21162a72c214ca49a920fa175bead2f7041152469`.

The trajectories continue from \(N=32\) to \(N=64\) with the calibration constants fixed. At quadrature order 20, the normalized-innovation condition holds on 27 of 32 transitions in the pure model and 28 of 32 in the regularized model. The per-transition values are included in the numerical records.

The projection-drop and quartic identity defects remain below $2\times10^{-12}$ and $10^{-16}$, respectively, in the order-16/20 cross-check.
