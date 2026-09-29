# S2. Finite-range estimates and continuation

The complete construction and field definitions are in [S2a: pool generation and source coefficients](S2_pool_generation.md). The associated data and configuration are available in the companion repository <https://github.com/RonzolYu/cga4PDE>.

The calibration range is $N=8$--$32$ for the pure and regularized one-dimensional quartic models. Candidate and reference pools, consumed masks, checkpoints, source coefficients, and state files are indexed by `config/revision_protocol.json` and the data manifests. The frozen protocol has SHA-256 `77a43ae6de7b5e2a2addc5d21162a72c214ca49a920fa175bead2f7041152469`.

A forward continuation to $N=64$ was performed from the verified $N=32$ checkpoints. At quadrature order 20, the normalized-innovation sufficient condition holds for 27/32 transitions in the pure model and 28/32 in the regularized model; the first transition not satisfying it is $33\to34$ in both. This continuation is numerical evidence on a finite index range and does not establish a uniform or asymptotic rate.

The projection-drop and quartic identity defects remain below $2\times10^{-12}$ and $10^{-16}$, respectively, in the order-16/20 cross-check.
