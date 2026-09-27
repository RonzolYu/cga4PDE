# S2. Finite-window certificate and continuation

The complete construction and field definitions are in [S2a: pool generation and source coefficients](S2_pool_generation.md).
All paths are resolved from the repository root. Candidate and reference pools, consumed masks,
checkpoints, source coefficients, and state files are archived under `data/raw/`; frozen results
are under `result/section85/` and `result/revision_20260914/`.

The calibration window is N=8--32 for the pure and regularized one-dimensional quartic models.
The frozen protocol is `config/revision_protocol.json` (SHA-256
`77a43ae6de7b5e2a2addc5d21162a72c214ca49a920fa175bead2f7041152469`).

A forward continuation to N=64 was performed from the verified N=32 checkpoints. At quadrature
order 20, the normalized-innovation sufficient condition passes 27/32 transitions for the pure
model and 28/32 for the regularized model; the first failed transition is 33 to 34 in both.
This continuation is diagnostic and does not establish a uniform or asymptotic rate.

The projection-drop and quartic identity defects remain below 2e-12 and 1e-16, respectively, in
the order-16/20 cross-check.
