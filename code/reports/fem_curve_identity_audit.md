# FEM C1/C2 curve-identity audit

## Result

The C1 and C2 FEM curves are not identical in the unrounded data.  They become visually indistinguishable at high resolution because the two cases use the same manufactured profile and their relative H1 errors are dominated by the same finite-element approximation component.  Separate model files and nonzero high-precision differences exclude a duplicated plotted column.

| Degree | Matched DOFs | Maximum absolute difference | Difference at largest matched DOF | Exactly identical? |
|---|---:|---:|---:|---|
| P1 | 7 | 2.40351447e-06 | 3.23440254e-10 | no |
| P2 | 7 | 8.82108352e-05 | 1.59784512e-13 | no |
| P3 | 8 | 2.61788258e+01 | 1.33942065e-14 | no |

## Provenance checks

- C1 and C2 carry different problem hashes in `fem_baseline_raw.csv`.
- P1, P2, and P3 use different mesh-level/DOF sequences, consistent with their polynomial degrees.
- The archived terminal model files for C1 and C2 have distinct SHA-256 hashes for every degree.
- The plot generator reads rows by `(case_id, variant, dof)` and does not reuse a C1 array for C2.

- P1 terminal models: C1 `0a129bb89016302d6ef4234c9171b503808e030fb0c8577fdccab697e78f57b5`, C2 `ff23a24af810ae385f38578c85d304567e0c26776d6a60889e4b536831f493da`.
- P2 terminal models: C1 `3e906ece998320cc9ec672636ca9d559106deef2b218b644cdc44fe60746c4d5`, C2 `b7347b3d647a7bd42346d8a2626dfe71fab5ad300e316c6517d8d106905d08a2`.
- P3 terminal models: C1 `6fe07fde2958685c0e697bcff9fa5cf694dd44c92b109b4ce9cd81d761a9d138`, C2 `8611d7fcda959b420149f6b400a2d048e43f74d742b082c6310bf83b0ae44ee9`.
