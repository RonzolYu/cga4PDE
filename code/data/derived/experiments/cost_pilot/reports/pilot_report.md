# Bounded cost pilot report

- Planned runs: 12
- Completed rows: 12
- Failed runs: 0
- Cases: C1, C2, C4
- Seeds: 201, 202
- Timing repetitions: 2
- Target accepted width: 2

## Scope

This pilot invokes the existing finite-pool CGA smoke runner. The outer monotonic timer measures end-to-end process work. Existing per-attempt selection and projection timers are copied into the phase table. Pool generation, setup, evaluator internals, and I/O remain an explicit unattributed remainder because the source solver was not modified.

The pilot is suitable for validating logging, failure retention, and time-accounting plumbing. It is not a same-machine comparison against RD-WOGA, Random-FC, RFM, or FEM, and it must not be used to claim speed.
