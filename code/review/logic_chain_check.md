# Consistency revision scope

Date: 2026-10-01

The main theoretical statements and canonical full proofs are unchanged.
The stale duplicate full theory is replaced by a relative symlink to
`paper/sisc_cga/supplement/S4_theory_details.tex`. The ten algebra checks pass;
this is a consistency check, not a new formal verification of all proofs.

The numerical definitions now distinguish the componentwise gradient Lp
seminorm used for pure/regularized cases from the full reaction-case W1p norm.
The PDE energy and V-map keep Euclidean gradient geometry. Poincare/norm
equivalence transfers rate exponents, without identifying relative error values.

Calibration on N=8--32 and continuation to N=64 have different claim scopes.
The 27/32 and 28/32 continuation pass counts are empirical checks and cannot
extend the same fixed-constant theorem guarantee through failed transitions.

Primary comparisons use newly fitted FEM/RFM states and refitted fixed CGA
prefixes. These are separate from original CGA training trajectories. The same
solve/sign/metric-specific one-percent quadrature policy applies to every method.
Incomplete RFM samples remain conditional, with ten prescribed seeds retained.
The endpoint table and conclusion no longer assert uniform CGA dominance:
The interpolated FEM P3 reference has a slightly smaller error at C4 N=128
under the revised evaluator; it is not a direct fit at exactly that DOF.

The numerical archive and fresh reevaluation evidence are reported separately
from artifact reconstruction. Earlier compression claims are historical records.
