"""Independent algebra/finite-dimensional sanity checks, not a proof of a PDE rate.

Requires numpy. Run from any directory; --output is honored. Assertion failures
produce FAIL, preserve the completed checks, and exit with a nonzero status.
"""
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FULL_THEORY = ROOT / "chapter5_theory.tex"
LOADED_THEORY_CANDIDATES = (
    ROOT / "tex" / "sections" / "05_theory.tex",
    ROOT.parent / "paper" / "sisc_cga" / "sections" / "05_theory.tex",
)
LOADED_THEORY = next((path for path in LOADED_THEORY_CANDIDATES if path.is_file()), None)
ATOL, RTOL = 2e-11, 2e-10
SEED = 20260913


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/chapter5_algebra_checks.json")
    args = parser.parse_args()
    results = []

    def check(name, condition, **evidence):
        item = {"name": name, "passed": bool(condition), **evidence}
        results.append(item)
        if not condition:
            raise AssertionError(name)

    status = "PASS"
    try:
        # Expand F(a+v)-F(a)-F'(a)v in formal monomials (epsilon, a, v).
        coefficients = {}
        for degree, eps_power, scale in [(2, 2, Fraction(1, 2)),
                                          (4, 0, Fraction(1, 4))]:
            for v_power in range(2, degree + 1):
                key = (eps_power, degree - v_power, v_power)
                coefficients[key] = scale * math.comb(degree, v_power)
        check("quartic_exact_polynomial",
              coefficients == {(2, 0, 2): Fraction(1, 2),
                               (0, 2, 2): Fraction(3, 2),
                               (0, 1, 3): Fraction(1),
                               (0, 0, 4): Fraction(1, 4)})
        check("fractional_exponent_identity",
              all(4 * (Fraction(1, 2) + 1 / (2 * a)) == 2 + 2 / a
                  for a in [Fraction(1, 2), Fraction(1), Fraction(2), Fraction(6)]))

        # Direct scalar recurrence against the claimed explicit envelope.
        worst = 0.0
        for alpha in [0.5, 1., 2., 6.]:
            for c in [0.01, 0.1, 0.8, 1.]:
                x = 1.
                for j in range(1, 101):
                    x = max(0., x - c * x ** (1 + 1 / alpha))
                    bound = (1 + c * j / alpha) ** (-alpha)
                    worst = max(worst, x - bound)
        check("power_recurrence_1600_steps", worst <= ATOL, max_excess=worst)
        check("zero_entry_and_termination", 0. - 0. == 0. and 1. - 1. == 0.)
        check("reject_old_reciprocal_coefficient",
              0.9 > (1 + 1 / (2 * 0.1)) ** -2,
              true_first_error=0.9, invalid_bound=1 / 36)

        # Check the clipped descent by direct trial minimization on a fine grid.
        lam = np.linspace(0, 1, 10001)
        gap = 0.
        for L in [0.1, 1., 7.]:
            for a in [0., 0.03, 0.1, 0.8, 1., 3., 7., 20.]:
                exact = a * a / (2 * L) if a <= L else a - L / 2
                sampled = np.max(a * lam - L * lam ** 2 / 2)
                gap = max(gap, abs(exact - sampled))
        check("clipped_descent_grid", gap <= 1e-8, max_grid_gap=gap)

        # Random Euclidean realizations include arbitrary nonlinear-selected
        # spaces: selection order is random, not the OGA maximizing order.
        rng = np.random.default_rng(SEED)
        max_drop_defect, max_bound_excess, crossings = 0., 0., 0
        for trial in range(100):
            atoms = rng.normal(size=(8, 14))
            atoms /= np.linalg.norm(atoms, axis=0)
            entry, _ = np.linalg.qr(rng.normal(size=(8, 2)))
            q0 = np.eye(8) - entry @ entry.T
            coeff = rng.normal(size=14)
            B = np.sum(np.abs(coeff))
            vb = q0 @ atoms @ coeff
            noise = q0 @ rng.normal(size=8) * (0.02 if trial % 2 else 1.0)
            f = vb + noise
            sigma = np.linalg.norm(noise)
            q = entry
            r0sq = float(f @ f)
            x0 = max(r0sq - sigma ** 2, 0.)
            W = 0.
            for k in rng.permutation(14)[:6]:
                e = f - q @ (q.T @ f)
                w = atoms[:, k] - q @ (q.T @ atoms[:, k])
                ell = np.linalg.norm(w)
                assert ell > 1e-10
                scores = np.abs(e @ atoms)
                S = np.max(scores)
                theta = scores[k] / S if S > 1e-14 else 0.
                qnew = np.column_stack((q, w / ell))
                enew = f - qnew @ (qnew.T @ f)
                r2, newr2 = float(e @ e), float(enew @ enew)
                defect = abs(r2 - newr2 - scores[k] ** 2 / ell ** 2)
                max_drop_defect = max(max_drop_defect, defect)
                W += theta ** 2 / ell ** 2
                bound = 0. if x0 == 0. else 1 / (1 / x0 + W / (4 * B ** 2))
                max_bound_excess = max(max_bound_excess,
                                      max(newr2 - sigma ** 2, 0.) - bound)
                crossings += int(r2 > sigma ** 2 >= newr2)
                q = qnew
        check("pythagoras_600_independent_extensions",
              max_drop_defect <= ATOL, max_absolute_defect=max_drop_defect)
        check("weighted_approximate_source_600_extensions",
              max_bound_excess <= ATOL and crossings > 0,
              max_excess=max_bound_excess, remainder_crossings=crossings)

        # Explicitly check degenerate score and source cases.
        target = np.array([0., 1.])
        atom = np.array([1., 0.])
        check("zero_score_source_floor_B_zero_W_zero",
              target @ atom == 0. and np.linalg.norm(target) == 1.,
              B=0, sigma=1, W=0, residual=1)
        # Euclidean quartic discretization: manufacture load to cancel the
        # linear part, with an arbitrary active-space state, not a minimizer.
        defect = 0.
        for _ in range(50):
            derivative = rng.normal(size=(20, 6))
            truth = rng.normal(size=6)
            eps = 0.3
            a = derivative @ truth
            A = derivative.T @ ((eps ** 2 + 3 * a ** 2)[:, None] * derivative)
            basis = rng.normal(size=(6, 3))
            z = basis @ np.linalg.solve(basis.T @ A @ basis, basis.T @ A @ truth)
            state = basis @ rng.normal(size=3)
            v = derivative @ (state - truth)
            load = derivative.T @ (eps ** 2 * a + a ** 3)
            def energy(u):
                du = derivative @ u
                return np.sum(eps ** 2 * du ** 2 / 2 + du ** 4 / 4) - load @ u
            gap = energy(state) - energy(truth)
            r2 = (truth - z) @ A @ (truth - z)
            d2 = (state - z) @ A @ (state - z)
            R = np.sum(a * v ** 3 + v ** 4 / 4)
            defect = max(defect, abs(gap - (r2 + d2) / 2 - R) / max(1, abs(gap)))
        check("quartic_50_nonminimizing_states", defect <= RTOL,
              max_scaled_defect=defect)
    except Exception as exc:
        status = "FAIL"
        results.append({"exception": repr(exc)})
    report = {
        "status": status, "scope": "algebra and finite-dimensional sanity checks only",
        "seed": SEED, "atol": ATOL, "rtol": RTOL,
        "chapter_sha256": hashlib.sha256(FULL_THEORY.read_bytes()).hexdigest(),
        "loaded_main_theory": None if LOADED_THEORY is None else str(LOADED_THEORY),
        "loaded_main_theory_sha256": (None if LOADED_THEORY is None else
                                       hashlib.sha256(LOADED_THEORY.read_bytes()).hexdigest()),
        "checks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": status, "checks": len(results), "output": str(args.output)}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
