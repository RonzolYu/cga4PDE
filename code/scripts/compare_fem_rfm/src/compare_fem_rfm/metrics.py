"""Common error and energy evaluator for FEM, RFM, and CGA."""

from __future__ import annotations

import numpy as np

from .problems import ProblemSpec, energy_density, exact_jet, forcing


def vp_map(z: np.ndarray, p: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        return np.abs(z) ** ((p - 2.0) / 2.0) * z
    norm = np.linalg.norm(z, axis=-1)
    return norm[..., None] ** ((p - 2.0) / 2.0) * z


def evaluate_fields(spec: ProblemSpec, points: np.ndarray, weights: np.ndarray,
                    u: np.ndarray, grad: np.ndarray) -> dict[str, float | None]:
    exact_u, exact_grad, _ = exact_jet(spec, points)
    f = forcing(spec, points)
    e_num = float(np.dot(weights, energy_density(spec, u, grad, f)))
    e_exact = float(np.dot(weights, energy_density(spec, exact_u, exact_grad, f)))
    if spec.natural_metric == "H1":
        numerator = np.dot(weights, (u - exact_u) ** 2 + np.sum((grad - exact_grad) ** 2, axis=1))
        denominator = np.dot(weights, exact_u**2 + np.sum(exact_grad**2, axis=1))
        natural = float(np.sqrt(numerator / denominator))
        v_error = None
    else:
        p = float(spec.p)
        numerator = np.dot(weights, np.sum(np.abs(grad - exact_grad) ** p, axis=1))
        denominator = np.dot(weights, np.sum(np.abs(exact_grad) ** p, axis=1))
        natural = float((numerator / denominator) ** (1.0 / p))
        va, vb = vp_map(grad, p), vp_map(exact_grad, p)
        v_error = float(np.sqrt(np.dot(weights, np.sum((va - vb) ** 2, axis=1)) /
                                np.dot(weights, np.sum(vb**2, axis=1))))
    l2 = float(np.sqrt(np.dot(weights, (u - exact_u) ** 2) /
                       np.dot(weights, exact_u**2)))
    return {
        # Keep the sign: a negative quadrature gap is an evaluation diagnostic,
        # not a positive approximation error.  Statistical validity is checked
        # separately for each metric by the shared quality policy.
        "energy_gap": e_num - e_exact,
        "energy_gap_signed": e_num - e_exact,
        "natural_error": natural,
        "v_error": v_error,
        "l2_error": l2,
        "numerical_energy": e_num,
        "exact_energy": e_exact,
    }


def max_relative_metric_delta(a: dict, b: dict) -> float:
    deltas = []
    for key in ("energy_gap", "natural_error", "v_error"):
        if a.get(key) is None or b.get(key) is None:
            continue
        deltas.append(abs(float(a[key]) - float(b[key])) / max(abs(float(b[key])), 1e-300))
    return max(deltas, default=0.0)


def relative_metric_deltas(a: dict, b: dict) -> dict[str, float | None]:
    result = {}
    for key in ("energy_gap", "natural_error", "v_error"):
        if a.get(key) is None or b.get(key) is None:
            result[key] = None
        else:
            result[key] = abs(float(a[key]) - float(b[key])) / max(abs(float(b[key])), 1e-300)
    return result
