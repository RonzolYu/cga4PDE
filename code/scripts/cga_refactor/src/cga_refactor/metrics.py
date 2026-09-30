"""Error, energy-gap, natural, and quasi metrics."""

from __future__ import annotations

import numpy as np


def bregman_divergence(problem: object, u: np.ndarray, grad_u: np.ndarray,
                       exact: tuple[np.ndarray, np.ndarray, np.ndarray],
                       rule: object) -> float:
    """Evaluate the Bregman divergence of the energy's convex part.

    The forcing term is affine and therefore cancels from the divergence.  The
    returned orientation is ``D_Phi((u,grad u),(u*,grad u*))``.
    """
    u_exact, grad_exact, _ = exact
    grad_sq = np.sum(grad_u * grad_u, axis=1)
    exact_grad_sq = np.sum(grad_exact * grad_exact, axis=1)

    if problem.name in {"linear", "cubic", "sinh"}:
        grad_potential = 0.5 * grad_sq
        exact_grad_potential = 0.5 * exact_grad_sq
        exact_flux = grad_exact
    elif problem.name == "regularized_p":
        eps = float(problem.epsilon)
        grad_potential = ((eps * eps + grad_sq) ** (problem.p / 2.0)
                          - eps ** problem.p) / problem.p
        exact_grad_potential = ((eps * eps + exact_grad_sq) ** (problem.p / 2.0)
                                - eps ** problem.p) / problem.p
        exact_flux = ((eps * eps + exact_grad_sq) ** ((problem.p - 2.0) / 2.0)
                      )[:, None] * grad_exact
    else:
        grad_potential = grad_sq ** (problem.p / 2.0) / problem.p
        exact_grad_potential = exact_grad_sq ** (problem.p / 2.0) / problem.p
        exact_flux = exact_grad_sq[:, None] ** ((problem.p - 2.0) / 2.0) * grad_exact

    density = (grad_potential - exact_grad_potential
               - np.sum(exact_flux * (grad_u - grad_exact), axis=1))
    if problem.name == "linear":
        density += 0.5 * u**2 - 0.5 * u_exact**2 - u_exact * (u - u_exact)
    elif problem.name == "cubic":
        density += (0.25 * u**4 - 0.25 * u_exact**4
                    - u_exact**3 * (u - u_exact))
    elif problem.name == "sinh":
        density += (np.cosh(u) - np.cosh(u_exact)
                    - np.sinh(u_exact) * (u - u_exact))
    elif problem.name == "reaction_p":
        density += (np.abs(u)**problem.p / problem.p
                    - np.abs(u_exact)**problem.p / problem.p
                    - np.abs(u_exact)**(problem.p - 2.0) * u_exact * (u - u_exact))
    return float(np.dot(rule.weights, density))


def _pair(absolute: float, denominator: float) -> dict[str, float | str]:
    if denominator <= np.finfo(float).tiny:
        return {"absolute": absolute, "relative": float("nan"), "status": "zero_reference"}
    return {"absolute": absolute, "relative": absolute / denominator, "status": "ok"}


def l2_error(u: np.ndarray, u_exact: np.ndarray, rule: object) -> dict[str, float | str]:
    absolute = float(np.sqrt(np.dot(rule.weights, (u - u_exact) ** 2)))
    denominator = float(np.sqrt(np.dot(rule.weights, u_exact**2)))
    return _pair(absolute, denominator)


def h1_error(u: np.ndarray, grad_u: np.ndarray, exact: tuple[np.ndarray, np.ndarray, np.ndarray],
             rule: object) -> dict[str, float | str]:
    u_exact, grad_exact, _ = exact
    absolute = float(np.sqrt(np.dot(rule.weights, (u-u_exact)**2 +
                                    np.sum((grad_u-grad_exact)**2, axis=1))))
    denominator = float(np.sqrt(np.dot(rule.weights, u_exact**2 +
                                       np.sum(grad_exact**2, axis=1))))
    return _pair(absolute, denominator)


def w1p_error(u: np.ndarray, grad_u: np.ndarray,
              exact: tuple[np.ndarray, np.ndarray, np.ndarray], p: float,
              rule: object, gradient_only: bool) -> dict[str, float | str]:
    u_exact, grad_exact, _ = exact
    density = np.sum(np.abs(grad_u-grad_exact)**p, axis=1)
    reference = np.sum(np.abs(grad_exact)**p, axis=1)
    if not gradient_only:
        density += np.abs(u-u_exact)**p
        reference += np.abs(u_exact)**p
    absolute = float(np.dot(rule.weights, density) ** (1.0/p))
    denominator = float(np.dot(rule.weights, reference) ** (1.0/p))
    return _pair(absolute, denominator)


def vp_map(z: np.ndarray, p: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        return np.abs(z) ** ((p-2.0)/2.0) * z
    norm = np.linalg.norm(z, axis=-1)
    return norm[..., None] ** ((p-2.0)/2.0) * z


def veps_map(z: np.ndarray, p: float, epsilon: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    norm_sq = np.sum(z*z, axis=-1)
    return (epsilon*epsilon + norm_sq)[..., None] ** ((p-2.0)/4.0) * z


def quasi_error(problem: object, u: np.ndarray, grad_u: np.ndarray,
                exact: tuple[np.ndarray, np.ndarray, np.ndarray],
                rule: object) -> dict[str, float | str] | None:
    if problem.name in {"linear", "cubic", "sinh"}:
        return None
    u_exact, grad_exact, _ = exact
    if problem.name == "regularized_p":
        a = veps_map(grad_u, problem.p, float(problem.epsilon))
        b = veps_map(grad_exact, problem.p, float(problem.epsilon))
    else:
        a, b = vp_map(grad_u, problem.p), vp_map(grad_exact, problem.p)
    density = np.sum((a-b)**2, axis=1)
    reference = np.sum(b*b, axis=1)
    if problem.name == "reaction_p":
        au, bu = vp_map(u, problem.p), vp_map(u_exact, problem.p)
        density += (au-bu)**2
        reference += bu**2
    absolute = float(np.sqrt(np.dot(rule.weights, density)))
    denominator = float(np.sqrt(np.dot(rule.weights, reference)))
    return _pair(absolute, denominator)


def energy_gap(numerical_energy: float, exact_energy: float,
               uncertainty: float = 0.0) -> dict[str, float | str]:
    gap = float(numerical_energy - exact_energy)
    if gap < -abs(uncertainty):
        status = "negative_unresolved"
    elif gap < 0.0:
        status = "within_quadrature_uncertainty"
    else:
        status = "ok"
    return {"raw": gap, "status": status}


def compute_metrics(problem: object, u: np.ndarray, grad_u: np.ndarray,
                    exact: tuple[np.ndarray, np.ndarray, np.ndarray], rule: object,
                    numerical_energy: float, exact_energy: float) -> dict[str, object]:
    l2 = l2_error(u, exact[0], rule)
    if problem.natural_metric == "H1":
        natural = h1_error(u, grad_u, exact, rule)
        full_w1p = None
    else:
        natural = w1p_error(u, grad_u, exact, problem.p, rule,
                            gradient_only=problem.name != "reaction_p")
        full_w1p = w1p_error(u, grad_u, exact, problem.p, rule, gradient_only=False)
    quasi = quasi_error(problem, u, grad_u, exact, rule)
    gap = energy_gap(numerical_energy, exact_energy)
    bregman = bregman_divergence(problem, u, grad_u, exact, rule)
    return {"energy_gap_raw": gap["raw"], "energy_gap_status": gap["status"],
            "bregman_gap": bregman,
            "l2_abs": l2["absolute"], "l2_rel": l2["relative"],
            "natural_abs": natural["absolute"], "natural_rel": natural["relative"],
            "w1p_full_abs": None if full_w1p is None else full_w1p["absolute"],
            "w1p_full_rel": None if full_w1p is None else full_w1p["relative"],
            "quasi_abs": None if quasi is None else quasi["absolute"],
            "quasi_rel": None if quasi is None else quasi["relative"]}
