"""The six variational Neumann problems and their shared analytic formulas."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    p: float = 4.0
    epsilon: float | None = None
    requires_zero_mean: bool = False
    natural_metric: str = "H1"
    exact_profile: str = "low_frequency"


def make_problem(name: str, dim: int, p: float = 4.0,
                 epsilon: float | None = None,
                 exact_profile: str = "low_frequency") -> Problem:
    if name not in {"linear", "cubic", "sinh", "pure_p", "regularized_p", "reaction_p"}:
        raise KeyError(name)
    if dim not in (1, 2) or p < 2.0:
        raise ValueError("dim must be 1 or 2 and p must be >= 2")
    if name == "regularized_p" and (epsilon is None or epsilon <= 0):
        raise ValueError("regularized_p requires epsilon > 0")
    if exact_profile not in {"low_frequency", "multifrequency"}:
        raise ValueError("unknown exact-solution profile")
    if dim == 2 and exact_profile != "low_frequency":
        raise ValueError("multifrequency is a 1D manufactured-solution profile")
    zero_mean = name in {"pure_p", "regularized_p"}
    natural = "H1" if name in {"linear", "cubic", "sinh"} else "W1p"
    return Problem(name, dim, float(p), epsilon, zero_mean, natural, exact_profile)


def exact_solution(problem: Problem, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    modes = ((1.0, 1.0),) if problem.exact_profile == "low_frequency" else (
        (1.0, 1.0), (0.2, 4.0), (0.1, 9.0))
    value_1d = np.zeros_like(x)
    first_1d = np.zeros_like(x)
    second_1d = np.zeros_like(x)
    for amplitude, frequency in modes:
        omega = 2.0 * np.pi * frequency
        value_1d += amplitude * np.cos(omega * x)
        first_1d -= amplitude * omega * np.sin(omega * x)
        second_1d -= amplitude * omega**2 * np.cos(omega * x)
    u = np.prod(value_1d, axis=1)
    q, dim = x.shape
    grad = np.empty((q, dim), dtype=np.float64)
    hess = np.empty((q, dim, dim), dtype=np.float64)
    for i in range(dim):
        factors = value_1d.copy()
        factors[:, i] = first_1d[:, i]
        grad[:, i] = np.prod(factors, axis=1)
        factors[:, i] = second_1d[:, i]
        hess[:, i, i] = np.prod(factors, axis=1)
        for j in range(i + 1, dim):
            factors = value_1d.copy()
            factors[:, i] = first_1d[:, i]
            factors[:, j] = first_1d[:, j]
            value = np.prod(factors, axis=1)
            hess[:, i, j] = value
            hess[:, j, i] = value
    return u, grad, hess


def flux(problem: Problem, u: np.ndarray, grad_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_u = np.asarray(grad_u, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    grad_sq = np.sum(grad_u * grad_u, axis=1)
    if problem.name in {"linear", "cubic", "sinh"}:
        diffusion = grad_u
    elif problem.name == "regularized_p":
        diffusion = (float(problem.epsilon) ** 2 + grad_sq)[:, None] ** ((problem.p - 2.0) / 2.0) * grad_u
    else:
        diffusion = grad_sq[:, None] ** ((problem.p - 2.0) / 2.0) * grad_u
    if problem.name == "linear":
        reaction = u
    elif problem.name == "cubic":
        reaction = u**3
    elif problem.name == "sinh":
        reaction = np.sinh(u)
    elif problem.name == "reaction_p":
        reaction = np.abs(u) ** (problem.p - 2.0) * u
    else:
        reaction = np.zeros_like(u)
    return diffusion, reaction


def source_from_jet(problem: Problem, u: np.ndarray, grad: np.ndarray,
                    hess: np.ndarray) -> np.ndarray:
    lap = np.trace(hess, axis1=1, axis2=2)
    if problem.name in {"linear", "cubic", "sinh"}:
        _, reaction = flux(problem, u, grad)
        return -lap + reaction
    grad_sq = np.sum(grad * grad, axis=1)
    quad = np.einsum("qi,qij,qj->q", grad, hess, grad, optimize=True)
    if problem.name == "regularized_p":
        eps2 = float(problem.epsilon) ** 2
        a = (eps2 + grad_sq) ** ((problem.p - 2.0) / 2.0)
        b = (problem.p - 2.0) * (eps2 + grad_sq) ** ((problem.p - 4.0) / 2.0)
    else:
        if problem.p == 2.0:
            a = np.ones_like(grad_sq)
            b = np.zeros_like(grad_sq)
        elif problem.p == 4.0:
            a = grad_sq
            b = np.full_like(grad_sq, 2.0)
        else:
            a = np.zeros_like(grad_sq)
            b = np.zeros_like(grad_sq)
            mask = grad_sq > 0.0
            a[mask] = grad_sq[mask] ** ((problem.p - 2.0) / 2.0)
            b[mask] = (problem.p - 2.0) * grad_sq[mask] ** ((problem.p - 4.0) / 2.0)
    source = -a * lap - b * quad
    if problem.name == "reaction_p":
        source += np.abs(u) ** (problem.p - 2.0) * u
    return source


def source_term(problem: Problem, x: np.ndarray) -> np.ndarray:
    return source_from_jet(problem, *exact_solution(problem, x))


def energy(problem: Problem, u: np.ndarray, grad_u: np.ndarray, f: np.ndarray,
           rule: object) -> float:
    grad_sq = np.sum(grad_u * grad_u, axis=1)
    if problem.name == "linear":
        density = 0.5 * grad_sq + 0.5 * u**2 - f * u
    elif problem.name == "cubic":
        density = 0.5 * grad_sq + 0.25 * u**4 - f * u
    elif problem.name == "sinh":
        density = 0.5 * grad_sq + np.cosh(u) - 1.0 - f * u
    elif problem.name == "regularized_p":
        eps = float(problem.epsilon)
        density = ((eps * eps + grad_sq) ** (problem.p / 2.0) - eps**problem.p) / problem.p - f * u
    elif problem.name == "reaction_p":
        density = (grad_sq ** (problem.p / 2.0) + np.abs(u) ** problem.p) / problem.p - f * u
    else:
        density = grad_sq ** (problem.p / 2.0) / problem.p - f * u
    return float(np.dot(rule.weights, density))


def first_variation(problem: Problem, u: np.ndarray, grad_u: np.ndarray,
                    v: np.ndarray, grad_v: np.ndarray, f: np.ndarray,
                    rule: object) -> np.ndarray | float:
    single = v.ndim == 1
    if single:
        v = v[:, None]
        grad_v = grad_v[:, None, :]
    diffusion, reaction = flux(problem, u, grad_u)
    result = np.einsum("qbd,qd,q->b", grad_v, diffusion, rule.weights, optimize=True)
    result += np.einsum("qb,q,q->b", v, reaction - f, rule.weights, optimize=True)
    return float(result[0]) if single else result


def _field(basis_values: np.ndarray, basis_grads: np.ndarray, c: np.ndarray,
           base: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    u = base[0] + basis_values @ c
    grad = base[1] + np.einsum("qmd,m->qd", basis_grads, c, optimize=True)
    return u, grad


def coefficient_objective(problem: Problem, basis_values: np.ndarray,
                          basis_grads: np.ndarray, c: np.ndarray,
                          base: tuple[np.ndarray, np.ndarray], f: np.ndarray,
                          rule: object) -> tuple[float, np.ndarray]:
    u, grad = _field(basis_values, basis_grads, c, base)
    value = energy(problem, u, grad, f, rule)
    gradient = first_variation(problem, u, grad, basis_values, basis_grads, f, rule)
    return value, np.asarray(gradient, dtype=np.float64)


def coefficient_hvp(problem: Problem, basis_values: np.ndarray,
                    basis_grads: np.ndarray, c: np.ndarray, direction: np.ndarray,
                    base: tuple[np.ndarray, np.ndarray], rule: object) -> np.ndarray:
    u, grad = _field(basis_values, basis_grads, c, base)
    z = basis_values @ direction
    grad_z = np.einsum("qmd,m->qd", basis_grads, direction, optimize=True)
    grad_sq = np.sum(grad * grad, axis=1)
    if problem.name in {"linear", "cubic", "sinh"}:
        flux_z = grad_z
        if problem.name == "linear":
            reaction_z = z
        elif problem.name == "cubic":
            reaction_z = 3.0 * u**2 * z
        else:
            reaction_z = np.cosh(u) * z
    else:
        if problem.name == "regularized_p":
            scale = float(problem.epsilon) ** 2 + grad_sq
        else:
            scale = grad_sq
        a = scale ** ((problem.p - 2.0) / 2.0)
        b = np.zeros_like(scale)
        if problem.p != 2.0:
            mask = scale > 0.0
            b[mask] = (problem.p - 2.0) * scale[mask] ** ((problem.p - 4.0) / 2.0)
        dot = np.sum(grad * grad_z, axis=1)
        flux_z = a[:, None] * grad_z + b[:, None] * dot[:, None] * grad
        reaction_z = ((problem.p - 1.0) * np.abs(u) ** (problem.p - 2.0) * z
                      if problem.name == "reaction_p" else np.zeros_like(z))
    result = np.einsum("qmd,qd,q->m", basis_grads, flux_z, rule.weights, optimize=True)
    result += np.einsum("qm,q,q->m", basis_values, reaction_z, rule.weights, optimize=True)
    return result


def weighted_mean(values: np.ndarray, rule: object) -> float:
    return float(np.dot(rule.weights, values) / np.sum(rule.weights))


def center_values(values: np.ndarray, mean: np.ndarray | float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) - np.asarray(mean, dtype=np.float64)


def check_compatibility(problem: Problem, f: np.ndarray, domain_rule: object,
                        boundary_flux: np.ndarray | None = None,
                        boundary_rule: object | None = None) -> dict[str, object]:
    residual = float(np.dot(domain_rule.weights, f))
    if boundary_flux is not None and boundary_rule is not None:
        residual -= float(np.dot(boundary_rule.weights, boundary_flux))
    scale = max(1.0, float(np.dot(domain_rule.weights, np.abs(f))))
    return {"required": problem.requires_zero_mean, "residual": residual,
            "passed": (not problem.requires_zero_mean) or abs(residual) <= 1e-7 * scale}


def check_zero_mean(problem: Problem, u: np.ndarray, rule: object) -> dict[str, object]:
    mean = weighted_mean(u, rule)
    return {"required": problem.requires_zero_mean, "mean": mean,
            "passed": (not problem.requires_zero_mean) or abs(mean) <= 1e-9}
