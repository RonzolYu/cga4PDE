"""Registered manufactured variational problems shared by all methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    case_id: str
    numerical_id: str
    model: str
    dim: int
    p: float | None
    relu_k: int
    exact_profile: str
    zero_mean: bool
    cga_target: int
    cga_expected_accepted: int

    @property
    def natural_metric(self) -> str:
        return "H1" if self.model in {"linear", "cubic", "sinh"} else "W1p"

    @property
    def problem_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


CASES: dict[str, ProblemSpec] = {
    "C1": ProblemSpec("C1", "01", "linear", 1, None, 3, "multifrequency", False, 256, 256),
    "C2": ProblemSpec("C2", "03", "cubic", 1, None, 3, "multifrequency", False, 256, 256),
    "C3": ProblemSpec("C3", "06", "sinh", 2, None, 3, "low_frequency", False, 512, 512),
    "C4": ProblemSpec("C4", "17", "pure_p", 1, 4.0, 3, "low_frequency", True, 256, 141),
    "C5": ProblemSpec("C5", "08", "pure_p", 2, 4.0, 3, "low_frequency", True, 512, 512),
}


def exact_jet(spec: ProblemSpec, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return manufactured solution, gradient, and Hessian on ``[0, 1]^d``."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    modes = ((1.0, 1.0),) if spec.exact_profile == "low_frequency" else (
        (1.0, 1.0), (0.2, 4.0), (0.1, 9.0)
    )
    value = np.zeros_like(x)
    first = np.zeros_like(x)
    second = np.zeros_like(x)
    for amplitude, frequency in modes:
        omega = 2.0 * np.pi * frequency
        value += amplitude * np.cos(omega * x)
        first -= amplitude * omega * np.sin(omega * x)
        second -= amplitude * omega**2 * np.cos(omega * x)
    u = np.prod(value, axis=1)
    q, dim = x.shape
    grad = np.empty((q, dim))
    hess = np.empty((q, dim, dim))
    for i in range(dim):
        factors = value.copy()
        factors[:, i] = first[:, i]
        grad[:, i] = np.prod(factors, axis=1)
        factors[:, i] = second[:, i]
        hess[:, i, i] = np.prod(factors, axis=1)
        for j in range(i + 1, dim):
            factors = value.copy()
            factors[:, i] = first[:, i]
            factors[:, j] = first[:, j]
            hess[:, i, j] = hess[:, j, i] = np.prod(factors, axis=1)
    return u, grad, hess


def forcing(spec: ProblemSpec, x: np.ndarray) -> np.ndarray:
    u, grad, hess = exact_jet(spec, x)
    lap = np.trace(hess, axis1=1, axis2=2)
    if spec.model == "linear":
        return -lap + u
    if spec.model == "cubic":
        return -lap + u**3
    if spec.model == "sinh":
        return -lap + np.sinh(u)
    if spec.model == "pure_p" and spec.p == 4.0:
        grad_sq = np.sum(grad * grad, axis=1)
        quad = np.einsum("qi,qij,qj->q", grad, hess, grad, optimize=True)
        return -grad_sq * lap - 2.0 * quad
    raise NotImplementedError((spec.model, spec.p))


def reaction(spec: ProblemSpec, u: np.ndarray) -> np.ndarray:
    if spec.model == "linear":
        return u
    if spec.model == "cubic":
        return u**3
    if spec.model == "sinh":
        return np.sinh(u)
    return np.zeros_like(u)


def reaction_derivative(spec: ProblemSpec, u: np.ndarray) -> np.ndarray:
    if spec.model == "linear":
        return np.ones_like(u)
    if spec.model == "cubic":
        return 3.0 * u**2
    if spec.model == "sinh":
        return np.cosh(u)
    return np.zeros_like(u)


def energy_density(spec: ProblemSpec, u: np.ndarray, grad: np.ndarray, f: np.ndarray) -> np.ndarray:
    grad_sq = np.sum(grad * grad, axis=1)
    if spec.model == "linear":
        return 0.5 * grad_sq + 0.5 * u**2 - f * u
    if spec.model == "cubic":
        return 0.5 * grad_sq + 0.25 * u**4 - f * u
    if spec.model == "sinh":
        return 0.5 * grad_sq + np.cosh(u) - 1.0 - f * u
    if spec.model == "pure_p":
        return grad_sq ** (float(spec.p) / 2.0) / float(spec.p) - f * u
    raise NotImplementedError(spec.model)

