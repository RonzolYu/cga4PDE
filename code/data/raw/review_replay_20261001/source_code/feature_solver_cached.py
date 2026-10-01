"""Convex outer-coefficient solvers for RFM and frozen-prefix CGA."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import linalg, optimize

from .problems import ProblemSpec


@dataclass
class FeatureSolveResult:
    coefficients: np.ndarray
    iterations: int
    residual: float
    success: bool
    message: str
    wall_time_sec: float
    hessian_backend: str = 'not_used'


def energy_and_gradient(spec: ProblemSpec, coefficients: np.ndarray, phi: np.ndarray,
                        grad_phi: np.ndarray, f: np.ndarray,
                        weights: np.ndarray) -> tuple[float, np.ndarray]:
    u = phi @ coefficients
    grad = np.einsum("qnd,n->qd", grad_phi, coefficients, optimize=True)
    grad_sq = np.sum(grad * grad, axis=1)
    if spec.model == "linear":
        density = 0.5 * grad_sq + 0.5 * u**2 - f * u
        flux, reaction = grad, u
    elif spec.model == "cubic":
        density = 0.5 * grad_sq + 0.25 * u**4 - f * u
        flux, reaction = grad, u**3
    elif spec.model == "sinh":
        density = 0.5 * grad_sq + np.cosh(u) - 1.0 - f * u
        flux, reaction = grad, np.sinh(u)
    elif spec.model == "pure_p":
        density = 0.25 * grad_sq**2 - f * u
        flux, reaction = grad_sq[:, None] * grad, np.zeros_like(u)
    else:
        raise NotImplementedError(spec.model)
    gradient = np.einsum("qnd,qd,q->n", grad_phi, flux, weights, optimize=True)
    gradient += np.einsum("qn,q,q->n", phi, reaction - f, weights, optimize=True)
    return float(np.dot(weights, density)), gradient


def hessian_vector_product(spec: ProblemSpec, coefficients: np.ndarray, direction: np.ndarray,
                           phi: np.ndarray, grad_phi: np.ndarray,
                           weights: np.ndarray) -> np.ndarray:
    u = phi @ coefficients
    grad_u = np.einsum("qnd,n->qd", grad_phi, coefficients, optimize=True)
    z = phi @ direction
    grad_z = np.einsum("qnd,n->qd", grad_phi, direction, optimize=True)
    if spec.model in {"cubic", "sinh"}:
        reaction_derivative = 3.0 * u**2 if spec.model == "cubic" else np.cosh(u)
        result = np.einsum("qnd,qd,q->n", grad_phi, grad_z, weights, optimize=True)
        result += np.einsum("qn,q,q->n", phi, reaction_derivative * z, weights, optimize=True)
        return result
    if spec.model == "pure_p":
        grad_sq = np.sum(grad_u * grad_u, axis=1)
        flux_z = grad_sq[:, None] * grad_z
        flux_z += 2.0 * np.sum(grad_u * grad_z, axis=1)[:, None] * grad_u
        return np.einsum("qnd,qd,q->n", grad_phi, flux_z, weights, optimize=True)
    raise NotImplementedError(spec.model)


class CachedHessianAction:
    """Reuse the exact quadrature Hessian while Newton-CG keeps its base point.

    The Gram form below is the derivative of energy_and_gradient.  It changes
    storage and arithmetic order, not the objective, coordinates or tolerances.
    Retain hessian_vector_product as a separate matrix-free reference.
    """

    def __init__(self, spec, phi, grad_phi, weights):
        self.spec, self.phi, self.grad_phi, self.weights = spec, phi, grad_phi, weights
        self.base = None
        self.matrix = None

    def __call__(self, coefficients, direction):
        if self.base is None or not np.array_equal(self.base, coefficients):
            grad = np.einsum('qnd,n->qd', self.grad_phi, coefficients, optimize=True)
            pure = self.spec.model == 'pure_p'
            density = np.sum(grad*grad,axis=1) if pure else np.ones_like(self.weights)
            root = np.sqrt(self.weights*density)
            n = len(coefficients)
            matrix = np.zeros((n,n))
            for axis in range(self.spec.dim):
                block = np.ascontiguousarray(root[:,None]*self.grad_phi[:,:,axis])
                matrix += block.T@block
            if pure:
                projection = np.einsum('qnd,qd->qn', self.grad_phi, grad, optimize=True)
                block = np.ascontiguousarray(np.sqrt(2*self.weights)[:,None]*projection)
            else:
                u = self.phi@coefficients
                reaction = (3*u*u if self.spec.model == 'cubic' else
                            np.cosh(u) if self.spec.model == 'sinh' else np.ones_like(u))
                block = np.ascontiguousarray(np.sqrt(self.weights*reaction)[:,None]*self.phi)
            matrix += block.T@block
            self.matrix = matrix
            self.base = coefficients.copy()
        return self.matrix@direction


def solve_feature_coefficients(spec: ProblemSpec, phi: np.ndarray, grad_phi: np.ndarray,
                               f: np.ndarray, weights: np.ndarray, cfg: dict,
                               initial: np.ndarray | None = None) -> FeatureSolveResult:
    started = time.perf_counter()
    n = phi.shape[1]
    if spec.model == "linear":
        sqrt_w = np.sqrt(weights)
        blocks = [phi * sqrt_w[:, None]]
        blocks.extend(grad_phi[:, :, j] * sqrt_w[:, None] for j in range(spec.dim))
        design = np.vstack(blocks)
        target = np.concatenate([sqrt_w * f] + [np.zeros_like(f) for _ in range(spec.dim)])
        coefficients, _, rank, _ = linalg.lstsq(
            design, target, cond=cfg["rank_rcond"], lapack_driver="gelsd"
        )
        _, gradient = energy_and_gradient(spec, coefficients, phi, grad_phi, f, weights)
        scale = max(1.0, float(np.linalg.norm(phi.T @ (weights * f))))
        residual = float(np.linalg.norm(gradient) / scale)
        return FeatureSolveResult(coefficients, 1, residual,
                                  residual < 2e-6, f"rank={rank}/{n}", time.perf_counter() - started)
    c0 = np.zeros(n) if initial is None else np.pad(initial, (0, n - initial.size))[:n]
    load_scale = max(1.0, float(np.linalg.norm(phi.T @ (weights * f))))

    # Reparameterize the same feature span in an H1-orthonormalized coordinate
    # system.  This is a solver preconditioner only; returned coefficients are
    # transformed back to the original frozen features.
    sqrt_w = np.sqrt(weights)
    gram = (phi * sqrt_w[:, None]).T @ (phi * sqrt_w[:, None])
    for axis in range(spec.dim):
        weighted_grad = grad_phi[:, :, axis] * sqrt_w[:, None]
        gram += weighted_grad.T @ weighted_grad
    eigenvalues, eigenvectors = linalg.eigh(gram, check_finite=False)
    floor = max(float(eigenvalues[-1]) * 1e-12, 1e-14)
    stabilized = np.maximum(eigenvalues, floor)
    transform = eigenvectors / np.sqrt(stabilized)[None, :]
    x0 = np.sqrt(stabilized) * (eigenvectors.T @ c0)
    phi_opt = phi @ transform
    q = grad_phi.shape[0]
    grad_flat = grad_phi.transpose(0, 2, 1).reshape(q * spec.dim, n)
    grad_opt = (grad_flat @ transform).reshape(q, spec.dim, n).transpose(0, 2, 1)

    def objective(a: np.ndarray) -> tuple[float, np.ndarray]:
        return energy_and_gradient(spec, a, phi_opt, grad_opt, f, weights)

    result = optimize.minimize(
        objective, x0, method="L-BFGS-B", jac=True,
        options={
            "maxiter": cfg["lbfgs_maxiter"],
            "gtol": cfg["lbfgs_gtol"],
            "ftol": cfg["lbfgs_ftol"],
            "maxls": 50,
            "maxcor": 30,
        },
    )
    coefficients = transform @ result.x
    _, gradient_original = energy_and_gradient(spec, coefficients, phi, grad_phi, f, weights)
    residual = float(np.linalg.norm(gradient_original) / load_scale)
    iterations = int(result.nit)
    messages = [f"L-BFGS-B: {result.message}"]
    used_backend = 'not_used'
    if residual > 2e-7:
        used_backend = cfg.get('hessian_backend','cached_matrix')
        if used_backend not in {'matrix_free','cached_matrix'}:
            raise ValueError(f'unknown Hessian backend: {used_backend}')
        if used_backend == 'matrix_free':
            def hessp(a: np.ndarray, direction: np.ndarray) -> np.ndarray:
                return hessian_vector_product(spec, a, direction, phi_opt, grad_opt, weights)
        else:
            hessp = CachedHessianAction(spec,phi_opt,grad_opt,weights)

        second = optimize.minimize(
            objective, result.x, method="Newton-CG", jac=True, hessp=hessp,
            options={"maxiter": cfg["newtoncg_maxiter"], "xtol": 1e-12},
        )
        iterations += int(second.nit)
        value_first, _ = objective(result.x)
        value_second, gradient_second = objective(second.x)
        if np.isfinite(value_second) and value_second <= value_first:
            result = second
            coefficients = transform @ result.x
            _, gradient_original = energy_and_gradient(spec, coefficients, phi, grad_phi, f, weights)
            residual = float(np.linalg.norm(gradient_original) / load_scale)
        messages.append(f"Newton-CG: {second.message}")
    return FeatureSolveResult(np.asarray(coefficients), iterations, residual,
                              residual < 2e-6,
                              "; ".join(messages), time.perf_counter() - started, used_backend)
