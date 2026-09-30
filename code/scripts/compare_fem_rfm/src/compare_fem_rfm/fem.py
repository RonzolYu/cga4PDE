"""Structured simplicial Lagrange P1/P2/P3 finite-element solver."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import (
    Basis,
    BilinearForm,
    ElementLineP1,
    ElementLineP2,
    ElementLinePp,
    ElementTriP1,
    ElementTriP2,
    ElementTriP3,
    LinearForm,
    MeshLine,
    MeshTri,
    asm,
)
from skfem.helpers import dot, grad

from .problems import ProblemSpec, forcing


@dataclass
class FEMModel:
    dim: int
    degree: int
    n_elements_axis: int
    coefficients: np.ndarray


@dataclass
class FEMResult:
    model: FEMModel
    full_dof: int
    dof: int
    iterations: int
    residual: float
    success: bool
    message: str
    wall_time_sec: float


def _mesh_and_element(dim: int, degree: int, n: int):
    grid = np.linspace(0.0, 1.0, n + 1)
    if dim == 1:
        mesh = MeshLine(grid)
        if degree == 1:
            element = ElementLineP1()
        elif degree == 2:
            element = ElementLineP2()
        else:
            element = ElementLinePp(degree)
    else:
        mesh = MeshTri.init_tensor(grid, grid)
        element = {1: ElementTriP1(), 2: ElementTriP2(), 3: ElementTriP3()}[degree]
    return mesh, element


def fem_dof(dim: int, degree: int, n: int, zero_mean: bool) -> int:
    mesh, element = _mesh_and_element(dim, degree, n)
    full = Basis(mesh, element).N
    return int(full - int(zero_mean))


def _form_forcing(spec: ProblemSpec, x: np.ndarray) -> np.ndarray:
    points = np.moveaxis(x, 0, -1).reshape(-1, spec.dim)
    return forcing(spec, points).reshape(x.shape[1:])


def _forms(spec: ProblemSpec, epsilon: float):
    @LinearForm
    def residual_form(v, w):
        u = w.uh
        f = _form_forcing(spec, w.x)
        if spec.model == "linear":
            return dot(u.grad, grad(v)) + (u - f) * v
        if spec.model == "cubic":
            return dot(u.grad, grad(v)) + (u**3 + epsilon * u - f) * v
        if spec.model == "sinh":
            return dot(u.grad, grad(v)) + (np.sinh(u) - f) * v
        grad_sq = dot(u.grad, u.grad)
        return (epsilon**2 + grad_sq) * dot(u.grad, grad(v)) - f * v

    @BilinearForm
    def jacobian_form(du, v, w):
        u = w.uh
        if spec.model == "linear":
            return dot(grad(du), grad(v)) + du * v
        if spec.model == "cubic":
            return dot(grad(du), grad(v)) + (3.0 * u**2 + epsilon) * du * v
        if spec.model == "sinh":
            return dot(grad(du), grad(v)) + np.cosh(u) * du * v
        grad_sq = dot(u.grad, u.grad)
        return ((epsilon**2 + grad_sq) * dot(grad(du), grad(v))
                + 2.0 * dot(u.grad, grad(du)) * dot(u.grad, grad(v)))

    return residual_form, jacobian_form


@LinearForm
def _mean_form(v, w):
    return v


def _stage_energy(spec: ProblemSpec, basis: Basis, coefficients: np.ndarray,
                  epsilon: float) -> float:
    uh = basis.interpolate(coefficients)
    u = np.asarray(uh)
    grad_u = np.asarray(uh.grad)
    grad_sq = np.sum(grad_u * grad_u, axis=0)
    f = _form_forcing(spec, basis.global_coordinates())
    if spec.model == "linear":
        density = 0.5 * grad_sq + 0.5 * u**2 - f * u
    elif spec.model == "cubic":
        density = 0.5 * grad_sq + 0.25 * u**4 + 0.5 * epsilon * u**2 - f * u
    elif spec.model == "sinh":
        density = 0.5 * grad_sq + np.cosh(u) - 1.0 - f * u
    else:
        density = 0.25 * ((epsilon**2 + grad_sq) ** 2 - epsilon**4) - f * u
    return float(np.sum(basis.dx * density))


def _projected_residual(residual: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
    if mean is None:
        return residual
    lam = -float(np.dot(mean, residual)) / float(np.dot(mean, mean))
    return residual + lam * mean


def solve_fem(spec: ProblemSpec, degree: int, n_elements_axis: int,
              cfg: dict) -> FEMResult:
    started = time.perf_counter()
    mesh, element = _mesh_and_element(spec.dim, degree, n_elements_axis)
    intorder = (cfg.get("assembly_order_1d_multifrequency", cfg["assembly_order"])
                if spec.dim == 1 and spec.exact_profile == "multifrequency"
                else cfg["assembly_order"])
    basis = Basis(mesh, element, intorder=intorder)
    coefficients = np.zeros(basis.N)
    mean = asm(_mean_form, basis) if spec.zero_mean else None
    fscale = max(1.0, float(np.linalg.norm(asm(
        LinearForm(lambda v, w: _form_forcing(spec, w.x) * v), basis
    ))))
    if spec.model == "pure_p":
        epsilon_stages = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0]
    elif spec.model == "cubic":
        epsilon_stages = [1.0, 0.1, 0.01, 0.0]
    else:
        epsilon_stages = [0.0]
    total_iterations = 0
    final_residual = float("inf")
    success = True
    message = "converged"
    for epsilon in epsilon_stages:
        residual_form, jacobian_form = _forms(spec, epsilon)
        stage_ok = False
        for _ in range(cfg["newton_maxiter"]):
            total_iterations += 1
            uh = basis.interpolate(coefficients)
            residual = np.asarray(asm(residual_form, basis, uh=uh))
            projected = _projected_residual(residual, mean)
            final_residual = float(np.linalg.norm(projected) / fscale)
            if final_residual <= cfg["newton_atol"] + cfg["newton_rtol"]:
                stage_ok = True
                break
            jacobian = asm(jacobian_form, basis, uh=uh).tocsr()
            if mean is None:
                delta = spsolve(jacobian, -residual)
            else:
                mcol = csr_matrix(mean[:, None])
                kkt = bmat([[jacobian, mcol], [mcol.T, csr_matrix((1, 1))]], format="csr")
                rhs = np.concatenate([-residual, [-float(np.dot(mean, coefficients))]])
                delta = spsolve(kkt, rhs)[:-1]
            if not np.all(np.isfinite(delta)):
                success, message = False, f"non-finite Newton step at epsilon={epsilon}"
                break
            energy0 = _stage_energy(spec, basis, coefficients, epsilon)
            directional = float(np.dot(residual, delta))
            step = 1.0
            accepted = False
            while step >= 2.0**-24:
                trial = coefficients + step * delta
                if mean is not None:
                    trial -= mean * (np.dot(mean, trial) / np.dot(mean, mean))
                with np.errstate(over="ignore", invalid="ignore"):
                    energy_trial = _stage_energy(spec, basis, trial, epsilon)
                if np.isfinite(energy_trial) and energy_trial <= energy0 + 1e-4 * step * directional:
                    coefficients = trial
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                if final_residual < 1e-8:
                    stage_ok = True
                    message = f"roundoff-limited at epsilon={epsilon}"
                else:
                    success, message = False, f"line search failed at epsilon={epsilon}"
                break
        if not stage_ok and success:
            if final_residual < 1e-8:
                stage_ok = True
                message = f"roundoff-limited at epsilon={epsilon}"
            else:
                success, message = False, f"Newton limit at epsilon={epsilon}; residual={final_residual:.3e}"
        if not success:
            break
    full_dof = int(basis.N)
    model = FEMModel(spec.dim, degree, n_elements_axis, coefficients)
    return FEMResult(model, full_dof, full_dof - int(spec.zero_mean), total_iterations,
                     final_residual, success, message, time.perf_counter() - started)


def _probe_gradient(basis: Basis, points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    x = points.T
    cells = basis.mesh.element_finder(mapping=basis.mapping)(*x)
    reference = basis.mapping.invF(x[:, :, None], tind=cells)
    rows = np.tile(np.arange(points.shape[0]), basis.Nbfun)
    cols = basis.element_dofs[:, cells].flatten()
    result = np.empty((points.shape[0], points.shape[1]))
    for axis in range(points.shape[1]):
        data = np.array([
            basis.elem.gbasis(basis.mapping, reference, k, tind=cells)[0].grad[axis]
            for k in range(basis.Nbfun)
        ]).flatten()
        matrix = coo_matrix((data, (rows, cols)), shape=(points.shape[0], basis.N))
        result[:, axis] = matrix @ coefficients
    return result


def evaluate_fem_model(model: FEMModel, points: np.ndarray, batch_size: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    mesh, element = _mesh_and_element(model.dim, model.degree, model.n_elements_axis)
    basis = Basis(mesh, element)
    values, gradients = [], []
    for start in range(0, points.shape[0], batch_size):
        batch = points[start:start + batch_size]
        values.append(np.asarray(basis.probes(batch.T) @ model.coefficients))
        gradients.append(_probe_gradient(basis, batch, model.coefficients))
    return np.concatenate(values), np.vstack(gradients)
