from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from cga_refactor.config import config_for
import cga_refactor.step2 as step2
from cga_refactor.step2 import (projected_residual, solve_newton_cg,
                                solve_linear, solve_trust_krylov, stable_transform)


def test_stable_transform_round_trip():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(30, 5))
    transform = stable_transform(matrix, 1e-12, 1e-10)
    c = rng.normal(size=5)
    y = transform["to_stable"] @ c
    recovered = transform["to_coeff"] @ y
    np.testing.assert_allclose(matrix @ recovered, matrix @ c, rtol=2e-12, atol=2e-12)


def test_projected_residual_gate():
    report = projected_residual(np.array([1e-10]), np.array([1.0]), 1e-9, 1e-7)
    assert report["passed"]
    report = projected_residual(np.array([1e-3]), np.array([1.0]), 1e-9, 1e-7)
    assert not report["passed"]


def test_hvp_fallback_uses_optimizer_iterate(monkeypatch):
    queried_points = []

    def fake_minimize(fun, y0, *, jac, hessp, method, options):
        hessp(y0, np.ones_like(y0))
        hessp(y0 + 0.75, np.ones_like(y0))
        return SimpleNamespace(x=y0, success=True, status=0, message="ok", nit=1, nfev=1)

    def objective(y):
        return float(0.5*np.dot(y, y)), y.copy()

    def hvp(point, direction):
        queried_points.append(point.copy())
        return (1.0 + point**2) * direction

    monkeypatch.setattr(step2, "minimize", fake_minimize)
    cfg = config_for("cubic", 1, profile="smoke")
    transform = {"r": np.eye(2)}
    y0 = np.array([0.2, -0.1])
    solve_trust_krylov(objective, hvp, transform, y0, cfg)
    assert len(queried_points) == 2
    np.testing.assert_allclose(queried_points[0], y0)
    np.testing.assert_allclose(queried_points[1], y0 + 0.75)


def test_newton_cg_strict_polish_reaches_residual_gate():
    def objective(y):
        return float(np.sum(0.5*y*y + 0.25*y**4)), y + y**3

    def hvp(point, direction):
        return (1.0 + 3.0*point**2) * direction

    cfg = config_for("cubic", 1, profile="smoke")
    y0 = np.array([2.0, -1.5, 0.8])
    result = solve_newton_cg(objective, hvp, {"r": np.eye(3)}, y0, cfg)
    gate = projected_residual(result["grad_stable"], result["initial_gradient"],
                              cfg.solver.projection_atol, cfg.solver.projection_rtol)
    assert gate["passed"]


def test_linear_iterative_refinement_reaches_strict_gate():
    hessian = np.diag([1.0, 1e-6, 1e-12])
    rhs = np.array([0.7, -2e-6, 3e-12])

    def objective(y):
        return float(0.5*y @ hessian @ y - rhs @ y), hessian @ y - rhs

    def hvp(_point, direction):
        return hessian @ direction

    cfg = config_for("linear", 1, profile="smoke")
    result = solve_linear(objective, hvp, {"r": np.eye(3)}, np.zeros(3), cfg)
    gate = projected_residual(result["grad_stable"], result["initial_gradient"],
                              cfg.solver.projection_atol, cfg.solver.projection_rtol)
    assert gate["passed"]
