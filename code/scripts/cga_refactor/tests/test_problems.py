from __future__ import annotations

import numpy as np
import pytest

from cga_refactor.problems import (coefficient_hvp, coefficient_objective, energy,
                                   exact_solution, first_variation, make_problem,
                                   source_term)
from cga_refactor.quadrature import segmented_gauss_1d, tensor_gauss_2d


@pytest.mark.parametrize("dim", [1, 2])
def test_cosine_profile_jet_against_finite_difference(dim):
    problem = make_problem("linear", dim)
    x = np.full((3, dim), 0.231)
    _, grad, _ = exact_solution(problem, x)
    h = 1e-6
    for axis in range(dim):
        xp, xm = x.copy(), x.copy()
        xp[:, axis] += h
        xm[:, axis] -= h
        fd = (exact_solution(problem, xp)[0] - exact_solution(problem, xm)[0]) / (2*h)
        np.testing.assert_allclose(grad[:, axis], fd, rtol=2e-6, atol=1e-8)


def test_multifrequency_profile_jet_and_neumann_boundary():
    problem = make_problem("cubic", 1, exact_profile="multifrequency")
    x = np.array([[0.231], [0.417], [0.683]])
    _, grad, _ = exact_solution(problem, x)
    h = 1e-7
    fd = (exact_solution(problem, x+h)[0] - exact_solution(problem, x-h)[0])/(2*h)
    np.testing.assert_allclose(grad[:, 0], fd, rtol=2e-6, atol=2e-7)
    boundary = np.array([[0.0], [1.0]])
    assert np.max(np.abs(exact_solution(problem, boundary)[1])) < 1e-11


@pytest.mark.parametrize("model", ["linear", "cubic", "sinh", "pure_p", "regularized_p", "reaction_p"])
@pytest.mark.parametrize("dim", [1, 2])
def test_manufactured_source_is_finite_and_neumann(model, dim):
    problem = make_problem(model, dim, epsilon=0.1 if model == "regularized_p" else None)
    rng = np.random.default_rng(2)
    x = rng.random((30, dim))
    assert source_term(problem, x).shape == (30,)
    assert np.all(np.isfinite(source_term(problem, x)))
    boundary = rng.random((10, dim))
    boundary[:, 0] = 0.0
    _, grad, _ = exact_solution(problem, boundary)
    assert np.max(np.abs(grad[:, 0])) < 1e-12


@pytest.mark.parametrize("model", ["linear", "cubic", "sinh", "pure_p", "regularized_p", "reaction_p"])
def test_energy_directional_derivative(model):
    problem = make_problem(model, 1, epsilon=0.1 if model == "regularized_p" else None)
    rule = segmented_gauss_1d(np.array([0.2, 0.6]), 10)
    x = rule.points
    u = 0.2*np.sin(2*np.pi*x[:, 0])
    grad = (0.4*np.pi*np.cos(2*np.pi*x[:, 0]))[:, None]
    v = np.cos(np.pi*x[:, 0])
    grad_v = (-np.pi*np.sin(np.pi*x[:, 0]))[:, None]
    f = source_term(problem, x)
    analytic = first_variation(problem, u, grad, v, grad_v, f, rule)
    h = 1e-6
    fd = (energy(problem, u+h*v, grad+h*grad_v, f, rule) -
          energy(problem, u-h*v, grad-h*grad_v, f, rule))/(2*h)
    np.testing.assert_allclose(analytic, fd, rtol=3e-5, atol=2e-7)


@pytest.mark.parametrize("p", [3.0, 5.0])
def test_p_laplacian_directional_derivative_for_requested_p(p):
    problem = make_problem("pure_p", 1, p=p)
    rule = segmented_gauss_1d(np.array([0.15, 0.55, 0.82]), 12)
    x = rule.points
    u = 0.15*np.sin(2*np.pi*x[:, 0])
    grad = (0.3*np.pi*np.cos(2*np.pi*x[:, 0]))[:, None]
    v = np.cos(np.pi*x[:, 0])
    grad_v = (-np.pi*np.sin(np.pi*x[:, 0]))[:, None]
    f = source_term(problem, x)
    analytic = first_variation(problem, u, grad, v, grad_v, f, rule)
    h = 2e-6
    fd = (energy(problem, u+h*v, grad+h*grad_v, f, rule) -
          energy(problem, u-h*v, grad-h*grad_v, f, rule))/(2*h)
    np.testing.assert_allclose(analytic, fd, rtol=8e-5, atol=3e-7)


def test_hvp_against_gradient_difference():
    problem = make_problem("reaction_p", 1)
    rule = segmented_gauss_1d(np.array([0.3]), 12)
    x = rule.points
    basis = np.column_stack([x[:, 0], x[:, 0]**2])
    grads = np.zeros((x.shape[0], 2, 1))
    grads[:, 0, 0] = 1.0
    grads[:, 1, 0] = 2*x[:, 0]
    c, z = np.array([0.3, -0.2]), np.array([0.4, 0.5])
    base = (np.zeros(x.shape[0]), np.zeros((x.shape[0], 1)))
    f = source_term(problem, x)
    analytic = coefficient_hvp(problem, basis, grads, c, z, base, rule)
    h = 1e-6
    gp = coefficient_objective(problem, basis, grads, c+h*z, base, f, rule)[1]
    gm = coefficient_objective(problem, basis, grads, c-h*z, base, f, rule)[1]
    np.testing.assert_allclose(analytic, (gp-gm)/(2*h), rtol=8e-5, atol=1e-7)
