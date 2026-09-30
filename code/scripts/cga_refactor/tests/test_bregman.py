import numpy as np
import pytest

from cga_refactor.metrics import bregman_divergence
from cga_refactor.problems import exact_solution, make_problem
from cga_refactor.quadrature import segmented_gauss_1d


@pytest.mark.parametrize(
    "name,epsilon", [("linear", None), ("cubic", None), ("sinh", None),
                     ("pure_p", None), ("regularized_p", 0.1),
                     ("reaction_p", None)],
)
def test_bregman_is_zero_at_exact_state(name, epsilon):
    problem = make_problem(name, 1, 4.0, epsilon)
    rule = segmented_gauss_1d(np.array([0.5]), 12)
    exact = exact_solution(problem, rule.points)
    assert bregman_divergence(problem, exact[0], exact[1], exact, rule) == pytest.approx(0.0)


@pytest.mark.parametrize("name,epsilon", [("pure_p", None), ("regularized_p", 0.1)])
def test_bregman_is_nonnegative_for_perturbed_state(name, epsilon):
    problem = make_problem(name, 1, 4.0, epsilon)
    rule = segmented_gauss_1d(np.array([0.5]), 12)
    exact = exact_solution(problem, rule.points)
    u = exact[0] + 0.05 * np.cos(4.0 * np.pi * rule.points[:, 0])
    grad = exact[1].copy()
    grad[:, 0] -= 0.2 * np.pi * np.sin(4.0 * np.pi * rule.points[:, 0])
    assert bregman_divergence(problem, u, grad, exact, rule) >= 0.0
