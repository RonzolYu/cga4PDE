from __future__ import annotations

import numpy as np
import pytest

from cga_refactor.problems import make_problem, source_term
from cga_refactor.quadrature import segmented_gauss_1d, sobol_rule, tensor_gauss_2d


def test_segmented_relu_integral():
    t = 0.37
    rule = segmented_gauss_1d(np.array([t]), 6)
    value = np.dot(rule.weights, np.maximum(rule.points[:, 0]-t, 0.0)**3)
    np.testing.assert_allclose(value, (1-t)**4/4, atol=2e-14)


def test_tensor_and_sobol_integrate_constant():
    for rule in (tensor_gauss_2d(5, 3), sobol_rule(2, 9, 4)):
        np.testing.assert_allclose(rule.weights.sum(), 1.0, atol=1e-14)
        assert rule.points.shape[1] == 2


def test_independent_sobol_rules_have_distinct_hashes():
    assert sobol_rule(2, 8, 1).hash != sobol_rule(2, 8, 2).hash


def test_fix124a_2d_rule_sizes_and_reproducibility():
    train = tensor_gauss_2d(64, 3)
    valid = sobol_rule(2, 17, 302)
    audit = sobol_rule(2, 19, 412)
    assert train.points.shape == (36_864, 2)
    assert valid.points.shape == (131_072, 2)
    assert audit.points.shape == (524_288, 2)
    assert valid.hash != audit.hash
    assert train.hash == tensor_gauss_2d(64, 3).hash
    assert train.points.dtype == np.float64 and train.weights.dtype == np.float64
    assert np.all(train.weights > 0.0)
    np.testing.assert_allclose(train.weights.sum(), 1.0, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("n_cells,order", [(0, 3), (4, 0), (-1, 2)])
def test_tensor_gauss_rejects_nonpositive_rule_parameters(n_cells, order):
    with pytest.raises(ValueError):
        tensor_gauss_2d(n_cells, order)


@pytest.mark.parametrize("p", [3.0, 5.0])
def test_p_laplacian_critical_point_segmentation_restores_compatibility(p):
    rule = segmented_gauss_1d(np.array([0.5]), 10)
    problem = make_problem("pure_p", 1, p=p)
    residual = np.dot(rule.weights, source_term(problem, rule.points))
    assert abs(residual) < 1e-11
