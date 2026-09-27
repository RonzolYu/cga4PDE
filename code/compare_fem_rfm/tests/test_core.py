import numpy as np
import pytest

from compare_fem_rfm.analysis import interpolate_series, loglog_interpolate
from compare_fem_rfm.features import sample_parameters
from compare_fem_rfm.fem import fem_dof
from compare_fem_rfm.metrics import evaluate_fields
from compare_fem_rfm.problems import CASES, exact_jet
from compare_fem_rfm.quadrature import composite_gauss


def test_exact_solution_has_zero_evaluation_error():
    for spec in CASES.values():
        rule = composite_gauss(spec.dim, 12, 4)
        u, grad, _ = exact_jet(spec, rule.points)
        metrics = evaluate_fields(spec, rule.points, rule.weights, u, grad)
        assert metrics["natural_error"] == pytest.approx(0.0, abs=1e-15)
        if spec.model == "pure_p":
            assert metrics["v_error"] == pytest.approx(0.0, abs=1e-15)


def test_loglog_interpolation_and_no_extrapolation():
    assert loglog_interpolate(8, 1.0, 32, 1.0 / 16.0, 16) == pytest.approx(0.25)
    rows = [{"dof": 8, "error": 1.0}, {"dof": 32, "error": 1.0 / 16.0}]
    assert interpolate_series(rows, "error", 16)[0] == pytest.approx(0.25)
    assert interpolate_series(rows, "error", 4) is None
    assert interpolate_series(rows, "error", 64) is None


def test_fem_zero_mean_dof_is_one_less():
    for dim in (1, 2):
        for degree in (1, 2, 3):
            assert fem_dof(dim, degree, 2, True) + 1 == fem_dof(dim, degree, 2, False)


def test_registered_c4_frozen_endpoint():
    assert CASES["C4"].cga_expected_accepted == 141
    assert CASES["C4"].cga_target == 256


def test_random_pool_is_deterministic_and_prefixable():
    cfg = {"crossing_fraction": 0.9, "sobol_fraction": 0.75, "anchor_padding": 0.05}
    w1, b1 = sample_parameters(2, 512, 201, cfg)
    w2, b2 = sample_parameters(2, 512, 201, cfg)
    assert np.array_equal(w1, w2)
    assert np.array_equal(b1, b2)
    assert w1[:128].shape == (128, 2)


def test_problem_hashes_are_unique_and_stable():
    hashes = [spec.problem_hash for spec in CASES.values()]
    assert len(set(hashes)) == len(hashes)
    assert all(len(value) == 64 for value in hashes)

