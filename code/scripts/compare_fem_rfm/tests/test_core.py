import numpy as np
import pytest

from compare_fem_rfm.analysis import interpolate_series, loglog_interpolate
from compare_fem_rfm.features import sample_parameters
from compare_fem_rfm.fem import FEMModel, evaluate_fem_model, fem_dof
from compare_fem_rfm.quality import metric_valid
from compare_fem_rfm.metrics import evaluate_fields
from compare_fem_rfm.problems import CASES, exact_jet, forcing
from compare_fem_rfm.quadrature import composite_gauss
from compare_fem_rfm.feature_solver import (CachedHessianAction, hessian_vector_product,
                                           energy_and_gradient, solve_feature_coefficients)


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


def test_cubic_fem_evaluation_is_invariant_to_probe_batch_size():
    rng = np.random.default_rng(28)
    model = FEMModel(1, 3, 5, rng.normal(size=fem_dof(1, 3, 5, False)))
    points = np.linspace(.011, .989, 39)[:, None]
    reference = evaluate_fem_model(model, points, batch_size=100)
    for size in (1, 3, 7, 13):
        values, grads = evaluate_fem_model(model, points, batch_size=size)
        np.testing.assert_allclose(values, reference[0], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(grads, reference[1], rtol=1e-13, atol=1e-13)
    h = 1e-7
    plus = evaluate_fem_model(model, points+h)[0]
    minus = evaluate_fem_model(model, points-h)[0]
    np.testing.assert_allclose(reference[1][:, 0], (plus-minus)/(2*h), rtol=2e-8, atol=2e-8)


def test_negative_energy_difference_retains_its_sign():
    spec = CASES["C1"]
    points = np.array([[.17]])
    exact, grad, _ = exact_jet(spec, points)
    # At one quadrature point, the minimizer of the value-dependent linear
    # potential need not be the continuous manufactured solution.
    metrics = evaluate_fields(spec, points, np.ones(1), forcing(spec, points), grad)
    assert metrics["energy_gap_signed"] < 0
    assert metrics["energy_gap"] == metrics["energy_gap_signed"]


def test_metric_validity_requires_its_own_successful_quadrature_audit():
    row = {"solver_success": "True", "energy_gap": -1e-8,
           "natural_error": .1, "v_error": .2, "audit_energy_gap_rel_delta": .001,
           "audit_natural_rel_delta": .008, "audit_v_rel_delta": .02}
    assert metric_valid(row, "natural_error")
    assert not metric_valid(row, "energy_gap")
    assert not metric_valid(row, "v_error")
    row.pop("audit_natural_rel_delta")
    assert not metric_valid(row, "natural_error")


@pytest.mark.parametrize('case', ['C2','C3','C4','C5'])
def test_cached_hessian_matches_matrix_free_action_and_directional_derivative(case):
    spec=CASES[case]
    rng=np.random.default_rng(91)
    phi=rng.normal(size=(41,7))
    grad=rng.normal(size=(41,7,spec.dim))
    weights=rng.uniform(.1,1,41)
    load=rng.normal(size=41)
    action=CachedHessianAction(spec,phi,grad,weights)
    for _ in range(2):
        coefficients=rng.normal(size=7)/3
        direction=rng.normal(size=7)
        expected=hessian_vector_product(spec,coefficients,direction,phi,grad,weights)
        actual=action(coefficients,direction)
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-12)
        h=1e-6
        plus=energy_and_gradient(spec,coefficients+h*direction,phi,grad,load,weights)[1]
        minus=energy_and_gradient(spec,coefficients-h*direction,phi,grad,load,weights)[1]
        np.testing.assert_allclose(actual,(plus-minus)/(2*h),rtol=1e-8,atol=2e-8)
        # A different direction at an unchanged point uses the same matrix.
        other=rng.normal(size=7)
        np.testing.assert_allclose(action(coefficients,other),
            hessian_vector_product(spec,coefficients,other,phi,grad,weights),rtol=2e-13,atol=2e-12)


@pytest.mark.parametrize('case', ['C4','C5'])
def test_hessian_backends_recover_a_known_coefficient_minimizer(case):
    spec=CASES[case]
    rng=np.random.default_rng(12)
    phi=rng.normal(size=(51,5))
    grad=rng.normal(size=(51,5,spec.dim))
    weights=np.full(51,1/51)
    exact=np.array([.2,-.1,.3,.05,-.15])
    _,load_vector=energy_and_gradient(spec,exact,phi,grad,np.zeros(51),weights)
    load=np.linalg.lstsq(phi.T*weights[None,:],load_vector,rcond=None)[0]
    config=dict(lbfgs_maxiter=1,lbfgs_gtol=1e-12,lbfgs_ftol=1e-14,newtoncg_maxiter=100)
    for backend in ('matrix_free','cached_matrix'):
        solved=solve_feature_coefficients(spec,phi,grad,load,weights,
                                         dict(config,hessian_backend=backend))
        assert solved.success
        assert solved.hessian_backend==backend
        # Coefficients have finite stopping accuracy under the declared 2e-6
        # scaled-gradient criterion; the manufactured minimum is independent.
        np.testing.assert_allclose(solved.coefficients,exact,rtol=0,atol=1e-6)
        value,_=energy_and_gradient(spec,solved.coefficients,phi,grad,load,weights)
        exact_value,_=energy_and_gradient(spec,exact,phi,grad,load,weights)
        assert value <= exact_value+1e-11
