"""Independent identities for the supported p >= 2 variational models."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cga_refactor.config import config_for, frozen_case_config, FROZEN_CASES
from cga_refactor.dictionary import build_pool, calibrate_pool, evaluate_atoms
from cga_refactor.metrics import bregman_divergence, vp_map
from cga_refactor.problems import (
    coefficient_hvp, coefficient_objective, energy, exact_solution,
    first_variation, flux, make_problem, source_term,
)
from cga_refactor.quadrature import tensor_gauss_2d
from cga_refactor.step1 import _pool_scores, feature_columns


@pytest.mark.parametrize("dim", [1, 2])
def test_batch_columns_equal_individual_columns_and_integral_gram(dim):
    rng = np.random.default_rng(13)
    values = rng.normal(size=(11, 4))
    grads = rng.normal(size=(11, 4, dim))
    rule = SimpleNamespace(weights=rng.uniform(.1, 1, 11))
    batch = feature_columns(values, grads, rule)
    individual = np.column_stack([
        feature_columns(values[:, j:j+1], grads[:, j:j+1], rule)[:, 0]
        for j in range(values.shape[1])
    ])
    expected = (np.einsum("qi,qj,q->ij", values, values, rule.weights)
                + np.einsum("qid,qjd,q->ij", grads, grads, rule.weights))
    np.testing.assert_array_equal(batch, individual)
    np.testing.assert_allclose(batch.T @ batch, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("p", [2., 3., 4., 5.])
def test_energy_derivative_and_bregman_identity_on_same_quadrature(dim, p):
    rule = tensor_gauss_2d(3, 3)
    if dim == 1:
        rule = SimpleNamespace(points=rule.points[:, :1], weights=rule.weights)
    problem = make_problem("pure_p", dim, p)
    exact = exact_solution(problem, rule.points)
    load = source_term(problem, rule.points)
    with np.errstate(all="raise"):
        diffusion, reaction = flux(problem, np.zeros(5), np.zeros((5, dim)))
        assert np.all(diffusion == 0) and np.all(reaction == 0)
        assert np.all(vp_map(np.zeros((5, dim)), p) == 0)
    u, grad = .7 * exact[0], .7 * exact[1]
    v, grad_v = .3 * exact[0], .3 * exact[1]
    derivative = first_variation(problem, u, grad, v, grad_v, load, rule)
    h = 1e-5
    finite_difference = (energy(problem, u+h*v, grad+h*grad_v, load, rule)
                         - energy(problem, u-h*v, grad-h*grad_v, load, rule)) / (2*h)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-8, atol=1e-9)
    # A manufactured continuous solution need not be stationary for a coarse
    # quadrature.  Keep its discrete first variation in the exact identity.
    linear_term = first_variation(problem, exact[0], exact[1],
                                  u-exact[0], grad-exact[1], load, rule)
    gap = energy(problem, u, grad, load, rule) - energy(problem, *exact[:2], load, rule)
    np.testing.assert_allclose(gap, bregman_divergence(problem, u, grad, exact, rule)
                               + linear_term, rtol=1e-12, atol=1e-12)


def test_unsupported_subquadratic_model_is_rejected_explicitly():
    with pytest.raises(ValueError, match="p must be >= 2"):
        make_problem("pure_p", 2, 1.5)


def test_hessian_vector_product_matches_directional_gradient_difference():
    rng = np.random.default_rng(31)
    problem = make_problem("pure_p", 2, 4.)
    rule = tensor_gauss_2d(3, 3)
    values = rng.normal(size=(rule.points.shape[0], 3))
    grads = rng.normal(size=(rule.points.shape[0], 3, 2))
    base = (rng.normal(size=rule.points.shape[0]), rng.normal(size=(rule.points.shape[0], 2)))
    c, direction = rng.normal(size=(2, 3))
    load = source_term(problem, rule.points)
    h = 1e-6
    plus = coefficient_objective(problem, values, grads, c+h*direction, base, load, rule)[1]
    minus = coefficient_objective(problem, values, grads, c-h*direction, base, load, rule)[1]
    np.testing.assert_allclose(coefficient_hvp(problem, values, grads, c, direction, base, rule),
                               (plus-minus)/(2*h), rtol=2e-8, atol=1e-8)


def test_fast_scores_equal_directional_derivatives():
    problem = make_problem("pure_p", 2, 4.)
    rule = tensor_gauss_2d(4, 3)
    cfg = replace(config_for("pure_p", 2, profile="smoke"), relu_power=3)
    pool = build_pool(problem, cfg, None, "candidate")
    calibrate_pool(problem, pool, rule)
    u, grad, _ = exact_solution(problem, rule.points)
    state = SimpleNamespace(u=.4*u, grad_u=.4*grad)
    load = source_term(problem, rule.points)
    scores = _pool_scores(problem, state, pool, rule, load, 8)
    idx = np.flatnonzero(pool.available)
    values, grads = evaluate_atoms(rule.points, pool.w[idx], pool.b[idx], pool.k)
    values -= pool.centers[idx]
    expected = np.abs(first_variation(problem, state.u, state.grad_u,
                                      values, grads, load, rule) / pool.scales[idx])
    np.testing.assert_allclose(scores[idx], expected, rtol=2e-13, atol=1e-14)


@pytest.mark.parametrize("case_id", list(FROZEN_CASES))
def test_frozen_case_constructor_matches_explicit_registry(case_id):
    cfg = frozen_case_config(case_id)
    assert cfg.p == FROZEN_CASES[case_id]["p"]
    assert cfg.relu_power == FROZEN_CASES[case_id]["relu_power"]


@pytest.mark.parametrize('profile', ['smoke', 'report', 'formal'])
def test_default_and_explicit_two_dimensional_configs_for_every_profile(profile):
    from cga_refactor.config import MODELS
    for model in MODELS:
        default = config_for(model, 2, profile=profile)
        assert default.relu_power == 3
        for power in (1, 3):
            explicit = config_for(model, 2, profile=profile, relu_power=power)
            assert explicit.relu_power == power


def test_two_dimensional_restart_recovers_missing_qr(tmp_path):
    """A legacy checkpoint rebuilt in batch must select the same next atom."""
    import json
    import shutil
    from pathlib import Path
    from cga_refactor.config import config_for
    from cga_refactor.solver import run_cga, resume_cga

    summary=run_cga(config_for('linear',2,profile='smoke',seed=17,
                              output_root=str(tmp_path/'initial')))
    original=Path(summary['run_dir'])
    runs=[]
    for name in ('saved_qr','reconstructed_qr'):
        run=tmp_path/name
        shutil.copytree(original,run)
        config=json.loads((run/'config.json').read_text())
        config['target_accepted']=3
        (run/'config.json').write_text(json.dumps(config))
        (run/'summary.json').unlink()
        runs.append(run)
    with np.load(runs[1]/'checkpoint.npz') as saved:
        legacy={name:saved[name] for name in saved.files if name not in {'q_basis','r_matrix'}}
    np.savez_compressed(runs[1]/'checkpoint.npz',**legacy)
    for run in runs:
        assert resume_cga(run)['accepted_atom_count']==3
    with np.load(runs[0]/'checkpoint.npz') as direct, np.load(runs[1]/'checkpoint.npz') as restored:
        np.testing.assert_array_equal(direct['accepted_indices'],restored['accepted_indices'])
        np.testing.assert_allclose(direct['coefficients'],restored['coefficients'],rtol=2e-10,atol=2e-12)
        np.testing.assert_allclose(direct['q_basis']@direct['q_basis'].T,
                                   restored['q_basis']@restored['q_basis'].T,rtol=2e-10,atol=2e-12)
