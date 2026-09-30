from __future__ import annotations

import numpy as np

from cga_refactor.config import config_for
from cga_refactor.dictionary import (build_pool, calibrate_pool, evaluate_atoms,
                                     pool_hash)
from cga_refactor.problems import make_problem
from cga_refactor.quadrature import segmented_gauss_1d
from cga_refactor.step1 import effective_rank, innovation
from cga_refactor.step1 import classify_step1_stop
from types import SimpleNamespace


def test_relu_power_value_and_gradient():
    x = np.array([[0.2], [0.7]])
    w, b = np.array([[1.0], [-1.0]]), np.array([-0.1, 0.9])
    values, grads = evaluate_atoms(x, w, b, 3)
    h = 1e-7
    vp = evaluate_atoms(x+h, w, b, 3)[0]
    vm = evaluate_atoms(x-h, w, b, 3)[0]
    np.testing.assert_allclose(grads[:, :, 0], (vp-vm)/(2*h), rtol=2e-6, atol=1e-8)
    assert values.shape == (2, 2)


def test_pool_seed_reproducibility_and_hash():
    cfg = config_for("linear", 1, profile="smoke")
    problem = make_problem("linear", 1)
    a = build_pool(problem, cfg, None, "candidate")
    b = build_pool(problem, cfg, None, "candidate")
    np.testing.assert_array_equal(a.w, b.w)
    np.testing.assert_array_equal(a.b, b.b)
    assert pool_hash(a) == pool_hash(b)


def test_pool_scales_and_innovation():
    cfg = config_for("pure_p", 1, profile="smoke")
    problem = make_problem("pure_p", 1)
    pool = build_pool(problem, cfg, None, "candidate")
    rule = segmented_gauss_1d(np.linspace(0.1, 0.9, 5), 6)
    calibrate_pool(problem, pool, rule, 8)
    assert np.all(pool.scales[pool.available] > 0)
    column = np.arange(10.0)
    q = (column/np.linalg.norm(column))[:, None]
    assert innovation(column, q)["relative"] < 1e-12


def test_effective_rank_known_spectrum():
    matrix = np.diag([1.0, 1e-4, 1e-13])
    rank, singular, condition = effective_rank(matrix, 1e-12, 1e-10)
    assert rank == 2
    assert condition == 1e4


def test_coverage_warning_does_not_stop_an_admissible_step():
    cfg = config_for("linear", 1, profile="smoke")
    state = SimpleNamespace(low_oracle_count=cfg.solver.oracle_patience - 1,
                            coverage_warning=False, had_coverage_warning=False,
                            initial_best_score=1.0)
    selection = {"admissible": True, "coverage_warning": True,
                 "dual_score_abs": 1.0}
    pool = SimpleNamespace(available=np.ones(1, dtype=bool))
    assert classify_step1_stop(selection, state, pool, cfg) is None
    assert state.coverage_warning
    assert state.had_coverage_warning
