from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from cga_refactor.config import config_for
from cga_refactor.output import create_run_dir
from cga_refactor.problems import make_problem
from cga_refactor.reporting import (aggregate_seeds, anchor_reference_line,
                                    cga_theory_reference, is_formal_eligible,
                                    local_orders, theory_rates)
from cga_refactor.run import exp0819_configs, exp_fix124a_configs


def test_run_directory_never_overwrites(tmp_path: Path):
    cfg = config_for("linear", 1, profile="smoke", output_root=str(tmp_path))
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = create_run_dir(tmp_path, cfg, stamp)
    second = create_run_dir(tmp_path, cfg, stamp)
    assert first != second
    assert first.exists() and second.exists()


def test_dyadic_local_order():
    n = 2**np.arange(5)
    errors = n**(-1.75)
    np.testing.assert_allclose(local_orders(errors)[1:], 1.75, atol=2e-14)
    reference = anchor_reference_line(n.astype(float), errors, 1.75)
    np.testing.assert_allclose(reference, errors)


def test_exp0819_matrix_contains_formal_base_and_requested_sensitivities(tmp_path: Path):
    configs = exp0819_configs(output_root=str(tmp_path))
    assert len(configs) == 17
    assert sum(cfg.phase == "formal_base" for cfg in configs) == 12
    assert all(cfg.target_accepted == (256 if cfg.dim == 1 else 512) for cfg in configs)
    signatures = {(cfg.model, cfg.p, cfg.relu_power, cfg.dim) for cfg in configs}
    for p in (3.0, 5.0):
        assert ("pure_p", p, 1, 1) in signatures
        assert ("pure_p", p, 3, 2) in signatures
    assert ("pure_p", 4.0, 3, 1) in signatures


def test_exp_fix124a_matrix_has_17_unique_cases_and_only_seed201(tmp_path: Path):
    configs = exp_fix124a_configs(output_root=str(tmp_path))
    assert len(configs) == 17
    assert len({(cfg.model, cfg.p, cfg.relu_power, cfg.dim) for cfg in configs}) == 17
    assert {cfg.seed for cfg in configs} == {201}
    assert all(cfg.target_accepted == (256 if cfg.dim == 1 else 512)
               for cfg in configs)
    for cfg in configs:
        if cfg.dim == 2:
            assert (cfg.quadrature.train_cells_2d, cfg.quadrature.train_order,
                    cfg.quadrature.validation_sobol_power,
                    cfg.quadrature.audit_sobol_power) == (64, 3, 17, 19)


def test_audit_failed_run_is_not_formal_or_aggregated(tmp_path: Path):
    run_dir = tmp_path / "failed"
    run_dir.mkdir()
    summary = {"run_dir": str(run_dir), "model": "pure_p", "p": 4.0,
               "relu_power": 3, "dim": 2, "epsilon": None, "seed": 201,
               "accepted_atom_count": 512, "trusted_atom_count": 512,
               "target_accepted": 512, "formal_status": "audit_failed",
               "audit_passed": False, "metrics": {"natural_rel": 1e-3}}
    import json
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert not is_formal_eligible(summary)
    assert aggregate_seeds([run_dir]) == []


def test_reference_rates_follow_local_hessian_woga_theory():
    linear = theory_rates(make_problem("linear", 1), dim=1, k=3)
    assert linear["natural"] == 3.0
    assert linear["energy"] == 6.0
    pure_p = theory_rates(make_problem("pure_p", 2, p=4), dim=2, k=3)
    assert pure_p["natural"] == 1.75
    assert pure_p["natural_global"] == 0.875
    assert pure_p["energy"] == 3.5
    assert pure_p["quasi"] == 1.75


def test_cga_theory_0820_reference_rates_are_kept_separate():
    semilinear = cga_theory_reference(make_problem("cubic", 1))
    assert semilinear["baseline"] == {"energy": 1.0, "natural": 0.5,
                                      "quasi": None}
    assert semilinear["enhanced"] is None

    expected = {
        3.0: {"energy": 3.0, "natural": 1.0, "quasi": 1.5},
        4.0: {"energy": 2.0, "natural": 0.5, "quasi": 1.0},
        5.0: {"energy": 5.0 / 3.0, "natural": 1.0 / 3.0,
              "quasi": 5.0 / 6.0},
    }
    for p, rates in expected.items():
        reference = cga_theory_reference(make_problem("pure_p", 1, p=p))
        assert reference["baseline"] == {"energy": 1.0, "natural": 1.0 / p,
                                          "quasi": 0.5}
        assert reference["enhanced"] == rates
