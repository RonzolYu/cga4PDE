from __future__ import annotations

from pathlib import Path
import numpy as np

from cga_refactor.config import SolverConfig, config_for
from cga_refactor.metrics import energy_gap, vp_map
from cga_refactor.solver import energy_consistency_warning, formal_status, resume_cga, run_cga


def test_energy_gap_negative_is_not_absolute():
    result = energy_gap(-1.0, -0.9, 1e-3)
    assert np.isclose(result["raw"], -0.1)
    assert result["status"] == "negative_unresolved"


def test_vp_map_zero_and_shape():
    z = np.zeros((4, 2))
    assert vp_map(z, 4.0).shape == z.shape
    assert np.all(vp_map(z, 4.0) == 0)


def test_energy_warning_is_invariant_to_additive_energy_constant():
    cfg = SolverConfig()
    base = energy_consistency_warning(-10.0, -10.2, -10.0, -9.99, cfg)
    shifted = energy_consistency_warning(999990.0, 999989.8,
                                         999990.0, 999990.01, cfg)
    assert base["quadrature_warning"] == shifted["quadrature_warning"]
    assert base["quadrature_warning_reason"] == shifted["quadrature_warning_reason"]
    np.testing.assert_allclose(base["train_energy_drop"], shifted["train_energy_drop"])
    np.testing.assert_allclose(base["validation_energy_drop"],
                               shifted["validation_energy_drop"])


def test_train_decreases_while_validation_increases_warns_without_exact_solution():
    result = energy_consistency_warning(-3.0, -3.1, -3.0, -2.99, SolverConfig())
    assert result["quadrature_warning"]
    assert result["quadrature_warning_reason"] == "validation_energy_increased"


def test_similar_train_and_validation_decreases_pass():
    result = energy_consistency_warning(-3.0, -3.1, -3.0, -3.099,
                                        SolverConfig(validation_gap_rtol=0.02))
    assert not result["quadrature_warning"]
    assert result["quadrature_warning_reason"] == "none"


def test_formal_status_requires_audit_and_target():
    assert formal_status(False, 512, 512) == (False, "audit_failed")
    assert formal_status(True, 300, 512) == (False, "trusted_partial")
    assert formal_status(True, 512, 512) == (True, "trusted_complete")


def test_tiny_solver_reaches_target(tmp_path: Path):
    cfg = config_for("linear", 1, profile="smoke", seed=12, output_root=str(tmp_path))
    summary = run_cga(cfg)
    assert summary["target_reached"]
    assert summary["accepted_atom_count"] == 2
    assert Path(summary["run_dir"], "history.csv").exists()
    resumed = resume_cga(summary["run_dir"])
    assert resumed["accepted_atom_count"] == summary["accepted_atom_count"]
