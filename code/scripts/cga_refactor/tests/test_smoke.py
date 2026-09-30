from __future__ import annotations

from pathlib import Path
import pytest

from cga_refactor.config import MODELS, config_for
from cga_refactor.reporting import make_run_report
from cga_refactor.solver import run_cga


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dim", [1, 2])
def test_all_twelve_cases_end_to_end(model, dim, tmp_path: Path):
    cfg = config_for(model, dim, profile="smoke", seed=17,
                     output_root=str(tmp_path/model/f"d{dim}"))
    summary = run_cga(cfg)
    assert summary["target_reached"]
    assert summary["accepted_atom_count"] == 2
    files = make_run_report(summary["run_dir"])
    assert all(path.exists() for path in files)
