#!/usr/bin/env python3
"""Write a hash inventory for the paper and its reproduction materials."""

from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reproducibility" / "manifests" / "public_source_inventory.csv"

INCLUDE = [
    "main.tex",
    "main.pdf",
    "supplement.tex",
    "References/*.bib",
    "sections/*.tex",
    "generated/*.tex",
    "figures/**/*.pdf",
    "figures/**/*.manifest.json",
    "data/raw/cga_cases/*/config.json",
    "data/raw/cga_cases/*/history.csv",
    "data/raw/cga_cases/*/summary.json",
    "data/raw/cga_cases/*/environment.json",
    "data/raw/cga_cases/*/pool_manifest.json",
    "data/raw/sensitivity/**/*.json",
    "data/raw/sensitivity/base_pure/*/states/*.npz",
    "data/raw/sensitivity/epsilon_01/*/states/*.npz",
    "data/raw/baseline/data/fem_raw.csv",
    "data/raw/baseline/data/models/*.npz",
    "data/derived/experiments/cga_metrics_long.csv",
    "data/derived/experiments/epsilon_scan_long.csv",
    "data/derived/experiments/ablation_long.csv",
    "data/derived/experiments/quadrature_sensitivity.csv",
    "data/derived/experiments/rfm_multiseed_raw.csv",
    "data/derived/experiments/rfm_multiseed_summary.csv",
    "data/derived/experiments/anomaly_and_stopping.csv",
    "data/derived/experiments/experiment_validation.json",
    "data/derived/experiments/*.json",
    "data/derived/experiments/*.csv",
    "data/derived/experiments/controlled_baselines/**/*",
    "data/derived/experiments/cost_pilot/**/*",
    "data/derived/experiments/fem_rfm_recomputed/**/*",
    "data/derived/window_diagnostics/*",
    "data/derived/baselines/*",
    "data/manifests/*",
    "reproducibility/configs/*",
    "reproducibility/environment.yml",
    "reproducibility/scripts/*.py",
    "reproducibility/scripts/cga_refactor/**/*",
    "reproducibility/scripts/compare_fem_rfm/**/*",
    "reproducibility/scripts/compare_fem_rfm/configs/*",
    "reproducibility/scripts/compare_fem_rfm/data/cga_cases/**/*",
    "reproducibility/scripts/compare_fem_rfm/data/cga_snapshot/*",
    "reproducibility/scripts/compare_fem_rfm/PACKAGE_LOCAL_README.md",
    "reports/artifact_crosswalk.md",
    "reports/claim_evidence_matrix.md",
    "reports/fem_curve_identity_audit.md",
    "reports/figure_table_numeric_validation.md",
    "reports/numerical_source_check.md",
    "reports/theorem_assumption_use_matrix.md",
    "reports/source/*",
    "README.md",
    "reproducibility/README.md",
]


def category(path: Path) -> str:
    rel = path.relative_to(ROOT)
    if rel.suffix == ".tex" or rel.parts[0] == "References":
        return "paper-source"
    if rel.parts[0] == "figures":
        return "figure"
    if rel.parts[0] == "data":
        return "data"
    if "configs" in rel.parts:
        return "configuration"
    if "scripts" in rel.parts:
        return "scripts"
    return "documentation"


def role(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("data/raw/cga_cases"):
        return "CGA accepted-state source"
    if rel.startswith("data/raw/baseline"):
        return "FEM/RFM source"
    if rel.startswith("data/derived/window_diagnostics"):
        return "finite-window diagnostic"
    if rel.startswith("data/derived/baselines"):
        return "baseline plot/table source"
    if rel.startswith("data/derived/experiments"):
        return "CGA and sensitivity source"
    if rel.startswith("figures"):
        return "generated figure or sidecar"
    if rel.startswith("generated"):
        return "generated TeX table"
    if rel.startswith("reproducibility/configs"):
        return "machine-readable protocol"
    if rel.startswith("reproducibility/code"):
        return "reproduction script or implementation"
    return "paper or package documentation"


def main() -> None:
    paths: set[Path] = set()
    for pattern in INCLUDE:
        paths.update(
            path for path in ROOT.glob(pattern)
            if path.is_file()
            and not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts)
            and path.suffix != ".pyc"
        )
    paths.discard(OUTPUT)
    rows = []
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        payload = path.read_bytes()
        rows.append({
            "schema_version": "cga-public-inventory-v1",
            "category": category(path),
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
            "role": role(path),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} entries to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
