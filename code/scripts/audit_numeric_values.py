#!/usr/bin/env python3
"""Audit the package-local numerical records used by the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/compare_fem_rfm/src'))
from compare_fem_rfm.quality import metric_valid, summarize_rfm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "reports" / "numerical_source_check.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    experiments = read_json(DATA / "derived" / "experiments" / "experiment_validation.json")
    diagnostics = read_json(
        DATA / "derived" / "window_diagnostics" / "window_certificate_validation.json"
    )
    artifacts = read_json(DATA / "derived" / "baselines" / "artifact_validation.json")
    rfm_raw = read_csv(DATA / "derived" / "experiments" / "rfm_multiseed_raw.csv")
    rfm_summary = read_csv(DATA / "derived" / "experiments" / "rfm_multiseed_summary.csv")
    budget = read_csv(DATA / "derived" / "window_diagnostics" / "transfer_budget.csv")
    common = read_csv(DATA / "derived" / "baselines" / "baseline_common_grid.csv")
    problems = read_json(ROOT / "config" / "problems.json")

    checks: list[tuple[str, bool, str]] = []
    checks.append(("experiment data validation", experiments.get("passed") is True,
                   "data/derived/experiments/experiment_validation.json"))
    checks.append(("RFM raw row count", len(rfm_raw) == experiments.get("rfm", {}).get("rows"),
                   "data/derived/experiments/rfm_multiseed_raw.csv"))
    checks.append(("RFM prescribed seed set", experiments.get("rfm", {}).get("seeds") == list(range(201, 211)),
                   "data/derived/experiments/experiment_validation.json"))
    checks.append(("RFM count conservation",
                   all(int(row["success_count"]) + int(row["failure_count"]) == int(row["seed_count"])
                       for row in rfm_summary),
                   "data/derived/experiments/rfm_multiseed_summary.csv"))

    c5 = [row for row in rfm_summary if row["case_id"] == "C5" and int(row["dof"]) == 256]
    checks.append(("C5 endpoint keeps prescribed denominator", len(c5) == 1
                   and int(c5[0]["seed_count"]) == 10
                   and int(c5[0]["natural_error_sample_count"]) <= int(c5[0]["success_count"]),
                   "data/derived/experiments/rfm_multiseed_summary.csv"))
    rebuilt = summarize_rfm(rfm_raw)
    stored = {(r['case_id'], int(r['dof'])):r for r in rfm_summary}
    checks.append(('metric-specific summary counts', all(
        all(str(stored[(r['case_id'], r['dof'])][m+'_sample_count']) == str(r[m+'_sample_count'])
            for m in ('energy_gap','natural_error','v_error')) for r in rebuilt),
        'data/derived/experiments/rfm_multiseed_summary.csv'))

    case17 = [case for case in problems["cases"] if case["paper_case_id"] == "17"]
    checks.append(("ID17 low-frequency profile", len(case17) == 1
                   and case17[0]["exact_profile"] == "low_frequency"
                   and int(case17[0]["accepted"]) == 141,
                   "config/problems.json"))

    checks.append(("finite-window diagnostic validation", diagnostics.get("passed") is True,
                   "data/derived/window_diagnostics/window_certificate_validation.json"))
    checks.append(("finite-window claim scope", diagnostics.get("continuous_certificate") is False
                   and diagnostics.get("finite_rate_certificate") is False,
                   "data/derived/window_diagnostics/window_certificate_validation.json"))
    checks.append(("finite-window transfer rows", len(budget) == 48,
                   "data/derived/window_diagnostics/transfer_budget.csv"))

    checks.append(("artifact validation", artifacts.get("passed") is True,
                   "data/derived/baselines/artifact_validation.json"))
    checks.append(("figure count metadata", isinstance(artifacts.get("figure_count_pdf"), int)
                   and artifacts.get("figure_count_pdf") > 0,
                   "data/derived/baselines/artifact_validation.json"))
    checks.append(("common-grid interpolation brackets", artifacts.get("unbracketed_interpolations") == 0
                   and all((row.get("is_interpolated") != "True")
                           or int(row["left_dof"]) < int(row["dof"]) < int(row["right_dof"])
                           for row in common),
                   "data/derived/baselines/baseline_common_grid.csv"))
    checks.append(("baseline plot scope", artifacts.get("baseline_theory_lines") is False
                   and artifacts.get("baseline_order_plots") is False
                   and artifacts.get("main_baseline_points_are_actual") is True,
                   "data/derived/baselines/artifact_validation.json"))
    eligible = {case for case, row in artifacts.get("rfm_endpoint_status", {}).items()
                if row.get("aggregate_ranking_eligible")}
    config = read_json(ROOT / 'config/plots.json')
    complete = {case for case, dof in config['baseline_statistical_endpoints'].items()
                if sum(metric_valid(r, 'natural_error') for r in rfm_raw
                       if r['case_id'] == case and int(r['dof']) == int(dof)) == 10}
    checks.append(("aggregate ranking uses complete endpoints", eligible == complete,
                   "data/derived/baselines/artifact_validation.json"))

    checks.append(("quadrature sensitivity rows", len(read_csv(DATA / "derived" / "experiments" / "quadrature_sensitivity.csv")) == 120,
                   "data/derived/experiments/quadrature_sensitivity.csv"))

    source_paths = [ROOT / "tex/main.tex"]
    source_paths.extend(sorted((ROOT / "tex/sections").glob("*.tex")))
    source_paths.extend(sorted((ROOT / "tex/generated").glob("*.tex")))
    internal_terms = [
        "mis" + r"sion[ _-]?[0-9v]*",
        r"according to (?:the )?(?:" + "re" + "viewer|re" + "quest)",
        "as " + "requested",
        r"we (?:have )?" + "revised",
        "previous" + " version",
        "old" + " version",
        "completion" + " report",
        "must be " + "corrected",
        "displayed rather than " + "removed",
        "remain " + "visible",
        "beau" + "tif",
        "cherry" + r"[ .-]?pick",
    ]
    internal_pattern = re.compile("|".join(internal_terms), re.IGNORECASE)
    hits = []
    for path in source_paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if internal_pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{line_number}")
    checks.append(("paper source contains no internal-process language", not hits,
                   "main.tex, supplement.tex, sections/*.tex, generated/*.tex"))

    failures = [name for name, ok, _ in checks if not ok]
    lines = [
        "# Numeric source check",
        "",
        "This audit checks the package-local validation records, statistical counts, interpolation brackets,",
        "claim scope, and source-language constraints used by the paper.",
        "",
        "| Check | Status | Source |",
        "|---|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if ok else 'FAIL'} | `{source}` |"
                 for name, ok, source in checks)
    lines.extend(["", f"Result: **{'PASS' if not failures else 'FAIL'}** "
                  f"({len(checks) - len(failures)}/{len(checks)} checks)."])
    if hits:
        lines.extend(["", "Internal-process language matches:", *[f"- `{hit}`" for hit in hits]])
    if failures:
        lines.extend(["", "Failed checks:", *[f"- {name}" for name in failures]])
    # Keep machine-readable checks and seed-level counts beside the human report.
    check_dir = ROOT / "result" / "review_checks"
    check_dir.mkdir(parents=True, exist_ok=True)
    check_payload = {"status": "PASS" if not failures else "FAIL",
                     "checks": [{"name": n, "passed": ok, "source": src}
                                for n, ok, src in checks]}
    (check_dir / "numeric_checks.json").write_text(json.dumps(check_payload, indent=2) + "\n")
    with (check_dir / "rfm_seed_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "dof", "seed_count", "success_count", "failure_count"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(rfm_summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{'PASS' if not failures else 'FAIL'}: {len(checks) - len(failures)}/{len(checks)} checks")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
