#!/usr/bin/env python3
"""Build an explicit, scope-labeled cost summary for the archived artifacts.

The selection-variant runs expose end-to-end process times, whereas the FEM/RFM
comparison runner exposes per-state solve times.  They are written to one CSV
with a scope column, but are never silently treated as the same timing measure.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[3]
BASELINE = PACKAGE_ROOT / "data" / "derived" / "experiments" / "controlled_baselines"
COMPARE = PACKAGE_ROOT / "data" / "derived" / "experiments" / "fem_rfm_recomputed"
PILOT = PACKAGE_ROOT / "data" / "derived" / "experiments" / "cost_pilot"
OUT = PILOT


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    if value in (None, "", "NA", "nan", "NaN"):
        return None
    return float(value)


def q(values: list[float]) -> tuple[float, float, float]:
    values = sorted(values)
    if not values:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0], values[0]
    return (
        statistics.quantiles(values, n=4, method="inclusive")[0],
        statistics.median(values),
        statistics.quantiles(values, n=4, method="inclusive")[2],
    )


def append_group(out: list[dict], case: str, method: str, width: int,
                 rows: list[dict[str, str]], *, scope: str, source: str,
                 dof_note: str = "") -> None:
    successful = [r for r in rows if str(r.get("status", "success")).lower() in {"success", "completed", "true"}
                  and str(r.get("success", "true")).lower() not in {"false", "0"}]
    time_values = [number(r.get("wall_end_to_end_sec") or r.get("wall_time_sec")) for r in successful]
    time_values = [x for x in time_values if x is not None]
    error_values = [number(r.get("final_h1_or_w1p_error") or r.get("natural_error")) for r in successful]
    error_values = [x for x in error_values if x is not None]
    energy_values = [number(r.get("final_energy_gap") or r.get("energy_gap")) for r in successful]
    energy_values = [x for x in energy_values if x is not None]
    tq1, tmed, tq3 = q(time_values)
    eq1, emed, eq3 = q(error_values)
    gq1, gmed, gq3 = q(energy_values)
    out.append({
        "case_id": case,
        "method": method,
        "target_width": width,
        "observed_dof": ";".join(sorted({str(r.get("dof", r.get("final_width", width))) for r in successful})),
        "n_total": len(rows),
        "n_success": len(successful),
        "success_rate": (len(successful) / len(rows)) if rows else math.nan,
        "wall_time_q1_sec": tq1,
        "wall_time_median_sec": tmed,
        "wall_time_q3_sec": tq3,
        "natural_error_median": emed,
        "energy_gap_median": gmed,
        "timing_scope": scope,
        "dof_note": dof_note,
        "source": source,
    })


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_rows = []
    for name in ("baseline_pilot_c1c2c4.csv", "baseline_c3c5.csv"):
        baseline_rows.extend(read_csv(BASELINE / name))
    rows: list[dict] = []
    for case in ("C1", "C2", "C3", "C4", "C5"):
        for method in ("CGA-FP", "RD-WOGA", "Random-FC"):
            group = [r for r in baseline_rows if r["case_id"] == case and r["method"] == method
                     and int(r["final_width"]) == 8]
            append_group(rows, case, method, 8, group,
                         scope="end-to-end run timing (selection campaign)",
                         source="baseline_campaign/formal_pilot",
                         dof_note="accepted width exactly 8")

    fem = read_csv(COMPARE / "fem_raw.csv")
    rfm = read_csv(COMPARE / "rfm_raw.csv")
    # The comparison runner reports actual FEM state DOFs.  Select the state
    # closest to eight and retain the observed DOF instead of interpolating a
    # wall time.  RFM has an exact width-eight state.
    for case in ("C1", "C2", "C3", "C4", "C5"):
        for variant, label in (("p1", "FEM P1"), ("p3", "FEM P3")):
            candidates = [r for r in fem if r["case_id"] == case and r["variant"] == variant]
            if candidates:
                nearest = min(candidates, key=lambda r: abs(int(r["dof"]) - 8))
                append_group(rows, case, label, 8, [nearest],
                             scope="per-state solver/evaluation timing (comparison runner)",
                             source="compare_fem_rfm/fem_raw.csv",
                             dof_note=f"nearest actual FEM DOF={nearest['dof']}; no time interpolation")
        candidates = [r for r in rfm if r["case_id"] == case and int(r["dof"]) == 8]
        append_group(rows, case, "RFM", 8, candidates,
                     scope="per-state solver/evaluation timing (comparison runner)",
                     source="compare_fem_rfm/rfm_raw.csv",
                     dof_note="feature width exactly 8")

    rows = [r for r in rows if r]
    fields = list(rows[0])
    with (OUT / "same_machine_cost_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    pilot = read_csv(PILOT / "pilot_summary.csv")
    report = [
        "# Cost accounting and timing scope",
        "",
        "The selection campaign and the FEM/RFM comparison were executed on the same ARM macOS host family, but they expose different timing boundaries. The selection campaign records end-to-end process wall time; the comparison runner records per-state solver/evaluation time. The combined CSV therefore retains a `timing_scope` field and is descriptive only; it is not a cross-method speed ranking.",
        "",
        "## Selection-variant runs",
        "",
        "CGA-FP, RD-WOGA, and Random-FC use the formal pilot at accepted width 8. The pilot contains 3 seeds for C1, C2, and C4 and 5 seeds for C3 and C5. Success and failure counts are retained in the CSV. Median and quartile wall times are end-to-end process times.",
        "",
        "## FEM/RFM comparison timing",
        "",
        "FEM P1/P3 and RFM rows are the actual width-eight or nearest available FEM states from the five-case comparison. Their `wall_time_sec` field is a per-state solver/evaluation measurement, not an end-to-end process measurement. FEM nearest-state rows retain their actual DOF and are not interpolated.",
        "",
        "## Instrumentation pilot",
        "",
        f"The bounded instrumentation pilot contains {sum(int(r['n_total']) for r in pilot)} CGA-only timing repetitions at target width 2 for C1, C2, and C4. Its known selection and correction phases are logged, while setup, pool generation, evaluator internals, serialization, and I/O remain an unattributed remainder. These data validate the logging schema and do not support an efficiency conclusion.",
        "",
        "## Reproducibility",
        "",
        "Raw records: `baseline_campaign/data/raw/`, `cost_campaign/pilot_20260908_final/data/raw/`, and `cost_campaign/scripts/compare_fem_rfm/data/`. Environment and protocol manifests are stored alongside each campaign.",
    ]
    (OUT / "same_machine_cost_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
