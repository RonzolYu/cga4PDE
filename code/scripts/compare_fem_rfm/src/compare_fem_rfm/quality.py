"""Metric-specific validity shared by replay, aggregation and artifact checks."""

import math
from collections import defaultdict

AUDIT_FIELDS = {
    "energy_gap": "audit_energy_gap_rel_delta",
    "natural_error": "audit_natural_rel_delta",
    "v_error": "audit_v_rel_delta",
}
DEFAULT_AUDIT_RTOL = 0.01


def metric_valid(row: dict, metric: str, rtol: float = DEFAULT_AUDIT_RTOL) -> bool:
    if str(row.get("solver_success")) != "True":
        return False
    try:
        value = float(row[metric])
        delta = float(row[AUDIT_FIELDS[metric]])
    except (KeyError, TypeError, ValueError):
        return False
    # Absence of a quadrature audit is not evidence of stability.
    return math.isfinite(value) and value >= 0 and math.isfinite(delta) and 0 <= delta <= rtol


def invalid_reason(row: dict, metric: str, rtol: float = DEFAULT_AUDIT_RTOL) -> str:
    if metric == "v_error" and row.get("case_id") not in {"C4", "C5"}:
        return "not_applicable"
    if str(row.get("solver_success")) != "True":
        return "coefficient_solve_failed"
    try:
        value = float(row[metric])
    except (KeyError, TypeError, ValueError):
        return "missing_or_nonfinite_metric"
    if not math.isfinite(value):
        return "missing_or_nonfinite_metric"
    if value < 0:
        return "negative_signed_energy" if metric == "energy_gap" else "negative_metric"
    try:
        delta = float(row[AUDIT_FIELDS[metric]])
    except (KeyError, TypeError, ValueError):
        return "quadrature_audit_missing"
    return "none" if math.isfinite(delta) and 0 <= delta <= rtol else "quadrature_audit_failed"


def summarize_rfm(raw: list[dict], rtol: float = DEFAULT_AUDIT_RTOL) -> list[dict]:
    """Count solves and valid observations separately for every error metric."""
    groups = defaultdict(list)
    identifiers = set()
    for row in raw:
        key = row["case_id"], int(row["dof"]), int(row["seed"])
        if key in identifiers:
            raise ValueError(f"duplicate RFM record {key}")
        identifiers.add(key)
        groups[key[:2]].append(row)

    def quantile(values, p):
        values = sorted(values)
        position = (len(values)-1)*p
        left = int(position)
        return values[left] + (position-left)*(values[min(left+1,len(values)-1)]-values[left])

    summaries = []
    for (case, dof), group in sorted(groups.items()):
        successes = [r for r in group if str(r["solver_success"]) == "True"]
        summary = dict(schema_version="cga-experiments-v2", case_id=case, dof=dof,
            seed_count=len(group), success_count=len(successes), failure_count=len(group)-len(successes),
            failure_reasons="; ".join(sorted({r["solver_message"] for r in group
                                              if str(r["solver_success"]) != "True"})) or "none",
            config_hashes=";".join(sorted({r["config_hash"] for r in group})))
        for metric in ("energy_gap", "natural_error", "v_error", "l2_error", "wall_time_sec"):
            included = [r for r in group if metric_valid(r, metric, rtol)] if metric in AUDIT_FIELDS else successes
            values = [float(r[metric]) for r in included if r.get(metric) not in (None,"","NA","nan","NaN")]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"nonfinite statistic {case}/{dof}/{metric}")
            summary[metric+"_sample_count"] = len(values)
            for p, suffix in [(.5,"median"),(.25,"q1"),(.75,"q3")]:
                summary[metric+"_"+suffix] = quantile(values,p) if values else "NA"
        summaries.append(summary)
    return summaries
