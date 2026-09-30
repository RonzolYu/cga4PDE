"""Common-grid interpolation, seed aggregation, orders, tables, and plots."""

from __future__ import annotations

import csv
from collections import defaultdict
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import load_config, project_root
from .problems import CASES


NUMERIC_FIELDS = {
    "seed", "dof", "mesh_level", "energy_gap", "energy_gap_signed", "natural_error",
    "v_error", "l2_error", "wall_time_sec", "solver_residual", "evaluation_audit_rel_delta",
    "audit_energy_gap_rel_delta", "audit_natural_rel_delta", "audit_v_rel_delta",
}


def read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in NUMERIC_FIELDS:
                if key in row and row[key] != "":
                    row[key] = float(row[key])
                elif key in row:
                    row[key] = None
            row["dof"] = int(row["dof"])
            rows.append(row)
    return rows


def quantiles(values: list[float]) -> tuple[float, float, float]:
    q = np.quantile(np.asarray(values), [0.25, 0.5, 0.75])
    return float(q[0]), float(q[1]), float(q[2])


def aggregate_rfm(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["dof"])].append(row)
    output = []
    for (case_id, dof), group in sorted(groups.items()):
        record = {"case_id": case_id, "dof": dof, "method": "rfm", "variant": "rfm_median",
                  "seed_count": len(group)}
        for metric in ("energy_gap", "natural_error", "v_error", "wall_time_sec"):
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            if values:
                q1, median, q3 = quantiles(values)
                record[metric], record[f"{metric}_q1"], record[f"{metric}_q3"] = median, q1, q3
            else:
                record[metric] = record[f"{metric}_q1"] = record[f"{metric}_q3"] = None
        output.append(record)
    return output


def loglog_interpolate(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if not (x0 <= x <= x1) or x0 == x1:
        raise ValueError("interpolation target is not bracketed")
    if min(x0, y0, x1, y1, x) <= 0.0:
        raise ValueError("log-log interpolation requires positive data")
    theta = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + theta * (math.log(y1) - math.log(y0)))


def interpolate_series(rows: list[dict], metric: str, target: int) -> tuple[float, bool] | None:
    points = sorted((int(row["dof"]), float(row[metric])) for row in rows if row.get(metric) is not None)
    exact = [value for dof, value in points if dof == target]
    if exact:
        return exact[-1], False
    lower = [(dof, value) for dof, value in points if dof < target]
    upper = [(dof, value) for dof, value in points if dof > target]
    if not lower or not upper:
        return None
    x0, y0 = lower[-1]
    x1, y1 = upper[0]
    return loglog_interpolate(x0, y0, x1, y1, target), True


def method_series(cga: list[dict], fem: list[dict], rfm: list[dict], case_id: str) -> dict[str, list[dict]]:
    result = {"CGA": [r for r in cga if r["case_id"] == case_id]}
    for degree in (1, 2, 3):
        result[f"FEM P{degree}"] = [r for r in fem if r["case_id"] == case_id and r["variant"] == f"p{degree}"]
    # ``rfm`` is already aggregated over the registered seeds; keep that
    # distinction visible in every downstream table and figure.
    result["RFM median"] = [r for r in rfm if r["case_id"] == case_id]
    return result


def build_common_grid(cga: list[dict], fem: list[dict], rfm: list[dict], cfg: dict) -> list[dict]:
    output = []
    for case_id, spec in CASES.items():
        targets = cfg[f"common_dofs_{spec.dim}d"]
        series = method_series(cga, fem, rfm, case_id)
        metrics = ["natural_error", "energy_gap"] + (["v_error"] if spec.model == "pure_p" else [])
        for metric in metrics:
            for method, rows in series.items():
                for target in targets:
                    value = interpolate_series(rows, metric, target)
                    if value is None:
                        continue
                    item = {"case_id": case_id, "metric": metric, "method": method,
                            "dof": target, "value": value[0], "is_interpolated": value[1],
                            "q1": None, "q3": None}
                    if method == "RFM median":
                        q1 = interpolate_series(rows, f"{metric}_q1", target)
                        q3 = interpolate_series(rows, f"{metric}_q3", target)
                        item["q1"] = None if q1 is None else q1[0]
                        item["q3"] = None if q3 is None else q3[0]
                    output.append(item)
    return output


def local_orders(common: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in common:
        groups[(row["case_id"], row["metric"], row["method"])].append(row)
    output = []
    for key, rows in groups.items():
        rows.sort(key=lambda row: row["dof"])
        for left, right in zip(rows[:-1], rows[1:]):
            if right["dof"] != 2 * left["dof"]:
                continue
            order = math.log(float(left["value"]) / float(right["value"]), 2.0)
            output.append({"case_id": key[0], "metric": key[1], "method": key[2],
                           "dof_left": left["dof"], "dof_right": right["dof"], "order": order,
                           "decreasing": right["value"] < left["value"]})
    return output


def order_summary(common: list[dict], orders: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in common:
        groups[(row["case_id"], row["metric"], row["method"])].append(row)
    order_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in orders:
        order_groups[(row["case_id"], row["metric"], row["method"])].append(row)
    output = []
    for key, rows in groups.items():
        rows.sort(key=lambda row: row["dof"])
        ords = sorted(order_groups.get(key, []), key=lambda row: row["dof_right"])
        last_raw = ords[-1]["order"] if ords else None
        valid = [row for row in ords if row["decreasing"] and row["order"] > 0.15]
        last_valid = valid[-1]["order"] if valid else None
        plateau = None
        for row in ords:
            if row["order"] <= 0.15:
                plateau = row["dof_right"]
                break
        fit_rows = [row for row in rows if row["dof"] >= 16 and (plateau is None or row["dof"] < plateau)]
        if len(fit_rows) >= 2:
            slope = -float(np.polyfit(np.log([r["dof"] for r in fit_rows]),
                                      np.log([r["value"] for r in fit_rows]), 1)[0])
        else:
            slope = None
        output.append({"case_id": key[0], "metric": key[1], "method": key[2],
                       "last_raw_local_order": last_raw, "last_valid_local_order": last_valid,
                       "window_fit_order": slope, "plateau_onset_dof": plateau})
    return output


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


COLORS = {"CGA": "#c0392b", "FEM P1": "#1f77b4", "FEM P2": "#2ca02c",
          "FEM P3": "#9467bd", "RFM median": "#e67e22"}
MARKERS = {"CGA": "o", "FEM P1": "s", "FEM P2": "^", "FEM P3": "D", "RFM median": "v"}


def _theory_order(method: str, metric: str, dim: int) -> float:
    if method == "CGA":
        natural = 3.0 if dim == 1 else 1.75
    elif method.startswith("FEM P"):
        natural = float(method[-1]) / dim
    else:
        natural = 0.5
    return 2.0 * natural if metric == "energy_gap" else natural


def plot_metric(case_id: str, metric: str, series: dict[str, list[dict]], out: Path,
                archive: list[dict]) -> None:
    spec = CASES[case_id]
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    archived = sorted([r for r in archive if r["case_id"] == case_id and r.get(metric) is not None],
                      key=lambda r: r["dof"])
    if archived:
        ax.loglog([r["dof"] for r in archived], [r[metric] for r in archived],
                  color="0.82", lw=0.8, zorder=0)
    for method in ["CGA", "FEM P1", "FEM P2", "FEM P3", "RFM median"]:
        rows = sorted([r for r in series[method] if r.get(metric) is not None], key=lambda r: r["dof"])
        if not rows:
            continue
        x = np.array([r["dof"] for r in rows])
        y = np.array([r[metric] for r in rows])
        ax.loglog(x, y, marker=MARKERS[method], ms=4.2, lw=1.6, color=COLORS[method], label=method)
        anchor_index = 0 if spec.dim == 1 else min(1, len(rows) - 1)
        alpha = _theory_order(method, metric, spec.dim)
        guide = y[anchor_index] * (x / x[anchor_index]) ** (-alpha)
        ax.loglog(x, guide, color=COLORS[method], alpha=0.28, ls=":", lw=0.9)
    ylabel = {"natural_error": "relative natural error", "energy_gap": "energy gap",
              "v_error": "relative V-map error"}[metric]
    ax.set(xlabel="DOF", ylabel=ylabel, title=f"{case_id}: {spec.model}, d={spec.dim}")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_orders(case_id: str, orders: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for method in ["CGA", "FEM P1", "FEM P2", "FEM P3", "RFM median"]:
        rows = sorted([r for r in orders if r["case_id"] == case_id and
                       r["metric"] == "natural_error" and r["method"] == method],
                      key=lambda r: r["dof_right"])
        if rows:
            ax.semilogx([r["dof_right"] for r in rows], [r["order"] for r in rows],
                        base=2, marker=MARKERS[method], color=COLORS[method], label=method)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set(xlabel="right endpoint DOF", ylabel="dyadic local order", title=f"{case_id}: natural-error orders")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def load_archive_history() -> list[dict]:
    path = project_root() / "data" / "cga_archive_history.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["accepted"] != "True":
                continue
            rows.append({"case_id": row["case_id"], "dof": int(row["accepted_atom_count"]),
                         "energy_gap": float(row["energy_gap"]),
                         "natural_error": float(row["natural_error"]),
                         "v_error": None if row["v_error"] == "" else float(row["v_error"])})
    return rows


def build_analysis() -> dict[str, list[dict]]:
    root = project_root()
    cfg, _ = load_config()
    cga = read_csv(root / "data" / "cga_evaluated.csv")
    fem = read_csv(root / "data" / "fem_raw.csv")
    rfm_raw = read_csv(root / "data" / "rfm_raw.csv")
    rfm = aggregate_rfm(rfm_raw)
    common = build_common_grid(cga, fem, rfm, cfg)
    orders = local_orders(common)
    summaries = order_summary(common, orders)
    write_dict_csv(root / "data" / "rfm_aggregated.csv", rfm)
    write_dict_csv(root / "data" / "common_grid.csv", common)
    write_dict_csv(root / "artifacts" / "tables" / "local_orders.csv", orders)
    write_dict_csv(root / "artifacts" / "tables" / "order_summary.csv", summaries)
    archive = load_archive_history()
    for case_id, spec in CASES.items():
        series = method_series(cga, fem, rfm, case_id)
        for metric in ["natural_error", "energy_gap"] + (["v_error"] if spec.model == "pure_p" else []):
            plot_metric(case_id, metric, series,
                        root / "artifacts" / "figures" / f"{case_id}_{metric}.pdf", archive)
        plot_orders(case_id, orders, root / "artifacts" / "figures" / f"{case_id}_orders.pdf")
    return {"cga": cga, "fem": fem, "rfm_raw": rfm_raw, "rfm": rfm,
            "common": common, "orders": orders, "order_summary": summaries}
