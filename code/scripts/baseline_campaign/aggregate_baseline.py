#!/usr/bin/env python3
"""Aggregate baseline runs and generate reproducible figures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value: object) -> float | None:
    if value in (None, "", "NA", "nan", "NaN"):
        return None
    try:
        x = float(value)
    except (ValueError, TypeError):
        return None
    return x if np.isfinite(x) else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def combine(inputs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in inputs:
        rows.extend(read_csv(path))
    keys = {(r["case_id"], r["method"], r["seed"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate case/method/seed rows in baseline inputs")
    return rows


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["method"])].append(row)
    out: list[dict[str, object]] = []
    for (case, method), group in sorted(groups.items()):
        metrics = {
            "energy_gap": "final_energy_gap",
            "sobolev_error": "final_h1_or_w1p_error",
            "v_error": "final_v_error",
            "wall_time_sec": "wall_end_to_end_sec",
        }
        item: dict[str, object] = {
            "schema_version": "sisc-baseline-summary-v1",
            "case_id": case, "method": method,
            "n_total": len(group),
            "n_success": sum(r.get("status") == "success" for r in group),
            "terminal_width": ";".join(sorted({r.get("final_width", "") for r in group})),
        }
        item["success_rate"] = float(item["n_success"]) / len(group) if group else 0.0
        for label, field in metrics.items():
            values = [num(r.get(field)) for r in group]
            values = [v for v in values if v is not None]
            if values:
                q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
                item[f"{label}_median"] = statistics.median(values)
                item[f"{label}_q1"] = q[0]
                item[f"{label}_q3"] = q[2]
            else:
                item[f"{label}_median"] = ""
                item[f"{label}_q1"] = ""
                item[f"{label}_q3"] = ""
        out.append(item)
    return out


def histories(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for run in rows:
        path = Path(run["run_dir"]) / "history.csv"
        if not path.exists():
            continue
        for record in read_csv(path):
            if record.get("accepted", "False") != "True":
                continue
            out.append({
                "schema_version": "sisc-baseline-history-v1",
                "case_id": run["case_id"], "method": run["method"], "seed": run["seed"],
                "accepted_width": record.get("accepted_atom_count", ""),
                "energy_gap": record.get("energy_gap_raw", ""),
                "sobolev_error": record.get("natural_rel", ""),
                "v_error": record.get("quasi_rel", ""),
                "wall_time_sec": record.get("elapsed_total_s", ""),
                "source_run": str(path.resolve()),
            })
    return out


def plot(rows: list[dict[str, object]], outdir: Path, metric: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    methods = ["CGA-FP", "RD-WOGA", "Random-FC"]
    colors = {"CGA-FP": "#2166AC", "RD-WOGA": "#B2182B", "Random-FC": "#4D9221"}
    cases = sorted({str(r["case_id"]) for r in rows})
    fig, axes = plt.subplots(1, len(cases), figsize=(3.1 * len(cases), 2.6), squeeze=False)
    for ax, case in zip(axes[0], cases):
        for method in methods:
            group = [r for r in rows if r["case_id"] == case and r["method"] == method]
            points: dict[int, list[float]] = defaultdict(list)
            for r in group:
                try:
                    x, y = int(r["accepted_width"]), num(r.get(metric))
                except (TypeError, ValueError):
                    continue
                if y is not None and y > 0:
                    points[x].append(y)
            if not points:
                continue
            xs = sorted(points)
            med = [statistics.median(points[x]) for x in xs]
            q1 = [min(points[x]) for x in xs]
            q3 = [max(points[x]) for x in xs]
            ax.loglog(xs, med, "o-", ms=3.2, lw=1.2, color=colors[method], label=method)
            if len(points[xs[0]]) > 1:
                ax.fill_between(xs, q1, q3, color=colors[method], alpha=0.12, linewidth=0)
        ax.set_title(case)
        ax.set_xlabel("Accepted atoms")
        ax.grid(True, which="both", alpha=0.25, lw=0.45)
    axes[0][0].set_ylabel(metric.replace("_", " "))
    axes[0][0].legend(fontsize=7, framealpha=0.9)
    fig.tight_layout(pad=0.7)
    fig.savefig(outdir / f"baseline_{metric}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"baseline_{metric}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = combine(args.inputs)
    summary = aggregate(rows)
    history = histories(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "baseline_summary.csv", summary)
    write_csv(args.out / "baseline_history.csv", history)
    for metric in ("energy_gap", "sobolev_error", "v_error"):
        plot(history, args.out / "figures", metric)
    manifest = {
        "schema_version": "sisc-baseline-artifacts-v1",
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p.read_bytes()).hexdigest()} for p in args.inputs],
        "summary": str((args.out / "baseline_summary.csv").resolve()),
        "history": str((args.out / "baseline_history.csv").resolve()),
        "methods": ["CGA-FP", "RD-WOGA", "Random-FC"],
        "plot_rule": "actual accepted states only; median and min/max over observed seeds; no theory lines",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
