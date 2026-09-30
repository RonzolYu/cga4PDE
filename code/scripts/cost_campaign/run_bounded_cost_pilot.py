#!/usr/bin/env python3
"""Bounded, non-invasive cost pilot for the existing CGA runner.

This script deliberately does not modify cga_refactor.  It times each smoke
run from an outer monotonic clock, reads the solver's existing history.csv
fields, and records the phase measurements that are already available
(selection and projection).  Uninstrumented phases are kept as an explicit
remainder rather than being inferred.

The pilot is a data-pipeline check, not a cross-method efficiency experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


# Keep numerical libraries single-threaded so the bounded pilot is repeatable.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[3]
SRC_ROOT = PACKAGE_ROOT / "reproducibility" / "scripts" / "cga_refactor" / "src"
sys.path.insert(0, str(SRC_ROOT))

from cga_refactor.config import config_for, config_hash  # noqa: E402
from cga_refactor.solver import run_cga  # noqa: E402


CASE_MAP: dict[str, tuple[str, int]] = {
    "C1": ("linear", 1),
    "C2": ("cubic", 1),
    "C3": ("sinh", 2),
    "C4": ("pure_p", 1),
    "C5": ("pure_p", 2),
}
DEFAULT_CASES = ("C1", "C2", "C4")
DEFAULT_SEEDS = (201, 202)
DEFAULT_REPETITIONS = 2
SCHEMA = "cost-pilot-v1"


def _finite(value: Any) -> float | None:
    if value in (None, "", "NA", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PACKAGE_ROOT,
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _rss_bytes() -> int | None:
    """Return max RSS in bytes on both macOS and Linux."""
    try:
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    if platform.system() == "Darwin":
        return raw
    return raw * 1024


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _environment() -> dict[str, Any]:
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        numpy_version = "unavailable"
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = "unavailable"
    return {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": numpy_version,
        "scipy": scipy_version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "threads": {name: os.environ.get(name, "unspecified") for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "git_sha": _git_sha(),
        "timer": "time.perf_counter_ns",
        "phase_scope": (
            "outer wall time plus existing history fields "
            "elapsed_selection_s and elapsed_projection_s"
        ),
        "unmeasured_phases": [
            "setup", "pool generation", "quadrature construction",
            "serialization and I/O", "independent evaluator internals"
        ],
    }


def _phase_rows(
    run: dict[str, Any],
    history: list[dict[str, str]],
    total_wall: float,
    total_cpu: float,
    rss: int | None,
) -> list[dict[str, Any]]:
    """Expand existing history to an explicit phase table.

    The solver stores per-attempt selection/projection times.  We preserve those
    fields and assign the remaining outer time to unattributed_remainder.
    This avoids presenting a derived remainder as a measured phase.
    """
    rows: list[dict[str, Any]] = []
    selection_sum = sum(_finite(item.get("elapsed_selection_s")) or 0.0 for item in history)
    projection_sum = sum(_finite(item.get("elapsed_projection_s")) or 0.0 for item in history)
    known = selection_sum + projection_sum
    remainder = max(0.0, total_wall - known)
    run_id = run["run_id"]
    common = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "case_id": run["case_id"],
        "method": run["method"],
        "seed": run["seed"],
        "timing_rep": run["timing_rep"],
        "rss_bytes_at_end": rss if rss is not None else "NA",
        "status": run["status"],
    }
    for item in history:
        attempt = item.get("attempted_iteration", "")
        for phase, key in (
            ("selection", "elapsed_selection_s"),
            ("correction", "elapsed_projection_s"),
        ):
            value = _finite(item.get(key))
            if value is None:
                continue
            rows.append({
                **common,
                "attempted_iteration": attempt,
                "accepted_atom_count": item.get("accepted_atom_count", ""),
                "phase": phase,
                "wall_time_sec": value,
                "cpu_time_sec": "NA",
                "operation_count": (
                    item.get("objective_evaluations", "NA")
                    if phase == "correction"
                    else "NA"
                ),
                "accepted": item.get("accepted", ""),
                "rollback": item.get("rollback", ""),
                "failure_reason": item.get("acceptance_reason", ""),
            })
    rows.append({
        **common,
        "attempted_iteration": "",
        "accepted_atom_count": run.get("final_width", ""),
        "phase": "unattributed_remainder",
        "wall_time_sec": remainder,
        "cpu_time_sec": "NA",
        "operation_count": "NA",
        "accepted": "",
        "rollback": "",
        "failure_reason": (
            "existing solver exposes no setup/atom/evaluator phase timers"
        ),
    })
    return rows


def _step_rows(run: dict[str, Any], history: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_selection = 0.0
    cumulative_projection = 0.0
    for item in history:
        cumulative_selection += _finite(item.get("elapsed_selection_s")) or 0.0
        cumulative_projection += _finite(item.get("elapsed_projection_s")) or 0.0
        elapsed_total = _finite(item.get("elapsed_total_s"))
        rows.append({
            "schema_version": SCHEMA,
            "run_id": run["run_id"],
            "case_id": run["case_id"],
            "method": run["method"],
            "seed": run["seed"],
            "timing_rep": run["timing_rep"],
            "attempted_iteration": item.get("attempted_iteration", ""),
            "accepted_atom_count": item.get("accepted_atom_count", ""),
            "effective_rank": item.get("effective_rank", ""),
            "status": "accepted" if _bool_text(item.get("accepted")) == "true" else "attempt",
            "accepted": item.get("accepted", ""),
            "rollback": item.get("rollback", ""),
            "stop_reason": item.get("acceptance_reason", ""),
            "failure_reason": (
                "" if _bool_text(item.get("accepted")) == "true"
                else item.get("acceptance_reason", "")
            ),
            "native_dof": item.get("accepted_atom_count", ""),
            "feature_count": item.get("accepted_atom_count", ""),
            "selected_score": item.get("dual_score_abs", ""),
            "selected_score_rank": item.get("candidate_score_rank", ""),
            "candidate_evaluations": item.get("remaining_candidate_count", "NA"),
            "score_evaluations": item.get("remaining_candidate_count", "NA"),
            "rejected_attempts": item.get("rejected_candidate_count", "NA"),
            "objective_calls": item.get("objective_evaluations", "NA"),
            "gradient_calls": "NA",
            "hessian_calls": "NA",
            "solver_iterations": item.get("optimizer_iterations", "NA"),
            "solver_residual": item.get("projected_residual_abs", "NA"),
            "solver_success": item.get("optimizer_success", "NA"),
            "train_energy": item.get("train_energy", ""),
            "eval_energy": item.get("validation_energy", ""),
            "energy_gap": item.get("energy_gap_raw", ""),
            "bregman_gap": item.get("bregman_gap", ""),
            "relative_h1_error": (
                item.get("natural_rel", "NA")
                if run.get("model") in {"linear", "cubic", "sinh"}
                else "NA"
            ),
            "relative_w1p_error": item.get("w1p_full_rel", "NA"),
            "relative_v_distance": item.get("quasi_rel", "NA"),
            "wall_time_cumulative_sec": elapsed_total if elapsed_total is not None else "NA",
            "known_selection_cumulative_sec": cumulative_selection,
            "known_correction_cumulative_sec": cumulative_projection,
        })
    return rows


def _aggregate(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (str(row["case_id"]), str(row["method"]), str(row["final_width"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (case_id, method, width), rows in sorted(grouped.items()):
        times = [_finite(row.get("total_wall_time_sec")) for row in rows]
        times = [value for value in times if value is not None]
        success = sum(_bool_text(row.get("success")) == "true" for row in rows)
        output.append({
            "schema_version": SCHEMA,
            "case_id": case_id,
            "method": method,
            "width": width,
            "n_success": success,
            "n_total": len(rows),
            "success_rate": success / len(rows) if rows else "NA",
            "median_wall_time_sec": statistics.median(times) if times else "NA",
            "q1_wall_time_sec": (
                statistics.quantiles(times, n=4, method="inclusive")[0]
                if len(times) >= 2 else (times[0] if times else "NA")
            ),
            "q3_wall_time_sec": (
                statistics.quantiles(times, n=4, method="inclusive")[2]
                if len(times) >= 2 else (times[0] if times else "NA")
            ),
            "total_known_phase_time_sec": sum(
                (_finite(row.get("known_phase_time_sec")) or 0.0) for row in rows
            ),
            "pilot_scope": "CGA smoke runner only; no cross-method claim",
        })
    return output


def run(args: argparse.Namespace) -> Path:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(exist_ok=True)
    environment = _environment()
    (output_root / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    protocol = {
        "schema_version": SCHEMA,
        "profile": "smoke",
        "cases": args.cases,
        "seeds": args.seeds,
        "timing_repetitions": args.repetitions,
        "target_accepted": args.target,
        "method": "finite-pool CGA",
        "phase_scope": environment["phase_scope"],
        "not_instrumented": environment["unmeasured_phases"],
    }
    (output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    run_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    planned = len(args.cases) * len(args.seeds) * args.repetitions
    completed = 0

    for case_id in args.cases:
        model, dim = CASE_MAP[case_id]
        for seed in args.seeds:
            for timing_rep in range(1, args.repetitions + 1):
                completed += 1
                run_id = f"{case_id}_{model}_d{dim}_seed{seed}_rep{timing_rep}"
                run_dir_root = output_root / "runs" / run_id
                run_dir_root.mkdir(parents=True, exist_ok=True)
                cfg = config_for(
                    model, dim, profile="smoke", seed=seed,
                    output_root=str(run_dir_root / "solver"),
                )
                cfg = replace(
                    cfg, phase="cost_pilot", target_accepted=args.target,
                )
                started_wall = time.perf_counter_ns()
                started_cpu = time.process_time_ns()
                status = "completed"
                failure_reason = ""
                summary: dict[str, Any] = {}
                try:
                    summary = run_cga(cfg)
                except Exception as exc:  # Keep failed runs in the denominator.
                    status = "failed"
                    failure_reason = f"{type(exc).__name__}: {exc}"
                    failures.append({
                        "run_id": run_id, "case_id": case_id, "seed": seed,
                        "timing_rep": timing_rep, "failure_reason": failure_reason,
                    })
                ended_cpu = time.process_time_ns()
                ended_wall = time.perf_counter_ns()
                total_wall = (ended_wall - started_wall) / 1e9
                total_cpu = (ended_cpu - started_cpu) / 1e9
                rss = _rss_bytes()
                run_dir = Path(summary["run_dir"]) if summary.get("run_dir") else None
                history = _read_history(run_dir / "history.csv") if run_dir else []
                final_width = summary.get("accepted_atom_count", 0)
                final_metrics = summary.get("metrics") or {}
                known_selection = sum(
                    _finite(item.get("elapsed_selection_s")) or 0.0 for item in history
                )
                known_correction = sum(
                    _finite(item.get("elapsed_projection_s")) or 0.0 for item in history
                )
                known_phase = known_selection + known_correction
                row = {
                    "schema_version": SCHEMA,
                    "run_id": run_id,
                    "case_id": case_id,
                    "model": model,
                    "dim": dim,
                    "method": "finite-pool CGA",
                    "variant": "smoke",
                    "seed": seed,
                    "timing_rep": timing_rep,
                    "config_hash": config_hash(cfg),
                    "status": status,
                    "success": status == "completed",
                    "solver_status": summary.get("formal_status", status),
                    "stop_reason": summary.get("stop_reason", "exception"),
                    "target_width": cfg.target_accepted,
                    "final_width": final_width,
                    "final_energy_gap": final_metrics.get("energy_gap_raw", "NA"),
                    "final_h1_or_natural_error": (
                        final_metrics.get("natural_rel")
                        or final_metrics.get("w1p_full_rel")
                        or final_metrics.get("l2_rel")
                        or "NA"
                    ),
                    "attempted_iterations": summary.get("attempted_iteration", "NA"),
                    "accepted_atoms": summary.get("accepted_atom_count", "NA"),
                    "rejected_attempts": (
                        sum(_bool_text(item.get("accepted")) != "true" for item in history)
                        if history else "NA"
                    ),
                    "score_evaluations": "NA",
                    "objective_evaluations": sum(
                        int(float(item["objective_evaluations"]))
                        for item in history
                        if _finite(item.get("objective_evaluations")) is not None
                    ) if history else "NA",
                    "solver_iterations": sum(
                        int(float(item["optimizer_iterations"]))
                        for item in history
                        if _finite(item.get("optimizer_iterations")) is not None
                    ) if history else "NA",
                    "total_wall_time_sec": total_wall,
                    "total_cpu_time_sec": total_cpu,
                    "known_selection_time_sec": known_selection,
                    "known_correction_time_sec": known_correction,
                    "known_phase_time_sec": known_phase,
                    "unattributed_remainder_sec": max(0.0, total_wall - known_phase),
                    "phase_coverage_ratio": (
                        known_phase / total_wall if total_wall > 0 else "NA"
                    ),
                    "peak_rss_bytes": rss if rss is not None else "NA",
                    "failure_reason": failure_reason,
                    "run_dir": str(run_dir) if run_dir else "NA",
                    "source_hash": _sha256(run_dir / "history.csv") if run_dir else "NA",
                }
                run_rows.append(row)
                step_rows.extend(_step_rows(row, history))
                phase_rows.extend(_phase_rows(row, history, total_wall, total_cpu, rss))
                print(
                    f"[{completed}/{planned}] {run_id} status={status} "
                    f"width={final_width} wall={total_wall:.3f}s",
                    flush=True,
                )

    run_fields = list(run_rows[0]) if run_rows else ["schema_version", "status"]
    step_fields = [
        "schema_version", "run_id", "case_id", "method", "seed", "timing_rep",
        "attempted_iteration", "accepted_atom_count", "effective_rank", "status",
        "accepted", "rollback", "stop_reason", "failure_reason", "native_dof",
        "feature_count", "selected_score", "selected_score_rank",
        "candidate_evaluations", "score_evaluations", "rejected_attempts",
        "objective_calls", "gradient_calls", "hessian_calls", "solver_iterations",
        "solver_residual", "solver_success", "train_energy", "eval_energy",
        "energy_gap", "bregman_gap", "relative_h1_error", "relative_w1p_error",
        "relative_v_distance", "wall_time_cumulative_sec",
        "known_selection_cumulative_sec", "known_correction_cumulative_sec",
    ]
    phase_fields = [
        "schema_version", "run_id", "case_id", "method", "seed", "timing_rep",
        "attempted_iteration", "accepted_atom_count", "phase", "wall_time_sec",
        "cpu_time_sec", "operation_count", "accepted", "rollback",
        "failure_reason", "rss_bytes_at_end", "status",
    ]
    _write_csv(output_root / "data/raw/pilot_runs.csv", run_rows, run_fields)
    _write_csv(output_root / "data/raw/pilot_steps.csv", step_rows, step_fields)
    _write_csv(output_root / "data/raw/pilot_phases.csv", phase_rows, phase_fields)
    aggregate_rows = _aggregate(run_rows)
    _write_csv(
        output_root / "data/derived/pilot_summary.csv",
        aggregate_rows,
        list(aggregate_rows[0]) if aggregate_rows else ["schema_version", "status"],
    )
    manifest = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str((output_root / "protocol.json").resolve()),
        "environment": str((output_root / "environment.json").resolve()),
        "planned_runs": planned,
        "completed_runs": len(run_rows),
        "failed_runs": len(failures),
        "failures": failures,
        "files": {
            "runs": str((output_root / "data/raw/pilot_runs.csv").resolve()),
            "steps": str((output_root / "data/raw/pilot_steps.csv").resolve()),
            "phases": str((output_root / "data/raw/pilot_phases.csv").resolve()),
            "summary": str((output_root / "data/derived/pilot_summary.csv").resolve()),
        },
        "scope": "bounded CGA smoke pilot; no cross-method efficiency claim",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_lines = [
        "# Bounded cost pilot report",
        "",
        f"- Planned runs: {planned}",
        f"- Completed rows: {len(run_rows)}",
        f"- Failed runs: {len(failures)}",
        f"- Cases: {', '.join(args.cases)}",
        f"- Seeds: {', '.join(str(value) for value in args.seeds)}",
        f"- Timing repetitions: {args.repetitions}",
        f"- Target accepted width: {args.target}",
        "",
        "## Scope",
        "",
        "This pilot invokes the existing finite-pool CGA smoke runner. The outer "
        "monotonic timer measures end-to-end process work. Existing per-attempt "
        "selection and projection timers are copied into the phase table. Pool "
        "generation, setup, evaluator internals, and I/O remain an explicit "
        "unattributed remainder because the source solver was not modified.",
        "",
        "The pilot is suitable for validating logging, failure retention, and "
        "time-accounting plumbing. It is not a same-machine comparison against "
        "RD-WOGA, Random-FC, RFM, or FEM, and it must not be used to claim speed.",
    ]
    if failures:
        report_lines.extend(["", "## Failures", ""])
        report_lines.extend(
            f"- {item['run_id']}: {item['failure_reason']}" for item in failures
        )
    (output_root / "reports").mkdir(exist_ok=True)
    (output_root / "reports/pilot_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=PACKAGE_ROOT / "data" / "derived" / "experiments" / "cost_pilot" / "regenerated",
    )
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASE_MAP), default=list(DEFAULT_CASES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--target", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repetitions < 1 or args.target < 1:
        raise SystemExit("repetitions and target must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("seeds must be distinct")
    out = run(args)
    print(json.dumps({"status": "completed", "output_root": str(out.resolve())},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
