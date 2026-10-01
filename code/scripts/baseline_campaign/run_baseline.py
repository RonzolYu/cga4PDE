#!/usr/bin/env python3
"""Run controlled CGA, RD-WOGA, and Random-FC experiments.

The implementation reuses the copied CGA solver.  The two additional
selection rules are kept here so that the original implementation remains
unchanged and every run records the method protocol explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[3]
PACKAGE = PACKAGE_ROOT / "code" / "scripts" / "cga_refactor" / "src"
sys.path.insert(0, str(PACKAGE))

from cga_refactor.config import config_for  # noqa: E402
from cga_refactor.dictionary import build_pool, calibrate_pool  # noqa: E402
from cga_refactor import solver as solver_mod  # noqa: E402
from cga_refactor.solver import run_cga  # noqa: E402
from cga_refactor.step1 import (  # noqa: E402
    _best_admissible,
    _pool_scores,
    dual_scores,
    feature_columns,
    innovation,
)

ORIGINAL_SELECT_ATOM = solver_mod.select_atom

METHODS = ("CGA-FP", "RD-WOGA", "Random-FC")
CASE_TO_CONFIG = {
    "C1": ("linear", 1),
    "C2": ("cubic", 1),
    "C3": ("sinh", 2),
    "C4": ("pure_p", 1),
    "C5": ("pure_p", 2),
}


def _digest_pool(pool: object) -> str:
    payload = (pool.w.tobytes() + pool.b.tobytes() +
               f"{pool.k}|{pool.seed}|{pool.sampler}|{pool.kind}|campaign-v1".encode())
    return sha256(payload).hexdigest()


def _append_archive_atom(archive: object, fresh: object, idx: int) -> int:
    """Append a selected fresh-pool atom to the persistent model dictionary."""
    old = archive.size
    # Pool is a mutable dataclass; update arrays in place at the wrapper level.
    archive.w = np.concatenate([archive.w, fresh.w[idx:idx + 1]], axis=0)
    archive.b = np.concatenate([archive.b, fresh.b[idx:idx + 1]], axis=0)
    archive.scales = np.concatenate([archive.scales, fresh.scales[idx:idx + 1]])
    archive.centers = np.concatenate([archive.centers, fresh.centers[idx:idx + 1]])
    archive.available = np.concatenate([archive.available, np.asarray([False])])
    archive.consumed = np.concatenate([archive.consumed, np.asarray([True])])
    archive.original_indices = np.concatenate([
        archive.original_indices, np.asarray([fresh.original_indices[idx]], dtype=np.int64)
    ])
    archive.hash = _digest_pool(archive)
    return old


def _select_random_fc(problem: object, state: object, train_rule: object,
                      cfg: object) -> dict[str, object]:
    """Select a uniformly random admissible atom after evaluating all scores."""
    started = time.perf_counter()
    pool = state.candidate_pool
    scores = _pool_scores(problem, state, pool, train_rule, state.f,
                          cfg.pool.candidate_batch_size)
    reference_scores = _pool_scores(problem, state, state.reference_pool, train_rule,
                                    state.f, cfg.pool.reference_batch_size)
    finite = np.flatnonzero(np.isfinite(scores) & pool.available & ~pool.consumed)
    if finite.size == 0:
        return {
            "admissible": False, "raw_winner_pool_index": -1,
            "nominal_pool_index": None, "candidate_score_rank": None,
            "dual_score_abs": 0.0, "signed_score": 0.0,
            "rejected_candidate_count": 0,
            "elapsed_selection_s": time.perf_counter() - started,
        }
    rng = np.random.default_rng(int(cfg.seed) + 104729 * (state.attempted_iteration + 1))
    order = finite[rng.permutation(finite.size)]
    raw_idx = int(np.argmax(scores))
    raw_oracle_idx = int(np.argmax(reference_scores)) if np.any(np.isfinite(reference_scores)) else -1
    raw_score = float(scores[raw_idx])
    raw_oracle_score = float(reference_scores[raw_oracle_idx]) if raw_oracle_idx >= 0 else 0.0
    rejected = 0
    for rank, idx0 in enumerate(order, start=1):
        idx = int(idx0)
        values, grads = solver_mod.evaluate_atoms(
            train_rule.points, pool.w[idx:idx + 1], pool.b[idx:idx + 1], pool.k
        )
        if problem.requires_zero_mean:
            values -= pool.centers[idx]
        scale = float(pool.scales[idx])
        values /= scale
        grads /= scale
        feat = feature_columns(values, grads, train_rule)[:, 0]
        info = innovation(feat, state.q_basis)
        threshold = cfg.solver.innovation_atol + cfg.solver.innovation_rtol * np.linalg.norm(feat)
        pool.consumed[idx] = True
        pool.available[idx] = False
        if info["absolute"] <= threshold:
            rejected += 1
            continue
        score_data = dual_scores(problem, state.u, state.grad_u, state.f,
                                 values, grads, np.ones(1), train_rule)
        rank_order = np.argsort(-scores, kind="stable")
        score_rank = int(np.flatnonzero(rank_order == idx)[0]) + 1
        return {
            "admissible": True, "raw_winner_pool_index": raw_idx,
            "nominal_pool_index": idx, "candidate_score_rank": score_rank,
            "pairing": float(score_data["pairing"][0]),
            "signed_score": float(score_data["signed_score"][0]),
            "dual_score_abs": float(score_data["absolute_score"][0]),
            "atom_scale": scale, "innovation_abs": info["absolute"],
            "innovation_rel": info["relative"], "innovation_residual": info["residual"],
            "innovation_projections": info["projections"],
            "values_train": values[:, 0], "grads_train": grads[:, 0, :],
            "feature": feat, "rejected_candidate_count": rejected,
            "raw_candidate_score": raw_score, "raw_oracle_pool_index": raw_oracle_idx,
            "raw_oracle_score": raw_oracle_score,
            "raw_oracle_ratio": raw_score / raw_oracle_score if raw_oracle_score > 0 else 1.0,
            "admissible_oracle_pool_index": raw_oracle_idx,
            "admissible_oracle_score": raw_oracle_score,
            "admissible_oracle_ratio": raw_score / raw_oracle_score if raw_oracle_score > 0 else 1.0,
            "oracle_pool_index": raw_oracle_idx, "oracle_score": raw_oracle_score,
            "oracle_ratio": raw_score / raw_oracle_score if raw_oracle_score > 0 else 1.0,
            "coverage_warning": False,
            "remaining_candidate_count": int(np.count_nonzero(pool.available)),
            "elapsed_selection_s": time.perf_counter() - started,
        }
    return {
        "admissible": False, "raw_winner_pool_index": raw_idx,
        "nominal_pool_index": None, "candidate_score_rank": None,
        "dual_score_abs": raw_score, "signed_score": 0.0,
        "rejected_candidate_count": rejected,
        "elapsed_selection_s": time.perf_counter() - started,
    }


def _select_rd_woga(problem: object, state: object, train_rule: object,
                    cfg: object) -> dict[str, object]:
    """Draw a fresh pool, select greedily inside it, and archive the winner."""
    fresh_cfg = replace(cfg, seed=int(cfg.seed + 1000003 + 7919 * state.attempted_iteration))
    fresh = build_pool(problem, fresh_cfg, None, "candidate")
    calibrate_pool(problem, fresh, train_rule, cfg.pool.candidate_batch_size,
                   scale_atol=cfg.solver.scale_atol, scale_rtol=cfg.solver.scale_rtol)
    # Use the copied CGA selector on the fresh pool.  Its reference score is kept
    # unchanged, so the pool-refresh cost is the only changed selection mechanism.
    original_pool = state.candidate_pool
    state.candidate_pool = fresh
    try:
        selected = ORIGINAL_SELECT_ATOM(problem, state, fresh, state.reference_pool,
                                        train_rule, cfg)
    finally:
        state.candidate_pool = original_pool
    if selected.get("admissible"):
        fresh_idx = int(selected["nominal_pool_index"])
        archive_idx = _append_archive_atom(original_pool, fresh, fresh_idx)
        selected["fresh_pool_index"] = fresh_idx
        selected["fresh_pool_hash"] = fresh.hash
        selected["nominal_pool_index"] = archive_idx
        selected["pool_mode"] = "resampled"
    else:
        selected["pool_mode"] = "resampled"
        selected["fresh_pool_hash"] = fresh.hash
    return selected


def _method_selector(method: str):
    original = ORIGINAL_SELECT_ATOM

    def selector(problem: object, state: object, candidate_pool: object,
                 reference_pool: object, train_rule: object, cfg: object) -> dict[str, object]:
        if method == "CGA-FP":
            out = original(problem, state, candidate_pool, reference_pool, train_rule, cfg)
            out["pool_mode"] = "fixed"
            return out
        if method == "Random-FC":
            out = _select_random_fc(problem, state, train_rule, cfg)
            out["pool_mode"] = "fixed"
            return out
        return _select_rd_woga(problem, state, train_rule, cfg)

    return original, selector


def run_one(case_id: str, method: str, seed: int, target: int,
            profile: str, output_root: Path, pool_size: int | None = None) -> dict[str, object]:
    model, dim = CASE_TO_CONFIG[case_id]
    cfg = config_for(model, dim, profile=profile, seed=seed, output_root=str(output_root))
    cfg = replace(cfg, target_accepted=target)
    if case_id == "C4":
        # The comparison case is ID17 (p=4, ReLU^3), not the default
        # one-dimensional pure-p variant used by the general campaign helper.
        cfg = replace(cfg, relu_power=3)
    if pool_size is not None:
        pool = replace(cfg.pool, candidate_size=max(pool_size, target),
                       reference_size=max(2 * pool_size, target + 1))
        cfg = replace(cfg, pool=pool)
    original, selector = _method_selector(method)
    solver_mod.select_atom = selector
    started = time.perf_counter()
    summary = None
    error = None
    try:
        summary = run_cga(cfg)
    except Exception as exc:  # retain failures as first-class observations
        error = f"{type(exc).__name__}: {exc}"
    finally:
        solver_mod.select_atom = original
    wall_end = time.perf_counter() - started
    if summary is None:
        return {"case_id": case_id, "method": method, "seed": seed, "target": target,
                "status": "failed", "failure_reason": error, "wall_end_to_end_sec": wall_end}
    run_dir = Path(summary["run_dir"])
    history = []
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    rows = [r for r in history if r.get("accepted", "False") == "True"]
    last = rows[-1] if rows else (history[-1] if history else {})
    score_sec = sum(float(r.get("elapsed_selection_s") or 0.0) for r in history)
    corr_sec = sum(float(r.get("elapsed_projection_s") or 0.0) for r in history)
    objective_calls = sum(int(float(r.get("objective_evaluations") or 0.0)) for r in history)
    solver_iterations = sum(int(float(r.get("optimizer_iterations") or 0.0)) for r in history)
    counters = {
        "attempted_atoms": len(history),
        "accepted_atoms": int(summary.get("accepted_atom_count", 0)),
        "rejected_attempts": max(0, len(history) - int(summary.get("accepted_atom_count", 0))),
        "score_evaluations": 0,
        "objective_evaluations": objective_calls,
        "active_solver_iterations": solver_iterations,
    }
    # The solver scans every available candidate and reference atom at each
    # attempt; this deterministic count is preferable to an inferred time.
    counters["score_evaluations"] = sum(
        int(cfg.pool.candidate_size) + int(cfg.pool.reference_size) for _ in history
    )
    row = {
        "case_id": case_id, "method": method, "seed": seed, "target": target,
        "status": "success" if summary.get("audit_passed") else "audit_failed",
        "stop_reason": summary.get("stop_reason"),
        "final_width": summary.get("accepted_atom_count"),
        "final_rank": summary.get("effective_rank"),
        "final_energy_gap": (last.get("energy_gap_raw") or ""),
        "final_h1_or_w1p_error": (last.get("natural_rel") or ""),
        "final_v_error": (last.get("quasi_rel") or ""),
        "wall_end_to_end_sec": wall_end,
        "wall_solver_sec": corr_sec,
        "wall_score_sec": score_sec,
        "wall_setup_atom_sec": max(0.0, float(summary.get("elapsed_seconds") or wall_end)
                                  - score_sec - corr_sec),
        "wall_total_internal_sec": summary.get("elapsed_seconds"),
        **counters,
        "run_dir": str(run_dir.resolve()),
        "config_hash": json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("config_sha256"),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["C1", "C2", "C4"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[201])
    parser.add_argument("--target", type=int, default=8)
    parser.add_argument("--profile", choices=["smoke", "report", "formal"], default="report")
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--output-root", type=Path,
                        default=PACKAGE_ROOT / "data" / "derived" / "experiments" / "controlled_baselines" / "runs")
    parser.add_argument("--csv", type=Path,
                        default=PACKAGE_ROOT / "data" / "derived" / "experiments" / "controlled_baselines" / "baseline_runs.csv")
    args = parser.parse_args()
    rows = []
    for case_id in args.cases:
        for method in args.methods:
            for seed in args.seeds:
                print(f"running {case_id} {method} seed={seed}", flush=True)
                rows.append(run_one(case_id, method, seed, args.target, args.profile,
                                    args.output_root, args.pool_size))
    write_csv(args.csv, rows)
    manifest = {
        "schema_version": "sisc-baseline-campaign-v1",
        "cases": args.cases, "methods": args.methods, "seeds": args.seeds,
        "target": args.target, "profile": args.profile, "pool_size": args.pool_size,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pid": os.getpid(),
        "csv": str(args.csv.resolve()),
    }
    args.csv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
