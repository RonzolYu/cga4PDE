"""Fixed-state numerical diagnostics for the p=4 parameter study.

The script only reads archived pools, states, and histories. It writes
CSV/JSON summaries under ``continuation_study/data/derived/section85``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parent
OUT = ROOT / "data" / "derived" / "section85"
SENSITIVITY = ROOT / "data" / "raw" / "sensitivity"
sys.path.insert(0, str(ROOT / "scripts" / "cga_refactor" / "src"))

from cga_refactor.config import PoolConfig, QuadratureConfig, RunConfig, SolverConfig
from cga_refactor.dictionary import breakpoints_1d, evaluate_atoms, load_pool
from cga_refactor.problems import make_problem, source_term
from cga_refactor.quadrature import segmented_gauss_1d
from cga_refactor.solver import evaluate_saved_model
from cga_refactor.step1 import _pool_scores, feature_columns
from cga_refactor.step2 import project_full_span


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_path(name: str) -> Path:
    matches = sorted((SENSITIVITY / name).glob("*/config.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one run for {name}, found {len(matches)}")
    return matches[0].parent


def load_config(run: Path) -> RunConfig:
    raw = json.loads((run / "config.json").read_text(encoding="utf-8"))
    return RunConfig(
        model=raw["model"],
        dim=raw["dim"],
        seed=raw["seed"],
        p=raw["p"],
        epsilon=raw["epsilon"],
        relu_power=raw["relu_power"],
        exact_profile=raw["exact_profile"],
        quadrature_level=raw["quadrature_level"],
        target_accepted=raw["target_accepted"],
        pool=PoolConfig(**raw["pool"]),
        quadrature=QuadratureConfig(**raw["quadrature"]),
        solver=SolverConfig(**raw["solver"]),
    )


def common_rule(candidate: object, reference: object, order: int, min_width: float) -> object:
    points = np.unique(np.r_[breakpoints_1d(candidate), breakpoints_1d(reference), 0.5])
    return segmented_gauss_1d(points, order, min_width)


def consumed_at_state(history: list[dict[str, str]], state: int) -> list[int]:
    consumed: list[int] = []
    if state == 0:
        return consumed
    for row in history:
        value = row.get("nominal_pool_index", "")
        if value not in ("", "None"):
            consumed.append(int(value))
        if row["accepted"] == "True" and int(row["accepted_atom_count"]) == state:
            break
    return sorted(set(consumed))


def score_rows_for_state(run: Path, state: int, orders: list[int]) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = load_config(run)
    problem = make_problem(config.model, config.dim, config.p, config.epsilon, config.exact_profile)
    candidate = load_pool(run / "candidate_pool.npz")
    reference = load_pool(run / "reference_pool.npz")
    history = read_csv(run / "history.csv")
    consumed = consumed_at_state(history, state)
    scores: dict[int, np.ndarray] = {}
    for order in orders:
        rule = common_rule(candidate, reference, order, config.quadrature.min_segment_width)
        u, grad = evaluate_saved_model(
            run / "states" / f"accepted_{state:04d}.npz", problem, rule.points
        )
        state_obj = SimpleNamespace(u=u, grad_u=grad)
        values = _pool_scores(
            problem,
            state_obj,
            candidate,
            rule,
            source_term(problem, rule.points),
            config.pool.candidate_batch_size,
        )
        values = np.asarray(values, dtype=np.float64)
        values[~np.isfinite(values)] = 0.0
        if consumed:
            values[np.asarray(consumed, dtype=int)] = 0.0
        scores[order] = values

    rows: list[dict[str, object]] = []
    order_keys = sorted(scores)
    for idx in range(candidate.size):
        if not np.isfinite(candidate.scales[idx]) or candidate.scales[idx] <= 1e-12:
            continue
        row: dict[str, object] = {"candidate_pool_index": idx}
        for order in order_keys:
            row[f"score_q{order}"] = float(scores[order][idx])
            row[f"rank_q{order}"] = int(np.argsort(-scores[order], kind="stable").tolist().index(idx))
        rows.append(row)

    summary: dict[str, object] = {
        "state_N": state,
        "consumed_count": len(consumed),
        "valid_candidate_count": len(rows),
    }
    for left, right in zip(order_keys[:-1], order_keys[1:]):
        a, b = scores[left], scores[right]
        top_a = int(np.argmax(a))
        top_b = int(np.argmax(b))
        rank_a = np.argsort(-a, kind="stable")
        rank_b = np.argsort(-b, kind="stable")
        top10_a, top10_b = set(rank_a[:10].tolist()), set(rank_b[:10].tolist())
        denom = max(float(np.max(np.abs(a))), np.finfo(float).tiny)
        summary.update({
            f"top1_q{left}": top_a,
            f"top1_q{right}": top_b,
            f"top1_same_q{left}_q{right}": top_a == top_b,
            f"top10_overlap_q{left}_q{right}": len(top10_a & top10_b),
            f"max_abs_score_diff_q{left}_q{right}": float(np.max(np.abs(a - b))),
            f"max_rel_score_diff_q{left}_q{right}": float(np.max(np.abs(a - b)) / denom),
        })
    return rows, summary


def candidate_ranking() -> None:
    run = run_path("base_pure")
    rows, summary = score_rows_for_state(run, 8, [4, 6, 10, 16, 20, 32])
    write_csv(OUT / "candidate_ranking_step8.csv", rows)
    (OUT / "candidate_ranking_step8_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def same_space_replay() -> None:
    run = run_path("base_pure")
    config = load_config(run)
    problem = make_problem(config.model, config.dim, config.p, config.epsilon, config.exact_profile)
    candidate = load_pool(run / "candidate_pool.npz")
    history = read_csv(run / "history.csv")
    accepted = [int(r["accepted_pool_index"]) for r in history if r["accepted"] == "True"][:8]
    indices = accepted + [398]
    rule = segmented_gauss_1d(
        np.r_[breakpoints_1d(candidate), 0.5],
        config.quadrature.train_order_1d,
        config.quadrature.min_segment_width,
    )
    valid = segmented_gauss_1d(
        np.r_[breakpoints_1d(candidate), 0.5],
        config.quadrature.validation_order_1d,
        config.quadrature.min_segment_width,
    )
    values, grads = evaluate_atoms(rule.points, candidate.w[indices], candidate.b[indices], candidate.k)
    values -= candidate.centers[indices][None, :]
    values /= candidate.scales[indices][None, :]
    grads /= candidate.scales[indices][None, :, None]
    values_valid, grads_valid = evaluate_atoms(
        valid.points, candidate.w[indices], candidate.b[indices], candidate.k
    )
    values_valid -= candidate.centers[indices][None, :]
    values_valid /= candidate.scales[indices][None, :]
    grads_valid /= candidate.scales[indices][None, :, None]
    with np.load(run / "states" / "accepted_0008.npz") as state:
        coefficients = state["coefficients"].copy()
    q_matrix = np.linalg.qr(feature_columns(values, grads, rule), mode="reduced")[0]
    # Hold the space, initial state, quadrature, and iteration caps fixed.
    # Only the relative stopping tolerance changes.
    trials = [(f"rtol={rtol:g}", SolverConfig(**{
        **config.solver.__dict__, "projection_rtol": rtol}))
        for rtol in (1e-4, 1e-6, 1e-8)]
    current_u = values[:, :len(coefficients)] @ coefficients
    current_grad = np.einsum("qmd,m->qd", grads[:, :len(coefficients), :], coefficients)
    rows = []
    for label, solver in trials:
        local = RunConfig(
            model=config.model,
            dim=config.dim,
            seed=config.seed,
            p=config.p,
            epsilon=config.epsilon,
            relu_power=config.relu_power,
            exact_profile=config.exact_profile,
            quadrature_level=config.quadrature_level,
            target_accepted=config.target_accepted,
            pool=config.pool,
            quadrature=config.quadrature,
            solver=solver,
        )
        result = project_full_span(
            problem,
            {
                "values_train": values,
                "grads_train": grads,
                "values_valid": values_valid,
                "grads_valid": grads_valid,
                "f_valid": source_term(problem, valid.points),
            },
            coefficients,
            (current_u, current_grad),
            source_term(problem, rule.points),
            rule,
            valid,
            local,
            r_matrix=None,
            q_matrix=q_matrix,
        )
        residual = result.get("residual", {})
        rows.append({
            "run_label": label,
            "attempt_index": 9,
            "candidate_pool_index": 398,
            "method_chain": ">".join(result.get("method_chain", [])),
            "projected_residual_abs": residual.get("absolute"),
            "projected_residual_rel": residual.get("relative"),
            "projected_residual_target": residual.get("target"),
            "residual_passed": residual.get("passed"),
            "energy_decrease": result.get("energy_decrease"),
            "final_train_energy": result.get("final_train_energy"),
            "objective_gradient_calls": result.get("evaluation_counts", {}).get("objective_gradient_calls"),
            "hvp_calls": result.get("evaluation_counts", {}).get("hvp_calls"),
            "replay_result": "accepted" if residual.get("passed") else "projected_residual",
            "historical_acceptance_reason": "projected_residual",
            "validation_level": "fixed-space replay with the installed solver; not a historical rerun",
        })
    write_csv(OUT / "same_space_replay.csv", rows)


def _pool_parameter_set(run: Path) -> np.ndarray:
    pool = load_pool(run / "candidate_pool.npz")
    return np.column_stack([pool.w, pool.b])


def pool_overlap() -> None:
    runs = {name: run_path(name) for name in ["base_pure", "pool_256", "pool_1024"]}
    values = {name: _pool_parameter_set(run) for name, run in runs.items()}
    rows = []
    for left, right in [("base_pure", "pool_256"), ("base_pure", "pool_1024"),
                        ("pool_256", "pool_1024")]:
        a, b = values[left], values[right]
        # 1D parameters are represented exactly up to the canonicalization tolerance.
        keys_a = {tuple(np.round(row, 12).tolist()) for row in a}
        keys_b = {tuple(np.round(row, 12).tolist()) for row in b}
        common = len(keys_a & keys_b)
        rows.append({
            "left": left,
            "right": right,
            "left_effective": len(keys_a),
            "right_effective": len(keys_b),
            "common_parameters": common,
            "left_common_fraction": common / max(len(keys_a), 1),
            "right_common_fraction": common / max(len(keys_b), 1),
            "nested_left_in_right": common == len(keys_a),
            "nested_right_in_left": common == len(keys_b),
        })
    write_csv(OUT / "pool_overlap.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["ranking", "same-space", "pool-overlap", "all"],
        default=["all"],
    )
    args = parser.parse_args()
    parts = set(args.parts)
    if "all" in parts or "ranking" in parts:
        candidate_ranking()
    if "all" in parts or "same-space" in parts:
        same_space_replay()
    if "all" in parts or "pool-overlap" in parts:
        pool_overlap()
    print("section85 sensitivity diagnostics written to", OUT)


if __name__ == "__main__":
    main()
