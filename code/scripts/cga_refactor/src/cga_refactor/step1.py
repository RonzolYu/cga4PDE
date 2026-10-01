"""Step 1: exhaustive finite-pool dual scoring with numerical admissibility."""

from __future__ import annotations

import time
import numpy as np

from .dictionary import evaluate_atoms, compute_atom_scales
from .problems import first_variation, flux


def dual_scores(problem: object, u: np.ndarray, grad_u: np.ndarray, f: np.ndarray,
                atom_values: np.ndarray, atom_grads: np.ndarray,
                scales: np.ndarray, rule: object) -> dict[str, np.ndarray]:
    pairing = np.asarray(first_variation(problem, u, grad_u, atom_values,
                                         atom_grads, f, rule))
    signed = -pairing / scales
    return {"pairing": pairing, "signed_score": signed, "absolute_score": np.abs(signed)}


def feature_columns(values: np.ndarray, grads: np.ndarray, rule: object) -> np.ndarray:
    """Keep one atom per column in the weighted value/gradient embedding.

    Gradient input axes are (quadrature point, atom, spatial component).
    Rows enumerate (quadrature point, spatial component), so move the atom
    axis last before flattening.  This also agrees with incremental columns
    and the inverse layout used by the stable-coordinate solver.
    """
    root = np.sqrt(rule.weights)
    gradient_rows = (root[:, None, None] * grads).transpose(0, 2, 1)
    return np.vstack([root[:, None] * values,
                      gradient_rows.reshape(values.shape[0] * grads.shape[2], values.shape[1])])


def innovation(feature: np.ndarray, q_basis: np.ndarray) -> dict[str, object]:
    column = np.asarray(feature, dtype=np.float64).reshape(-1)
    residual = column.copy()
    projections = np.zeros(0, dtype=np.float64)
    if q_basis.shape[1]:
        projections = q_basis.T @ residual
        residual -= q_basis @ projections
        correction = q_basis.T @ residual
        projections += correction
        residual -= q_basis @ correction
    abs_value = float(np.linalg.norm(residual))
    rel_value = abs_value / max(float(np.linalg.norm(column)), np.finfo(float).tiny)
    return {"absolute": abs_value, "relative": rel_value, "residual": residual,
            "projections": projections}


def update_qr(q_basis: np.ndarray, residual_column: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(residual_column))
    if norm == 0.0:
        return q_basis.copy()
    return np.column_stack([q_basis, residual_column / norm])


def effective_rank(feature_matrix: np.ndarray, atol: float,
                   rtol: float) -> tuple[int, np.ndarray, float]:
    if feature_matrix.shape[1] == 0:
        return 0, np.zeros(0), 1.0
    singular = np.linalg.svd(feature_matrix, compute_uv=False)
    threshold = atol + rtol * singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    condition = float(singular[0] / singular[rank - 1]) if rank else float("inf")
    return rank, singular, condition


def _pool_scores(problem: object, state: object, pool: object, rule: object,
                 f: np.ndarray, batch_size: int) -> np.ndarray:
    scores = np.full(pool.size, -np.inf, dtype=np.float64)
    usable = np.flatnonzero(pool.available & ~pool.consumed)
    diffusion, reaction = flux(problem, state.u, state.grad_u)
    weighted_diffusion = rule.weights[:, None] * diffusion
    weighted_reaction = rule.weights * (reaction - f)
    for chunk in range(0, usable.size, batch_size):
        idx = usable[chunk:chunk + batch_size]
        z = rule.points @ pool.w[idx].T + pool.b[idx][None, :]
        positive = np.maximum(z, 0.0)
        values = positive ** pool.k
        if problem.requires_zero_mean:
            values -= pool.centers[idx][None, :]
        scales = pool.scales[idx]
        directional_flux = weighted_diffusion @ pool.w[idx].T
        derivative = ((z > 0.0).astype(np.float64) if pool.k == 1
                      else pool.k * positive ** (pool.k - 1))
        pairing = np.sum(derivative * directional_flux, axis=0)
        pairing += values.T @ weighted_reaction
        scores[idx] = np.abs(-pairing / scales)
    return scores


def _best_admissible(problem: object, state: object, pool: object,
                     scores: np.ndarray, rule: object, cfg: object
                     ) -> tuple[int, float]:
    """Return the best pool atom that passes the same innovation gate as Step 1."""
    for idx in np.argsort(-scores, kind="stable"):
        if not np.isfinite(scores[idx]):
            break
        values, grads = evaluate_atoms(rule.points, pool.w[idx:idx+1],
                                       pool.b[idx:idx+1], pool.k)
        if problem.requires_zero_mean:
            values -= pool.centers[idx]
        values /= float(pool.scales[idx])
        grads /= float(pool.scales[idx])
        feat = feature_columns(values, grads, rule)[:, 0]
        info = innovation(feat, state.q_basis)
        threshold = cfg.solver.innovation_atol + cfg.solver.innovation_rtol * np.linalg.norm(feat)
        if float(info["absolute"]) > threshold:
            return int(idx), float(scores[idx])
    return -1, 0.0


def select_atom(problem: object, state: object, candidate_pool: object,
                reference_pool: object, train_rule: object, cfg: object) -> dict[str, object]:
    started = time.perf_counter()
    candidate_scores = _pool_scores(problem, state, candidate_pool, train_rule, state.f,
                                    cfg.pool.candidate_batch_size)
    reference_scores = _pool_scores(problem, state, reference_pool, train_rule, state.f,
                                    cfg.pool.reference_batch_size)
    raw_idx = int(np.argmax(candidate_scores)) if np.any(np.isfinite(candidate_scores)) else -1
    raw_oracle_idx = int(np.argmax(reference_scores)) if np.any(np.isfinite(reference_scores)) else -1
    raw_score = float(candidate_scores[raw_idx]) if raw_idx >= 0 else 0.0
    raw_oracle_score = float(reference_scores[raw_oracle_idx]) if raw_oracle_idx >= 0 else 0.0
    raw_ratio = raw_score / raw_oracle_score if raw_oracle_score > 0.0 else 1.0
    admissible_oracle_idx, admissible_oracle_score = _best_admissible(
        problem, state, reference_pool, reference_scores, train_rule, cfg)
    order = np.argsort(-candidate_scores, kind="stable")
    selected = None
    rejected = 0
    for score_rank, idx in enumerate(order, start=1):
        if not np.isfinite(candidate_scores[idx]):
            break
        values, grads = evaluate_atoms(train_rule.points, candidate_pool.w[idx:idx+1],
                                       candidate_pool.b[idx:idx+1], candidate_pool.k)
        if problem.requires_zero_mean:
            values -= candidate_pool.centers[idx]
        scale = float(candidate_pool.scales[idx])
        values /= scale
        grads /= scale
        score_data = dual_scores(problem, state.u, state.grad_u, state.f, values, grads,
                                 np.ones(1), train_rule)
        feat = feature_columns(values, grads, train_rule)[:, 0]
        info = innovation(feat, state.q_basis)
        threshold = cfg.solver.innovation_atol + cfg.solver.innovation_rtol * np.linalg.norm(feat)
        candidate_pool.consumed[idx] = True
        candidate_pool.available[idx] = False
        if info["absolute"] <= threshold:
            rejected += 1
            continue
        selected = {"admissible": True, "raw_winner_pool_index": raw_idx,
                    "nominal_pool_index": int(idx), "candidate_score_rank": score_rank,
                    "pairing": float(score_data["pairing"][0]),
                    "signed_score": float(score_data["signed_score"][0]),
                    "dual_score_abs": float(score_data["absolute_score"][0]),
                    "atom_scale": scale, "innovation_abs": info["absolute"],
                    "innovation_rel": info["relative"], "innovation_residual": info["residual"],
                    "innovation_projections": info["projections"],
                    "values_train": values[:, 0], "grads_train": grads[:, 0, :],
                    "feature": feat, "rejected_candidate_count": rejected}
        break
    if selected is None:
        selected = {"admissible": False, "raw_winner_pool_index": raw_idx,
                    "nominal_pool_index": None, "candidate_score_rank": None,
                    "dual_score_abs": raw_score, "signed_score": 0.0,
                    "rejected_candidate_count": rejected}
    admissible_candidate_score = float(selected.get("dual_score_abs", 0.0))
    admissible_ratio = (admissible_candidate_score / admissible_oracle_score
                        if admissible_oracle_score > 0.0 else 1.0)
    coverage_warning = (bool(selected["admissible"]) and
                        admissible_ratio < cfg.solver.oracle_ratio_floor)
    selected.update({"raw_candidate_score": raw_score,
                     "raw_oracle_pool_index": raw_oracle_idx,
                     "raw_oracle_score": raw_oracle_score,
                     "raw_oracle_ratio": raw_ratio,
                     "admissible_oracle_pool_index": admissible_oracle_idx,
                     "admissible_oracle_score": admissible_oracle_score,
                     "admissible_oracle_ratio": admissible_ratio,
                     "coverage_warning": coverage_warning,
                     # Backward-compatible aliases now refer to the admissible diagnostic.
                     "oracle_pool_index": admissible_oracle_idx,
                     "oracle_score": admissible_oracle_score,
                     "oracle_ratio": admissible_ratio,
                     "remaining_candidate_count": int(np.count_nonzero(candidate_pool.available)),
                     "elapsed_selection_s": time.perf_counter() - started})
    return selected


def classify_step1_stop(selection: dict[str, object], state: object, pool: object,
                        cfg: object) -> str | None:
    if selection["admissible"]:
        if bool(selection["coverage_warning"]):
            state.low_oracle_count += 1
        else:
            state.low_oracle_count = 0
        state.coverage_warning = state.low_oracle_count >= cfg.solver.oracle_patience
        state.had_coverage_warning = state.had_coverage_warning or state.coverage_warning
        # Coverage is a quality warning.  It must not terminate an otherwise
        # admissible CGA run.
        return None
    if not np.any(pool.available):
        return "pool_exhausted"
    initial = max(state.initial_best_score, 1.0)
    if float(selection["dual_score_abs"]) <= cfg.solver.score_atol + cfg.solver.score_rtol * initial:
        return "fixed_pool_stationary"
    return "rank_saturation"
