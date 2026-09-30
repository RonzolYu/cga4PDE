"""Straight-line CGA loop with explicit tentative commit and rollback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np

from .config import (PoolConfig, QuadratureConfig, RunConfig, SolverConfig,
                     config_from_dict, validate_config)
from .dictionary import (Pool, build_pool, calibrate_pool, evaluate_atoms,
                         load_pool)
from .metrics import compute_metrics
from .output import (append_history, create_run_dir, environment_info, save_checkpoint,
                     save_model, save_pool_manifest, write_run_header, write_summary)
from .problems import (Problem, check_compatibility, check_zero_mean, energy,
                       exact_solution, make_problem, source_term)
from .quadrature import make_quadratures
from .step1 import classify_step1_stop, feature_columns, select_atom, update_qr
from .step2 import accept_projection, project_full_span


@dataclass
class CGAState:
    cfg: RunConfig
    problem: Problem
    candidate_pool: Pool
    reference_pool: Pool
    quadratures: dict[str, object]
    f: np.ndarray
    f_valid: np.ndarray
    u: np.ndarray
    grad_u: np.ndarray
    validation_u: np.ndarray
    validation_grad_u: np.ndarray
    train_energy: float
    validation_energy: float
    exact_train_energy: float
    exact_validation_energy: float
    basis_values_train: np.ndarray
    basis_grads_train: np.ndarray
    basis_values_valid: np.ndarray
    basis_grads_valid: np.ndarray
    coefficients: np.ndarray
    q_basis: np.ndarray
    r_matrix: np.ndarray
    attempted_iteration: int = 0
    nominal_selected_count: int = 0
    accepted_atom_count: int = 0
    effective_rank: int = 0
    accepted_indices: list[int] = field(default_factory=list)
    initial_best_score: float = 0.0
    low_oracle_count: int = 0
    coverage_warning: bool = False
    had_coverage_warning: bool = False
    quadrature_warning_count: int = 0
    quadrature_warning_start: int | None = None
    first_quadrature_warning: int | None = None
    trusted_atom_count: int = 0
    trusted_metrics: dict[str, object] = field(default_factory=dict)
    history_rows: int = 0
    stop_reason: str = "not_started"
    final_metrics: dict[str, object] = field(default_factory=dict)


def initialize_state(cfg: RunConfig, problem: Problem, pools: tuple[Pool, Pool],
                     quadratures: dict[str, object]) -> CGAState:
    train, valid = quadratures["train"], quadratures["valid"]
    f, f_valid = source_term(problem, train.points), source_term(problem, valid.points)
    u = np.zeros(train.points.shape[0], dtype=np.float64)
    grad = np.zeros((train.points.shape[0], cfg.dim), dtype=np.float64)
    uv = np.zeros(valid.points.shape[0], dtype=np.float64)
    gradv = np.zeros((valid.points.shape[0], cfg.dim), dtype=np.float64)
    exact_t, exact_v = exact_solution(problem, train.points), exact_solution(problem, valid.points)
    e = energy(problem, u, grad, f, train)
    ev = energy(problem, uv, gradv, f_valid, valid)
    exact_e = energy(problem, exact_t[0], exact_t[1], f, train)
    exact_ev = energy(problem, exact_v[0], exact_v[1], f_valid, valid)
    feature_rows = train.points.shape[0] * (cfg.dim + 1)
    state = CGAState(cfg, problem, pools[0], pools[1], quadratures, f, f_valid,
                    u, grad, uv, gradv, e, ev, exact_e, exact_ev,
                    np.zeros((train.points.shape[0], 0)),
                    np.zeros((train.points.shape[0], 0, cfg.dim)),
                    np.zeros((valid.points.shape[0], 0)),
                    np.zeros((valid.points.shape[0], 0, cfg.dim)),
                    np.zeros(0), np.zeros((feature_rows, 0)), np.zeros((0, 0)))
    state.final_metrics = compute_metrics(problem, uv, gradv, exact_v, valid, ev, exact_ev)
    state.trusted_metrics = dict(state.final_metrics)
    return state


def evaluate_solution(state: CGAState, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not state.accepted_indices:
        return np.zeros(x.shape[0]), np.zeros((x.shape[0], state.problem.dim))
    idx = np.asarray(state.accepted_indices)
    values, grads = evaluate_atoms(x, state.candidate_pool.w[idx], state.candidate_pool.b[idx],
                                   state.candidate_pool.k)
    values -= state.candidate_pool.centers[idx][None, :]
    values /= state.candidate_pool.scales[idx][None, :]
    grads /= state.candidate_pool.scales[idx][None, :, None]
    return values @ state.coefficients, np.einsum("qmd,m->qd", grads, state.coefficients, optimize=True)


def evaluate_saved_model(model_path: Path, problem: Problem,
                         x: np.ndarray, batch_size: int = 4096
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a saved model without forming a prohibitive audit-points by atoms array."""
    with np.load(model_path, allow_pickle=False) as model:
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        if coefficients.size == 0:
            return np.zeros(x.shape[0]), np.zeros((x.shape[0], problem.dim))
        w, b = np.asarray(model["w"]), np.asarray(model["b"])
        scales, centers, k = np.asarray(model["scales"]), np.asarray(model["centers"]), int(model["k"])
        u = np.empty(x.shape[0], dtype=np.float64)
        grad = np.empty((x.shape[0], problem.dim), dtype=np.float64)
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            values, grads = evaluate_atoms(x[start:stop], w, b, k)
            values -= centers[None, :]
            values /= scales[None, :]
            grads /= scales[None, :, None]
            u[start:stop] = values @ coefficients
            grad[start:stop] = np.einsum("qmd,m->qd", grads, coefficients, optimize=True)
    return u, grad


def energy_consistency_warning(previous_train_energy: float, train_energy: float,
                               previous_validation_energy: float,
                               validation_energy: float,
                               cfg: SolverConfig) -> dict[str, float | bool | str]:
    """Compare accepted-step energy drops without using an exact solution.

    Using energy drops makes the decision invariant under an additive constant
    in the variational energy.  Only the harmful one-sided discrepancy, where
    training looks more favorable than validation, is used for the warning.
    """
    train_drop = float(previous_train_energy - train_energy)
    validation_drop = float(previous_validation_energy - validation_energy)
    drop_gap = float(train_drop - validation_drop)
    scale = float(max(abs(train_drop), abs(validation_drop),
                      cfg.energy_drop_scale_floor))
    limit = float(cfg.validation_gap_atol + cfg.validation_gap_rtol * scale)
    if validation_drop < -cfg.validation_increase_atol:
        warning, reason = True, "validation_energy_increased"
    elif drop_gap > limit:
        warning, reason = True, "train_valid_drop_disagreed"
    else:
        warning, reason = False, "none"
    return {
        "train_energy_drop": train_drop,
        "validation_energy_drop": validation_drop,
        "train_validation_drop_gap": drop_gap,
        "energy_drop_scale": scale,
        "energy_drop_gap_limit": limit,
        "quadrature_warning": warning,
        "quadrature_warning_reason": reason,
    }


def audit_saved_model(run_dir: Path, problem: Problem,
                      quadratures: dict[str, object],
                      cfg: SolverConfig | None = None) -> dict[str, object]:
    """Re-evaluate the primary (trusted) model on train, validation, and audit rules."""
    cfg = cfg or SolverConfig()
    evaluations: dict[str, dict[str, object]] = {}
    for name in ("train", "valid", "audit"):
        rule = quadratures[name]
        u, grad = evaluate_saved_model(run_dir / "model.npz", problem, rule.points)
        f = source_term(problem, rule.points)
        exact = exact_solution(problem, rule.points)
        numerical_energy = energy(problem, u, grad, f, rule)
        exact_energy = energy(problem, exact[0], exact[1], f, rule)
        evaluations[name] = {
            "numerical_energy": numerical_energy,
            "exact_energy": exact_energy,
            "metrics": compute_metrics(problem, u, grad, exact, rule,
                                       numerical_energy, exact_energy),
        }
    checks: dict[str, dict[str, float | bool] | None] = {}
    gaps: dict[str, float | None] = {}
    for metric in ("energy_gap_raw", "natural_rel", "quasi_rel"):
        valid_value = evaluations["valid"]["metrics"].get(metric)
        audit_value = evaluations["audit"]["metrics"].get(metric)
        if valid_value is None or audit_value is None:
            checks[metric] = None
            gaps[metric] = None
        else:
            valid_float, audit_float = float(valid_value), float(audit_value)
            difference = abs(valid_float - audit_float)
            absolute_tolerance = (cfg.audit_energy_gap_atol
                                  if metric == "energy_gap_raw"
                                  else cfg.audit_metric_atol)
            limit = absolute_tolerance + cfg.audit_rtol * max(
                abs(valid_float), abs(audit_float))
            passed = bool(difference <= limit)
            scale = max(abs(valid_float), abs(audit_float), np.finfo(float).tiny)
            gaps[metric] = difference / scale
            checks[metric] = {"valid": valid_float, "audit": audit_float,
                              "difference": difference, "limit": limit,
                              "passed": passed}
    audit_passed = all(check is None or bool(check["passed"])
                       for check in checks.values())
    return {"rules": evaluations, "metric_checks": checks,
            "validation_audit_relative_gaps": gaps,
            "passed": audit_passed,
            "passed_2pct": audit_passed}


def formal_status(audit_passed: bool, trusted_atom_count: int,
                  target_accepted: int) -> tuple[bool, str]:
    """Return the reporting status; this function never changes the solver."""
    if not audit_passed:
        return False, "audit_failed"
    if trusted_atom_count >= target_accepted:
        return True, "trusted_complete"
    return False, "trusted_partial"


def commit_atom(state: CGAState, selection: dict[str, object],
                projection: dict[str, object], tentative: dict[str, np.ndarray]) -> None:
    state.accepted_indices.append(int(selection["nominal_pool_index"]))
    state.coefficients = np.asarray(projection["coefficients"])
    state.basis_values_train = tentative["values_train"]
    state.basis_grads_train = tentative["grads_train"]
    state.basis_values_valid = tentative["values_valid"]
    state.basis_grads_valid = tentative["grads_valid"]
    state.u = state.basis_values_train @ state.coefficients
    state.grad_u = np.einsum("qmd,m->qd", state.basis_grads_train,
                             state.coefficients, optimize=True)
    state.validation_u = state.basis_values_valid @ state.coefficients
    state.validation_grad_u = np.einsum("qmd,m->qd", state.basis_grads_valid,
                                        state.coefficients, optimize=True)
    state.train_energy = float(projection["final_train_energy"])
    state.validation_energy = float(projection["final_validation_energy"])
    state.q_basis = tentative["q_matrix"]
    state.r_matrix = tentative["r_matrix"]
    state.accepted_atom_count += 1
    state.effective_rank = int(projection["rank"])


def rollback_attempt(state: CGAState, selection: dict[str, object], reason: str) -> None:
    state.stop_reason = "running"


def one_attempt(state: CGAState) -> tuple[dict[str, object], str | None]:
    state.attempted_iteration += 1
    selection = select_atom(state.problem, state, state.candidate_pool, state.reference_pool,
                            state.quadratures["train"], state.cfg)
    if state.attempted_iteration == 1:
        state.initial_best_score = float(selection["dual_score_abs"])
    stop = classify_step1_stop(selection, state, state.candidate_pool, state.cfg)
    record: dict[str, object] = {
        "attempted_iteration": state.attempted_iteration,
        "nominal_selected_count": state.nominal_selected_count,
        "accepted_atom_count": state.accepted_atom_count,
        "effective_rank": state.effective_rank,
        "raw_winner_pool_index": selection.get("raw_winner_pool_index"),
        "nominal_pool_index": selection.get("nominal_pool_index"),
        "accepted_pool_index": None, "candidate_score_rank": selection.get("candidate_score_rank"),
        "signed_score": selection.get("signed_score"), "dual_score_abs": selection.get("dual_score_abs"),
        "atom_scale": selection.get("atom_scale"), "innovation_abs": selection.get("innovation_abs"),
        "innovation_rel": selection.get("innovation_rel"),
        "rejected_candidate_count": selection.get("rejected_candidate_count"),
        "oracle_pool_index": selection.get("oracle_pool_index"), "oracle_score": selection.get("oracle_score"),
        "oracle_ratio": selection.get("oracle_ratio"),
        "raw_oracle_pool_index": selection.get("raw_oracle_pool_index"),
        "raw_oracle_score": selection.get("raw_oracle_score"),
        "raw_oracle_ratio": selection.get("raw_oracle_ratio"),
        "admissible_oracle_pool_index": selection.get("admissible_oracle_pool_index"),
        "admissible_oracle_score": selection.get("admissible_oracle_score"),
        "admissible_oracle_ratio": selection.get("admissible_oracle_ratio"),
        "coverage_warning": selection.get("coverage_warning", False),
        "coverage_warning_streak": state.low_oracle_count,
        "remaining_candidate_count": selection.get("remaining_candidate_count"),
        "train_energy": state.train_energy, "validation_energy": state.validation_energy,
        "accepted": False, "rollback": False, "acceptance_reason": stop or "selection_failed",
        "method_chain": "", "optimizer_success": None, "optimizer_iterations": None,
        "objective_evaluations": None, "projected_residual_abs": None,
        "projected_residual_rel": None, "projected_residual_target": None,
        "energy_decrease": None, "condition_estimate": None,
        "zero_mean_residual": None, "compatibility_residual": None,
        "train_validation_gap": state.validation_energy - state.train_energy,
        "train_energy_drop": None, "validation_energy_drop": None,
        "train_validation_drop_gap": None, "energy_drop_scale": None,
        "energy_drop_gap_limit": None,
        "quadrature_warning": False, "quadrature_warning_streak": state.quadrature_warning_count,
        "quadrature_warning_reason": "none",
        "trusted": False,
        "elapsed_selection_s": selection.get("elapsed_selection_s"), "elapsed_projection_s": 0.0,
        "energy_gap_raw": None, "energy_gap_status": None, "bregman_gap": None,
        "l2_abs": None, "l2_rel": None, "natural_abs": None, "natural_rel": None,
        "w1p_full_abs": None, "w1p_full_rel": None, "quasi_abs": None, "quasi_rel": None,
    }
    if not selection["admissible"] or stop is not None:
        return record, stop
    state.nominal_selected_count += 1
    record["nominal_selected_count"] = state.nominal_selected_count
    idx = int(selection["nominal_pool_index"])
    values_v, grads_v = evaluate_atoms(state.quadratures["valid"].points,
                                       state.candidate_pool.w[idx:idx+1],
                                       state.candidate_pool.b[idx:idx+1],
                                       state.candidate_pool.k)
    if state.problem.requires_zero_mean:
        values_v -= state.candidate_pool.centers[idx]
    values_v /= state.candidate_pool.scales[idx]
    grads_v /= state.candidate_pool.scales[idx]
    tentative = {
        "values_train": np.column_stack([state.basis_values_train, selection["values_train"]]),
        "grads_train": np.concatenate([state.basis_grads_train,
                                       np.asarray(selection["grads_train"])[:, None, :]], axis=1),
        "values_valid": np.column_stack([state.basis_values_valid, values_v[:, 0]]),
        "grads_valid": np.concatenate([state.basis_grads_valid, grads_v], axis=1),
        "f_valid": state.f_valid,
    }
    old_size = state.r_matrix.shape[0]
    tentative_r = np.zeros((old_size + 1, old_size + 1), dtype=np.float64)
    tentative_r[:old_size, :old_size] = state.r_matrix
    tentative_r[:old_size, old_size] = np.asarray(selection["innovation_projections"])
    tentative_r[old_size, old_size] = float(selection["innovation_abs"])
    tentative["r_matrix"] = tentative_r
    tentative["q_matrix"] = update_qr(
        state.q_basis, np.asarray(selection["innovation_residual"]))
    projection = project_full_span(state.problem, tentative, state.coefficients,
                                   (np.zeros_like(state.u), np.zeros_like(state.grad_u)),
                                   state.f, state.quadratures["train"],
                                   state.quadratures["valid"], state.cfg,
                                   r_matrix=tentative_r,
                                   q_matrix=tentative["q_matrix"])
    if "coefficients" in projection:
        u_trial = tentative["values_train"] @ projection["coefficients"]
        constraint = check_zero_mean(state.problem, u_trial, state.quadratures["train"])
    else:
        constraint = {"passed": False, "mean": None}
    accepted, reason = accept_projection(state.train_energy, projection, constraint, state.cfg)
    record.update({"method_chain": ">".join(projection.get("method_chain", [])),
                   "optimizer_success": projection.get("optimizer_success"),
                   "optimizer_iterations": projection.get("nit"),
                   "objective_evaluations": projection.get("nfev"),
                   "projected_residual_abs": projection.get("residual", {}).get("absolute"),
                   "projected_residual_rel": projection.get("residual", {}).get("relative"),
                   "projected_residual_target": projection.get("residual", {}).get("target"),
                   "energy_decrease": projection.get("energy_decrease"),
                   "condition_estimate": projection.get("condition"),
                   "zero_mean_residual": constraint.get("mean"),
                   "elapsed_projection_s": projection.get("elapsed_projection_s", 0.0),
                   "rollback": not accepted, "acceptance_reason": reason})
    if accepted:
        previous_train_energy = state.train_energy
        previous_validation_energy = state.validation_energy
        commit_atom(state, selection, projection, tentative)
        record.update({"accepted": True, "rollback": False,
                       "accepted_pool_index": idx,
                       "accepted_atom_count": state.accepted_atom_count,
                       "effective_rank": state.effective_rank,
                       "train_energy": state.train_energy,
                       "validation_energy": state.validation_energy})
        exact_v = exact_solution(state.problem, state.quadratures["valid"].points)
        metrics = compute_metrics(state.problem, state.validation_u, state.validation_grad_u,
                                  exact_v, state.quadratures["valid"], state.validation_energy,
                                  state.exact_validation_energy)
        state.final_metrics = metrics
        record.update(metrics)
        record["train_validation_gap"] = state.validation_energy - state.train_energy
        consistency = energy_consistency_warning(
            previous_train_energy, state.train_energy,
            previous_validation_energy, state.validation_energy,
            state.cfg.solver)
        record.update(consistency)
        warning = bool(consistency["quadrature_warning"])
        if warning:
            if state.quadrature_warning_count == 0:
                state.quadrature_warning_start = state.accepted_atom_count
            state.quadrature_warning_count += 1
        else:
            state.quadrature_warning_count = 0
            state.quadrature_warning_start = None
            state.trusted_atom_count = state.accepted_atom_count
            state.trusted_metrics = dict(metrics)
            record["trusted"] = True
        record["quadrature_warning_streak"] = state.quadrature_warning_count
        if state.quadrature_warning_count >= state.cfg.solver.quadrature_warning_patience:
            if state.first_quadrature_warning is None:
                state.first_quadrature_warning = state.quadrature_warning_start
            return record, "quadrature_invalid"
    else:
        rollback_attempt(state, selection, reason)
    return record, None


def should_stop(state: CGAState) -> tuple[bool, str]:
    if state.accepted_atom_count >= state.cfg.target_accepted:
        if state.quadrature_warning_count:
            reason = "target_reached_with_quadrature_warning"
        else:
            reason = ("target_reached_with_coverage_warning"
                      if state.coverage_warning else "target_reached")
        return True, reason
    if state.attempted_iteration >= state.cfg.max_attempts:
        return True, "max_attempts"
    return False, "running"


def run_cga(cfg: RunConfig) -> dict[str, object]:
    validate_config(cfg)
    started = time.perf_counter()
    problem = make_problem(cfg.model, cfg.dim, cfg.p, cfg.epsilon, cfg.exact_profile)
    candidate = build_pool(problem, cfg, None, "candidate")
    reference = build_pool(problem, cfg, None, "reference")
    quadratures = make_quadratures(cfg, candidate, reference)
    calibrate_pool(problem, candidate, quadratures["train"], cfg.pool.candidate_batch_size,
                   scale_atol=cfg.solver.scale_atol, scale_rtol=cfg.solver.scale_rtol)
    calibrate_pool(problem, reference, quadratures["train"], cfg.pool.reference_batch_size,
                   scale_atol=cfg.solver.scale_atol, scale_rtol=cfg.solver.scale_rtol)
    state = initialize_state(cfg, problem, (candidate, reference), quadratures)
    compatibility = check_compatibility(problem, state.f, quadratures["train"])
    if not compatibility["passed"]:
        raise ValueError(f"Neumann compatibility failed: {compatibility['residual']:.3e}")
    run_dir = create_run_dir(Path(cfg.output_root) / cfg.phase, cfg)
    write_run_header(run_dir, cfg, environment_info())
    save_pool_manifest(run_dir, candidate, reference)
    save_model(run_dir, state, "trusted_model.npz")
    state.stop_reason = "running"
    while True:
        record, step_stop = one_attempt(state)
        record["compatibility_residual"] = compatibility["residual"]
        record["elapsed_total_s"] = time.perf_counter() - started
        append_history(run_dir, record)
        state.history_rows += 1
        if bool(record.get("accepted")):
            save_model(run_dir, state,
                       f"states/accepted_{state.accepted_atom_count:04d}.npz")
        if bool(record.get("accepted")) and bool(record.get("trusted")):
            save_model(run_dir, state, "trusted_model.npz")
        if state.attempted_iteration % cfg.solver.checkpoint_every_attempts == 0:
            save_checkpoint(run_dir, state)
        stop, reason = should_stop(state)
        if step_stop is not None:
            stop, reason = True, step_stop
        if stop:
            state.stop_reason = reason
            break
    save_checkpoint(run_dir, state)
    save_model(run_dir, state, "diagnostic_model.npz")
    if state.quadrature_warning_count == 0:
        save_model(run_dir, state, "model.npz")
    else:
        # The primary model is always the last quadrature-trusted checkpoint.
        trusted = run_dir / "trusted_model.npz"
        (run_dir / "model.npz").write_bytes(trusted.read_bytes())
    audit = audit_saved_model(run_dir, problem, quadratures, cfg.solver)
    audit_passed = bool(audit["passed"])
    complete, status = formal_status(audit_passed, state.trusted_atom_count,
                                     cfg.target_accepted)
    trusted_metrics = dict(audit["rules"]["valid"]["metrics"])
    summary = {"run_dir": str(run_dir.resolve()), "phase": cfg.phase,
               "model": cfg.model, "p": cfg.p, "epsilon": cfg.epsilon,
               "relu_power": cfg.relu_power, "dim": cfg.dim, "seed": cfg.seed,
               "target_accepted": cfg.target_accepted,
               "target_reached": state.accepted_atom_count >= cfg.target_accepted,
               "trusted_target_reached": state.trusted_atom_count >= cfg.target_accepted,
               "audit_passed": audit_passed,
               "formal_complete": complete,
               "formal_status": status,
               "stop_reason": state.stop_reason,
               "attempted_iteration": state.attempted_iteration,
               "nominal_selected_count": state.nominal_selected_count,
               "accepted_atom_count": state.accepted_atom_count,
               "effective_rank": state.effective_rank,
               "trusted_atom_count": state.trusted_atom_count,
               "first_quadrature_warning": state.first_quadrature_warning,
               "quadrature_warning_streak": state.quadrature_warning_count,
               "coverage_warning": state.coverage_warning,
               "had_coverage_warning": state.had_coverage_warning,
               "train_energy": state.train_energy,
               "validation_energy": state.validation_energy,
               "exact_validation_energy": state.exact_validation_energy,
               "compatibility_residual": compatibility["residual"],
               "elapsed_seconds": time.perf_counter() - started,
               "metrics": trusted_metrics,
               "terminal_metrics": state.final_metrics,
               "quadrature_audit": audit}
    write_summary(run_dir, summary)
    return summary


def resume_cga(run_dir: str | Path) -> dict[str, object]:
    import json
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved_summary.get("target_reached"):
            return saved_summary
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = config_from_dict(raw)
    problem = make_problem(cfg.model, cfg.dim, cfg.p, cfg.epsilon, cfg.exact_profile)
    candidate = load_pool(run_dir / "candidate_pool.npz")
    reference = load_pool(run_dir / "reference_pool.npz")
    quadratures = make_quadratures(cfg, candidate, reference)
    checkpoint = np.load(run_dir / "checkpoint.npz", allow_pickle=False)
    candidate.consumed[:] = checkpoint["candidate_consumed"]
    candidate.available &= ~candidate.consumed
    state = initialize_state(cfg, problem, (candidate, reference), quadratures)
    state.attempted_iteration = int(checkpoint["attempted"])
    state.nominal_selected_count = int(checkpoint["nominal"])
    state.accepted_atom_count = int(checkpoint["accepted"])
    state.effective_rank = int(checkpoint["rank"])
    state.accepted_indices = checkpoint["accepted_indices"].astype(int).tolist()
    state.coefficients = np.asarray(checkpoint["coefficients"], dtype=np.float64)
    saved_q = (np.asarray(checkpoint["q_basis"], dtype=np.float64)
               if "q_basis" in checkpoint.files else None)
    state.q_basis = (saved_q if saved_q is not None
                     else np.zeros((quadratures["train"].points.shape[0] * (cfg.dim + 1), 0)))
    state.r_matrix = (np.asarray(checkpoint["r_matrix"], dtype=np.float64)
                      if "r_matrix" in checkpoint.files else np.zeros((0, 0)))
    state.history_rows = int(checkpoint["history_rows"])
    state.initial_best_score = float(checkpoint["initial_best_score"]) if "initial_best_score" in checkpoint.files else 0.0
    state.low_oracle_count = int(checkpoint["low_oracle_count"]) if "low_oracle_count" in checkpoint.files else 0
    state.coverage_warning = bool(checkpoint["coverage_warning"]) if "coverage_warning" in checkpoint.files else False
    state.had_coverage_warning = bool(checkpoint["had_coverage_warning"]) if "had_coverage_warning" in checkpoint.files else False
    state.quadrature_warning_count = int(checkpoint["quadrature_warning_count"]) if "quadrature_warning_count" in checkpoint.files else 0
    warning_start = int(checkpoint["quadrature_warning_start"]) if "quadrature_warning_start" in checkpoint.files else -1
    state.quadrature_warning_start = None if warning_start < 0 else warning_start
    first_warning = int(checkpoint["first_quadrature_warning"]) if "first_quadrature_warning" in checkpoint.files else -1
    state.first_quadrature_warning = None if first_warning < 0 else first_warning
    state.trusted_atom_count = int(checkpoint["trusted_atom_count"]) if "trusted_atom_count" in checkpoint.files else state.accepted_atom_count
    if state.accepted_indices:
        idx = np.asarray(state.accepted_indices, dtype=int)
        vt, gt = evaluate_atoms(quadratures["train"].points, candidate.w[idx],
                                candidate.b[idx], candidate.k)
        vv, gv = evaluate_atoms(quadratures["valid"].points, candidate.w[idx],
                                candidate.b[idx], candidate.k)
        vt -= candidate.centers[idx][None, :]
        vv -= candidate.centers[idx][None, :]
        vt /= candidate.scales[idx][None, :]
        vv /= candidate.scales[idx][None, :]
        gt /= candidate.scales[idx][None, :, None]
        gv /= candidate.scales[idx][None, :, None]
        state.basis_values_train, state.basis_grads_train = vt, gt
        state.basis_values_valid, state.basis_grads_valid = vv, gv
        state.u = vt @ state.coefficients
        state.grad_u = np.einsum("qmd,m->qd", gt, state.coefficients, optimize=True)
        state.validation_u = vv @ state.coefficients
        state.validation_grad_u = np.einsum("qmd,m->qd", gv, state.coefficients, optimize=True)
        if saved_q is None:
            state.q_basis, state.r_matrix = np.linalg.qr(
                feature_columns(vt, gt, quadratures["train"]), mode="reduced")
        state.train_energy = energy(problem, state.u, state.grad_u, state.f, quadratures["train"])
        state.validation_energy = energy(problem, state.validation_u, state.validation_grad_u,
                                         state.f_valid, quadratures["valid"])
        exact_v = exact_solution(problem, quadratures["valid"].points)
        state.final_metrics = compute_metrics(problem, state.validation_u, state.validation_grad_u,
                                              exact_v, quadratures["valid"],
                                              state.validation_energy,
                                              state.exact_validation_energy)
        state.trusted_metrics = dict(state.final_metrics)
    compatibility = check_compatibility(problem, state.f, quadratures["train"])
    while True:
        stop, reason = should_stop(state)
        if stop:
            state.stop_reason = reason
            break
        record, step_stop = one_attempt(state)
        record["compatibility_residual"] = compatibility["residual"]
        record["elapsed_total_s"] = None
        append_history(run_dir, record)
        state.history_rows += 1
        if bool(record.get("accepted")):
            save_model(run_dir, state,
                       f"states/accepted_{state.accepted_atom_count:04d}.npz")
        if bool(record.get("accepted")) and bool(record.get("trusted")):
            save_model(run_dir, state, "trusted_model.npz")
        save_checkpoint(run_dir, state)
        if step_stop is not None:
            state.stop_reason = step_stop
            break
    save_model(run_dir, state, "diagnostic_model.npz")
    if state.quadrature_warning_count == 0:
        save_model(run_dir, state)
    elif (run_dir / "trusted_model.npz").exists():
        (run_dir / "model.npz").write_bytes((run_dir / "trusted_model.npz").read_bytes())
    audit = audit_saved_model(run_dir, problem, quadratures, cfg.solver)
    audit_passed = bool(audit["passed"])
    complete, status = formal_status(audit_passed, state.trusted_atom_count,
                                     cfg.target_accepted)
    trusted_metrics = dict(audit["rules"]["valid"]["metrics"])
    exact_v = exact_solution(problem, quadratures["valid"].points)
    state.final_metrics = compute_metrics(problem, state.validation_u, state.validation_grad_u,
                                          exact_v, quadratures["valid"], state.validation_energy,
                                          state.exact_validation_energy)
    summary = {"run_dir": str(run_dir.resolve()), "phase": cfg.phase, "model": cfg.model,
               "p": cfg.p, "epsilon": cfg.epsilon, "relu_power": cfg.relu_power,
               "dim": cfg.dim, "seed": cfg.seed, "target_accepted": cfg.target_accepted,
               "target_reached": state.accepted_atom_count >= cfg.target_accepted,
               "trusted_target_reached": state.trusted_atom_count >= cfg.target_accepted,
               "audit_passed": audit_passed,
               "formal_complete": complete,
               "formal_status": status,
               "stop_reason": state.stop_reason, "attempted_iteration": state.attempted_iteration,
               "nominal_selected_count": state.nominal_selected_count,
               "accepted_atom_count": state.accepted_atom_count, "effective_rank": state.effective_rank,
               "trusted_atom_count": state.trusted_atom_count,
               "first_quadrature_warning": state.first_quadrature_warning,
               "quadrature_warning_streak": state.quadrature_warning_count,
               "coverage_warning": state.coverage_warning,
               "had_coverage_warning": state.had_coverage_warning,
               "train_energy": state.train_energy, "validation_energy": state.validation_energy,
               "exact_validation_energy": state.exact_validation_energy,
               "compatibility_residual": compatibility["residual"], "elapsed_seconds": None,
               "metrics": trusted_metrics,
               "terminal_metrics": state.final_metrics,
               "quadrature_audit": audit, "resumed": True}
    write_summary(run_dir, summary)
    return summary
