"""Step 2: stable-coordinate full-span correction and acceptance gates."""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.linalg import LinAlgError, solve, solve_triangular
from scipy.sparse.linalg import LinearOperator, cg

from .problems import coefficient_hvp, coefficient_objective, energy
from .step1 import feature_columns


def build_feature_matrix(basis_values: np.ndarray, basis_grads: np.ndarray,
                         rule: object) -> np.ndarray:
    return feature_columns(basis_values, basis_grads, rule)


def stable_transform(matrix: np.ndarray, atol: float, rtol: float) -> dict[str, object]:
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    threshold = atol + (rtol * singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > threshold))
    if rank == 0:
        return {"u": u[:, :0], "singular_values": singular, "v": vt.T[:, :0],
                "rank": 0, "to_coeff": np.zeros((matrix.shape[1], 0)),
                "to_stable": np.zeros((0, matrix.shape[1])), "condition": float("inf")}
    v = vt[:rank].T
    to_coeff = v / singular[:rank][None, :]
    to_stable = singular[:rank, None] * vt[:rank]
    return {"u": u[:, :rank], "singular_values": singular, "v": v, "rank": rank,
            "to_coeff": to_coeff, "to_stable": to_stable,
            "condition": float(singular[0] / singular[rank - 1])}


def qr_transform(r_matrix: np.ndarray, atol: float, rtol: float) -> dict[str, object]:
    """Use the incrementally constructed feature QR without refactorizing a tall matrix."""
    diagonal = np.abs(np.diag(r_matrix))
    scale = float(np.max(diagonal)) if diagonal.size else 0.0
    threshold = atol + rtol * scale
    rank = int(np.count_nonzero(diagonal > threshold))
    condition = float(scale / np.min(diagonal[diagonal > threshold])) if rank else float("inf")
    return {"r": np.asarray(r_matrix, dtype=np.float64), "rank": rank,
            "condition": condition, "singular_values": diagonal}


def _to_coeff(transform: dict[str, object], y: np.ndarray) -> np.ndarray:
    if "r" in transform:
        return solve_triangular(transform["r"], y, lower=False, check_finite=False)
    return transform["to_coeff"] @ y


def _to_stable(transform: dict[str, object], c: np.ndarray) -> np.ndarray:
    if "r" in transform:
        return transform["r"] @ c
    return transform["to_stable"] @ c


def _grad_to_stable(transform: dict[str, object], gradient: np.ndarray) -> np.ndarray:
    if "r" in transform:
        return solve_triangular(transform["r"], gradient, trans="T", lower=False,
                                check_finite=False)
    return transform["to_coeff"].T @ gradient


def armijo_start(objective: object, y0: np.ndarray, direction: np.ndarray,
                 cfg: object) -> tuple[np.ndarray, int]:
    value, grad = objective(y0)
    slope = float(np.dot(grad, direction))
    if slope >= 0.0:
        direction = -direction
        slope = -slope
    step = 1.0
    for count in range(12):
        trial = y0 + step * direction
        trial_value, _ = objective(trial)
        if np.isfinite(trial_value) and trial_value <= value + 1e-4 * step * slope:
            return trial, count + 1
        step *= 0.5
    return y0, 12


def projected_residual(gradient: np.ndarray, initial_gradient: np.ndarray,
                       atol: float, rtol: float) -> dict[str, float | bool]:
    absolute = float(np.linalg.norm(gradient))
    initial = float(np.linalg.norm(initial_gradient))
    target = atol + rtol * initial
    return {"absolute": absolute, "relative": absolute / max(initial, np.finfo(float).tiny),
            "target": target, "passed": absolute <= target}


def _result(name: str, scipy_result: object, objective: object, y_initial: np.ndarray,
            transform: dict[str, object], initial_gradient: np.ndarray) -> dict[str, object]:
    value, grad = objective(scipy_result.x)
    return {"method": name, "optimizer_success": bool(scipy_result.success),
            "status": int(scipy_result.status), "message": str(scipy_result.message),
            "nit": int(getattr(scipy_result, "nit", 0)),
            "nfev": int(getattr(scipy_result, "nfev", 0)), "value": float(value),
            "grad_stable": np.asarray(grad), "stable_coordinates": np.asarray(scipy_result.x),
            "coefficients": _to_coeff(transform, scipy_result.x),
            "initial_gradient": initial_gradient}


def solve_linear(objective: object, hvp_y: object, transform: dict[str, object],
                 y0: np.ndarray, cfg: object, hessian_is_identity: bool = False
                 ) -> dict[str, object]:
    value0, grad0 = objective(y0)
    rank = y0.size
    hessian = None
    if hessian_is_identity:
        # The linear energy Hessian is precisely the weighted H1 Gram matrix.
        # In the incrementally orthonormal Q coordinates it is the identity.
        delta = -grad0
    else:
        hessian = np.column_stack([hvp_y(y0, np.eye(rank)[:, j]) for j in range(rank)])
        hessian = 0.5 * (hessian + hessian.T)
        try:
            delta = solve(hessian, -grad0, assume_a="pos", check_finite=False)
        except LinAlgError:
            delta, *_ = np.linalg.lstsq(hessian, -grad0, rcond=cfg.solver.rank_rtol)
    y = y0 + delta
    target = (cfg.solver.projection_atol +
              cfg.solver.projection_rtol * float(np.linalg.norm(grad0)))
    refinements = 0
    for refinements in range(1, 6):
        _, residual_gradient = objective(y)
        if np.linalg.norm(residual_gradient) <= target:
            refinements -= 1
            break
        if hessian_is_identity:
            correction = -residual_gradient
        else:
            try:
                correction = solve(hessian, -residual_gradient, assume_a="pos",
                                   check_finite=False)
            except LinAlgError:
                correction, *_ = np.linalg.lstsq(hessian, -residual_gradient,
                                                  rcond=cfg.solver.rank_rtol)
        y += correction
    value, grad = objective(y)
    return {"method": "exact-linear", "optimizer_success": True, "status": 0,
            "message": "symmetric linear solve with iterative refinement",
            "nit": 1 + refinements, "nfev": 2 + refinements,
            "value": float(value), "grad_stable": grad, "stable_coordinates": y,
            "coefficients": _to_coeff(transform, y), "initial_gradient": grad0}


def solve_lbfgs(objective: object, transform: dict[str, object], y0: np.ndarray,
                cfg: object) -> dict[str, object]:
    _, grad0 = objective(y0)
    residual_target = (cfg.solver.projection_atol +
                       cfg.solver.projection_rtol * float(np.linalg.norm(grad0)))
    # L-BFGS-B's gtol controls the infinity norm, whereas the acceptance gate
    # controls the Euclidean norm.  Scale it so an optimizer stop is capable of
    # satisfying the unchanged projected-residual gate in an m-dimensional span.
    infinity_target = residual_target / max(1.0, np.sqrt(float(y0.size)))
    result = minimize(lambda y: objective(y), y0, jac=True, method="L-BFGS-B",
                      options={"maxiter": cfg.solver.max_lbfgs_iterations,
                               "gtol": infinity_target,
                               "ftol": 0.0, "maxls": 40})
    return _result("L-BFGS-B", result, objective, y0, transform, grad0)


def solve_newton_cg(objective: object, hvp_y: object, transform: dict[str, object],
                    y0: np.ndarray, cfg: object) -> dict[str, object]:
    _, grad0 = objective(y0)
    result = minimize(lambda y: objective(y), y0, jac=True, hessp=hvp_y,
                      method="Newton-CG", options={"maxiter": cfg.solver.max_newton_iterations,
                                                   "xtol": min(cfg.solver.projection_atol * 0.1,
                                                               cfg.solver.projection_rtol)})
    target = (cfg.solver.projection_atol +
              cfg.solver.projection_rtol * float(np.linalg.norm(grad0)))
    value, gradient = objective(result.x)
    if np.linalg.norm(gradient) > target:
        y, value, gradient = np.asarray(result.x).copy(), float(value), np.asarray(gradient)
        extra_iterations = 0
        for extra_iterations in range(1, cfg.solver.max_newton_iterations + 1):
            if np.linalg.norm(gradient) <= target:
                break
            size = y.size
            if size <= 64:
                eye = np.eye(size)
                hessian = np.column_stack([hvp_y(y, eye[:, j]) for j in range(size)])
                hessian = 0.5 * (hessian + hessian.T)
                curvature_scale = max(1.0, float(np.linalg.norm(hessian, ord=np.inf)))
                direction = None
                for damping_index in range(8):
                    damping = (0.0 if damping_index == 0 else
                               1e-10 * curvature_scale * 10.0**(damping_index - 1))
                    try:
                        trial_direction = np.linalg.solve(
                            hessian + damping * np.eye(size), -gradient)
                    except np.linalg.LinAlgError:
                        continue
                    if (np.all(np.isfinite(trial_direction)) and
                            np.dot(gradient, trial_direction) < 0.0):
                        direction = trial_direction
                        break
            else:
                probe = hvp_y(y, gradient)
                curvature_scale = max(1.0, float(np.linalg.norm(probe)) /
                                      max(float(np.linalg.norm(gradient)),
                                          np.finfo(float).tiny))
                direction = None
                for damping_index in range(8):
                    damping = (0.0 if damping_index == 0 else
                               1e-10 * curvature_scale * 10.0**(damping_index - 1))
                    operator = LinearOperator((size, size),
                                              matvec=lambda z, d=damping:
                                              hvp_y(y, z) + d*z,
                                              dtype=np.float64)
                    trial_direction, _ = cg(
                        operator, -gradient, rtol=1e-10,
                        atol=min(target * 0.1, float(np.linalg.norm(gradient)) * 1e-8),
                        maxiter=min(4 * size, 600))
                    if (np.all(np.isfinite(trial_direction)) and
                            np.dot(gradient, trial_direction) < 0.0):
                        direction = trial_direction
                        break
            if direction is None:
                direction = -gradient
            slope = float(np.dot(gradient, direction))
            step = 1.0
            accepted_step = False
            for _ in range(24):
                trial_y = y + step * direction
                trial_value, trial_gradient = objective(trial_y)
                if (np.isfinite(trial_value) and
                        trial_value <= value + 1e-4 * step * slope):
                    y, value = trial_y, float(trial_value)
                    gradient = np.asarray(trial_gradient)
                    accepted_step = True
                    break
                step *= 0.5
            if not accepted_step:
                break
        result = OptimizeResult(
            x=y, success=bool(np.linalg.norm(gradient) <= target),
            status=0 if np.linalg.norm(gradient) <= target else 1,
            message="Newton-CG with strict HVP polish",
            nit=int(getattr(result, "nit", 0)) + extra_iterations,
            nfev=int(getattr(result, "nfev", 0)) + extra_iterations)
    return _result("Newton-CG", result, objective, y0, transform, grad0)


def solve_trust_krylov(objective: object, hvp_y: object, transform: dict[str, object],
                       y0: np.ndarray, cfg: object) -> dict[str, object]:
    _, grad0 = objective(y0)
    result = minimize(lambda y: objective(y), y0, jac=True, hessp=hvp_y,
                      method="trust-krylov", options={"maxiter": cfg.solver.max_newton_iterations,
                                                      "gtol": cfg.solver.projection_atol})
    return _result("trust-krylov", result, objective, y0, transform, grad0)


def project_full_span(problem: object, basis: dict[str, np.ndarray], c_old: np.ndarray,
                      base: tuple[np.ndarray, np.ndarray], f: np.ndarray,
                      train_rule: object, valid_rule: object, cfg: object,
                      r_matrix: np.ndarray | None = None,
                      q_matrix: np.ndarray | None = None) -> dict[str, object]:
    started = time.perf_counter()
    if r_matrix is None:
        feature = build_feature_matrix(basis["values_train"], basis["grads_train"], train_rule)
        transform = stable_transform(feature, cfg.solver.rank_atol, cfg.solver.rank_rtol)
    else:
        transform = qr_transform(r_matrix, cfg.solver.rank_atol, cfg.solver.rank_rtol)
    if transform["rank"] < basis["values_train"].shape[1]:
        return {"accepted": False, "reason": "rank_no_growth", "rank": transform["rank"],
                "condition": transform["condition"], "elapsed_projection_s": time.perf_counter()-started}
    y0 = _to_stable(transform, np.pad(c_old, (0, 1)))
    stable_values = stable_grads = None
    if q_matrix is not None:
        n_points = train_rule.points.shape[0]
        n_atoms = basis["values_train"].shape[1]
        root_weights = np.sqrt(train_rule.weights)
        stable_values = q_matrix[:n_points] / root_weights[:, None]
        weighted_grads = q_matrix[n_points:].reshape(n_points, problem.dim, n_atoms)
        stable_grads = (weighted_grads / root_weights[:, None, None]).transpose(0, 2, 1)

    def objective(y: np.ndarray) -> tuple[float, np.ndarray]:
        if stable_values is not None and stable_grads is not None:
            return coefficient_objective(problem, stable_values, stable_grads,
                                         y, base, f, train_rule)
        c = _to_coeff(transform, y)
        val, grad_c = coefficient_objective(problem, basis["values_train"],
                                             basis["grads_train"], c, base, f, train_rule)
        return val, _grad_to_stable(transform, grad_c)
    def hvp_y(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
        if stable_values is not None and stable_grads is not None:
            return coefficient_hvp(problem, stable_values, stable_grads, point,
                                   direction, base, train_rule)
        c = _to_coeff(transform, point)
        h = coefficient_hvp(problem, basis["values_train"], basis["grads_train"], c,
                            _to_coeff(transform, direction), base, train_rule)
        return _grad_to_stable(transform, h)
    initial_value, initial_grad = objective(y0)
    new_direction_c = np.zeros(basis["values_train"].shape[1])
    new_direction_c[-1] = 1.0
    new_direction_y = _to_stable(transform, new_direction_c)
    y_start, armijo_steps = armijo_start(objective, y0, new_direction_y, cfg)
    if problem.name == "linear":
        result = solve_linear(objective, hvp_y, transform, y_start, cfg,
                              hessian_is_identity=stable_values is not None)
        chain = ["exact-linear"]
    else:
        result = solve_lbfgs(objective, transform, y_start, cfg)
        residual = projected_residual(result["grad_stable"], initial_grad,
                                      cfg.solver.projection_atol, cfg.solver.projection_rtol)
        chain = ["L-BFGS-B"]
        if not residual["passed"]:
            result = solve_newton_cg(objective, hvp_y, transform,
                                     result["stable_coordinates"], cfg)
            residual = projected_residual(result["grad_stable"], initial_grad,
                                          cfg.solver.projection_atol, cfg.solver.projection_rtol)
            chain.append("Newton-CG")
        if not residual["passed"]:
            result = solve_trust_krylov(objective, hvp_y, transform,
                                        result["stable_coordinates"], cfg)
            chain.append("trust-krylov")
    residual = projected_residual(result["grad_stable"], initial_grad,
                                  cfg.solver.projection_atol, cfg.solver.projection_rtol)
    c = np.asarray(result["coefficients"])
    u_valid = basis["values_valid"] @ c
    grad_valid = np.einsum("qmd,m->qd", basis["grads_valid"], c, optimize=True)
    f_valid = basis["f_valid"]
    valid_energy = energy(problem, u_valid, grad_valid, f_valid, valid_rule)
    result.update({"method_chain": chain, "residual": residual,
                   "initial_train_energy": float(initial_value),
                   "final_train_energy": float(result["value"]),
                   "final_validation_energy": float(valid_energy),
                   "energy_decrease": float(initial_value - result["value"]),
                   "rank": transform["rank"], "condition": transform["condition"],
                   "armijo_steps": armijo_steps,
                   "elapsed_projection_s": time.perf_counter() - started})
    return result


def accept_projection(old_energy: float, projection: dict[str, object],
                      constraint_report: dict[str, object], cfg: object) -> tuple[bool, str]:
    if "coefficients" not in projection:
        return False, str(projection.get("reason", "projection_failed"))
    finite = (np.isfinite(projection["final_train_energy"]) and
              np.isfinite(projection["final_validation_energy"]) and
              np.all(np.isfinite(projection["coefficients"])))
    if not finite:
        return False, "nonfinite"
    if not bool(projection["residual"]["passed"]):
        return False, "projected_residual"
    floor = 64.0 * np.finfo(float).eps * max(1.0, abs(old_energy))
    if projection["final_train_energy"] > old_energy - floor:
        return False, "energy_not_decreased"
    if not bool(constraint_report.get("passed", True)):
        return False, "constraint"
    return True, "accepted"
