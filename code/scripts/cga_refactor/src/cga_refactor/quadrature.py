"""Independent training, validation, audit, and boundary quadrature rules."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class QuadratureRule:
    points: np.ndarray
    weights: np.ndarray
    name: str
    hash: str
    normals: np.ndarray | None = None


def _rule(points: np.ndarray, weights: np.ndarray, name: str,
          normals: np.ndarray | None = None) -> QuadratureRule:
    points = np.ascontiguousarray(points, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    digest = sha256(points.tobytes() + weights.tobytes() + name.encode()).hexdigest()
    return QuadratureRule(points, weights, name, digest, normals)


def segmented_gauss_1d(breakpoints: np.ndarray, order: int,
                       min_width: float = 1e-10) -> QuadratureRule:
    bp = np.asarray(breakpoints, dtype=np.float64)
    bp = np.unique(np.clip(bp[(bp > 0.0) & (bp < 1.0)], 0.0, 1.0))
    edges = [0.0]
    for value in bp:
        if value - edges[-1] >= min_width:
            edges.append(float(value))
    if 1.0 - edges[-1] < min_width and len(edges) > 1:
        edges[-1] = 1.0
    else:
        edges.append(1.0)
    nodes, base_weights = np.polynomial.legendre.leggauss(order)
    pts, ws = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        pts.append((0.5 * (right - left) * nodes + 0.5 * (left + right))[:, None])
        ws.append(0.5 * (right - left) * base_weights)
    return _rule(np.vstack(pts), np.concatenate(ws), f"SG{order}-{len(edges)-1}")


def tensor_gauss_2d(n_cells: int, order: int) -> QuadratureRule:
    if not isinstance(n_cells, (int, np.integer)) or int(n_cells) < 1:
        raise ValueError("n_cells must be a positive integer")
    if not isinstance(order, (int, np.integer)) or int(order) < 1:
        raise ValueError("order must be a positive integer")
    n_cells, order = int(n_cells), int(order)
    nodes, base_weights = np.polynomial.legendre.leggauss(order)
    one_x, one_w = [], []
    for cell in range(n_cells):
        left, right = cell / n_cells, (cell + 1) / n_cells
        one_x.append(0.5 * (right - left) * nodes + 0.5 * (left + right))
        one_w.append(0.5 * (right - left) * base_weights)
    x = np.concatenate(one_x)
    w = np.concatenate(one_w)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    ww = np.multiply.outer(w, w)
    rule = _rule(np.column_stack([xx.ravel(), yy.ravel()]), ww.ravel(),
                 f"TG{n_cells}-{order}")
    if (not np.all(np.isfinite(rule.weights)) or np.any(rule.weights <= 0.0)
            or not np.isclose(np.sum(rule.weights), 1.0, rtol=0.0, atol=1e-13)):
        raise ValueError("invalid tensor Gauss weights")
    return rule


def sobol_rule(dim: int, n_power: int, seed: int) -> QuadratureRule:
    points = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(n_power)
    weights = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    return _rule(points, weights, f"SQ{n_power}-seed{seed}")


def boundary_rule(dim: int, n_cells: int, order: int) -> QuadratureRule:
    if dim == 1:
        return _rule(np.array([[0.0], [1.0]]), np.ones(2), "boundary-1d",
                     np.array([[-1.0], [1.0]]))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xs, ws = [], []
    for cell in range(n_cells):
        left, right = cell / n_cells, (cell + 1) / n_cells
        xs.append(0.5 * (right - left) * nodes + 0.5 * (left + right))
        ws.append(0.5 * (right - left) * weights)
    t, w = np.concatenate(xs), np.concatenate(ws)
    points = np.vstack([np.column_stack([np.zeros_like(t), t]),
                        np.column_stack([np.ones_like(t), t]),
                        np.column_stack([t, np.zeros_like(t)]),
                        np.column_stack([t, np.ones_like(t)])])
    normals = np.vstack([np.tile([-1.0, 0.0], (t.size, 1)),
                         np.tile([1.0, 0.0], (t.size, 1)),
                         np.tile([0.0, -1.0], (t.size, 1)),
                         np.tile([0.0, 1.0], (t.size, 1))])
    return _rule(points, np.tile(w, 4), f"boundary-{n_cells}-{order}", normals)


def make_quadratures(cfg: object, candidate_pool: object,
                     reference_pool: object) -> dict[str, QuadratureRule]:
    qcfg = cfg.quadrature
    if cfg.dim == 1:
        from .dictionary import breakpoints_1d
        bp = np.concatenate([breakpoints_1d(candidate_pool), breakpoints_1d(reference_pool)])
        # For non-even p, the manufactured p-Laplacian source is only piecewise
        # smooth where grad(cos(2*pi*x)) vanishes.  Make the known interior
        # critical point an integration edge instead of asking one Gauss panel
        # to cross it.
        if cfg.model in {"pure_p", "regularized_p", "reaction_p"}:
            bp = np.append(bp, 0.5)
        train = segmented_gauss_1d(bp, qcfg.train_order_1d, qcfg.min_segment_width)
        valid = segmented_gauss_1d(bp, qcfg.validation_order_1d, qcfg.min_segment_width)
        audit = segmented_gauss_1d(bp, qcfg.audit_order_1d, qcfg.min_segment_width)
    else:
        train = tensor_gauss_2d(qcfg.train_cells_2d, qcfg.train_order)
        valid = sobol_rule(2, qcfg.validation_sobol_power, cfg.seed + 101)
        audit = sobol_rule(2, qcfg.audit_sobol_power, cfg.seed + 211)
    return {"train": train, "valid": valid, "audit": audit,
            "boundary": boundary_rule(cfg.dim, qcfg.train_cells_2d, qcfg.train_order)}


def quadrature_audit(evaluator: object, train_rule: QuadratureRule,
                     audit_rule: QuadratureRule) -> dict[str, float | bool]:
    train = float(evaluator(train_rule))
    audit = float(evaluator(audit_rule))
    gap = abs(train - audit)
    rel = gap / max(1.0, abs(audit))
    return {"train": train, "audit": audit, "absolute_gap": gap,
            "relative_gap": rel, "warning": rel > 2e-2}
