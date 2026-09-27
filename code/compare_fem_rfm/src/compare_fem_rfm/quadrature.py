"""Deterministic and QMC quadrature rules used for training and evaluation."""

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


def _make(points: np.ndarray, weights: np.ndarray, name: str) -> QuadratureRule:
    points = np.ascontiguousarray(points, dtype=float)
    weights = np.ascontiguousarray(weights, dtype=float)
    digest = sha256(points.tobytes() + weights.tobytes() + name.encode()).hexdigest()
    if np.any(weights <= 0.0) or not np.isclose(weights.sum(), 1.0, atol=2e-13):
        raise ValueError(f"invalid quadrature rule {name}")
    return QuadratureRule(points, weights, name, digest)


def composite_gauss(dim: int, cells: int, order: int) -> QuadratureRule:
    nodes, base_weights = np.polynomial.legendre.leggauss(order)
    one_x, one_w = [], []
    for cell in range(cells):
        left, right = cell / cells, (cell + 1) / cells
        one_x.append(0.5 * (right - left) * nodes + 0.5 * (left + right))
        one_w.append(0.5 * (right - left) * base_weights)
    x, w = np.concatenate(one_x), np.concatenate(one_w)
    if dim == 1:
        return _make(x[:, None], w, f"CG{cells}x{order}-d1")
    xx, yy = np.meshgrid(x, x, indexing="ij")
    ww = np.multiply.outer(w, w)
    return _make(np.column_stack([xx.ravel(), yy.ravel()]), ww.ravel(),
                 f"CG{cells}x{order}-d2")


def segmented_gauss_1d(breakpoints: np.ndarray, order: int) -> QuadratureRule:
    """Gauss rule split at all ReLU ridge breakpoints."""
    bp = np.asarray(breakpoints, dtype=float)
    bp = np.unique(bp[(bp > 0.0) & (bp < 1.0)])
    edges = np.concatenate([[0.0], bp, [1.0]])
    nodes, base_weights = np.polynomial.legendre.leggauss(order)
    points, weights = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        if right - left <= 1e-12:
            continue
        points.append(0.5 * (right - left) * nodes + 0.5 * (left + right))
        weights.append(0.5 * (right - left) * base_weights)
    return _make(np.concatenate(points)[:, None], np.concatenate(weights),
                 f"SG{order}-{len(points)}segments")


def sobol_rule(dim: int, power: int, seed: int) -> QuadratureRule:
    points = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(power)
    weights = np.full(points.shape[0], 1.0 / points.shape[0])
    return _make(points, weights, f"SQ{power}-seed{seed}-d{dim}")


def make_rule(spec: object, cfg: dict, purpose: str) -> QuadratureRule:
    qcfg = cfg["quadrature"]
    if purpose == "train":
        if spec.dim == 1:
            return composite_gauss(1, qcfg["train_1d_cells"], qcfg["train_1d_order"])
        return composite_gauss(2, qcfg["train_2d_cells"], qcfg["train_2d_order"])
    if purpose == "evaluation":
        return composite_gauss(spec.dim,
                               qcfg[f"evaluation_{spec.dim}d_cells"],
                               qcfg[f"evaluation_{spec.dim}d_order"])
    if purpose == "audit":
        return composite_gauss(spec.dim,
                               qcfg[f"audit_{spec.dim}d_cells"],
                               qcfg[f"audit_{spec.dim}d_order"])
    raise KeyError(purpose)
