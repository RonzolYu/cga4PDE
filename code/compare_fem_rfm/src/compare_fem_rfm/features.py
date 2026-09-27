"""Nested random ReLU-cubed ridge features and frozen CGA model loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc


@dataclass
class FeatureSet:
    w: np.ndarray
    b: np.ndarray
    k: int
    scales: np.ndarray
    centers: np.ndarray

    @property
    def size(self) -> int:
        return int(self.b.size)


def _sobol_points(n: int, dim: int, seed: int) -> np.ndarray:
    power = int(np.ceil(np.log2(max(1, n))))
    return qmc.Sobol(dim, scramble=True, seed=seed).random_base2(power)[:n]


def sample_parameters(dim: int, size: int, seed: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Match the pool-small direction/offset distribution without selection."""
    rng = np.random.default_rng(seed)
    n_cross = min(size, max(1, int(round(size * cfg["crossing_fraction"]))))
    n_active = size - n_cross
    if dim == 1:
        strata = (np.arange(n_cross) + rng.random(n_cross)) / n_cross
        w_cross = np.where(np.arange(n_cross) % 2 == 0, 1.0, -1.0)[:, None]
        b_cross = -w_cross[:, 0] * strata
        if n_active:
            w_active = np.where(np.arange(n_active) % 2 == 0, 1.0, -1.0)[:, None]
            t = np.where(w_active[:, 0] > 0.0, -cfg["anchor_padding"],
                         1.0 + cfg["anchor_padding"])
            w = np.vstack([w_cross, w_active])
            b = np.concatenate([b_cross, -w_active[:, 0] * t])
        else:
            w, b = w_cross, b_cross
        order = rng.permutation(size)
        return w[order], b[order]
    n_sobol = int(round(size * cfg["sobol_fraction"]))
    directions = np.empty((size, dim))
    if n_sobol:
        directions[:n_sobol] = 2.0 * _sobol_points(n_sobol, dim, seed) - 1.0
    if n_sobol < size:
        directions[n_sobol:] = rng.normal(size=(size - n_sobol, dim))
    norms = np.linalg.norm(directions, axis=1)
    directions[norms < 1e-14, 0] = 1.0
    norms[norms < 1e-14] = 1.0
    w = directions / norms[:, None]
    b = np.empty(size)
    for i in range(size):
        lo = float(np.minimum(w[i], 0.0).sum())
        hi = float(np.maximum(w[i], 0.0).sum())
        if i < n_cross:
            b[i] = -rng.uniform(lo, hi)
        else:
            b[i] = -lo + cfg["anchor_padding"]
    order = rng.permutation(size)
    return w[order], b[order]


def evaluate_raw(points: np.ndarray, w: np.ndarray, b: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    z = points @ w.T + b[None, :]
    positive = np.maximum(z, 0.0)
    values = positive**k
    derivative = k * positive ** (k - 1) if k > 1 else (z > 0.0).astype(float)
    grads = derivative[:, :, None] * w[None, :, :]
    return values, grads


def calibrate_features(spec: object, w: np.ndarray, b: np.ndarray, k: int,
                       points: np.ndarray, weights: np.ndarray) -> FeatureSet:
    values, grads = evaluate_raw(points, w, b, k)
    centers = np.einsum("qn,q->n", values, weights, optimize=True) if spec.zero_mean else np.zeros(b.size)
    values = values - centers[None, :]
    if spec.natural_metric == "H1":
        density = values**2 + np.sum(grads**2, axis=2)
        scales = np.sqrt(np.einsum("qn,q->n", density, weights, optimize=True))
    else:
        density = np.sum(np.abs(grads) ** float(spec.p), axis=2)
        scales = np.einsum("qn,q->n", density, weights, optimize=True) ** (1.0 / float(spec.p))
    valid = np.isfinite(scales) & (scales > 1e-13)
    if not np.any(valid):
        raise ValueError("all sampled features are degenerate")
    return FeatureSet(w[valid], b[valid], k, scales[valid], centers[valid])


def evaluate_features(features: FeatureSet, points: np.ndarray, width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = features.size if width is None else int(width)
    values, grads = evaluate_raw(points, features.w[:n], features.b[:n], features.k)
    values = (values - features.centers[None, :n]) / features.scales[None, :n]
    grads = grads / features.scales[None, :n, None]
    return values, grads


def load_cga_features(path: str | Path) -> tuple[FeatureSet, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        features = FeatureSet(data["w"], data["b"], int(data["k"]),
                              data["scales"], data["centers"])
        coefficients = np.asarray(data["coefficients"], dtype=float)
    return features, coefficients
