"""ReLU ridge atoms and deterministic fixed candidate/reference pools."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from scipy.stats import qmc


@dataclass
class Pool:
    w: np.ndarray
    b: np.ndarray
    k: int
    seed: int
    sampler: str
    kind: str
    hash: str
    available: np.ndarray
    consumed: np.ndarray
    scales: np.ndarray
    centers: np.ndarray
    original_indices: np.ndarray

    @property
    def size(self) -> int:
        return int(self.b.size)


def relu_power(z: np.ndarray, k: int) -> np.ndarray:
    return np.maximum(z, 0.0) ** int(k)


def evaluate_atoms(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                   k: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = x @ w.T + b[None, :]
    values = relu_power(z, k)
    if k == 1:
        dz = (z > 0.0).astype(np.float64)
    else:
        dz = k * relu_power(z, k - 1)
    grads = dz[:, :, None] * w[None, :, :]
    return values, grads


def _sobol_points(n: int, dim: int, seed: int) -> np.ndarray:
    power = int(np.ceil(np.log2(max(1, n))))
    return qmc.Sobol(dim, scramble=True, seed=seed).random_base2(power)[:n]


def sample_pool_parameters(dim: int, size: int, sampler: str, seed: int,
                           offset_cfg: object) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_cross = int(round(size * float(offset_cfg.crossing_fraction)))
    n_cross = min(size, max(1, n_cross))
    n_active = size - n_cross
    if dim == 1:
        strata = (np.arange(n_cross) + rng.random(n_cross)) / n_cross
        w_cross = np.where(np.arange(n_cross) % 2 == 0, 1.0, -1.0)[:, None]
        b_cross = -w_cross[:, 0] * strata
        if n_active:
            w_active = np.where(np.arange(n_active) % 2 == 0, 1.0, -1.0)[:, None]
            pad = float(offset_cfg.anchor_padding)
            t = np.where(w_active[:, 0] > 0.0, -pad, 1.0 + pad)
            b_active = -w_active[:, 0] * t
            w = np.vstack([w_cross, w_active])
            b = np.concatenate([b_cross, b_active])
        else:
            w, b = w_cross, b_cross
        order = rng.permutation(size)
        return w[order], b[order]
    n_sobol = int(round(size * float(offset_cfg.sobol_fraction)))
    directions = np.empty((size, dim), dtype=np.float64)
    if n_sobol:
        u = _sobol_points(n_sobol, dim, seed)
        directions[:n_sobol] = 2.0 * u - 1.0
    if n_sobol < size:
        directions[n_sobol:] = rng.normal(size=(size - n_sobol, dim))
    norms = np.linalg.norm(directions, axis=1)
    tiny = norms < 1e-14
    directions[tiny, 0] = 1.0
    norms[tiny] = 1.0
    directions /= norms[:, None]
    w = directions
    b = np.empty(size, dtype=np.float64)
    for i in range(size):
        lo = float(np.sum(np.minimum(w[i], 0.0)))
        hi = float(np.sum(np.maximum(w[i], 0.0)))
        if i < n_cross:
            threshold = rng.uniform(lo, hi)
            b[i] = -threshold
        else:
            b[i] = -lo + float(offset_cfg.anchor_padding)
    order = rng.permutation(size)
    return w[order], b[order]


def canonicalize_pool(w: np.ndarray, b: np.ndarray, tol: float = 1e-13,
                      min_spacing: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep, keys = [], set()
    for i, (wi, bi) in enumerate(zip(w, b)):
        scale = max(tol, min_spacing if w.shape[1] == 1 else tol)
        key = tuple(np.rint(np.concatenate([wi, [bi]]) / scale).astype(np.int64))
        if key not in keys:
            keys.add(key)
            keep.append(i)
    idx = np.asarray(keep, dtype=np.int64)
    return np.ascontiguousarray(w[idx]), np.ascontiguousarray(b[idx]), idx


def compute_atom_scales(problem: object, values: np.ndarray, grads: np.ndarray,
                        rule: object) -> np.ndarray:
    if problem.natural_metric == "H1":
        density = values * values + np.sum(grads * grads, axis=2)
        return np.sqrt(np.einsum("qb,q->b", density, rule.weights, optimize=True))
    density = np.sum(np.abs(grads) ** problem.p, axis=2)
    if problem.name == "reaction_p":
        density += np.abs(values) ** problem.p
    return np.einsum("qb,q->b", density, rule.weights, optimize=True) ** (1.0 / problem.p)


def pool_hash(pool: Pool) -> str:
    payload = (pool.w.tobytes() + pool.b.tobytes() +
               f"{pool.k}|{pool.seed}|{pool.sampler}|{pool.kind}|pool-v1".encode())
    return sha256(payload).hexdigest()


def build_pool(problem: object, cfg: object, rule_for_scale: object | None,
               kind: str) -> Pool:
    pool_cfg = cfg.pool
    size = pool_cfg.candidate_size if kind == "candidate" else pool_cfg.reference_size
    seed = int(cfg.seed + (1009 if kind == "reference" else 0))
    w, b = sample_pool_parameters(problem.dim, size, pool_cfg.sampler, seed, pool_cfg)
    w, b, original = canonicalize_pool(w, b)
    pool = Pool(w, b, int(cfg.relu_power), seed, pool_cfg.sampler, kind,
                "", np.ones(b.size, dtype=bool), np.zeros(b.size, dtype=bool),
                np.full(b.size, np.nan), np.full(b.size, np.nan), original)
    pool.hash = pool_hash(pool)
    if rule_for_scale is not None:
        calibrate_pool(problem, pool, rule_for_scale)
    return pool


def calibrate_pool(problem: object, pool: Pool, rule: object,
                   batch_size: int = 256, *, scale_atol: float = 1e-14,
                   scale_rtol: float = 1e-12) -> None:
    for start in range(0, pool.size, batch_size):
        stop = min(pool.size, start + batch_size)
        values, grads = evaluate_atoms(rule.points, pool.w[start:stop], pool.b[start:stop], pool.k)
        if problem.requires_zero_mean:
            centers = np.einsum("qb,q->b", values, rule.weights, optimize=True) / np.sum(rule.weights)
            values = values - centers[None, :]
            pool.centers[start:stop] = centers
        else:
            pool.centers[start:stop] = 0.0
        pool.scales[start:stop] = compute_atom_scales(problem, values, grads, rule)
    threshold = scale_atol + scale_rtol * max(1.0, float(np.nanmax(pool.scales)))
    pool.available &= np.isfinite(pool.scales) & (pool.scales > threshold)


def breakpoints_1d(pool: Pool) -> np.ndarray:
    if pool.w.shape[1] != 1:
        return np.zeros(0, dtype=np.float64)
    t = -pool.b / pool.w[:, 0]
    return np.sort(t[(t > 0.0) & (t < 1.0)])


def save_pool(path: str | Path, pool: Pool) -> None:
    np.savez_compressed(path, w=pool.w, b=pool.b, k=pool.k, seed=pool.seed,
                        sampler=pool.sampler, kind=pool.kind, hash=pool.hash,
                        scales=pool.scales, centers=pool.centers,
                        available=pool.available,
                        original_indices=pool.original_indices)


def load_pool(path: str | Path) -> Pool:
    with np.load(path, allow_pickle=False) as data:
        available = data["available"] if "available" in data.files else np.ones(data["b"].size, bool)
        pool = Pool(data["w"], data["b"], int(data["k"]), int(data["seed"]),
                    str(data["sampler"]), str(data["kind"]), str(data["hash"]),
                    available, np.zeros(data["b"].size, bool),
                    data["scales"], data["centers"], data["original_indices"])
    if pool_hash(pool) != pool.hash:
        raise ValueError("pool hash mismatch")
    return pool


def pool_manifest(pool: Pool) -> dict[str, object]:
    return {"kind": pool.kind, "size": pool.size, "seed": pool.seed,
            "sampler": pool.sampler, "relu_power": pool.k, "hash": pool.hash,
            "available": int(np.count_nonzero(pool.available)),
            "scale_min": float(np.nanmin(pool.scales)),
            "scale_max": float(np.nanmax(pool.scales))}
