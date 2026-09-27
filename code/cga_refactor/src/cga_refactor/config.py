"""Small immutable configuration objects for one CGA run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import math
import tomllib


MODELS = ("linear", "cubic", "sinh", "pure_p", "regularized_p", "reaction_p")
FREEZE_PROFILE = "neumann-cga-v1"
SCHEMA_VERSION = "cga-refactor-1"
EXACT_PROFILES = ("low_frequency", "multifrequency")
QUADRATURE_LEVELS = ("Q0", "Q1", "Q2")


@dataclass(frozen=True)
class PoolConfig:
    candidate_size: int = 256
    reference_size: int = 512
    candidate_batch_size: int = 128
    reference_batch_size: int = 128
    sampler: str = "hybrid_qmc_iid"
    sobol_fraction: float = 0.75
    crossing_fraction: float = 0.90
    anchor_padding: float = 0.05


@dataclass(frozen=True)
class QuadratureConfig:
    train_cells_2d: int = 12
    train_order: int = 3
    validation_sobol_power: int = 12
    audit_sobol_power: int = 13
    train_order_1d: int = 6
    validation_order_1d: int = 10
    audit_order_1d: int = 14
    min_segment_width: float = 1e-10


@dataclass(frozen=True)
class SolverConfig:
    max_attempt_multiplier: int = 3
    scale_atol: float = 1e-14
    scale_rtol: float = 1e-12
    innovation_atol: float = 1e-12
    innovation_rtol: float = 1e-10
    rank_atol: float = 1e-12
    rank_rtol: float = 1e-10
    score_atol: float = 1e-13
    score_rtol: float = 1e-10
    oracle_ratio_floor: float = 0.5
    oracle_patience: int = 8
    projection_atol: float = 1e-9
    projection_rtol: float = 1e-7
    max_lbfgs_iterations: int = 120
    max_newton_iterations: int = 50
    validation_gap_atol: float = 1e-10
    validation_gap_rtol: float = 2e-2
    validation_increase_atol: float = 1e-10
    energy_drop_scale_floor: float = 1e-12
    quadrature_warning_patience: int = 3
    audit_energy_gap_atol: float = 1e-10
    audit_metric_atol: float = 1e-12
    audit_rtol: float = 2e-2
    checkpoint_every_attempts: int = 1


@dataclass(frozen=True)
class RunConfig:
    model: str
    dim: int
    seed: int = 201
    p: float = 4.0
    epsilon: float | None = None
    relu_power: int = 3
    exact_profile: str = "low_frequency"
    quadrature_level: str = "Q0"
    target_accepted: int = 32
    phase: str = "report"
    dtype: str = "float64"
    output_root: str = "cga_refactor/results"
    pool: PoolConfig = field(default_factory=PoolConfig)
    quadrature: QuadratureConfig = field(default_factory=QuadratureConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    freeze_profile: str = FREEZE_PROFILE
    schema_version: str = SCHEMA_VERSION

    @property
    def max_attempts(self) -> int:
        return self.target_accepted * self.solver.max_attempt_multiplier


def validate_config(cfg: RunConfig) -> None:
    if cfg.model not in MODELS:
        raise ValueError(f"unknown model {cfg.model!r}")
    if cfg.dim not in (1, 2):
        raise ValueError("dim must be 1 or 2")
    if cfg.dtype != "float64":
        raise ValueError("formal and report runs require float64")
    if cfg.p < 2.0:
        raise ValueError("this implementation supports p >= 2")
    if cfg.model == "regularized_p" and (cfg.epsilon is None or cfg.epsilon <= 0.0):
        raise ValueError("regularized_p requires epsilon > 0")
    if cfg.model != "regularized_p" and cfg.epsilon is not None:
        raise ValueError("epsilon is only valid for regularized_p")
    if cfg.relu_power < 1:
        raise ValueError("relu_power must be positive")
    if cfg.exact_profile not in EXACT_PROFILES:
        raise ValueError(f"exact_profile must be one of {EXACT_PROFILES}")
    if cfg.quadrature_level not in QUADRATURE_LEVELS:
        raise ValueError(f"quadrature_level must be one of {QUADRATURE_LEVELS}")
    if cfg.dim == 2 and cfg.exact_profile != "low_frequency":
        raise ValueError("the formal 2D protocol uses the low-frequency manufactured solution")
    if cfg.target_accepted < 1 or cfg.pool.candidate_size < cfg.target_accepted:
        raise ValueError("candidate pool must be at least as large as the target")
    if cfg.pool.reference_size < 1:
        raise ValueError("reference pool must be nonempty")
    if not (0.0 <= cfg.pool.sobol_fraction <= 1.0):
        raise ValueError("sobol_fraction must lie in [0, 1]")
    warning_values = (
        cfg.solver.validation_gap_atol,
        cfg.solver.validation_gap_rtol,
        cfg.solver.validation_increase_atol,
        cfg.solver.energy_drop_scale_floor,
        cfg.solver.audit_energy_gap_atol,
        cfg.solver.audit_metric_atol,
        cfg.solver.audit_rtol,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in warning_values):
        raise ValueError("energy-warning and audit tolerances must be finite and nonnegative")
    if cfg.solver.validation_gap_rtol >= 1.0 or cfg.solver.audit_rtol >= 1.0:
        raise ValueError("relative energy-warning and audit tolerances must be below one")
    if cfg.solver.quadrature_warning_patience < 1:
        raise ValueError("quadrature_warning_patience must be positive")
    if cfg.quadrature.train_cells_2d < 1 or cfg.quadrature.train_order < 1:
        raise ValueError("2D tensor quadrature cells and order must be positive")


def config_to_dict(cfg: RunConfig) -> dict[str, Any]:
    return asdict(cfg)


def config_hash(cfg: RunConfig) -> str:
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    pool = PoolConfig(**raw.pop("pool", {}))
    quadrature = QuadratureConfig(**raw.pop("quadrature", {}))
    solver = SolverConfig(**raw.pop("solver", {}))
    cfg = RunConfig(pool=pool, quadrature=quadrature, solver=solver, **raw)
    validate_config(cfg)
    return cfg


def config_for(model: str, dim: int, *, profile: str = "report", seed: int = 201,
               output_root: str = "cga_refactor/results",
               quadrature_level: str = "Q0") -> RunConfig:
    """Construct either the exact frozen budget or a documented report budget."""
    if profile not in {"report", "formal", "smoke"}:
        raise ValueError("profile must be report, formal, or smoke")
    p_family = model in {"pure_p", "regularized_p", "reaction_p"}
    epsilon = 0.1 if model == "regularized_p" else None
    relu_power = 1 if model == "pure_p" and dim == 1 else 3
    if profile == "formal":
        pool = PoolConfig(
            candidate_size=2048 if dim == 1 else 4096,
            reference_size=4096 if dim == 1 else 8192,
            candidate_batch_size=256 if dim == 1 else 128,
            reference_batch_size=256 if dim == 1 else 128,
            sampler="stratified_breakpoint" if dim == 1 else "hybrid_qmc_iid",
        )
        levels = {
            "Q0": (40, 16, 18),
            "Q1": (48, 17, 18),
            "Q2": (64, 17, 19),
        }
        if quadrature_level not in levels:
            raise ValueError("quadrature_level must be Q0, Q1, or Q2")
        cells, validation_power, audit_power = levels[quadrature_level]
        quad = QuadratureConfig(train_cells_2d=cells,
                                validation_sobol_power=validation_power,
                                audit_sobol_power=audit_power)
        target = 256 if dim == 1 else 512
    elif profile == "smoke":
        pool = PoolConfig(candidate_size=24, reference_size=32, candidate_batch_size=16,
                          reference_batch_size=16,
                          sampler="stratified_breakpoint" if dim == 1 else "hybrid_qmc_iid")
        quad = QuadratureConfig(train_cells_2d=4, validation_sobol_power=8,
                                audit_sobol_power=9, train_order_1d=4,
                                validation_order_1d=6, audit_order_1d=8)
        target = 2
    else:
        pool = PoolConfig(candidate_size=256 if dim == 1 else 512,
                          reference_size=512 if dim == 1 else 1024,
                          candidate_batch_size=128, reference_batch_size=128,
                          sampler="stratified_breakpoint" if dim == 1 else "hybrid_qmc_iid")
        quad = QuadratureConfig(train_cells_2d=12 if not p_family else 16,
                                validation_sobol_power=12 if not p_family else 13,
                                audit_sobol_power=13 if not p_family else 14)
        target = 32 if dim == 1 else 64
    solver = SolverConfig(
        scale_rtol=1e-6 if dim == 2 and relu_power >= 3 else 1e-12,
        projection_atol=1e-11 if model == "linear" else 1e-9,
        projection_rtol=1e-10 if model == "linear" else (1e-6 if p_family else 1e-7),
    )
    exact_profile = ("multifrequency"
                     if dim == 1 and not p_family and profile == "formal"
                     else "low_frequency")
    cfg = RunConfig(model=model, dim=dim, seed=seed, epsilon=epsilon,
                    relu_power=relu_power, target_accepted=target, phase=profile,
                    exact_profile=exact_profile, quadrature_level=quadrature_level,
                    output_root=output_root, pool=pool, quadrature=quad, solver=solver)
    validate_config(cfg)
    return cfg
