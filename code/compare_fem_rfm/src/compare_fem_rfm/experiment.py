"""Experiment orchestration and append-only raw result storage."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

from .feature_solver import solve_feature_coefficients
from .features import (
    FeatureSet,
    calibrate_features,
    evaluate_features,
    load_cga_features,
    sample_parameters,
)
from .fem import FEMModel, evaluate_fem_model, solve_fem
from .metrics import evaluate_fields, max_relative_metric_delta, relative_metric_deltas
from .problems import CASES, ProblemSpec, forcing
from .quadrature import QuadratureRule, make_rule, segmented_gauss_1d


RESULT_FIELDS = [
    "case_id", "numerical_id", "method", "variant", "seed", "dof", "mesh_level",
    "accepted_atoms", "feature_count", "energy_gap", "energy_gap_signed",
    "natural_error", "v_error", "l2_error", "numerical_energy", "exact_energy",
    "solver_iterations", "solver_residual", "solver_success", "solver_message",
    "wall_time_sec", "evaluation_rule", "evaluation_hash", "evaluation_audit_rel_delta",
    "audit_energy_gap_rel_delta", "audit_natural_rel_delta", "audit_v_rel_delta",
    "plateau_flag", "source", "source_path", "problem_hash", "config_hash",
    "archived_energy_gap", "archived_natural_error", "archived_v_error", "model_path",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_path(path: Path) -> str:
    """Return a portable provenance path relative to the comparison package."""
    root = project_root()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_config() -> tuple[dict, str]:
    path = project_root() / "configs" / "experiment.json"
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


class ResultStore:
    def __init__(self, path: Path, force: bool = False):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if force and path.exists():
            path.unlink()
        self.keys: set[tuple[str, str, str, str, int]] = set()
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self.keys.add((row["case_id"], row["method"], row["variant"],
                                   row["seed"], int(row["dof"])))

    def contains(self, case_id: str, method: str, variant: str,
                 seed: int | None, dof: int) -> bool:
        return (case_id, method, variant, "" if seed is None else str(seed), dof) in self.keys

    def append(self, row: dict) -> None:
        serial = {key: row.get(key) for key in RESULT_FIELDS}
        serial["seed"] = "" if serial["seed"] is None else serial["seed"]
        serial["v_error"] = "" if serial["v_error"] is None else serial["v_error"]
        serial["evaluation_audit_rel_delta"] = "" if serial["evaluation_audit_rel_delta"] is None else serial["evaluation_audit_rel_delta"]
        serial["archived_v_error"] = "" if serial["archived_v_error"] is None else serial["archived_v_error"]
        write_header = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(serial)
        self.keys.add((str(serial["case_id"]), str(serial["method"]), str(serial["variant"]),
                       str(serial["seed"]), int(serial["dof"])))


def _metric_record(spec: ProblemSpec, method: str, variant: str, dof: int,
                   metrics: dict, rule: QuadratureRule, config_hash: str, **kwargs) -> dict:
    row = {
        "case_id": spec.case_id,
        "numerical_id": spec.numerical_id,
        "method": method,
        "variant": variant,
        "seed": kwargs.pop("seed", None),
        "dof": int(dof),
        "mesh_level": kwargs.pop("mesh_level", ""),
        "accepted_atoms": kwargs.pop("accepted_atoms", ""),
        "feature_count": kwargs.pop("feature_count", ""),
        **metrics,
        "evaluation_rule": rule.name,
        "evaluation_hash": rule.hash,
        "evaluation_audit_rel_delta": kwargs.pop("evaluation_audit_rel_delta", None),
        "audit_energy_gap_rel_delta": kwargs.pop("audit_energy_gap_rel_delta", None),
        "audit_natural_rel_delta": kwargs.pop("audit_natural_rel_delta", None),
        "audit_v_rel_delta": kwargs.pop("audit_v_rel_delta", None),
        "plateau_flag": False,
        "problem_hash": spec.problem_hash,
        "config_hash": config_hash,
    }
    row.update(kwargs)
    return row


def _evaluate_feature_solution(spec: ProblemSpec, features: FeatureSet,
                               coefficients: np.ndarray, rule: QuadratureRule,
                               batch_size: int = 12000) -> dict:
    u_parts, grad_parts = [], []
    for start in range(0, rule.points.shape[0], batch_size):
        points = rule.points[start:start + batch_size]
        phi, grad_phi = evaluate_features(features, points, coefficients.size)
        u_parts.append(phi @ coefficients)
        grad_parts.append(np.einsum("qnd,n->qd", grad_phi, coefficients, optimize=True))
    return evaluate_fields(spec, rule.points, rule.weights,
                           np.concatenate(u_parts), np.vstack(grad_parts))


def _feature_train_rule(spec: ProblemSpec, cfg: dict, w: np.ndarray, b: np.ndarray) -> QuadratureRule:
    if spec.dim != 1:
        return make_rule(spec, cfg, "train")
    breakpoints = -b / w[:, 0]
    if spec.model == "pure_p":
        breakpoints = np.append(breakpoints, 0.5)
    return segmented_gauss_1d(breakpoints, cfg["quadrature"]["train_1d_order"])


def _source_dir(spec: ProblemSpec, cfg: dict) -> Path:
    root = project_root()
    if spec.case_id == "C4":
        return (root / cfg["c4_frozen_root"]).resolve()
    pool = (root / cfg["cga_source_root"]).resolve()
    matches = sorted(pool.glob(f"{spec.numerical_id}_*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one CGA source for {spec.case_id}, got {matches}")
    return matches[0]


def _load_snapshot(cfg: dict) -> dict[tuple[str, int], dict]:
    path = (project_root() / cfg["cga_snapshot"]).resolve()
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            numerical_id = row["case"].split("_", 1)[0]
            if numerical_id not in {spec.numerical_id for spec in CASES.values()}:
                continue
            result[(numerical_id, int(row["N"]))] = row
    return result


def _optional_float(value: object) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def import_cga_history(cfg: dict) -> None:
    output = project_root() / "data" / "cga_archive_history.csv"
    fields = ["case_id", "numerical_id", "attempted_iteration", "accepted_atom_count",
              "energy_gap", "natural_error", "v_error", "accepted", "trusted", "source_path"]
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for spec in CASES.values():
            source = _source_dir(spec, cfg)
            with (source / "history.csv").open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    writer.writerow({
                        "case_id": spec.case_id,
                        "numerical_id": spec.numerical_id,
                        "attempted_iteration": row["attempted_iteration"],
                        "accepted_atom_count": row["accepted_atom_count"],
                        "energy_gap": row["energy_gap_raw"],
                        "natural_error": row["natural_rel"],
                        "v_error": row["quasi_rel"],
                        "accepted": row["accepted"],
                        "trusted": row["trusted"],
                        "source_path": package_path(source / "history.csv"),
                    })


def run_cga(case_ids: list[str], cfg: dict, config_hash: str, force: bool) -> None:
    store = ResultStore(project_root() / "data" / "cga_evaluated.csv", force)
    snapshot = _load_snapshot(cfg)
    for case_id in case_ids:
        spec = CASES[case_id]
        source = _source_dir(spec, cfg)
        model_file = source / "diagnostic_model.npz"
        features, archived_coefficients = load_cga_features(model_file)
        if features.size != spec.cga_expected_accepted:
            raise RuntimeError(f"{case_id}: expected {spec.cga_expected_accepted} atoms, got {features.size}")
        widths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        widths += [spec.cga_expected_accepted] if spec.cga_expected_accepted not in widths else []
        widths = [n for n in widths if n <= spec.cga_expected_accepted]
        train = _feature_train_rule(spec, cfg, features.w, features.b)
        evaluation, audit = (make_rule(spec, cfg, name) for name in ("evaluation", "audit"))
        phi_full, grad_full = evaluate_features(features, train.points)
        f = forcing(spec, train.points)
        for width in widths:
            if store.contains(case_id, "cga", "relu3", 201, width):
                continue
            archived = snapshot.get((spec.numerical_id, width), {})
            if width < spec.cga_expected_accepted:
                # The archive contains no coefficient checkpoint at intermediate
                # dyadic widths.  Preserve the preregistered SQ17 value rather than
                # silently fitting an under-resolved surrogate on a different rule.
                archive_rule = QuadratureRule(
                    np.empty((0, spec.dim)), np.empty(0), "pool-small SQ17 archive",
                    sha256(b"pool-small-SQ17-seed302").hexdigest(),
                )
                metrics = {
                    "energy_gap": _optional_float(archived.get("energy_gap_raw")),
                    "energy_gap_signed": _optional_float(archived.get("energy_gap_raw")),
                    "natural_error": _optional_float(archived.get("natural_rel")),
                    "v_error": _optional_float(archived.get("quasi_rel")),
                    "l2_error": None, "numerical_energy": None, "exact_energy": None,
                }
                row = _metric_record(
                    spec, "cga", "relu3", width, metrics, archive_rule, config_hash,
                    seed=201, accepted_atoms=width, solver_iterations=0, solver_residual=0.0,
                    solver_success=True, solver_message="read-only archived dyadic metric",
                    wall_time_sec=0.0, evaluation_audit_rel_delta=None,
                    source="cga_accepted_dyadic", source_path=package_path(source / "dyadic.csv"),
                    archived_energy_gap=archived.get("energy_gap_raw", ""),
                    archived_natural_error=archived.get("natural_rel", ""),
                    archived_v_error=_optional_float(archived.get("quasi_rel")), model_path="",
                )
                store.append(row)
                print(f"CGA {case_id} N={width}: archived natural={metrics['natural_error']:.3e}", flush=True)
                continue
            # The archived full coefficient vector is a deterministic warm start.
            # Reoptimization still uses only the frozen prefix ``[:width]``.
            initial = archived_coefficients[:width]
            if width == spec.cga_expected_accepted:
                solved_coefficients = initial
                solved_iterations, solved_residual, solved_success = 0, 0.0, True
                solved_message, solved_wall_time = "archived terminal coefficients", 0.0
            else:
                solved = solve_feature_coefficients(spec, phi_full[:, :width], grad_full[:, :width],
                                                    f, train.weights, cfg["solver"], initial)
                solved_coefficients = solved.coefficients
                solved_iterations, solved_residual = solved.iterations, solved.residual
                solved_success, solved_message = solved.success, solved.message
                solved_wall_time = solved.wall_time_sec
            metrics = _evaluate_feature_solution(spec, features, solved_coefficients, evaluation)
            audit_delta = None
            audit_deltas = {"energy_gap": None, "natural_error": None, "v_error": None}
            if width == spec.cga_expected_accepted:
                audited = _evaluate_feature_solution(spec, features, solved_coefficients, audit)
                audit_delta = max_relative_metric_delta(metrics, audited)
                audit_deltas = relative_metric_deltas(metrics, audited)
                model_out = project_root() / "data" / "models" / f"{case_id}_cga_{width}.npz"
                model_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(model_out, w=features.w[:width], b=features.b[:width], k=features.k,
                                    scales=features.scales[:width], centers=features.centers[:width],
                                    coefficients=solved_coefficients)
            else:
                model_out = ""
            row = _metric_record(
                spec, "cga", "relu3", width, metrics, evaluation, config_hash,
                seed=201, accepted_atoms=width, feature_count="",
                solver_iterations=solved_iterations, solver_residual=solved_residual,
                solver_success=solved_success, solver_message=solved_message,
                wall_time_sec=solved_wall_time, evaluation_audit_rel_delta=audit_delta,
                audit_energy_gap_rel_delta=audit_deltas["energy_gap"],
                audit_natural_rel_delta=audit_deltas["natural_error"],
                audit_v_rel_delta=audit_deltas["v_error"],
                source="cga_terminal_common_evaluator", source_path=package_path(model_file),
                archived_energy_gap=archived.get("energy_gap_raw", ""),
                archived_natural_error=archived.get("natural_rel", ""),
                archived_v_error=_optional_float(archived.get("quasi_rel")),
                model_path="" if not model_out else package_path(model_out),
            )
            store.append(row)
            print(f"CGA {case_id} N={width}: natural={metrics['natural_error']:.3e}", flush=True)


def run_fem(case_ids: list[str], cfg: dict, config_hash: str, force: bool) -> None:
    store = ResultStore(project_root() / "data" / "fem_raw.csv", force)
    for case_id in case_ids:
        spec = CASES[case_id]
        evaluation, audit = (make_rule(spec, cfg, name) for name in ("evaluation", "audit"))
        levels_by_degree = cfg["fem"][f"elements_{spec.dim}d"]
        for degree in cfg["fem"]["degrees"]:
            levels = levels_by_degree[str(degree)]
            for n in levels:
                variant = f"p{degree}"
                solved = solve_fem(spec, degree, n, cfg["fem"])
                if store.contains(case_id, "fem", variant, None, solved.dof):
                    continue
                u, grad_u = evaluate_fem_model(solved.model, evaluation.points)
                metrics = evaluate_fields(spec, evaluation.points, evaluation.weights, u, grad_u)
                audit_delta = None
                audit_deltas = {"energy_gap": None, "natural_error": None, "v_error": None}
                if n == levels[-1]:
                    ua, ga = evaluate_fem_model(solved.model, audit.points)
                    audited = evaluate_fields(spec, audit.points, audit.weights, ua, ga)
                    audit_delta = max_relative_metric_delta(metrics, audited)
                    audit_deltas = relative_metric_deltas(metrics, audited)
                    model_out = project_root() / "data" / "models" / f"{case_id}_fem_{variant}_n{n}.npz"
                    model_out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(model_out, dim=spec.dim, degree=degree,
                                        n_elements_axis=n, coefficients=solved.model.coefficients)
                else:
                    model_out = ""
                row = _metric_record(
                    spec, "fem", variant, solved.dof, metrics, evaluation, config_hash,
                    mesh_level=n, solver_iterations=solved.iterations,
                    solver_residual=solved.residual, solver_success=solved.success,
                    solver_message=solved.message, wall_time_sec=solved.wall_time_sec,
                    evaluation_audit_rel_delta=audit_delta, source="newly_computed",
                    audit_energy_gap_rel_delta=audit_deltas["energy_gap"],
                    audit_natural_rel_delta=audit_deltas["natural_error"],
                    audit_v_rel_delta=audit_deltas["v_error"],
                    source_path="", archived_energy_gap="", archived_natural_error="",
                    archived_v_error=None,
                    model_path="" if not model_out else package_path(model_out),
                )
                store.append(row)
                print(f"FEM {case_id} {variant} dof={solved.dof}: natural={metrics['natural_error']:.3e}", flush=True)


def run_rfm(case_ids: list[str], cfg: dict, config_hash: str, force: bool,
            seeds: list[int] | None = None) -> None:
    store = ResultStore(project_root() / "data" / "rfm_raw.csv", force)
    seeds = cfg["rfm"]["seeds"] if seeds is None else seeds
    for case_id in case_ids:
        spec = CASES[case_id]
        widths = cfg["rfm"][f"widths_{spec.dim}d"]
        evaluation, audit = (make_rule(spec, cfg, name) for name in ("evaluation", "audit"))
        for seed in seeds:
            sampled_size = widths[-1] if spec.dim == 1 else 2 * widths[-1]
            w, b = sample_parameters(spec.dim, sampled_size, seed, cfg["rfm"])
            train = _feature_train_rule(spec, cfg, w, b)
            f = forcing(spec, train.points)
            features = calibrate_features(spec, w, b, spec.relu_k, train.points, train.weights)
            if features.size < widths[-1]:
                raise RuntimeError(f"{case_id} seed={seed}: only {features.size} valid features")
            features = FeatureSet(features.w[:widths[-1]], features.b[:widths[-1]], features.k,
                                  features.scales[:widths[-1]], features.centers[:widths[-1]])
            phi_full, grad_full = evaluate_features(features, train.points)
            previous = None
            for width in widths:
                if store.contains(case_id, "rfm", "rfm_relu3", seed, width):
                    continue
                solved = solve_feature_coefficients(spec, phi_full[:, :width], grad_full[:, :width],
                                                    f, train.weights, cfg["solver"], previous)
                previous = solved.coefficients
                metrics = _evaluate_feature_solution(spec, features, solved.coefficients, evaluation)
                audit_delta = None
                audit_deltas = {"energy_gap": None, "natural_error": None, "v_error": None}
                if width == widths[-1]:
                    audited = _evaluate_feature_solution(spec, features, solved.coefficients, audit)
                    audit_delta = max_relative_metric_delta(metrics, audited)
                    audit_deltas = relative_metric_deltas(metrics, audited)
                    model_out = project_root() / "data" / "models" / f"{case_id}_rfm_seed{seed}_n{width}.npz"
                    model_out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(model_out, w=features.w, b=features.b, k=features.k,
                                        scales=features.scales, centers=features.centers,
                                        coefficients=solved.coefficients)
                else:
                    model_out = ""
                row = _metric_record(
                    spec, "rfm", "rfm_relu3", width, metrics, evaluation, config_hash,
                    seed=seed, feature_count=width, solver_iterations=solved.iterations,
                    solver_residual=solved.residual, solver_success=solved.success,
                    solver_message=solved.message, wall_time_sec=solved.wall_time_sec,
                    evaluation_audit_rel_delta=audit_delta, source="newly_computed_nested_prefix",
                    audit_energy_gap_rel_delta=audit_deltas["energy_gap"],
                    audit_natural_rel_delta=audit_deltas["natural_error"],
                    audit_v_rel_delta=audit_deltas["v_error"],
                    source_path="", archived_energy_gap="", archived_natural_error="",
                    archived_v_error=None,
                    model_path="" if not model_out else package_path(model_out),
                )
                store.append(row)
                print(f"RFM {case_id} seed={seed} N={width}: natural={metrics['natural_error']:.3e}", flush=True)


def write_manifest(cfg: dict, config_hash: str) -> None:
    root = project_root()
    inputs = {}
    for spec in CASES.values():
        source = _source_dir(spec, cfg)
        for name in ("history.csv", "dyadic.csv", "diagnostic_model.npz", "summary.json"):
            path = source / name
            inputs[package_path(path)] = sha256(path.read_bytes()).hexdigest()
    repo = root.parent
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--short"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_status = "unavailable", ""

    def package_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "unavailable"

    manifest = {
        "schema_version": cfg["schema_version"],
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": package_version("matplotlib"),
        "scikit_fem": package_version("scikit-fem"),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status_sha256": sha256(git_status.encode("utf-8")).hexdigest(),
        "config_sha256": config_hash,
        "problem_hashes": {case_id: spec.problem_hash for case_id, spec in CASES.items()},
        "seeds": cfg["rfm"]["seeds"],
        "input_sha256": inputs,
    }
    (root / "artifacts" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["cga", "fem", "rfm", "all"], default="all")
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=sorted(CASES))
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg, config_hash = load_config()
    import_cga_history(cfg)
    write_manifest(cfg, config_hash)
    if args.method in {"cga", "all"}:
        run_cga(args.cases, cfg, config_hash, args.force)
    if args.method in {"fem", "all"}:
        run_fem(args.cases, cfg, config_hash, args.force)
    if args.method in {"rfm", "all"}:
        run_rfm(args.cases, cfg, config_hash, args.force, args.seeds)


if __name__ == "__main__":
    main()
