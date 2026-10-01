"""Non-overwriting run directories and structured experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import numpy as np

from .config import config_hash, config_to_dict


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _json_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_value(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def create_run_dir(root: str | Path, cfg: object,
                   timestamp: datetime | None = None) -> Path:
    stamp = (timestamp or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")
    method = "exact-linear" if cfg.model == "linear" else "lbfgs"
    name = (f"{cfg.model}_p{cfg.p:g}_k{cfg.relu_power}_d{cfg.dim}_neumann_"
            f"{cfg.pool.sampler}_{method}_seed{cfg.seed:04d}_{stamp}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for suffix in [""] + [f"_{i:03d}" for i in range(1, 1000)]:
        path = root / f"{name}{suffix}"
        try:
            path.mkdir()
            (path / "figures").mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique run directory")


def environment_info() -> dict[str, Any]:
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        scipy_version = "unavailable"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                             stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        git_commit, git_dirty = "unavailable", None
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy_version,
            "platform": platform.platform(), "processor": platform.processor(),
            "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unspecified"),
            "git_commit": git_commit, "git_dirty": git_dirty,
            "started": datetime.now().astimezone().isoformat()}


def write_run_header(run_dir: Path, cfg: object, environment: dict[str, Any]) -> None:
    payload = config_to_dict(cfg)
    payload["config_sha256"] = config_hash(cfg)
    _write_json(run_dir / "config.json", payload)
    _write_json(run_dir / "environment.json", environment)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    lines = [f"start={environment['started']}", f"freeze_profile={cfg.freeze_profile}",
             f"phase={cfg.phase}", f"config_sha256={config_hash(cfg)}",
             f"problem={cfg.model} dim={cfg.dim} boundary=Neumann",
             f"p={cfg.p} epsilon={cfg.epsilon} relu_power={cfg.relu_power}",
             f"target={cfg.target_accepted} max_attempts={cfg.max_attempts}",
             f"dtype={cfg.dtype} python={environment['python'].split()[0]} numpy={environment['numpy']} scipy={environment['scipy']}"]
    (run_dir / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_history(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / "history.csv"
    row = {k: ("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else v)
           for k, v in _json_value(record).items()}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    with (run_dir / "log.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"attempt={record['attempted_iteration']} accepted={record['accepted']} "
                     f"atoms={record['accepted_atom_count']} energy={record.get('train_energy')} "
                     f"reason={record.get('acceptance_reason')}\n")


def save_checkpoint(run_dir: Path, state: object) -> None:
    target = run_dir / "checkpoint.npz"
    fd, temporary = tempfile.mkstemp(prefix="checkpoint-", suffix=".npz", dir=run_dir)
    os.close(fd)
    np.savez_compressed(temporary, attempted=state.attempted_iteration,
                        nominal=state.nominal_selected_count,
                        accepted=state.accepted_atom_count, rank=state.effective_rank,
                        accepted_indices=np.asarray(state.accepted_indices, dtype=np.int64),
                        coefficients=state.coefficients,
                        q_basis=state.q_basis,
                        r_matrix=state.r_matrix,
                        candidate_consumed=state.candidate_pool.consumed,
                        history_rows=state.history_rows,
                        initial_best_score=state.initial_best_score,
                        low_oracle_count=state.low_oracle_count,
                        coverage_warning=int(state.coverage_warning),
                        had_coverage_warning=int(state.had_coverage_warning),
                        quadrature_warning_count=state.quadrature_warning_count,
                        quadrature_warning_start=(-1 if state.quadrature_warning_start is None
                                                  else state.quadrature_warning_start),
                        first_quadrature_warning=(-1 if state.first_quadrature_warning is None
                                                  else state.first_quadrature_warning),
                        trusted_atom_count=state.trusted_atom_count)
    os.replace(temporary, target)


def load_checkpoint(run_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(run_dir) / "checkpoint.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verify_resume_config(saved: dict[str, Any], cfg: object, pools: tuple[object, object]) -> None:
    if saved.get("config_sha256") != config_hash(cfg):
        raise ValueError("resume config hash mismatch")
    if saved.get("candidate_pool_hash") != pools[0].hash or saved.get("reference_pool_hash") != pools[1].hash:
        raise ValueError("resume pool hash mismatch")


def save_model(run_dir: Path, state: object, filename: str = "model.npz") -> None:
    idx = np.asarray(state.accepted_indices, dtype=np.int64)
    pool = state.candidate_pool
    target = run_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, model=state.problem.name, dim=state.problem.dim,
                        p=state.problem.p, epsilon=-1.0 if state.problem.epsilon is None else state.problem.epsilon,
                        exact_profile=state.problem.exact_profile,
                        w=pool.w[idx], b=pool.b[idx], k=pool.k, scales=pool.scales[idx],
                        centers=pool.centers[idx], coefficients=state.coefficients)


def save_pool_manifest(run_dir: Path, candidate: object, reference: object) -> None:
    from .dictionary import pool_manifest, save_pool
    save_pool(run_dir / "candidate_pool.npz", candidate)
    save_pool(run_dir / "reference_pool.npz", reference)
    _write_json(run_dir / "pool_manifest.json",
                {"candidate": pool_manifest(candidate), "reference": pool_manifest(reference)})


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    summary = dict(summary)
    summary["finished"] = datetime.now().astimezone().isoformat()
    _write_json(run_dir / "summary.json", summary)
