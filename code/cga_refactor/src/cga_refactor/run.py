"""CLI and the twelve-case campaign."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import json

from .config import MODELS, RunConfig, config_for, config_to_dict
from .reporting import make_exp0822_report, make_report, make_run_report, render_tex
from .solver import resume_cga, run_cga


def benchmark_configs(base_cfg: RunConfig | None = None, *, profile: str = "report",
                      seed: int = 201, output_root: str = "cga_refactor/results") -> list[RunConfig]:
    if base_cfg is not None:
        profile, seed, output_root = base_cfg.phase, base_cfg.seed, base_cfg.output_root
    return [config_for(model, dim, profile=profile, seed=seed, output_root=output_root)
            for model in MODELS for dim in (1, 2)]


def run_one(cfg: RunConfig) -> dict[str, object]:
    summary = run_cga(cfg)
    make_run_report(summary["run_dir"])
    return summary


def run_campaign(base_cfg: RunConfig | None = None, seeds: list[int] | None = None,
                 *, profile: str = "report", seed: int = 201,
                 output_root: str = "cga_refactor/results") -> dict[str, object]:
    seeds = list(seeds) if seeds else [seed]
    seed_tag = "-".join(str(value) for value in seeds)
    root = Path(output_root) / f"campaign_{profile}_seeds{seed_tag}_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    root.mkdir(parents=True)
    run_root = root / "runs"
    configs = [cfg for value in seeds for cfg in benchmark_configs(
        base_cfg, profile=profile, seed=value, output_root=str(run_root))]
    summaries, run_dirs = [], []
    for index, cfg in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] seed={cfg.seed} {cfg.model} d={cfg.dim} profile={cfg.phase}", flush=True)
        summary = run_one(cfg)
        summaries.append(summary)
        run_dirs.append(summary["run_dir"])
        (root / "progress.json").write_text(json.dumps({"completed": index,
                                                         "last": summary}, indent=2), encoding="utf-8")
    manifest = {"profile": profile, "seeds": seeds, "run_dirs": run_dirs,
                "completed": len(summaries), "summaries": summaries}
    (root / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = make_report(root)
    return {"campaign_dir": str(root.resolve()), "report": str(report.resolve()),
            "summaries": summaries}


def exp0819_configs(*, seed: int = 201,
                    output_root: str = "cga_refactor/results",
                    quadrature_level: str = "Q0") -> list[RunConfig]:
    """The 17 formal runs requested for the 19 August experiment."""
    base = [replace(cfg, phase="formal_base")
            for cfg in [config_for(model, dim, profile="formal", seed=seed,
                                   output_root=output_root,
                                   quadrature_level=quadrature_level)
                        for model in MODELS for dim in (1, 2)]]
    p_sensitivity = [
        replace(config_for("pure_p", dim, profile="formal", seed=seed,
                           output_root=output_root, quadrature_level=quadrature_level),
                p=p, phase="formal_p_sensitivity")
        for p in (3.0, 5.0) for dim in (1, 2)
    ]
    k_sensitivity = [
        replace(config_for("pure_p", 1, profile="formal", seed=seed,
                           output_root=output_root, quadrature_level=quadrature_level), relu_power=3,
                phase="formal_k_sensitivity")
    ]
    return base + p_sensitivity + k_sensitivity


def run_exp0819(*, seeds: list[int] | tuple[int, ...] = (201, 203, 207),
                output_root: str = "cga_refactor/results", workers: int = 3,
                quadrature_level: str = "Q0",
                pilot_decision: str | Path | None = None) -> dict[str, object]:
    """Run the formal 12-case campaign and five requested p/k additions."""
    if workers < 1:
        raise ValueError("workers must be positive")
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    seeds = [int(value) for value in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    pilot = None
    if pilot_decision is not None:
        pilot = json.loads(Path(pilot_decision).read_text(encoding="utf-8"))
        if not bool(pilot.get("passed")):
            raise ValueError("formal campaign blocked: quadrature pilot did not pass")
        if pilot.get("selected_level") != quadrature_level:
            raise ValueError("formal quadrature level does not match the frozen pilot decision")
    seed_tag = "-".join(str(value) for value in seeds)
    root = Path(output_root) / f"exp_0819_seeds{seed_tag}_{quadrature_level}_{stamp}"
    root.mkdir(parents=True)
    configs = [cfg for seed in seeds for cfg in exp0819_configs(
        seed=seed, output_root=str(root / "runs"), quadrature_level=quadrature_level)]
    planned = [config_to_dict(cfg) for cfg in configs]
    manifest_path = root / "campaign_manifest.json"
    manifest_path.write_text(json.dumps({
        "profile": "formal_0819", "seeds": seeds,
        "quadrature_level": quadrature_level, "status": "running",
        "pilot_decision": pilot,
        "planned_runs": planned, "run_dirs": [], "completed": 0,
        "failures": [], "summaries": [],
    }, indent=2), encoding="utf-8")

    ordered: list[dict[str, object] | None] = [None] * len(configs)
    failures: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(configs))) as executor:
        futures = {executor.submit(run_one, cfg): (index, cfg)
                   for index, cfg in enumerate(configs)}
        for completed, future in enumerate(as_completed(futures), start=1):
            index, cfg = futures[future]
            try:
                ordered[index] = future.result()
                outcome = {"index": index + 1, "model": cfg.model, "p": cfg.p,
                           "k": cfg.relu_power, "dim": cfg.dim, "status": "completed"}
            except Exception as exc:
                failure = {"index": index + 1, "config": config_to_dict(cfg),
                           "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                outcome = {"index": index + 1, "model": cfg.model, "p": cfg.p,
                           "k": cfg.relu_power, "dim": cfg.dim, "status": "failed",
                           "error": failure["error"]}
            print(f"[{completed}/{len(configs)}] {outcome}", flush=True)
            summaries_now = [item for item in ordered if item is not None]
            progress = {"completed_futures": completed,
                        "successful_runs": len(summaries_now),
                        "failed_runs": len(failures), "last": outcome}
            (root / "progress.json").write_text(json.dumps(progress, indent=2),
                                                 encoding="utf-8")
            manifest_path.write_text(json.dumps({
                "profile": "formal_0819", "seeds": seeds,
                "quadrature_level": quadrature_level, "status": "running",
                "pilot_decision": pilot,
                "planned_runs": planned,
                "run_dirs": [item["run_dir"] for item in summaries_now],
                "completed": len(summaries_now), "failures": failures,
                "summaries": summaries_now,
            }, indent=2), encoding="utf-8")

    summaries = [item for item in ordered if item is not None]
    manifest = {"profile": "formal_0819", "seeds": seeds,
                "quadrature_level": quadrature_level,
                "pilot_decision": pilot,
                "status": "complete" if not failures else "complete_with_failures",
                "planned_runs": planned,
                "run_dirs": [item["run_dir"] for item in summaries],
                "completed": len(summaries), "failures": failures,
                "summaries": summaries}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    markdown_report = make_report(root)
    tex_report = render_tex(root, summaries, output_name="exp_0819.tex")
    return {"campaign_dir": str(root.resolve()), "report": str(tex_report.resolve()),
            "markdown_report": str(markdown_report.resolve()),
            "completed": len(summaries), "failures": failures,
            "summaries": summaries}


def _case_key(item: dict[str, object]) -> tuple[object, ...]:
    """Fields that identify one of the 17 registered cases for one seed."""
    return (int(item["seed"]), str(item["phase"]), str(item["model"]),
            float(item["p"]), int(item["relu_power"]), int(item["dim"]))


def run_exp0820(*, source_campaign: str | Path,
                output_root: str = "cga_refactor/results", workers: int = 3) -> dict[str, object]:
    """Finish the 0819 matrix in a new 0820 campaign.

    Runs with final summaries are reused.  Every source run without a summary is
    deliberately restarted from iteration zero in the new campaign directory.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    source_root = Path(source_campaign).resolve()
    source_manifest = json.loads((source_root / "campaign_manifest.json").read_text(encoding="utf-8"))
    seeds = [int(value) for value in source_manifest["seeds"]]
    quadrature_level = str(source_manifest["quadrature_level"])
    pilot = source_manifest.get("pilot_decision")
    if not pilot or not bool(pilot.get("passed")) or pilot.get("selected_level") != quadrature_level:
        raise ValueError("source campaign does not contain a passing frozen pilot decision")

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    seed_tag = "-".join(str(value) for value in seeds)
    root = Path(output_root) / f"exp_0820_seeds{seed_tag}_{quadrature_level}_{stamp}"
    root.mkdir(parents=True)
    configs = [cfg for seed in seeds for cfg in exp0819_configs(
        seed=seed, output_root=str(root / "runs"), quadrature_level=quadrature_level)]
    planned = [config_to_dict(cfg) for cfg in configs]

    source_summaries: dict[tuple[object, ...], dict[str, object]] = {}
    for path in source_manifest.get("run_dirs", []):
        summary_path = Path(path) / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            source_summaries[_case_key(summary)] = summary
    ordered: list[dict[str, object] | None] = [
        source_summaries.get(_case_key(item)) for item in planned
    ]
    reused_indices = [index + 1 for index, item in enumerate(ordered) if item is not None]
    restart_indices = [index + 1 for index, item in enumerate(ordered) if item is None]
    manifest_path = root / "campaign_manifest.json"
    failures: list[dict[str, object]] = []

    def write_manifest(status: str) -> None:
        summaries_now = [item for item in ordered if item is not None]
        manifest_path.write_text(json.dumps({
            "profile": "formal_0820", "seeds": seeds,
            "quadrature_level": quadrature_level, "status": status,
            "pilot_decision": pilot, "source_campaign": str(source_root),
            "restart_policy": "source runs without summary restart from iteration zero",
            "reused_indices": reused_indices,
            "restarted_from_scratch_indices": restart_indices,
            "planned_runs": planned,
            "run_dirs": [item["run_dir"] for item in summaries_now],
            "completed": len(summaries_now), "failures": failures,
            "summaries": summaries_now,
        }, indent=2), encoding="utf-8")

    write_manifest("running")
    pending = [(index, cfg) for index, (cfg, item) in enumerate(zip(configs, ordered))
               if item is None]
    completed_new = 0
    with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
        futures = {executor.submit(run_one, cfg): (index, cfg) for index, cfg in pending}
        for future in as_completed(futures):
            index, cfg = futures[future]
            completed_new += 1
            try:
                ordered[index] = future.result()
                outcome = {"index": index + 1, "seed": cfg.seed, "model": cfg.model,
                           "p": cfg.p, "k": cfg.relu_power, "dim": cfg.dim,
                           "status": "completed"}
            except Exception as exc:
                failure = {"index": index + 1, "config": config_to_dict(cfg),
                           "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                outcome = {"index": index + 1, "seed": cfg.seed, "model": cfg.model,
                           "p": cfg.p, "k": cfg.relu_power, "dim": cfg.dim,
                           "status": "failed", "error": failure["error"]}
            print(f"[{completed_new}/{len(pending)} new; "
                  f"{sum(item is not None for item in ordered)}/{len(configs)} total] {outcome}", flush=True)
            (root / "progress.json").write_text(json.dumps({
                "completed_new_futures": completed_new,
                "successful_total": sum(item is not None for item in ordered),
                "failed_new_runs": len(failures), "last": outcome,
            }, indent=2), encoding="utf-8")
            write_manifest("running")

    summaries = [item for item in ordered if item is not None]
    status = "complete" if len(summaries) == len(configs) and not failures else "complete_with_failures"
    write_manifest(status)
    markdown_report = make_report(root, report_stem="exp_0820", report_date="20 August 2026")
    return {"campaign_dir": str(root.resolve()),
            "report": str((root / "exp_0820.tex").resolve()),
            "markdown_report": str(markdown_report.resolve()),
            "completed": len(summaries), "failures": failures,
            "summaries": summaries}


def exp_fix124a_configs(*, seed: int = 201,
                        output_root: str = "cga_refactor/results") -> list[RunConfig]:
    """The approved 17-case, one-seed Q2 matrix for the 22 August rerun."""
    if seed != 201:
        raise ValueError("exp0822 is frozen to the single pre-registered seed 201")
    configs = []
    for cfg in exp0819_configs(seed=seed, output_root=output_root,
                               quadrature_level="Q2"):
        solver = replace(cfg.solver, max_lbfgs_iterations=400,
                         max_newton_iterations=100)
        configs.append(replace(cfg, phase="formal_fix124a", solver=solver,
                               freeze_profile="neumann-cga-v1-fix124a",
                               schema_version="cga-refactor-2"))
    signatures = {(cfg.model, cfg.p, cfg.relu_power, cfg.dim) for cfg in configs}
    if len(configs) != 17 or len(signatures) != 17:
        raise RuntimeError("exp0822 must contain exactly 17 unique cases")
    if any(cfg.seed != 201 for cfg in configs):
        raise RuntimeError("exp0822 contains an unapproved seed")
    for cfg in configs:
        expected_target = 256 if cfg.dim == 1 else 512
        if cfg.target_accepted != expected_target:
            raise RuntimeError("exp0822 target mismatch")
        if cfg.dim == 2:
            q = cfg.quadrature
            if (q.train_cells_2d, q.train_order, q.validation_sobol_power,
                    q.audit_sobol_power) != (64, 3, 17, 19):
                raise RuntimeError("exp0822 2D quadrature must be TG64-3/SQ17/SQ19")
    return configs


def run_exp0822(*, seed: int = 201,
                output_root: str = "cga_refactor/results",
                workers: int = 1) -> dict[str, object]:
    """Run all 17 fix124a cases sequentially from iteration zero."""
    if workers != 1:
        raise ValueError("exp0822 is deliberately sequential; workers must equal one")
    if seed != 201:
        raise ValueError("exp0822 is frozen to seed 201")
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    root = Path(output_root) / f"exp_0822_seed201_Q2_{stamp}"
    root.mkdir(parents=True)
    configs = exp_fix124a_configs(seed=seed, output_root=str(root / "runs"))
    planned = [config_to_dict(cfg) for cfg in configs]
    manifest_path = root / "campaign_manifest.json"
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    def write_manifest(status: str) -> None:
        manifest_path.write_text(json.dumps({
            "profile": "formal_fix124a", "seeds": [201],
            "quadrature_level": "Q2", "quadrature_protocol": "TG64-3/SQ17/SQ19",
            "status": status, "restart_policy": "all cases start from iteration zero",
            "planned_runs": planned,
            "run_dirs": [summary["run_dir"] for summary in summaries],
            "completed": len(summaries), "failures": failures,
            "summaries": summaries,
        }, indent=2), encoding="utf-8")

    write_manifest("running")
    for index, cfg in enumerate(configs, start=1):
        try:
            summary = run_one(cfg)
            summaries.append(summary)
            print(f"[{index}/17] completed {cfg.model} p={cfg.p:g} k={cfg.relu_power} "
                  f"d={cfg.dim}: status={summary['formal_status']} "
                  f"trusted={summary['trusted_atom_count']}/{cfg.target_accepted} "
                  f"audit={summary['audit_passed']}", flush=True)
        except Exception as exc:
            failure = {"index": index, "config": config_to_dict(cfg),
                       "formal_status": "solver_failed",
                       "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[{index}/17] failed {cfg.model} p={cfg.p:g} k={cfg.relu_power} "
                  f"d={cfg.dim}: {failure['error']}", flush=True)
        (root / "progress.json").write_text(json.dumps({
            "finished_cases": index, "successful_runs": len(summaries),
            "failed_runs": len(failures),
            "last_case": config_to_dict(cfg),
        }, indent=2), encoding="utf-8")
        write_manifest("running")

    status = "complete" if not failures else "complete_with_failures"
    write_manifest(status)
    markdown_report = make_exp0822_report(root)
    return {"campaign_dir": str(root.resolve()),
            "report": str((root / "exp0822.tex").resolve()),
            "pdf": str((root / "exp0822.pdf").resolve()),
            "markdown_report": str(markdown_report.resolve()),
            "completed": len(summaries), "failures": failures,
            "summaries": summaries}


def check_pilot_gates(summaries: list[dict[str, object]]) -> dict[str, object]:
    complete = sum(bool(s.get("target_reached")) for s in summaries)
    return {"passed": complete == len(summaries), "completed": complete,
            "total": len(summaries), "action": "none" if complete == len(summaries) else "block"}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="cga-refactor")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run")
    one.add_argument("model", choices=MODELS)
    one.add_argument("dim", type=int, choices=(1,2))
    campaign = sub.add_parser("campaign")
    for target in (one, campaign):
        target.add_argument("--profile", choices=("smoke", "report", "formal"), default="report")
        target.add_argument("--seed", type=int, default=201)
        target.add_argument("--output-root", default="cga_refactor/results")
    report = sub.add_parser("report")
    report.add_argument("campaign_dir")
    report.add_argument("--name", default="exp_0819")
    report.add_argument("--date", default="19 August 2026")
    resume = sub.add_parser("resume")
    resume.add_argument("run_dir")
    exp0819 = sub.add_parser("exp0819")
    exp0819.add_argument("--seeds", type=int, nargs="+", default=[201, 203, 207])
    exp0819.add_argument("--quadrature-level", choices=("Q0", "Q1", "Q2"), default="Q0")
    exp0819.add_argument("--pilot-decision")
    exp0819.add_argument("--output-root", default="cga_refactor/results")
    exp0819.add_argument("--workers", type=int, default=2)
    exp0820 = sub.add_parser("exp0820")
    exp0820.add_argument("--source-campaign", required=True)
    exp0820.add_argument("--output-root", default="cga_refactor/results")
    exp0820.add_argument("--workers", type=int, default=3)
    exp0822 = sub.add_parser("exp0822")
    exp0822.add_argument("--seed", type=int, default=201)
    exp0822.add_argument("--output-root", default="cga_refactor/results")
    exp0822.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        cfg = config_for(args.model, args.dim, profile=args.profile, seed=args.seed,
                         output_root=args.output_root)
        print(json.dumps(run_one(cfg), indent=2))
    elif args.command == "campaign":
        print(json.dumps(run_campaign(profile=args.profile, seed=args.seed,
                                      output_root=args.output_root), indent=2))
    elif args.command == "report":
        print(make_report(args.campaign_dir, report_stem=args.name,
                          report_date=args.date))
    elif args.command == "exp0819":
        print(json.dumps(run_exp0819(seeds=args.seeds, output_root=args.output_root,
                                    workers=args.workers,
                                    quadrature_level=args.quadrature_level,
                                    pilot_decision=args.pilot_decision), indent=2))
    elif args.command == "exp0820":
        print(json.dumps(run_exp0820(source_campaign=args.source_campaign,
                                    output_root=args.output_root,
                                    workers=args.workers), indent=2))
    elif args.command == "exp0822":
        print(json.dumps(run_exp0822(seed=args.seed,
                                    output_root=args.output_root,
                                    workers=args.workers), indent=2))
    else:
        print(json.dumps(resume_cga(args.run_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
