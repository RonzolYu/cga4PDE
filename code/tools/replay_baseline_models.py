#!/usr/bin/env python3
"""Replay and archive every comparison state under one frozen evaluation policy.

This produces a new batch; it does not fill historical missing paths with models
from a different run.  CLI subsets are resumable and raw observations remain
available even when coefficient or quadrature checks fail.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/compare_fem_rfm/src"))

import numpy as np
from compare_fem_rfm.experiment import _evaluate_feature_solution
from compare_fem_rfm.feature_solver import solve_feature_coefficients
from compare_fem_rfm.features import (FeatureSet, calibrate_features,
    evaluate_features, load_cga_features, sample_parameters)
from compare_fem_rfm.fem import FEMModel, evaluate_fem_model, solve_fem
from compare_fem_rfm.metrics import evaluate_fields, relative_metric_deltas
from compare_fem_rfm.problems import CASES, forcing
from compare_fem_rfm.quadrature import composite_gauss, segmented_gauss_1d
from compare_fem_rfm.quality import metric_valid, invalid_reason


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open(newline='',encoding='utf-8') as f:
            columns=next(csv.reader(f))
        if set(columns)!=set(row):
            raise ValueError(f'CSV schema differs at {path}; migrate before appending')
    else:
        columns=list(row)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_rule(spec, cfg, w, b):
    if spec.dim == 2:
        return composite_gauss(2, cfg["quadrature"]["train_2d_cells"], cfg["quadrature"]["train_2d_order"])
    knots = -b / w[:, 0]
    if spec.model == "pure_p":
        knots = np.append(knots, .5)
    return segmented_gauss_1d(knots, cfg["quadrature"]["train_1d_order"])


def audited_metrics(spec, evaluate, cfg):
    levels = cfg["quadrature"][f"evaluation_{spec.dim}d"]
    previous = None
    for cells, order in levels:
        rule = composite_gauss(spec.dim, cells, order)
        current = evaluate(rule)
        if previous is not None:
            deltas = relative_metric_deltas(previous, current)
            applicable = ["natural_error"] + (["v_error"] if spec.model == "pure_p" else [])
            # Energy is refined too, but a nearly cancelled signed gap need not
            # be certifiable even when the other error metrics are stable.
            needed = applicable + ["energy_gap"]
            if all(deltas[k] is not None and deltas[k] <= cfg["validity"]["audit_relative_tolerance"]
                   for k in needed) and current["energy_gap_signed"] >= 0:
                break
        previous = current
    if previous is current:
        # At the last permitted rule, compare with the preceding rule, not itself.
        preceding = composite_gauss(spec.dim, *levels[-2])
        previous = evaluate(preceding)
    deltas = relative_metric_deltas(previous, current)
    return current, rule, deltas


def record(spec, method, variant, seed, dof, solved, metrics, rule, deltas,
           model_path, cfg_hash, source, mesh=""):
    row = dict(schema_version="cga-review-replay-v1", case_id=spec.case_id,
        numerical_id=spec.numerical_id, method=method, variant=variant,
        seed="" if seed is None else seed, dof=dof, mesh_level=mesh,
        accepted_atoms=dof if method=="cga" else "", feature_count=dof if method=="rfm" else "",
        energy_gap=metrics["energy_gap"], energy_gap_signed=metrics["energy_gap_signed"],
        natural_error=metrics["natural_error"], v_error=metrics["v_error"] if metrics["v_error"] is not None else "",
        l2_error=metrics["l2_error"], numerical_energy=metrics["numerical_energy"], exact_energy=metrics["exact_energy"],
        solver_iterations=solved.iterations, solver_residual=solved.residual, solver_success=solved.success,
        solver_message=solved.message, wall_time_sec=solved.wall_time_sec,
        evaluation_rule=rule.name, evaluation_hash=rule.hash,
        evaluation_audit_rel_delta=max(x for x in deltas.values() if x is not None),
        audit_energy_gap_rel_delta=deltas["energy_gap"], audit_natural_rel_delta=deltas["natural_error"],
        audit_v_rel_delta=deltas["v_error"] if deltas["v_error"] is not None else "",
        plateau_flag=False, source=source, source_path="", problem_hash=spec.problem_hash,
        config_hash=cfg_hash, archived_energy_gap="", archived_natural_error="", archived_v_error="",
        model_path=model_path.relative_to(ROOT).as_posix(), model_sha256=sha256(model_path.read_bytes()).hexdigest(),
        metric_definition="relative_H1" if spec.natural_metric=="H1" else "relative_gradient_component_Lp_seminorm",
        record_provenance="review_replay_20261001",
        hessian_backend=getattr(solved,'hessian_backend','not_applicable'))
    for metric in ("energy_gap", "natural_error", "v_error"):
        row[metric+"_valid"] = metric_valid(row, metric)
        row[metric+"_invalid_reason"] = invalid_reason(row, metric)
    return row


def run_features(spec, cfg, output, method, cfg_hash, selected_seeds):
    csv_path = output/f"{method}_raw.csv"
    done = {(r["case_id"], str(r["seed"]), int(r["dof"])):r for r in read_csv(csv_path)} if csv_path.exists() else {}
    seeds = cfg["seeds"] if method=="rfm" else [201]
    if selected_seeds:
        seeds = [s for s in seeds if s in selected_seeds]
    for seed in seeds:
        widths = list(cfg["widths"][spec.case_id])
        if method=="rfm":
            maximum = widths[-1]
            w, b = sample_parameters(spec.dim, maximum if spec.dim==1 else 2*maximum, seed, cfg["rfm"])
            train = train_rule(spec, cfg, w, b)
            features = calibrate_features(spec, w, b, spec.relu_k, train.points, train.weights)
            features = FeatureSet(features.w[:maximum], features.b[:maximum], features.k,
                                  features.scales[:maximum], features.centers[:maximum])
            if features.size != maximum:
                raise ValueError(f"{spec.case_id}/{seed}: fewer than {maximum} nondegenerate features")
        else:
            source = ROOT/f"data/raw/baseline/data/models/{spec.case_id}_cga_{spec.cga_expected_accepted}.npz"
            features, _ = load_cga_features(source)
            train = train_rule(spec, cfg, features.w, features.b)
            widths = [n for n in widths if n<=features.size]
            if features.size not in widths:
                widths.append(features.size)
        values, grads = evaluate_features(features, train.points)
        load = forcing(spec, train.points)
        previous = None
        for width in widths:
            key = spec.case_id, str(seed), width
            if key in done:
                saved = ROOT/done[key]["model_path"]
                with np.load(saved) as model:
                    previous = model["coefficients"]
                continue
            start = time.monotonic()
            solved = solve_feature_coefficients(spec, values[:,:width], grads[:,:width], load, train.weights,
                                                cfg["solver"], previous)
            previous = solved.coefficients
            model_path = output/"models"/f"{spec.case_id}_{method}_seed{seed}_n{width}.npz"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(model_path, w=features.w[:width], b=features.b[:width], k=features.k,
                                scales=features.scales[:width], centers=features.centers[:width], coefficients=solved.coefficients)
            evaluate = lambda rule: _evaluate_feature_solution(spec, features, solved.coefficients, rule, batch_size=4096)
            metrics, rule, deltas = audited_metrics(spec, evaluate, cfg)
            row = record(spec, method, "rfm_relu3" if method=="rfm" else "relu3", seed, width, solved,
                         metrics, rule, deltas, model_path, cfg_hash,
                         "new_nested_rfm_coefficient_replay" if method=="rfm" else "frozen_greedy_prefix_coefficient_replay")
            append_csv(csv_path, row)
            print(f"{method} {spec.case_id} seed={seed} N={width} solve={solved.success} "
                  f"natural={metrics['natural_error']:.7g} valid={row['natural_error_valid']} "
                  f"audit={deltas['natural_error']:.3g} elapsed={time.monotonic()-start:.1f}s", flush=True)


def run_fem(spec, cfg, output, cfg_hash):
    path = output/"fem_raw.csv"
    done = {(r["case_id"], r["variant"], int(r["mesh_level"])) for r in read_csv(path)} if path.exists() else set()
    historical = read_csv(ROOT/"data/raw/baseline/data/fem_raw.csv")
    for old in historical:
        if old["case_id"] != spec.case_id:
            continue
        key = spec.case_id, old["variant"], int(old["mesh_level"])
        if key in done:
            continue
        solved = solve_fem(spec, int(old["variant"][1:]), key[2], cfg["fem"])
        model_path = output/"models"/f"{spec.case_id}_fem_{old['variant']}_mesh{key[2]}.npz"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model_path, dim=solved.model.dim, degree=solved.model.degree,
                            n_elements_axis=solved.model.n_elements_axis, coefficients=solved.model.coefficients)
        def evaluate(rule):
            u, grad = evaluate_fem_model(solved.model, rule.points, batch_size=4096)
            return evaluate_fields(spec, rule.points, rule.weights, u, grad)
        metrics, rule, deltas = audited_metrics(spec, evaluate, cfg)
        row = record(spec, "fem", old["variant"], None, solved.dof, solved, metrics, rule, deltas,
                     model_path, cfg_hash, "new_mesh_coefficient_replay", mesh=key[2])
        append_csv(path, row)
        print(f"fem {spec.case_id} {old['variant']} mesh={key[2]} solve={solved.success} "
              f"natural={metrics['natural_error']:.7g} valid={row['natural_error_valid']}", flush=True)


def refresh_evaluations(cases, methods, cfg, output):
    """Write updated evaluations separately, so active replay CSVs stay intact."""
    for method in methods:
        updates = output/f"evaluation_updates_{method}.csv"
        previous = read_csv(updates) if updates.exists() else []
        previous = [r for r in previous if r["case_id"] not in cases]
        for row in read_csv(output/f"{method}_raw.csv"):
            if row["case_id"] not in cases:
                continue
            spec = CASES[row["case_id"]]
            path = ROOT/row["model_path"]
            with np.load(path) as saved:
                if method == "fem":
                    model = FEMModel(int(saved["dim"]), int(saved["degree"]),
                                     int(saved["n_elements_axis"]), saved["coefficients"])
                else:
                    features = FeatureSet(saved["w"],saved["b"],int(saved["k"]),saved["scales"],saved["centers"])
                    coefficients = saved["coefficients"]
            def evaluate(rule):
                if method == "fem":
                    u, grad = evaluate_fem_model(model, rule.points, batch_size=4096)
                    return evaluate_fields(spec, rule.points, rule.weights, u, grad)
                return _evaluate_feature_solution(spec, features, coefficients, rule, batch_size=4096)
            metrics, rule, deltas = audited_metrics(spec, evaluate, cfg)
            for key,value in metrics.items():
                row[key] = "" if value is None else value
            row.update(evaluation_rule=rule.name, evaluation_hash=rule.hash,
                       evaluation_audit_rel_delta=max(d for d in deltas.values() if d is not None),
                       audit_energy_gap_rel_delta=deltas["energy_gap"], audit_natural_rel_delta=deltas["natural_error"],
                       audit_v_rel_delta=deltas["v_error"] if deltas["v_error"] is not None else "")
            for metric in ("energy_gap","natural_error","v_error"):
                row[metric+"_valid"] = metric_valid(row,metric)
                row[metric+"_invalid_reason"] = invalid_reason(row,metric)
            previous.append(row)
        with updates.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(previous[0]),lineterminator='\n');writer.writeheader();writer.writerows(previous)
        print(f"Updated evaluations for {method}: {len(previous)} states",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--methods", nargs="+", choices=["rfm","cga","fem"], default=["rfm","cga","fem"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path,
                        help="Package-relative batch directory, useful for disjoint seed workers.")
    parser.add_argument("--refresh-evaluations", action="store_true")
    args=parser.parse_args()
    config_path = ROOT/"config/review_replay.json"
    cfg=json.loads(config_path.read_text());cfg_hash=sha256(config_path.read_bytes()).hexdigest()
    output=ROOT/(args.output_dir or Path("data/raw/review_replay_20261001"))
    output=output.resolve()
    if not output.is_relative_to(ROOT/'data/raw'):
        raise ValueError('Replay output must be under code/data/raw')
    output.mkdir(parents=True, exist_ok=True)
    target=output/"protocol.json"
    if target.exists() and sha256(target.read_bytes()).hexdigest()!=cfg_hash:
        raise RuntimeError("Existing replay uses another frozen protocol; choose a new batch directory.")
    target.write_bytes(config_path.read_bytes())
    if args.refresh_evaluations:
        refresh_evaluations(args.cases,args.methods,cfg,output)
        return
    for case in args.cases:
        for method in args.methods:
            if method=="fem": run_fem(CASES[case],cfg,output,cfg_hash)
            else: run_features(CASES[case],cfg,output,method,cfg_hash,args.seeds)


if __name__=="__main__":
    main()
