#!/usr/bin/env python3
"""Rebuild artifacts or recompute diagnostics in an isolated package copy."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv, hashlib, json, math, os
from pathlib import Path
import re, shutil, subprocess, sys, tempfile

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def aggregate_rfm(stage, options):
    import numpy as np
    base = stage / "data/derived/experiments"
    raw = read_csv(base / "rfm_multiseed_raw.csv")
    groups, identifiers = defaultdict(list), set()
    for row in raw:
        key = row["case_id"], int(row["dof"]), int(row["seed"])
        require(key not in identifiers, f"duplicate RFM record {key}")
        require(row["solver_success"] in ("True", "False"), f"invalid success flag {key}")
        identifiers.add(key)
        groups[key[:2]].append(row)
    expected = json.loads((base / "experiment_validation.json").read_text())["rfm"]
    failures = [r for r in raw if r["solver_success"] == "False"]
    require(len(raw) == expected["rows"] and len(failures) == expected["failures"], "RFM counts differ from declared protocol")
    require(sorted({int(r["seed"]) for r in raw}) == expected["seeds"], "RFM seed set differs")
    oldpath = base / "rfm_multiseed_summary.csv"
    old = {(r["case_id"], int(r["dof"])): r for r in read_csv(oldpath)} if oldpath.exists() else {}
    summaries = []
    for (case, dof), rows in sorted(groups.items()):
        ok = [r for r in rows if r["solver_success"] == "True"]
        row = dict(schema_version="cga-experiments-v1", case_id=case, dof=dof, seed_count=len(rows),
                   success_count=len(ok), failure_count=len(rows)-len(ok),
                   failure_reasons="; ".join(sorted({r["solver_message"] for r in rows if r["solver_success"] == "False"})) or "none",
                   config_hashes=";".join(sorted({r["config_hash"] for r in rows})))
        for metric in ("energy_gap", "natural_error", "v_error", "l2_error", "wall_time_sec"):
            values = [float(r[metric]) for r in ok if r[metric] not in ("", "NA", "nan", "NaN")]
            require(all(math.isfinite(v) for v in values), f"nonfinite {case}/{dof}/{metric}")
            q = np.quantile(values, [.5, .25, .75], method="linear") if values else ["NA"]*3
            for suffix, value in zip(("median", "q1", "q3"), q):
                field = metric + "_" + suffix
                row[field] = value
                if (case, dof) in old and value != "NA":
                    require(math.isclose(float(old[(case, dof)][field]), value, rel_tol=options["summary_rtol"], abs_tol=options["summary_atol"]),
                            f"stored summary mismatch: {case}/{dof}/{field}")
        summaries.append(row)
    write_csv(oldpath, summaries)
    write_csv(stage / "result/review_checks/rfm_failures.csv", failures)
    return dict(rows=len(raw), successes=len(raw)-len(failures), failures=len(failures), groups=len(summaries),
                quantile_method="numpy.quantile(method=linear); successful rows only; incomplete groups retained")

def continuation_table(stage):
    rows = [r for r in read_csv(stage / "result/continuation_20260914/continuation_summary.csv") if r["order"] == "20"]
    require({r["model"] for r in rows} == {"base_pure", "epsilon_01"}, "continuation table model set")
    lines = [r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{Continuation from $N=32$ to $N=64$ with constants fixed on $N=8$--$32$. Pass counts refer to the normalized-innovation sufficient condition; the first failed transition starts at the reported $N$.}",
             r"\label{tab:continuation-window}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Model & New steps & Pass & First failed $N$ & $\min c_N/c_0$ & $r_{64}^2$ \\", r"\midrule"]
    for r in rows:
        label = r"Pure $p=4$" if r["model"] == "base_pure" else r"$\varepsilon=0.1$"
        m, e = f'{float(r["terminal_r2"]):.3e}'.split("e")
        lines.append(f'{label} & {r["new_transitions"]} & {r["power_pass"]} & {r["first_failed_transition"]} & {float(r["min_c_ratio"]):.3f} & $' + m + rf'\times10^{{{int(e)}}}$ \\')
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table = "\n".join(lines) + "\n"
    (stage / "tex/generated/continuation_summary.tex").write_text(table)

def tex_dependencies(stage):
    seen, figures, generated = set(), set(), set()
    def visit(path):
        path = path.resolve()
        if path in seen:
            return
        require(path.is_file(), f"missing TeX input: {path}")
        seen.add(path)
        text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())
        for name in re.findall(r"\\input\{([^{}]+)\}", text):
            target = (stage / "tex" / name).with_suffix(".tex")
            if "generated/" in name:
                generated.add(target.resolve())
            visit(target)
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", text):
            candidates = [p for p in (stage / "tex/figures").rglob(Path(name).name)]
            require(len(candidates) == 1, f"missing or ambiguous regenerated figure {name}")
            figures.add(candidates[0].resolve())
    visit(stage / "tex/main.tex")
    return {"tables": len(generated), "figure_files": len(figures),
            "paths": sorted(str(p.relative_to(stage)) for p in figures | generated)}

def execute(args):
    root = Path(args.package_root).resolve() if args.package_root else Path(__file__).resolve().parents[1]
    out = Path(args.output_root).resolve() if args.output_root else root
    config = Path(args.config) if args.config else Path("config/reproduction.json")
    if not config.is_absolute():
        config = root / config
    report_path = out / f"reports/reproduction_{args.mode}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"mode": args.mode, "status": "FAIL", "commands": [], "output_root": str(out)}
    try:
        options = json.loads(config.read_text())
        require(options["schema_version"] == "reproduction-v1", "unsupported reproduction config")
        require(set(options) == {"schema_version", "summary_rtol", "summary_atol", "subprocess_timeout_seconds"}, "unknown reproduction config fields")
        require(0 <= options["summary_rtol"] <= 1e-6 and 0 <= options["summary_atol"] <= 1e-8, "invalid summary tolerances")
        require(isinstance(options['subprocess_timeout_seconds'], int) and options['subprocess_timeout_seconds'] > 0, 'invalid subprocess timeout')
        report["config_sha256"] = sha(config)
        for name in ("data/derived/experiments/experiment_validation.json",
                     "data/derived/window_diagnostics/window_certificate_validation.json",
                     "result/continuation_20260914/validation.json"):
            require(json.loads((root / name).read_text()).get("passed") is True, f"input validation failed: {name}")
        if args.mode == "solver":
            raise ValueError("The solver mode does not replay solver trajectories. Use the documented frozen-prefix experiment workflow; no solver run was performed.")
        with tempfile.TemporaryDirectory(prefix="cga_reproduction_") as folder:
            stage = Path(folder).resolve()
            for name in ("scripts", "config", "data"):
                shutil.copytree(root / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
            shutil.copytree(root / "result/section85", stage / "result/section85")
            shutil.copytree(root / "result/continuation_20260914", stage / "result/continuation_20260914")
            shutil.copytree(root / "tex", stage / "tex", ignore=shutil.ignore_patterns("*.aux", "*.log", "*.out", "*.synctex.gz"))
            shutil.copy2(root / "chapter5_theory.tex", stage / "chapter5_theory.tex")
            (stage / "tex/generated").mkdir(exist_ok=True)
            # These two tables are audited manuscript inputs rather than products of the
            # artifact generator; retain them when rebuilding the disposable stage.
            for static_name in ("transfer_margin_example.tex",):
                static_source = root / "tex/generated" / static_name
                require(static_source.is_file(), f"missing static TeX input: {static_source}")
                shutil.copy2(static_source, stage / "tex/generated" / static_name)
            (stage / "reports").mkdir()
            report["rfm"] = aggregate_rfm(stage, options)
            def run(script, *extra):
                cmd = [sys.executable, str(stage / "scripts" / script), *extra]
                env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                           VECLIB_MAXIMUM_THREADS="1", MPLCONFIGDIR=str(stage / ".mpl"))
                result = subprocess.run(cmd, cwd=stage, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        timeout=options["subprocess_timeout_seconds"])
                logrel = f"reports/reproduction_logs/{args.mode}_{len(report['commands']):02d}_{Path(script).stem}.log"
                log = out / logrel
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(result.stdout)
                report["commands"].append({"command": ["python", "scripts/"+script, *extra],
                                           "exit_code": result.returncode, "log": logrel})
                require(result.returncode == 0, f"{script} failed; see {logrel}")
            run("verify_continuation_prefix.py")
            if args.mode == "artifacts":
                run("generate_artifacts.py")
                run("analyze_fractional_innovation.py")
                continuation_table(stage)
                report["manuscript_dependencies"] = tex_dependencies(stage)
                products = list((stage / "tex/generated").glob("*.tex")) + list((stage / "tex/figures").rglob("*"))
                products += list((stage / "data/derived/baselines").glob("*"))
                products += list((stage / "result/fractional_innovation").glob("*"))
                products += [stage / "manifest/figure_table_manifest.csv"]
            else:
                run("section85_analysis.py")
                run("continuation_experiments.py", "analyze")
                run("audit_numeric_values.py")
                new, frozen = stage / "data/derived/section85", stage / "result/section85"
                protocol = json.loads((stage / "config/continuation_protocol.json").read_text())
                maximum = 0.
                for name in ("base_pure", "epsilon_01"):
                    for order in (16, 20):
                        fname = f"{name}_finite_q{order}.csv"
                        computed, reference = read_csv(new/fname), read_csv(frozen/fname)
                        require([r["N"] for r in computed] == [r["N"] for r in reference], "state grid changed")
                        for a, b in zip(computed, reference):
                            for key in ("r2", "D2", "energy_gap", "B", "sigma", "W"):
                                require(math.isclose(float(a[key]), float(b[key]), rel_tol=protocol["state_comparison_rtol"], abs_tol=protocol["state_comparison_atol"]),
                                        f"prefix value mismatch {fname}/{a['N']}/{key}")
                            maximum = max(maximum, abs(float(a["decomposition_defect"])))
                            require(abs(float(a["decomposition_defect"])) < protocol["quartic_identity_atol"], "quartic identity failed")
                            if a["projection_drop_defect"] not in ("", "None"):
                                require(abs(float(a["projection_drop_defect"])) < protocol["drop_identity_atol"], "projection identity failed")
                report["prefix_recomputation"] = {"state_rule_pairs": 100, "maximum_quartic_defect": maximum}
                products = [new / name for name in ('common_quadrature_states.csv', 'common_quadrature_terminal.csv',
                            'source_quadrature_crosscheck.csv', 'finite_trajectory_summary.csv')]
                for name in ('base_pure', 'epsilon_01'):
                    for order in (16, 20):
                        products += [new / f'{name}_finite_q{order}.csv', new / f'{name}_source_q{order}.npz',
                                     new / f'{name}_sources_q{order}.json',
                                     stage / 'result/continuation_20260914' / f'{name}_states_q{order}.csv']
                products += [stage / 'result/continuation_20260914' / n for n in ('continuation_summary.csv', 'validation.json')]
            products += [stage / "data/derived/experiments/rfm_multiseed_summary.csv"]
            products += list((stage / "result/review_checks").glob("*"))
            products += [stage / "result/continuation_20260914/prefix_verification.json"]
            report["outputs"] = {}
            # Directories returned by figure rglob are the only non-file entries.
            for candidate in products:
                require(candidate.exists(), f'missing generated output: {candidate.relative_to(stage)}')
            for source in sorted(set(p for p in products if p.is_file())):
                rel = source.relative_to(stage)
                target = out / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                report["outputs"][str(rel)] = sha(target)
            report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(report_path), "error": report.get("error")}))
    return 0 if report["status"] == "PASS" else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("artifacts", "diagnostics", "solver"), required=True)
    parser.add_argument("--package-root")
    parser.add_argument("--output-root")
    parser.add_argument("--config")
    return execute(parser.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
