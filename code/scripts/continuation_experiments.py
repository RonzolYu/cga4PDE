"""Frozen-prefix continuation and direct quadrature checks for the revised paper.

Run prepare before solver, then analyze. All outputs are package relative.
The solver works on new copies; original trajectories are read-only.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import shutil
import sys
import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cga_refactor/src"))
from cga_refactor.config import RunConfig, PoolConfig, QuadratureConfig, SolverConfig, config_hash
from cga_refactor.dictionary import load_pool, evaluate_atoms, breakpoints_1d
from cga_refactor.quadrature import segmented_gauss_1d
from cga_refactor.problems import make_problem, exact_solution
from cga_refactor import solver
from cga_refactor.output import environment_info

OUT = ROOT / "result/continuation_20260914"
RAW = ROOT / "data/raw/continuation_20260914"
PROTOCOL = ROOT / "config/continuation_protocol.json"
NAMES = ["base_pure", "epsilon_01"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def old_run(name):
    paths = list((ROOT / "data/raw/sensitivity" / name).glob("*/config.json"))
    assert len(paths) == 1, (name, paths)
    return paths[0].parent


def config(run):
    raw = json.loads((run / "config.json").read_text())
    raw.pop("config_sha256", None)
    return RunConfig(**{**raw, "pool": PoolConfig(**raw["pool"]),
                        "quadrature": QuadratureConfig(**raw["quadrature"]),
                        "solver": SolverConfig(**raw["solver"])})


def prepare():
    if PROTOCOL.exists():
        raise RuntimeError("Protocol already frozen; inspect existing results instead of overwriting.")
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in NAMES:
        run = old_run(name)
        checkpoint = np.load(run / "checkpoint.npz", allow_pickle=False)
        final = np.load(run / "states/accepted_0032.npz", allow_pickle=False)
        history = read_csv(run / "history.csv")
        pool = load_pool(run / "candidate_pool.npz")
        accepted = [int(row["accepted_pool_index"]) for row in history if row["accepted"] == "True"]
        consumed = {int(row["nominal_pool_index"]) for row in history
                    if row["nominal_pool_index"] not in ("", "None")}
        assert int(checkpoint["accepted"]) == 32
        assert int(checkpoint["history_rows"]) == len(history) == int(checkpoint["attempted"])
        assert np.array_equal(checkpoint["accepted_indices"], accepted)
        assert set(np.flatnonzero(checkpoint["candidate_consumed"])) == consumed
        assert np.array_equal(checkpoint["coefficients"], final["coefficients"])
        assert np.array_equal(pool.w[accepted], final["w"])
        assert np.array_equal(pool.b[accepted], final["b"])
        assert np.array_equal(pool.scales[accepted], final["scales"])
        assert all((run / "states" / f"accepted_{n:04d}.npz").exists() for n in range(1, 33))
        prefix = read_csv(ROOT / "result/section85" / f"{name}_finite_q20.csv")
        x = np.array([float(r["r2"]) for r in prefix])
        c0 = float(np.min((x[:-1] - x[1:]) / x[:-1] ** (7 / 6)))
        qmax = max((float(r["D2"]) / float(r["r2"])) ** .5 for r in prefix)
        rho = max(abs(float(r["remainder"])) / (float(r["r2"]) + float(r["D2"])) for r in prefix)
        source = ROOT / "result/section85" / f"{name}_source_q20.npz"
        required = [run / "config.json", run / "checkpoint.npz", run / "candidate_pool.npz",
                    run / "reference_pool.npz", run / "history.csv", source]
        required += sorted((run / "states").glob("accepted_*.npz"))
        records[name] = {
            "parent": str(run.relative_to(ROOT)), "checkpoint_verified": True,
            "attempts": len(history), "accepted": 32, "consumed": len(consumed),
            "alpha": 6., "c0": c0, "q_corr": qmax, "rho": rho,
            "energy_factor": (.5 + rho) * (1 + qmax * qmax),
            "B": float(prefix[0]["B"]), "sigma": float(prefix[0]["sigma"]),
            "entry_r2": x[0], "r32_squared": x[-1],
            "source_vector": str(source.relative_to(ROOT)),
            "input_sha256": {str(f.relative_to(ROOT)): digest(f) for f in required},
        }
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_states": [8, 32], "unseen_selection_steps": [32, 63],
        "target": 64, "quadrature_orders": [16, 20],
        "state_comparison_atol": 2e-12, "state_comparison_rtol": 5e-7,
        "drop_identity_atol": 1e-10, "quartic_identity_atol": 1e-11,
        "source_policy": "reuse entry-only q20 coefficients; do not fit to new states",
        "failure_policy": "report every stopped run and failed sufficient condition; do not retune constants",
        "scope": "forward extension of archived prefixes, not independent seed replication",
        "resume_state": "fixed saved pools and consumed mask; fresh coefficient solve per step; "
                        "QR reconstructed from saved active atoms; no online random draws",
        "runs": records,
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n")
    print("FROZEN", digest(PROTOCOL), flush=True)


def run_solver():
    protocol = json.loads(PROTOCOL.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        spec = protocol["runs"][name]
        for relative, expected in spec["input_sha256"].items():
            assert digest(ROOT / relative) == expected, relative
        run = RAW / name
        if not run.exists():
            shutil.copytree(ROOT / spec["parent"], run)
            (run / "summary.json").rename(run / "parent_summary.json")
            raw = json.loads((run / "config.json").read_text())
            raw.update(target_accepted=64, phase="frozen_prefix",
                       output_root="data/raw/continuation_20260914")
            (run / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
            raw["config_sha256"] = config_hash(config(run))
            (run / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
            (run / "environment_continuation.json").write_text(json.dumps(environment_info(), indent=2) + "\n")
            (run / "protocol_sha256.txt").write_text(digest(PROTOCOL) + "\n")
        elif (run / "summary.json").exists():
            print(name, "already completed; retaining result", flush=True)
            continue
        start = len(read_csv(run / "history.csv"))
        result = solver.resume_cga(run)
        assert (run / "history.csv").read_bytes().startswith((old_run(name) / "history.csv").read_bytes())
        saved = list((run / "states").glob("accepted_*.npz"))
        assert len(saved) == result["accepted_atom_count"]
        print(name, "new attempts", len(read_csv(run / "history.csv")) - start,
              "accepted", result["accepted_atom_count"], "stop", result["stop_reason"], flush=True)
    (OUT / "solver_status.json").write_text(json.dumps({
        n: json.loads((RAW / n / "summary.json").read_text()) for n in NAMES}, indent=2) + "\n")


def evaluate(name, order):
    """Reconstruct every active projection and the saved source from physical atoms."""
    protocol = json.loads(PROTOCOL.read_text())
    spec = protocol["runs"][name]
    run = RAW / name
    cfg = config(run)
    problem = make_problem(cfg.model, 1, 4, cfg.epsilon, cfg.exact_profile)
    candidate, reference = load_pool(run / "candidate_pool.npz"), load_pool(run / "reference_pool.npz")
    source = np.load(ROOT / spec["source_vector"])
    ci, ri, coeff = source["candidate_indices"], source["reference_indices"], source["coefficients"]
    rule = segmented_gauss_1d(np.unique(np.r_[breakpoints_1d(candidate), breakpoints_1d(reference), .5]), order)
    b = exact_solution(problem, rule.points)[1][:, 0]
    rw = np.sqrt(rule.weights * ((cfg.epsilon or 0) ** 2 + 3 * b * b))
    dc = evaluate_atoms(rule.points, candidate.w[ci], candidate.b[ci], candidate.k)[1][:, :, 0]
    dr = evaluate_atoms(rule.points, reference.w[ri], reference.b[ri], reference.k)[1][:, :, 0]
    F = rw[:, None] * np.column_stack((dc, dr))
    F /= np.linalg.norm(F, axis=0)
    target = rw * b
    models = sorted(p for p in (run / "states").glob("accepted_*.npz") if int(p.stem.split("_")[1]) >= 8)
    rows = []
    W = 0.
    B = float(np.abs(coeff).sum())
    for index, model in enumerate(models):
        n = int(model.stem.split("_")[1])
        z = np.load(model)
        active = evaluate_atoms(rule.points, z["w"], z["b"], int(z["k"]))[1][:, :, 0] / z["scales"]
        A = rw[:, None] * active
        Q, R = qr(A, mode="economic")
        singular = np.linalg.svd(R, compute_uv=False)
        assert singular[-1] > 1e-13 * singular[0], ("unresolved full rank", name, n, order)
        projected = Q @ (Q.T @ target)
        e = target - projected
        x = float(e @ e)
        actual = A @ z["coefficients"]
        D2 = float(np.sum((actual - projected) ** 2))
        h = active @ z["coefficients"] - b
        rem = float(rule.weights @ (b * h ** 3 + .25 * h ** 4))
        gap = float(rule.weights @ (.25 * h ** 2 * ((h + 2 * b) ** 2 + 2 * b ** 2)
                                   + .5 * (cfg.epsilon or 0) ** 2 * h ** 2))
        if index == 0:
            sigma = float(np.linalg.norm(e - (F - Q @ (Q.T @ F)) @ coeff))
            initial = x
        positive_initial = max(initial - sigma * sigma, 0.)
        ub = sigma ** 2 + 1 / (1 / positive_initial + W / (4 * B * B)) if positive_initial > 0 else initial
        old_envelope = (spec["entry_r2"] ** (-1 / 6) + spec["c0"] * (n - 8) / 6) ** -6
        qval = (D2 / x) ** .5 if x > 0 else float("inf")
        rhoval = abs(rem) / (x + D2) if x + D2 else 0.
        row = dict(model=name, order=order, N=n, r2=x, D2=D2, remainder=rem,
                   energy_gap=gap, B=B, sigma=sigma, W=W, source_bound=ub,
                   source_bound_ratio=ub / x, sigma2_over_r2=sigma * sigma / x,
                   q_corr=qval, rho=rhoval, q_prefix_pass=qval <= spec["q_corr"] + 1e-10,
                   rho_prefix_pass=rhoval <= spec["rho"] + 1e-10,
                   prefix_power_bound=old_envelope, prefix_power_bound_ratio=old_envelope / x,
                   prefix_energy_bound=spec["energy_factor"] * old_envelope,
                   quartic_defect=gap - .5 * (x + D2) - rem, condition=singular[0] / singular[-1],
                   theta="", ell2="", c_observed="", c_ratio="", power_condition_pass="",
                   projection_drop_defect="")
        if index + 1 < len(models):
            nz = np.load(models[index + 1])
            d = rw * evaluate_atoms(rule.points, nz["w"][-1:], nz["b"][-1:], int(nz["k"]))[1][:, 0, 0]
            d /= np.linalg.norm(d)
            ell = d - Q @ (Q.T @ d)
            ell2 = float(ell @ ell)
            score = abs(float(e @ d))
            S = float(np.max(np.abs(F.T @ e)))
            assert S > 0 and ell2 > 0
            theta = score / S
            drop = score * score / ell2
            c = drop / x ** (7 / 6)
            row.update(theta=theta, ell2=ell2, c_observed=c, c_ratio=c / spec["c0"],
                       power_condition_pass=c >= spec["c0"])
            W += theta * theta / ell2
        if rows:
            previous = rows[-1]
            pd = previous["c_observed"] * previous["r2"] ** (7 / 6)
            previous["projection_drop_defect"] = previous["r2"] - x - pd
        rows.append(row)
    write_csv(OUT / f"{name}_states_q{order}.csv", rows)
    # Prefix check is evaluated before trusting any new-state result.
    old = {int(r["N"]): r for r in read_csv(ROOT / "result/section85" / f"{name}_finite_q{order}.csv")}
    for r in rows:
        if r["N"] <= 32:
            assert np.isclose(r["r2"], float(old[r["N"]]["r2"]), atol=protocol["state_comparison_atol"],
                              rtol=protocol["state_comparison_rtol"]), (name, order, r["N"])
    assert max(abs(r["quartic_defect"]) for r in rows) < protocol["quartic_identity_atol"]
    assert max(abs(r["projection_drop_defect"]) for r in rows[:-1]) < protocol["drop_identity_atol"]
    tail = [r for r in rows if r["N"] >= 32 and r["c_observed"] != ""]
    end = rows[-1]
    return dict(model=name, order=order, terminal_N=end["N"], new_transitions=len(tail),
                power_pass=sum(r["power_condition_pass"] for r in tail),
                min_c_ratio=min(r["c_ratio"] for r in tail),
                first_failed_transition=next((r["N"] for r in tail if not r["power_condition_pass"]), ""),
                terminal_r2=end["r2"], terminal_energy=end["energy_gap"],
                source_bound_ratio=end["source_bound_ratio"], sigma2_over_r2=end["sigma2_over_r2"],
                power_bound_ratio=end["prefix_power_bound_ratio"],
                q_max=max(r["q_corr"] for r in rows if r["N"] > 32),
                rho_max=max(r["rho"] for r in rows if r["N"] > 32),
                max_drop_defect=max(abs(r["projection_drop_defect"]) for r in rows[:-1]),
                max_quartic_defect=max(abs(r["quartic_defect"]) for r in rows))


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name in NAMES:
        for order in [16, 20]:
            result = evaluate(name, order)
            results.append(result)
            print(json.dumps(result), flush=True)
    write_csv(OUT / "continuation_summary.csv", results)
    (OUT / "validation.json").write_text(json.dumps({
        "passed": True, "scope": "identities, prefix reconstruction and fixed-constant tests; "
                                "negative sufficient conditions are reported outcomes",
        "protocol_sha256": digest(PROTOCOL), "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["prepare", "solver", "analyze"])
    args = parser.parse_args()
    {"prepare": prepare, "solver": run_solver, "analyze": analyze}[args.part]()
