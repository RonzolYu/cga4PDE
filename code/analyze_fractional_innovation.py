"""Recompute the merged finite-window tables from the saved projection states.

Uses Python's standard library; run from any directory. No solver is rerun.
Inputs are result/section85/{base_pure,epsilon_01}_finite_q{16,20}.csv.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result" / "fractional_innovation"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "tex" / "generated").mkdir(parents=True, exist_ok=True)
MODELS = {"base_pure": "Pure", "epsilon_01": r"\(\varepsilon=0.1\)"}


def sci(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"


summaries, input_hashes, transitions = {}, {}, []
for model in MODELS:
    grids = {}
    for order in (16, 20):
        source = ROOT / "result" / "section85" / f"{model}_finite_q{order}.csv"
        input_hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open() as stream:
            rows = [row for row in csv.DictReader(stream) if 8 <= int(row["N"]) <= 32]
        assert [int(row["N"]) for row in rows] == list(range(8, 33))
        grids[order] = rows
    rows = grids[20]
    x = [float(row["r2"]) for row in rows]
    drops = [left - right for left, right in zip(x, x[1:])]
    assert all(value > 0 for value in drops)
    c = [drop / value ** (7 / 6) for value, drop in zip(x, drops)]
    lx = list(map(math.log, x[:-1]))
    ly = [math.log(math.sqrt(value)) for value in drops]
    mx, my = statistics.mean(lx), statistics.mean(ly)
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
    intercept = my - slope * mx
    cmin = min(c)
    projection_bound = (x[0] ** (-1 / 6) + cmin * 24 / 6) ** (-6)
    q = max(math.sqrt(float(row["D2"]) / float(row["r2"])) for row in rows)
    rho = max(abs(float(row["remainder"])) / (float(row["r2"]) + float(row["D2"])) for row in rows)
    assert rho < 0.5
    factor = (0.5 + rho) * (1 + q * q)
    other_x = [float(row["r2"]) for row in grids[16]]
    other_c = [(a - b) / a ** (7 / 6) for a, b in zip(other_x, other_x[1:])]
    summaries[model] = {
        "states": len(rows), "transitions": len(c), "alpha_benchmark": 6,
        "c_min": cmin, "c_min_N": 8 + c.index(cmin),
        "nu_fit_gamma_vs_r2": slope, "fit_intercept": intercept,
        "alpha_from_fit": 1 / (2 * slope - 1), "q_max": q, "rho_max": rho,
        "B": float(rows[0]["B"]), "sigma": float(rows[0]["sigma"]),
        "W_terminal": float(rows[-1]["W"]), "r2_terminal": x[-1],
        "source_projection_bound": float(rows[-1]["projection_upper_init_estimate"]),
        "fractional_projection_bound": projection_bound,
        "fractional_projection_bound_ratio": projection_bound / x[-1],
        "energy_gap_terminal": float(rows[-1]["energy_gap"]),
        "source_energy_bound": float(rows[-1]["energy_upper_init_estimate"]),
        "fractional_energy_bound": factor * projection_bound,
        "max_c_relative_quadrature_difference": max(abs(a - b) / a for a, b in zip(c, other_c)),
        "max_projection_drop_defect": max(abs(float(row["projection_drop_defect"])) for row in rows[:-1]),
        "max_energy_identity_defect": max(abs(float(row["decomposition_defect"])) for row in rows),
        "evidence": "Floating-point evaluation of saved states; no interval enclosure.",
    }
    for i, (value, drop, ci) in enumerate(zip(x, drops, c)):
        j = i + 1
        envelope = (x[0] ** (-1 / 6) + cmin * j / 6) ** (-6)
        assert x[j] <= envelope * (1 + 1e-12)
        transitions.append({"model": model, "N": 8 + i, "r2": value,
                            "projection_drop": drop, "gamma": math.sqrt(drop), "c_N": ci,
                            "next_projection_bound": envelope})

with (OUT / "transitions.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(transitions[0]))
    writer.writeheader()
    writer.writerows(transitions)
(OUT / "summary.json").write_text(json.dumps({"inputs_sha256": input_hashes, "models": summaries}, indent=2) + "\n")

def table(caption, label, lines):
    return ("\\begin{table}[!htbp]\n\\centering\n\\small\n"
            + "\\caption{" + caption + "}\n\\label{" + label + "}\n"
            + "\\begin{tabularx}{\\linewidth}{@{}Xrr@{}}\n\\toprule\n"
            + r"Quantity & Pure \(p=4\) & \(\varepsilon=0.1\) \\" + "\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")

conditions = [
    (r"Source budget \(B\)", "B"), (r"Source remainder \(\sigma\)", "sigma"),
    (r"\(\mathcal W_{24}\)", "W_terminal"), (r"\(q=\max D_N/r_N\)", "q_max"),
    (r"\(\rho=\max |R_N|/(r_N^2+D_N^2)\)", "rho_max"),
    (r"Computed \(r_{32}^2\)", "r2_terminal"),
    (r"Initial-value source bound for \(r_{32}^2\)", "source_projection_bound"),
    (r"Computed energy gap at \(N=32\)", "energy_gap_terminal"),
    (r"Initial-value source energy bound", "source_energy_bound"),
]
lines = [label + " & " + " & ".join(sci(summaries[m][key]) for m in MODELS) + r" \\" for label, key in conditions]
(ROOT / "tex" / "generated" / "merged_trajectory_conditions.tex").write_text(table(
    r"Source and quartic comparison quantities on all accepted states \(N=8\)--32. The source is fixed at entry; \(q\) and \(\rho\) are full-window maxima. Values are floating-point estimates.",
    "tab:merged-trajectory-conditions", lines))

checks = [
    (r"\(c_{\min}\) over all 24 transitions", "c_min", sci),
    (r"Transition attaining \(c_{\min}\)", "c_min_N", lambda n: rf"\({n}\to{n+1}\)"),
    (r"Fitted slope \(\nu\) of \(\log\gamma_N\) versus \(\log x_N\)", "nu_fit_gamma_vs_r2", lambda x: f"{x:.4f}"),
    (r"\(1/(2\nu-1)\) from the fit", "alpha_from_fit", lambda x: f"{x:.2f}"),
    (r"Fractional bound for \(r_{32}^2\), \(\alpha=6\)", "fractional_projection_bound", sci),
    (r"Fractional bound / computed \(r_{32}^2\)", "fractional_projection_bound_ratio", sci),
    (r"Fractional energy bound at \(N=32\)", "fractional_energy_bound", sci),
]
lines = [label + " & " + " & ".join(fmt(summaries[m][key]) for m in MODELS) + r" \\" for label, key, fmt in checks]
(ROOT / "tex" / "generated" / "fractional_innovation_summary.tex").write_text(table(
    r"Fractional-innovation quantities for \(\alpha=6\), computed from every transition \(N\to N+1\), \(N=8,\ldots,31\). The fitted exponent describes the innovation score and is not used to choose the benchmark exponent or its bound.",
    "tab:fractional-p4", lines))
print(json.dumps(summaries, indent=2))
