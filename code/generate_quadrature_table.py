#!/usr/bin/env python3
"""Generate the terminal quadrature-sensitivity table from archived evaluations."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "derived" / "experiments" / "quadrature_sensitivity.csv"
OUTPUT = ROOT / "generated" / "quadrature_sensitivity.tex"

CASE_ORDER = ["01", "03", "06", "17", "08", "10", "12", "15"]
CASE_LABEL = {
    "01": "Linear 1D",
    "03": "Cubic 1D",
    "06": "Sinh 2D",
    "17": "Pure p4 1D",
    "08": "Pure p4 2D",
    "10": "Reg. p4 2D",
    "12": "React. p4 2D",
    "15": "Pure p5 1D",
}


def sci(value: str) -> str:
    if value == "":
        return "--"
    x = float(value)
    if x == 0.0:
        return "$0$"
    exponent = int(f"{abs(x):.1e}".split("e")[1])
    mantissa = x / (10.0**exponent)
    return rf"${mantissa:.3f}\times10^{{{exponent}}}$"


def relative_change(a: str, b: str) -> str:
    if a == "" or b == "":
        return "--"
    x, y = float(a), float(b)
    return sci(str(abs(y - x) / max(abs(x), abs(y), 1.0e-300)))


def metric_rows(case_id: str, sample: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    rows = [
        ("direct energy gap", "energy_gap_raw"),
        ("Bregman gap", "bregman_gap"),
    ]
    if case_id in {"01", "03", "06"}:
        rows.append((r"relative $H^1$ error", "natural_rel"))
    elif case_id == "12":
        rows.extend(
            [
                (r"relative $W^{1,4}$ error", "natural_rel"),
                (r"relative combined $V$-distance", "quasi_rel"),
            ]
        )
    else:
        p = "5" if case_id == "15" else "4"
        rows.extend(
            [
                (rf"relative gradient $W^{{1,{p}}}$ error", "natural_rel"),
                (r"relative gradient $V$-distance", "quasi_rel"),
            ]
        )
    return rows


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    terminal: dict[str, dict[str, dict[str, str]]] = {}
    for row in source_rows:
        if row["state"] != "primary_terminal" or row["case_id"] not in CASE_LABEL:
            continue
        terminal.setdefault(row["case_id"], {})[row["evaluator_level"]] = row

    missing = [case for case in CASE_ORDER if set(terminal.get(case, {})) != {"low", "medium", "high"}]
    if missing:
        raise RuntimeError(f"Incomplete terminal quadrature records for cases: {missing}")

    lines = [
        r"\begingroup\scriptsize",
        r"\begin{longtable}{@{}llrrrr@{}}",
        r"\caption{Quadrature sensitivity at the principal terminal state.  The last column is ",
        r"$|q_{\mathrm H}-q_{\mathrm M}|/\max(|q_{\mathrm H}|,|q_{\mathrm M}|)$.  The direct energy gap ",
        r"is signed; the Bregman gap is evaluated in its nonnegative form.}\label{tab:quadrature-sensitivity}\\",
        r"\toprule",
        r"Case & quantity & low & medium & high & M--H change \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Case & quantity & low & medium & high & M--H change \\",
        r"\midrule",
        r"\endhead",
    ]

    for index, case_id in enumerate(CASE_ORDER):
        sample = terminal[case_id]
        for row_index, (name, field) in enumerate(metric_rows(case_id, sample)):
            prefix = CASE_LABEL[case_id] if row_index == 0 else ""
            low = sample["low"][field]
            medium = sample["medium"][field]
            high = sample["high"][field]
            lines.append(
                f"{prefix} & {name} & {sci(low)} & {sci(medium)} & {sci(high)} & "
                f"{relative_change(medium, high)} \\\\"
            )
        if index + 1 != len(CASE_ORDER):
            lines.append(r"\addlinespace[2pt]")

    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
