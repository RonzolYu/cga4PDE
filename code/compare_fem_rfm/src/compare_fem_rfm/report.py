"""Generate machine-backed CSV/TeX tables and the Chinese experiment report."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess

from .analysis import build_analysis, write_dict_csv
from .experiment import load_config, project_root
from .problems import CASES


METHODS = ["CGA", "FEM P1", "FEM P2", "FEM P3", "RFM median"]
METRIC_LABELS = {"natural_error": "相对自然误差", "energy_gap": "能量差", "v_error": "相对 V-map 误差"}
CASE_TITLES = {
    "C1": "线性反应--扩散，$d=1$，多频制造解",
    "C2": "三次半线性问题，$d=1$，多频制造解",
    "C3": "双曲正弦半线性问题，$d=2$",
    "C4": "纯 $p$-Laplacian，$p=4,d=1$",
    "C5": "纯 $p$-Laplacian，$p=4,d=2$",
}


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}e}"


def _common_rows(data: list[dict], case_id: str, metric: str) -> list[dict]:
    return [r for r in data if r["case_id"] == case_id and r["metric"] == metric]


def _lookup(common: list[dict], case_id: str, metric: str, method: str, dof: int) -> dict | None:
    matches = [r for r in common if r["case_id"] == case_id and r["metric"] == metric
               and r["method"] == method and r["dof"] == dof]
    return matches[0] if matches else None


def _max_shared_dof(common: list[dict], case_id: str, metric: str) -> int:
    sets = []
    for method in METHODS:
        sets.append({r["dof"] for r in common if r["case_id"] == case_id and
                     r["metric"] == metric and r["method"] == method})
    shared = set.intersection(*sets)
    if not shared:
        raise RuntimeError(f"no shared DOF for {case_id} {metric}")
    return max(shared)


def write_matched_tables(common: list[dict]) -> dict[tuple[str, str], str]:
    root = project_root()
    outputs = {}
    for case_id, spec in CASES.items():
        metrics = ["natural_error", "energy_gap"] + (["v_error"] if spec.model == "pure_p" else [])
        for metric in metrics:
            dofs = sorted({r["dof"] for r in _common_rows(common, case_id, metric)})
            wide = []
            for dof in dofs:
                row = {"dof": dof}
                for method in METHODS:
                    item = _lookup(common, case_id, metric, method, dof)
                    row[method] = None if item is None else item["value"]
                    row[f"{method}_interpolated"] = None if item is None else item["is_interpolated"]
                    if method == "RFM median":
                        row["RFM_q1"] = None if item is None else item["q1"]
                        row["RFM_q3"] = None if item is None else item["q3"]
                wide.append(row)
            csv_path = root / "artifacts" / "tables" / f"{case_id}_{metric}.csv"
            write_dict_csv(csv_path, wide)
            tex_path = root / "artifacts" / "tables" / f"{case_id}_{metric}.tex"
            lines = ["\\begin{tabular}{rccccc}", "\\toprule",
                     "DOF & CGA & FEM P1 & FEM P2 & FEM P3 & RFM median \\\\", "\\midrule"]
            for row in wide:
                values = [str(row["dof"])] + [fmt(row[m]) for m in METHODS]
                lines.append(" & ".join(values) + " \\\\")
            lines += ["\\bottomrule", "\\end{tabular}"]
            tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            outputs[(case_id, metric)] = tex_path.name
    return outputs


def validate_results(data: dict, cfg: dict) -> dict:
    all_rows = data["cga"] + data["fem"] + data["rfm_raw"]
    hashes = {case_id: sorted({r["problem_hash"] for r in all_rows if r["case_id"] == case_id})
              for case_id in CASES}
    evaluator_hashes = {case_id: sorted({r["evaluation_hash"] for r in all_rows if r["case_id"] == case_id})
                        for case_id in CASES}
    terminal_evaluator_hashes = {
        case_id: sorted({r["evaluation_hash"] for r in all_rows if r["case_id"] == case_id and
                         r.get("source") != "cga_accepted_dyadic"})
        for case_id in CASES
    }
    seed_checks = {}
    for case_id, spec in CASES.items():
        expected_widths = cfg["rfm"][f"widths_{spec.dim}d"]
        seed_checks[case_id] = {
            str(width): sorted({int(r["seed"]) for r in data["rfm_raw"]
                                if r["case_id"] == case_id and r["dof"] == width})
            for width in expected_widths
        }
    c4_max = max(r["dof"] for r in data["cga"] if r["case_id"] == "C4")
    solver_failures = [{"case_id": r["case_id"], "method": r["method"], "variant": r["variant"],
                        "seed": r["seed"], "dof": r["dof"], "residual": r["solver_residual"],
                        "message": r["solver_message"]}
                       for r in all_rows if str(r["solver_success"]).lower() != "true"]
    audit_values = [r["evaluation_audit_rel_delta"] for r in all_rows
                    if r.get("evaluation_audit_rel_delta") is not None]
    audit_natural = [r["audit_natural_rel_delta"] for r in all_rows
                     if r.get("audit_natural_rel_delta") is not None]
    audit_energy = [r["audit_energy_gap_rel_delta"] for r in all_rows
                    if r.get("audit_energy_gap_rel_delta") is not None]
    audit_v = [r["audit_v_rel_delta"] for r in all_rows if r.get("audit_v_rel_delta") is not None]
    return {
        "problem_hash_consistent": all(len(v) == 1 for v in hashes.values()),
        "problem_hashes": hashes,
        "common_evaluator_per_case": all(len(v) == 1 for v in evaluator_hashes.values()),
        "evaluation_hashes": evaluator_hashes,
        "common_terminal_evaluator_per_case": all(len(v) == 1 for v in terminal_evaluator_hashes.values()),
        "terminal_evaluation_hashes": terminal_evaluator_hashes,
        "c4_max_cga_dof_is_141": c4_max == 141,
        "c4_max_cga_dof": c4_max,
        "rfm_has_registered_seeds": all(
            values == cfg["rfm"]["seeds"] for case in seed_checks.values() for values in case.values()
        ),
        "rfm_seed_sets": seed_checks,
        "solver_failure_count": len(solver_failures),
        "solver_failures": solver_failures,
        "max_terminal_evaluation_audit_relative_delta": max(audit_values, default=0.0),
        "max_terminal_natural_audit_relative_delta": max(audit_natural, default=0.0),
        "max_terminal_energy_audit_relative_delta": max(audit_energy, default=0.0),
        "max_terminal_v_audit_relative_delta": max(audit_v, default=0.0),
        "no_common_grid_extrapolation": all(
            row["dof"] <= max(r["dof"] for r in
                              (data["rfm"] if row["method"] == "RFM median" else
                               data["cga"] if row["method"] == "CGA" else data["fem"])
                              if r["case_id"] == row["case_id"] and
                              (row["method"] in {"CGA", "RFM median"} or
                               r["variant"] == row["method"].lower().replace("fem ", "")))
            for row in data["common"]
        ),
    }


def _case_finding(common: list[dict], case_id: str) -> dict:
    dof = _max_shared_dof(common, case_id, "natural_error")
    values = {method: _lookup(common, case_id, "natural_error", method, dof)["value"] for method in METHODS}
    best = min(values, key=values.get)
    crossover = [n for n in sorted({r["dof"] for r in common if r["case_id"] == case_id})
                 if _lookup(common, case_id, "natural_error", "CGA", n) is not None and
                 _lookup(common, case_id, "natural_error", "FEM P3", n) is not None and
                 _lookup(common, case_id, "natural_error", "FEM P3", n)["value"] <=
                 _lookup(common, case_id, "natural_error", "CGA", n)["value"]]
    rfm = _lookup(common, case_id, "natural_error", "RFM median", dof)
    return {
        "dof": dof,
        "values": values,
        "best": best,
        "rfm_over_cga": values["RFM median"] / values["CGA"],
        "p3_over_cga": values["FEM P3"] / values["CGA"],
        "p3_crossover": None if not crossover else crossover[0],
        "rfm_iqr_factor": None if rfm["q1"] in (None, 0) else rfm["q3"] / rfm["q1"],
    }


def _summary_markdown(data: dict, validation: dict) -> str:
    common = data["common"]
    findings = {case_id: _case_finding(common, case_id) for case_id in CASES}
    cfg, _ = load_config()
    seed_text = ", ".join(str(seed) for seed in cfg["rfm"]["seeds"])
    lines = [
        "# FEM / RFM / CGA 五算例等自由度对比实验报告",
        "",
        "## 摘要",
        "",
        f"本实验在统一的可训练全局系数（DOF）口径下比较连续 Lagrange FEM P1/P2/P3、冻结随机 ReLU³ 特征的 RFM，以及冻结 pool-small 原子轨迹的 CGA。RFM 使用配置中登记的 seeds={seed_text}，表中给出成功运行的中位数及四分位区间。FEM、RFM 与 CGA 终点由本项目评价器重算；缺少系数快照的 CGA 中间 dyadic 点保留只读归档评价值并显式标注。",
        "",
    ]
    for case_id, finding in findings.items():
        lines.append(
            f"- {case_id} 在五种变体共同可比的最大 DOF={finding['dof']} 时，"
            f"自然误差最小的是 {finding['best']}；RFM median/CGA={finding['rfm_over_cga']:.2f}，"
            f"FEM P3/CGA={finding['p3_over_cga']:.2f}。"
        )
    lines += [
        "",
        "## 公平性与实现",
        "",
        "- PDE、制造解、载荷、能量和误差定义由同一 `ProblemSpec` 路径提供。精确解只用于载荷构造与最终评价。",
        "- FEM 使用 `[0,1]` 或 `[0,1]^2` 上的拟一致结构化网格；二维网格固定剖分为三角形，连续 P1/P2/P3 共用网格族。纯 p 模型用一个全局零均值约束，因此 DOF=`dim(Vh)-1`。",
        "- RFM 对每个种子一次生成最大特征池，小宽度严格取嵌套前缀；特征按自然度量归一化，纯 p 情形先中心化。",
        "- CGA 不重新选原子。归档未保存中间前缀的系数快照，因此中间 dyadic 点采用带来源标记的归档评价值；冻结终点诊断模型由共同评价器重算。原始历史另存为 `data/cga_archive_history.csv`，不会被覆盖。",
        "- 公共表格只做被实际点包围的 log-log 插值，绝不外推。C4 的 CGA 公共表截止 128，完整轨迹终止于 141。",
        "",
        "## 主要科学问题",
        "",
        "### 1. 自适应字典选择的收益",
        "",
    ]
    for case_id, f in findings.items():
        lines.append(f"- {case_id}: 在 DOF={f['dof']}，RFM median/CGA 自然误差比为 {f['rfm_over_cga']:.2f}；该中位数来自登记种子 {seed_text}，不代表更大规模的随机总体。")
    lines += ["", "### 2. 高阶 FEM 是否追上 CGA？", ""]
    for case_id, f in findings.items():
        cross = "未在合法公共网格发生" if f["p3_crossover"] is None else f"首次出现在 DOF={f['p3_crossover']}"
        lines.append(f"- {case_id}: FEM P3/CGA 末端误差比 {f['p3_over_cga']:.2f}，P3≤CGA 的 crossover {cross}。")
    lines += [
        "",
        "### 3. 收敛阶与平台",
        "",
        "`artifacts/tables/order_summary.csv` 同时给出最后原始局部阶、最后有效下降阶和预注册窗口拟合阶。理论参考线使用代码中固定的 CGA Hessian--WOGA、FEM 标准 H1 和 RFM Monte-Carlo 斜率，只做竖直平移，没有从数据拟合理论斜率。末段上升或局部阶≤0.15 被标成平台诊断，不被解释为算法本征阶。",
        "",
        "### 4. 数值审计与限制",
        "",
        f"- problem hash 一致：{validation['problem_hash_consistent']}；除明确标注的 CGA 中间归档点外，终点共同评价器一致：{validation['common_terminal_evaluator_per_case']}；C4 冻结终点正确：{validation['c4_max_cga_dof_is_141']}。",
        f"- RFM 注册种子 {seed_text} 完整：{validation['rfm_has_registered_seeds']}；求解器未通过残差门槛的记录数：{validation['solver_failure_count']}。",
        f"- 末端评价规则加密的最大相对变化：自然误差 {validation['max_terminal_natural_audit_relative_delta']:.3e}，V-map {validation['max_terminal_v_audit_relative_delta']:.3e}，能量差 {validation['max_terminal_energy_audit_relative_delta']:.3e}。能量差接近积分误差底时显著更敏感。",
        "- 二维 RFM 训练使用与最终评价相互独立的确定性复合 Gauss 规则；CGA 中间点沿用带来源标记的归档评价规则。本文比较的是有限 DOF 窗口，不能据此宣称连续字典的无条件渐近定理。",
        "",
        "## 文件索引",
        "",
        "- `data/*_raw.csv`：新计算原始记录；`data/cga_evaluated.csv`：冻结 CGA 的归档中间点与共同重评终点。",
        "- `data/common_grid.csv`：只用于表格的合法插值记录。",
        "- `artifacts/figures/`：误差、能量、V-map 与局部阶图。",
        "- `artifacts/manifest.json` 与 `artifacts/validation.json`：环境、输入哈希和验收检查。",
    ]
    return "\n".join(lines) + "\n"


def _tex_document(data: dict, validation: dict, table_files: dict) -> str:
    common = data["common"]
    findings = {case_id: _case_finding(common, case_id) for case_id in CASES}
    cfg, _ = load_config()
    seed_text = ", ".join(str(seed) for seed in cfg["rfm"]["seeds"])
    sections = []
    for case_id, spec in CASES.items():
        f = findings[case_id]
        metric_blocks = []
        for metric in ["natural_error", "energy_gap"] + (["v_error"] if spec.model == "pure_p" else []):
            metric_blocks.append(rf"""
\subsubsection*{{{METRIC_LABELS[metric]}}}
\begin{{center}}\includegraphics[width=0.78\textwidth]{{../figures/{case_id}_{metric}.pdf}}\end{{center}}
\begin{{center}}\small\input{{../tables/{table_files[(case_id, metric)]}}}\end{{center}}
""")
        sections.append(rf"""
\subsection{{{case_id}: {CASE_TITLES[case_id]}}}
在所有方法共同覆盖的最大合法 checkpoint $N={f['dof']}$，自然误差最小的方法为 {f['best']}。
此处 RFM median（登记种子 {seed_text}）与 CGA 的误差比值为 {f['rfm_over_cga']:.2f}，FEM P3 与 CGA 的比值为 {f['p3_over_cga']:.2f}。
{''.join(metric_blocks)}
\subsubsection*{{自然误差局部阶}}
\begin{{center}}\includegraphics[width=0.78\textwidth]{{../figures/{case_id}_orders.pdf}}\end{{center}}
""")
    return rf"""\documentclass[11pt,fontset=mac]{{ctexart}}
\usepackage[a4paper,margin=2.1cm]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,longtable,hyperref,xcolor}}
\hypersetup{{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}}
\title{{FEM / RFM / CGA 五算例等自由度公平对比实验}}
\author{{自动化可复现实验流水线}}
\date{{2026年9月}}
\begin{{document}}
\maketitle
\begin{{abstract}}
本文在统一 DOF 口径下比较连续 Lagrange FEM P1/P2/P3、冻结随机 ReLU$^3$ 特征 RFM 与冻结 pool-small 原子轨迹 CGA。五个制造解算例覆盖线性、半线性和退化 $p$-growth 模型。RFM 使用登记种子 {seed_text}，表中报告成功运行的中位数和四分位区间。FEM、RFM 与 CGA 终点由共同评价器重算；无系数快照的 CGA 中间点保留归档评价值。
\end{{abstract}}

\section{{实验设置与公平协议}}
定义 CGA 的 DOF 为接受原子数，RFM 的 DOF 为冻结特征数，FEM 的 DOF 为 $\dim V_h$；纯 $p$ 模型施加一个零均值约束，故减一。二维 FEM 使用固定结构化三角网格族，不做自适应加密。RFM 在每个种子内严格使用最大池的嵌套前缀。CGA 的选择顺序保持冻结；由于归档没有中间前缀的系数快照，中间 dyadic 点沿用带来源标记的归档评价值，冻结终点诊断模型由共同评价器重算。公共 dyadic 表仅作 log--log 内插，禁止外推；因此 C4 的 CGA 表截止 $N=128$，图中保留其 $N=141$ 终点。

评价积分为一维复合 Gauss 规则和二维确定性复合 Gauss 规则；RFM 的训练积分与最终评价相互独立。理论线的斜率由预注册公式给出：CGA 自然/V 误差为 $3$（1D）或 $7/4$（2D），FEM P$r$ 为 $r/d$，RFM guide 为 $1/2$；能量参考斜率取相应自然误差斜率的两倍。这些是有限窗口参考线，不是由数据回归得到的定理。

\section{{逐算例结果}}
{''.join(sections)}

\section{{跨算例解释}}
在共同最大 DOF 上，五个算例的 RFM/CGA 自然误差比分别为
{', '.join(f'{case_id}: {findings[case_id]["rfm_over_cga"]:.2f}' for case_id in CASES)}。
高阶 FEM 的结论不能预先写死：FEM P3/CGA 比分别为
{', '.join(f'{case_id}: {findings[case_id]["p3_over_cga"]:.2f}' for case_id in CASES)}。
完整局部阶、有效下降阶、窗口拟合阶和平台起点见 \texttt{{artifacts/tables/order\_summary.csv}}。

\section{{数值审计与局限}}
Problem hash 跨方法一致：{validation['problem_hash_consistent']}；除明确标注的 CGA 中间归档点外，终点共同评价器一致：{validation['common_terminal_evaluator_per_case']}；RFM 登记种子 {seed_text} 完整：{validation['rfm_has_registered_seeds']}；C4 CGA 冻结终点为 141：{validation['c4_max_cga_dof_is_141']}。未通过求解器残差门槛的记录数为 {validation['solver_failure_count']}。末端加密审计的最大相对变化为：自然误差 {validation['max_terminal_natural_audit_relative_delta']:.3e}，V-map {validation['max_terminal_v_audit_relative_delta']:.3e}，能量差 {validation['max_terminal_energy_audit_relative_delta']:.3e}。接近积分误差底的能量差应谨慎解释。

二维神经特征训练为控制内存使用固定的确定性复合 Gauss 规则，最终评价使用另一套更密的确定性复合 Gauss 规则。CGA 的浅灰细线来自只读归档历史，仅用于显示实际接受轨迹；dyadic marker 中的中间点同样是归档评价值，冻结终点才由共同评价器重算，具体来源记录在 CSV 中。本文结论限于预注册有限 DOF 窗口，不能替代连续字典渐近分析。

\appendix
\section{{复现信息}}
配置位于 \texttt{{configs/experiment.json}}，环境和输入 SHA-256 位于 \texttt{{artifacts/manifest.json}}，机器验收位于 \texttt{{artifacts/validation.json}}。原始数据、公共网格数据、图和表均由同一流水线生成。
\end{{document}}
"""


def build_report(compile_pdf: bool = True) -> None:
    root = project_root()
    cfg, _ = load_config()
    data = build_analysis()
    table_files = write_matched_tables(data["common"])
    validation = validate_results(data, cfg)
    (root / "artifacts" / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = _summary_markdown(data, validation)
    (root / "artifacts" / "report.md").write_text(markdown, encoding="utf-8")
    tex = _tex_document(data, validation, table_files)
    tex_path = root / "artifacts" / "tex" / "baseline_compare.tex"
    tex_path.write_text(tex, encoding="utf-8")
    if compile_pdf:
        result = subprocess.run(
            ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, text=True, capture_output=True,
        )
        (tex_path.parent / "latexmk.stdout.log").write_text(result.stdout + "\n" + result.stderr,
                                                             encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"LaTeX compilation failed; see {tex_path.parent / 'latexmk.stdout.log'}")
