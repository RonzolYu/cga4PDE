"""Clean standalone and original-preamble compilation in an external directory."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    workspace = Path(tempfile.mkdtemp(prefix="cga_chapter5_clean_"))
    logs = ROOT / "tmp/chapter5_clean_build"
    logs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "chapter5_theory.tex", workspace / "chapter5_theory.tex")
    preamble = (ROOT / "tex/main.tex").read_text().split(r"\begin{document}", 1)[0]
    (workspace / "integration.tex").write_text(preamble + r"""
\begin{document}
\setcounter{section}{4}
\def\CGAChapterFiveEmbedded{1}
\input{chapter5_theory}
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
""")
    shutil.copy2(ROOT / "tex/References/references.bib", workspace / "references.bib")
    records = []
    for stem in ["chapter5_theory", "integration"]:
        command = ["latexmk", "-xelatex", "-interaction=nonstopmode",
                   "-halt-on-error", stem + ".tex"]
        process = subprocess.run(command, cwd=workspace, capture_output=True, text=True)
        (logs / (stem + "_build.txt")).write_text(process.stdout + process.stderr)
        logpath = workspace / (stem + ".log")
        log = logpath.read_text(errors="replace") if logpath.exists() else ""
        bad = [line for line in log.splitlines() if any(term in line for term in
               ["undefined", "multiply defined", "Overfull", "! LaTeX Error",
                "Missing character"])]
        pdf = workspace / (stem + ".pdf")
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                              text=True) if pdf.exists() else None
        records.append({
            "file": stem + ".tex", "command": command,
            "exit_code": process.returncode, "blocking_log_lines": bad,
            "pdfinfo": info.stdout if info else "",
            "passed": process.returncode == 0 and not bad and info is not None
                      and info.returncode == 0,
        })
        for ext in ["log", "pdf", "aux"]:
            source = workspace / (stem + "." + ext)
            if source.exists():
                shutil.copy2(source, logs / source.name)
    passed = all(record["passed"] for record in records)
    if passed:
        shutil.copy2(workspace / "chapter5_theory.pdf", ROOT / "chapter5_theory.pdf")
    report = {
        "status": "PASS" if passed else "FAIL",
        "external_workspace": str(workspace),
        "chapter_sha256": hashlib.sha256((ROOT / "chapter5_theory.tex").read_bytes()).hexdigest(),
        "scope": "isolated standalone and original-main-preamble integration; "
                 "not a full-manuscript replacement or mathematical proof check",
        "builds": records,
    }
    (ROOT / "reports/chapter5_build_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "workspace": str(workspace)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
