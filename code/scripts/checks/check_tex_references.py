"""Static checks for the V3C English manuscript dependency tree."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


_BASE = Path(__file__).resolve().parents[2]
if (_BASE / "manuscript" / "main.tex").exists():
    ROOT = _BASE
elif (_BASE.parent / "main.tex").exists():
    ROOT = _BASE.parent
else:
    raise RuntimeError("cannot locate manuscript root")
MANUSCRIPT = ROOT / "manuscript" if (ROOT / "manuscript" / "main.tex").exists() else ROOT
MAIN = MANUSCRIPT / "main.tex"
REPORT = ROOT / "reports" / "cross_reference_check.md"

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|cref|Cref|autoref)\{([^}]+)\}")
CITE_RE = re.compile(r"\\(?:cite|citep|citet|Cite|parencite|textcite)\{([^}]+)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
BIB_ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)")


def resolve_input(name: str) -> Path:
    path = (MANUSCRIPT / name).with_suffix(".tex")
    return path.resolve()


def dependency_tree() -> list[Path]:
    seen: set[Path] = set()
    queue = [MAIN.resolve()]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix != ".tex" or not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for name in INPUT_RE.findall(text):
            queue.append(resolve_input(name))
    return sorted(seen)


def graphics_paths() -> list[Path]:
    return sorted(path for path in (MANUSCRIPT / "figures").iterdir() if path.is_dir())


def bibliography() -> Path:
    text = MAIN.read_text(encoding="utf-8")
    match = re.search(r"\\bibliography\{([^}]+)\}", text)
    if not match:
        raise RuntimeError("no bibliography declaration")
    return (MANUSCRIPT / match.group(1)).with_suffix(".bib")


def main() -> None:
    files = dependency_tree()
    labels: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    references: list[tuple[Path, int, str]] = []
    citations: list[tuple[Path, int, str]] = []
    graphics: list[tuple[Path, int, str]] = []

    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines, 1):
            for label in LABEL_RE.findall(line):
                labels[label].append((path, index))
            for ref in REF_RE.findall(line):
                for item in ref.split(","):
                    references.append((path, index, item.strip()))
            for cite in CITE_RE.findall(line):
                for item in cite.split(","):
                    citations.append((path, index, item.strip()))
            for graphic in GRAPHIC_RE.findall(line):
                graphics.append((path, index, graphic))

    bib_path = bibliography()
    bib_keys = set(BIB_ENTRY_RE.findall(bib_path.read_text(encoding="utf-8")))
    duplicate_labels = {key: value for key, value in labels.items() if len(value) > 1}
    undefined_refs = [(path, line, key) for path, line, key in references if key not in labels]
    undefined_cites = [(path, line, key) for path, line, key in citations if key not in bib_keys]

    search_dirs = graphics_paths()
    missing_graphics = []
    for path, line, name in graphics:
        candidates = [(directory / name).resolve() for directory in search_dirs]
        candidates.append((MANUSCRIPT / "figures" / name).resolve())
        if not any(candidate.exists() for candidate in candidates):
            missing_graphics.append((path, line, name))

    report = [
        "# Cross-reference and dependency check",
        "",
        f"- Manuscript dependency files: {len(files)}",
        f"- Labels: {len(labels)}",
        f"- References: {len(references)}",
        f"- Citations: {len(citations)}",
        f"- Graphics: {len(graphics)}",
        f"- Duplicate labels: {len(duplicate_labels)}",
        f"- Undefined references: {len(undefined_refs)}",
        f"- Undefined citations: {len(undefined_cites)}",
        f"- Missing graphics: {len(missing_graphics)}",
        "",
        "## Details",
        "",
    ]
    for name, entries in sorted(duplicate_labels.items()):
        locations = ", ".join(f"{path.relative_to(ROOT)}:{line}" for path, line in entries)
        report.append(f"- duplicate label `{name}`: {locations}")
    for path, line, key in undefined_refs:
        report.append(f"- undefined reference `{key}` at {path.relative_to(ROOT)}:{line}")
    for path, line, key in undefined_cites:
        report.append(f"- undefined citation `{key}` at {path.relative_to(ROOT)}:{line}")
    for path, line, name in missing_graphics:
        report.append(f"- missing graphic `{name}` at {path.relative_to(ROOT)}:{line}")
    if len(report) == 12:
        report.append("- No static dependency problems found.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    status = {
        "files": len(files),
        "labels": len(labels),
        "references": len(references),
        "citations": len(citations),
        "graphics": len(graphics),
        "duplicate_labels": {key: [f"{p.relative_to(ROOT)}:{n}" for p, n in value]
                             for key, value in duplicate_labels.items()},
        "undefined_references": [f"{p.relative_to(ROOT)}:{n}:{key}" for p, n, key in undefined_refs],
        "undefined_citations": [f"{p.relative_to(ROOT)}:{n}:{key}" for p, n, key in undefined_cites],
        "missing_graphics": [f"{p.relative_to(ROOT)}:{n}:{name}" for p, n, name in missing_graphics],
        "passed": not (duplicate_labels or undefined_refs or undefined_cites or missing_graphics),
    }
    (ROOT / "data" / "derived" / "cross_reference_status.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "derived" / "cross_reference_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    if not status["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
