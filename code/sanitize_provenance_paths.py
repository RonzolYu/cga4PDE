#!/usr/bin/env python3
"""Make copied provenance records portable without changing numerical fields.

The submission package contains snapshots copied from several workstation
directories.  This utility rewrites only path-valued strings in JSON/CSV
records.  External locations become explicit ``source_archive/...`` labels;
paths inside the package remain package-relative.  It is deliberately
append-free and can be rerun safely.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
_UNIX_HOME = "/" + "Users/"
_DRIVE_ROOT = r"[A-Z]:" + re.escape("\\")
ABSOLUTE = re.compile(
    r"(?:" + re.escape(_UNIX_HOME) + r"[^,;\"\s]+|" + _DRIVE_ROOT + r"[^,;\"\s]+)"
)


def portable_path(value: str) -> str:
    """Map a workstation path to a stable package-relative provenance label."""
    if not ABSOLUTE.search(value):
        return value
    text = value.replace("\\", "/")
    # Preserve useful experiment provenance while avoiding machine names.
    markers = (
        "/revision/baseline_campaign/",
        "/revision/cost_campaign/",
        "/revision/data/raw/",
        "/revision/data/derived/",
        "/reexp_small/",
    )
    for marker in markers:
        if marker in text:
            suffix = text.split(marker, 1)[1].lstrip("/")
            prefix = marker.strip("/").replace("/", "_")
            if marker == "/revision/data/raw/":
                prefix = "data_raw"
            elif marker == "/revision/data/derived/":
                prefix = "data_derived"
            elif marker == "/reexp_small/":
                prefix = "reexp_small"
            return f"source_archive/{prefix}/{suffix}"
    # Executable paths are environment metadata, not source files.
    if value.endswith("/python") or value.endswith("/python3"):
        return "python"
    return "source_archive/" + text.rstrip("/").split("/")[-1]


def rewrite_json_value(value: object) -> object:
    if isinstance(value, str):
        return portable_path(value)
    if isinstance(value, list):
        return [rewrite_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_json_value(item) for key, item in value.items()}
    return value


def rewrite_json(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    rewritten = rewrite_json_value(data)
    if rewritten == data:
        return False
    path.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def rewrite_csv(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return False
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    changed = False
    for row in rows:
        for key in fieldnames:
            value = row.get(key, "")
            rewritten = ABSOLUTE.sub(lambda match: portable_path(match.group(0)), value)
            if rewritten != value:
                row[key] = rewritten
                changed = True
    if not changed:
        return False
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def main() -> None:
    changed = []
    for path in sorted(ROOT.rglob("*.json")):
        if ".pytest_cache" in path.parts:
            continue
        if rewrite_json(path):
            changed.append(path.relative_to(ROOT).as_posix())
    for path in sorted(ROOT.rglob("*.csv")):
        if ".pytest_cache" in path.parts:
            continue
        if rewrite_csv(path):
            changed.append(path.relative_to(ROOT).as_posix())
    print(f"rewrote {len(changed)} provenance files")
    for item in changed:
        print(item)


if __name__ == "__main__":
    main()
