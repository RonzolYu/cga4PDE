#!/usr/bin/env python3
"""Validate the authoritative experiment index and its package-relative paths."""
from pathlib import Path
import csv, json, sys

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "manifest/experiment_index.csv"
PATHS = ROOT / "config/experiment_paths.json"

def main() -> int:
    rows = list(csv.DictReader(INDEX.open(newline="", encoding="utf-8")))
    required = {"case_id", "run_id", "model", "dim", "p", "relu_power", "seed", "target_accepted", "config_path", "status"}
    if not rows or not required.issubset(rows[0]):
        print("FAIL: index schema")
        return 1
    duplicate = len({r["case_id"] for r in rows}) != len(rows)
    missing = [r["config_path"] for r in rows if not (ROOT / r["config_path"]).exists()]
    if duplicate or missing:
        print(json.dumps({"status": "FAIL", "duplicate_case_ids": duplicate, "missing_config_paths": missing}))
        return 1
    print(json.dumps({"status": "PASS", "rows": len(rows), "index": str(INDEX.relative_to(ROOT))}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
