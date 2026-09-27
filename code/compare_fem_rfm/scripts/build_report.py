#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "_vendor")]

from compare_fem_rfm.report import build_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    build_report(not args.no_compile)

