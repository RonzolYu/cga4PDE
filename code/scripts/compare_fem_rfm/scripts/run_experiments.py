#!/usr/bin/env python3
"""Run one or all experiment families from a source checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "_vendor")]

from compare_fem_rfm.experiment import main


if __name__ == "__main__":
    main()

