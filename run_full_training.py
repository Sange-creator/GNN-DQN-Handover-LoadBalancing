#!/usr/bin/env python3
"""Compatibility wrapper for the clean multi-scenario UE-only training config."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
extra = sys.argv[1:]
sys.argv = [
    str(ROOT / "scripts" / "train.py"),
    "--config",
    str(ROOT / "configs" / "experiments" / "multiscenario_ue.json"),
    *extra,
]
runpy.run_path(str(ROOT / "scripts" / "train.py"), run_name="__main__")
