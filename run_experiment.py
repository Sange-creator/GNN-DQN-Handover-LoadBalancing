#!/usr/bin/env python3
"""Compatibility wrapper for quick smoke training.

Use `scripts/train.py --config configs/experiments/<name>.json` for new runs.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Legacy wrapper around scripts/train.py")
parser.add_argument("--config", type=Path)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--train-episodes", type=int)
parser.add_argument("--steps", type=int)
parser.add_argument("--test-episodes", type=int)
args, unknown = parser.parse_known_args()

if unknown:
    raise SystemExit(f"Unsupported legacy arguments: {unknown}")

config_path = args.config or (ROOT / "configs" / "experiments" / "smoke_ue.json")
if args.train_episodes is not None or args.steps is not None or args.test_episodes is not None:
    cfg = json.loads(config_path.read_text())
    cfg["run_name"] = "legacy_run_experiment"
    cfg["out_dir"] = "results/runs/legacy_run_experiment"
    if args.train_episodes is not None:
        cfg["episodes"] = args.train_episodes
    if args.steps is not None:
        cfg["steps_per_episode"] = args.steps
    if args.test_episodes is not None:
        cfg["eval_seeds"] = args.test_episodes
    generated = ROOT / "results" / "runs" / "legacy_run_experiment" / "config.generated.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(json.dumps(cfg, indent=2))
    config_path = generated

sys.argv = [
    str(ROOT / "scripts" / "train.py"),
    "--config",
    str(config_path),
]
runpy.run_path(str(ROOT / "scripts" / "train.py"), run_name="__main__")
