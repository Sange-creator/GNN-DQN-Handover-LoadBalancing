#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.rl.training import evaluate_and_write, load_gnn_checkpoint
from handover_gnn_dqn.topology import get_test_scenarios, get_training_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved GNN-DQN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    args = parser.parse_args()

    agent, meta, _payload = load_gnn_checkpoint(args.checkpoint)
    cfg = meta["config"]

    seed = int(cfg.get("seed", 42))
    scenarios = []
    if args.split in {"train", "all"}:
        scenarios.extend(get_training_scenarios(seed=seed))
    if args.split in {"test", "all"}:
        scenarios.extend(get_test_scenarios(seed=seed + 57))

    seeds = [seed + 20_000 + i * 37 for i in range(args.seeds)]
    evaluate_and_write(
        scenarios,
        agent,
        args.out_dir,
        feature_mode=cfg.get("feature_mode", "ue_only"),
        prb_available=bool(cfg.get("prb_available", cfg.get("feature_mode") != "ue_only")),
        steps=args.steps,
        seeds=seeds,
    )
    print(f"Wrote evaluation CSVs: {args.out_dir}")


if __name__ == "__main__":
    main()
