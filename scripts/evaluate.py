#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.rl.training import evaluate_and_write, load_gnn_checkpoint, son_config_from_dict
from handover_gnn_dqn.topology import (
    get_stress_scenarios,
    get_test_scenarios,
    get_training_scenarios,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved GNN-DQN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--son-max-step", type=float, default=None, help="Override max_cio_step_db in SON config")
    parser.add_argument("--son-update-interval", type=int, default=None, help="Override update_interval_steps in SON config")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenario names to evaluate (filter from split)")
    parser.add_argument(
        "--split",
        choices=["train", "test", "stress", "all", "all_plus_stress"],
        default="all",
        help=(
            "Which scenario set to evaluate against. "
            "'stress' = high-congestion eval-only scenarios; "
            "'all_plus_stress' = train+test+stress."
        ),
    )
    args = parser.parse_args()

    agent, meta, _payload = load_gnn_checkpoint(args.checkpoint)
    cfg = meta["config"]
    son_cfg_dict = dict(cfg.get("son_config", {}))
    if args.son_max_step is not None:
        son_cfg_dict["max_cio_step_db"] = args.son_max_step
    if args.son_update_interval is not None:
        son_cfg_dict["update_interval_steps"] = args.son_update_interval
    son_cfg = son_config_from_dict(son_cfg_dict)

    seed = int(cfg.get("seed", 42))
    scenarios = []
    if args.split in {"train", "all", "all_plus_stress"}:
        scenarios.extend(get_training_scenarios(seed=seed))
    if args.split in {"test", "all", "all_plus_stress"}:
        scenarios.extend(get_test_scenarios(seed=seed + 57))
    if args.split in {"stress", "all_plus_stress"}:
        scenarios.extend(get_stress_scenarios(seed=seed + 137))
    if args.scenarios is not None:
        keep = {s.strip() for s in args.scenarios.split(",")}
        scenarios = [s for s in scenarios if s.name in keep]
        if not scenarios:
            print(f"ERROR: none of {keep} matched any scenario in split '{args.split}'")
            sys.exit(1)

    seeds = [seed + 20_000 + i * 37 for i in range(args.seeds)]
    evaluate_and_write(
        scenarios,
        agent,
        args.out_dir,
        feature_mode=cfg.get("feature_mode", "ue_only"),
        prb_available=bool(cfg.get("prb_available", cfg.get("feature_mode") != "ue_only")),
        steps=args.steps,
        seeds=seeds,
        son_config=son_cfg,
    )
    print(f"Wrote evaluation CSVs: {args.out_dir}")


if __name__ == "__main__":
    main()
