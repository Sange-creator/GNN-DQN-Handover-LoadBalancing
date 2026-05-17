import argparse
import logging
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.handover_gnn_dqn.rl.training import evaluate_and_write, load_gnn_checkpoint, son_config_from_dict
from src.handover_gnn_dqn.topology import get_stress_scenarios, get_test_scenarios, get_training_scenarios

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved GNN-DQN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    agent, meta, _payload = load_gnn_checkpoint(args.checkpoint)
    cfg = meta["config"]
    son_cfg = son_config_from_dict(dict(cfg.get("son_config", {})))

    seed = int(cfg.get("seed", 42))
    scenarios = []
    scenarios.extend(get_training_scenarios(seed=seed))
    scenarios.extend(get_test_scenarios(seed=seed + 57))
    scenarios.extend(get_stress_scenarios(seed=seed + 137))
    
    seeds = [seed + 20_000 + i * 37 for i in range(args.seeds)]
    
    # We monkey-patch the default_policy_factories to only return the 4 we care about to save time.
    import src.handover_gnn_dqn.metrics.experiment as exp
    orig_factories = exp.default_policy_factories
    def targeted_factories(*fargs, **fkwargs):
        all_pols = orig_factories(*fargs, **fkwargs)
        return {k: v for k, v in all_pols.items() if k in ["a3_ttt", "gnn_dqn", "son_gnn_dqn", "adaptive_son_gnn_dqn"]}
    exp.default_policy_factories = targeted_factories

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
