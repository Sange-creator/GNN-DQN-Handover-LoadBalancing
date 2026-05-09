#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.oran import build_oran_decision
from handover_gnn_dqn.rl import load_gnn_checkpoint, make_env_from_scenario
from handover_gnn_dqn.topology import get_test_scenarios, get_training_scenarios


def scenario_by_name(name: str):
    scenarios = get_training_scenarios() + get_test_scenarios()
    by_name = {s.name: s for s in scenarios}
    if name not in by_name:
        raise ValueError(f"Unknown scenario {name!r}. Choices: {sorted(by_name)}")
    return by_name[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight O-RAN/xApp-style inference demo.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", default="kathmandu_real")
    parser.add_argument("--num-ues", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    agent, meta, _payload = load_gnn_checkpoint(
        args.checkpoint,
        expected_feature_profile="oran_e2",
        expected_feature_dim=15,
    )

    scenario = scenario_by_name(args.scenario)
    env = make_env_from_scenario(scenario, feature_mode="oran_e2", prb_available=True)
    env.reset(args.seed)
    edge_index, edge_weight = env.edge_data

    decisions = []
    for ue_id in range(min(args.num_ues, env.cfg.num_ues)):
        state = env.build_state(ue_id)
        target = agent.act(
            torch.from_numpy(state).float(),
            edge_index,
            edge_weight,
            valid_mask=env.valid_actions(ue_id),
        )
        decision = build_oran_decision(
            ue_id=ue_id,
            serving_cell=int(env.serving[ue_id]),
            target_cell=target,
            state=state,
        )
        decisions.append(asdict(decision))

    print(json.dumps({"scenario": scenario.name, "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
