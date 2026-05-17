"""Test whether SON is the bottleneck between training performance and deployment.

Compares raw gnn_dqn (direct cell selection) vs son_gnn_dqn (CIO-mediated)
using the current best checkpoint, across multiple SON configurations.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from pathlib import Path
from handover_gnn_dqn.rl.training import load_gnn_checkpoint, son_config_from_dict, make_env_from_scenario
from handover_gnn_dqn.metrics import default_policy_factories, evaluate_policies
from handover_gnn_dqn.son import SONConfig
from handover_gnn_dqn.topology.scenarios import get_training_scenarios

CHECKPOINT = Path("results/runs/highway_sonv3/checkpoints/resume/resume_ep0600.pt")
SCENARIOS = ["dense_urban", "overloaded_event", "highway", "suburban"]
SEEDS = 3
STEPS = 60

# SON configs to test
SON_CONFIGS = {
    "training_default": SONConfig(
        max_cio_step_db=2.0, update_interval_steps=6,
        cio_min_db=-6.0, cio_max_db=6.0,
        preference_threshold=0.10, rollback_pingpong_floor=0.01,
        rollback_throughput_drop_frac=0.30,
    ),
    "fast_updates": SONConfig(
        max_cio_step_db=2.0, update_interval_steps=2,
        cio_min_db=-6.0, cio_max_db=6.0,
        preference_threshold=0.08, rollback_pingpong_floor=0.01,
        rollback_throughput_drop_frac=0.30,
    ),
    "instant_cio": SONConfig(
        max_cio_step_db=6.0, update_interval_steps=2,
        cio_min_db=-6.0, cio_max_db=6.0,
        preference_threshold=0.05, rollback_pingpong_floor=0.02,
        rollback_throughput_drop_frac=0.50,
    ),
    "no_rollback": SONConfig(
        max_cio_step_db=6.0, update_interval_steps=2,
        cio_min_db=-6.0, cio_max_db=6.0,
        preference_threshold=0.05, rollback_pingpong_floor=1.0,
        rollback_throughput_drop_frac=1.0,
    ),
}


def run_test():
    print("=" * 70)
    print("SON BOTTLENECK TEST")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Seeds: {SEEDS}, Steps: {STEPS}")
    print()

    agent, metadata, _ = load_gnn_checkpoint(CHECKPOINT, strict_metadata=False)
    all_scenarios = get_training_scenarios()
    scenario_map = {s.name: s for s in all_scenarios}

    for scenario_name in SCENARIOS:
        print(f"\n{'='*50}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*50}")

        scenario = scenario_map[scenario_name]
        env = make_env_from_scenario(scenario, feature_mode="ue_only", prb_available=False)

        # Test each SON config
        results = {}
        for config_name, son_cfg in SON_CONFIGS.items():
            policies = default_policy_factories(
                gnn_agent=agent, son_config=son_cfg, include_true_prb=False
            )
            rows = evaluate_policies(env.cfg, policies, steps=STEPS, seeds=list(range(SEEDS)))
            by_method = {r["method"]: r for r in rows}
            results[config_name] = by_method

        # Print comparison table
        a3 = results["training_default"].get("a3_ttt", {})
        a3_thr = a3.get("avg_ue_throughput_mbps", 0)
        a3_p5 = a3.get("p5_ue_throughput_mbps", 0)

        print(f"\n  A3-TTT baseline: thr={a3_thr:.3f}, p5={a3_p5:.3f}")
        print(f"\n  {'Config':<20} {'thr':>8} {'vs A3':>8} {'p5':>8} {'vs A3':>8} {'pp':>8} {'load_std':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        # Raw gnn_dqn (no SON)
        raw = results["training_default"].get("gnn_dqn", {})
        if raw:
            raw_thr = raw.get("avg_ue_throughput_mbps", 0)
            raw_p5 = raw.get("p5_ue_throughput_mbps", 0)
            raw_pp = raw.get("pingpong_rate", 0)
            raw_ls = raw.get("load_std", 0)
            margin = ((raw_thr - a3_thr) / max(a3_thr, 1e-6)) * 100
            p5_margin = ((raw_p5 - a3_p5) / max(a3_p5, 1e-6)) * 100
            print(f"  {'RAW gnn_dqn':<20} {raw_thr:>8.3f} {margin:>+7.1f}% {raw_p5:>8.3f} {p5_margin:>+7.1f}% {raw_pp:>8.3f} {raw_ls:>8.3f}")

        # Each SON config
        for config_name in SON_CONFIGS:
            son = results[config_name].get("son_gnn_dqn", {})
            if son:
                son_thr = son.get("avg_ue_throughput_mbps", 0)
                son_p5 = son.get("p5_ue_throughput_mbps", 0)
                son_pp = son.get("pingpong_rate", 0)
                son_ls = son.get("load_std", 0)
                margin = ((son_thr - a3_thr) / max(a3_thr, 1e-6)) * 100
                p5_margin = ((son_p5 - a3_p5) / max(a3_p5, 1e-6)) * 100
                print(f"  {config_name:<20} {son_thr:>8.3f} {margin:>+7.1f}% {son_p5:>8.3f} {p5_margin:>+7.1f}% {son_pp:>8.3f} {son_ls:>8.3f}")

    print("\n\nINTERPRETATION:")
    print("  - If RAW gnn_dqn >> a3_ttt but all SON configs ≈ a3_ttt: SON IS the bottleneck")
    print("  - If instant_cio/no_rollback >> training_default: SON params too conservative")
    print("  - If RAW gnn_dqn ≈ a3_ttt: model itself isn't learning useful preferences")


if __name__ == "__main__":
    run_test()
