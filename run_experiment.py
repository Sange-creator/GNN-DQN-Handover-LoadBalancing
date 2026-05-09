from __future__ import annotations

import argparse
import time
from pathlib import Path

from handover_gnn_dqn.experiment import (
    attach_improvement_vs_regular,
    default_policy_factories,
    evaluate_policies,
    format_table,
    write_summary_csv,
)
from handover_gnn_dqn.flat_dqn import train_flat_dqn
from handover_gnn_dqn.gnn_dqn import DQNConfig, train_gnn_dqn
from handover_gnn_dqn.simulator import LTEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate GNN-DQN LTE handover optimization.")
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--num-cells", type=int, default=9)
    parser.add_argument("--num-ues", type=int, default=54)
    parser.add_argument("--skip-flat", action="store_true", help="Skip flat-DQN training")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.train_episodes = 5
        args.steps = 30
        args.test_episodes = 3

    lte_cfg = LTEConfig(num_cells=args.num_cells, num_ues=args.num_ues)
    dqn_cfg = DQNConfig(
        epsilon_decay_episodes=max(1, int(0.7 * args.train_episodes)),
    )

    print(f"=== GNN-DQN Handover Optimization ===")
    print(f"Cells: {lte_cfg.num_cells}, UEs: {lte_cfg.num_ues}")
    print(f"Training: {args.train_episodes} episodes × {args.steps} steps")
    from handover_gnn_dqn.simulator import CellularNetworkEnv
    env_tmp = CellularNetworkEnv(lte_cfg)
    print(f"Features per node: {env_tmp.feature_dim} (UE-observable)")
    print(f"Architecture: 3-layer GCN + Dueling DQN")
    print()

    t0 = time.time()
    print("Training GNN-DQN...")
    gnn_agent, gnn_history = train_gnn_dqn(
        lte_cfg,
        dqn_cfg,
        train_episodes=args.train_episodes,
        steps_per_episode=args.steps,
        seed=args.seed,
        verbose=True,
    )
    gnn_time = time.time() - t0
    print(f"GNN-DQN training completed in {gnn_time:.1f}s")

    param_count = sum(p.numel() for p in gnn_agent.parameters())
    print(f"GNN-DQN parameters: {param_count:,}")

    flat_agent = None
    if not args.skip_flat:
        print("\nTraining Flat-DQN (baseline)...")
        t1 = time.time()
        flat_agent, flat_history = train_flat_dqn(
            lte_cfg,
            dqn_cfg,
            train_episodes=args.train_episodes,
            steps_per_episode=args.steps,
            seed=args.seed,
            verbose=True,
        )
        flat_time = time.time() - t1
        print(f"Flat-DQN training completed in {flat_time:.1f}s")
        flat_params = sum(p.numel() for p in flat_agent.parameters())
        print(f"Flat-DQN parameters: {flat_params:,}")

    print(f"\nEvaluating all policies ({args.test_episodes} seeds)...")
    test_seeds = [args.seed + 10_000 + i * 37 for i in range(args.test_episodes)]
    rows = evaluate_policies(
        lte_cfg,
        default_policy_factories(gnn_agent=gnn_agent, flat_agent=flat_agent),
        steps=args.steps,
        seeds=test_seeds,
    )
    rows = sorted(rows, key=lambda r: r["avg_ue_throughput_mbps"], reverse=True)
    write_summary_csv(rows, args.out)

    print()
    print(format_table(rows))
    print(f"\nWrote {args.out}")

    gains = attach_improvement_vs_regular(rows)
    if gains:
        print(f"\n=== GNN-DQN vs Best Traditional ({gains['baseline']}) ===")
        print(f"  Avg throughput:  {gains['avg_throughput_gain_pct']:+.1f}%")
        print(f"  P5 throughput:   {gains['p5_throughput_gain_pct']:+.1f}%")
        print(f"  Load balance:    {gains['load_std_reduction_pct']:+.1f}% lower std")
        if gains.get("gnn_vs_flat_throughput_pct") is not None:
            print(f"\n=== GNN-DQN vs Flat-DQN (proves GNN value) ===")
            print(f"  Throughput:      {gains['gnn_vs_flat_throughput_pct']:+.1f}%")
            print(f"  Load balance:    {gains['gnn_vs_flat_load_std_pct']:+.1f}% lower std")

    total_time = time.time() - t0
    print(f"\nTotal experiment time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
