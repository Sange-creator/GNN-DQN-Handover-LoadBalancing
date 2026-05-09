"""Overnight training script: GNN-DQN on real Pokhara topology (75 cells).

Run with:
    caffeinate -i python3 run_overnight.py

Expected time: 5-8 hours on M1 Pro.
Results saved to: results/overnight/
"""
from __future__ import annotations

import time
import json
from pathlib import Path

import numpy as np
import torch

from handover_gnn_dqn.gnn_dqn import DQNConfig, GnnDQNAgent, train_gnn_dqn
from handover_gnn_dqn.flat_dqn import train_flat_dqn
from handover_gnn_dqn.topology import build_adjacency_from_positions, get_area_size
from handover_gnn_dqn.simulator import LTEConfig, CellularNetworkEnv
from handover_gnn_dqn.experiment import (
    default_policy_factories,
    evaluate_policies,
    format_table,
    write_summary_csv,
    attach_improvement_vs_regular,
)


def main():
    out_dir = Path("results/overnight")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load real Pokhara positions, use 15-cell subset for training
    all_positions = np.load("data/opencellid/pokhara_positions_clean.npy")
    rng_select = np.random.default_rng(42)
    idx = rng_select.choice(len(all_positions), size=15, replace=False)
    positions = all_positions[idx]
    area = get_area_size(positions)
    num_cells = len(positions)
    num_ues = num_cells * 6  # 6 UEs per cell = 90 UEs

    print("=" * 60)
    print("  GNN-DQN TRAINING")
    print("  Real Pokhara Topology (15 cells, 90 UEs)")
    print("=" * 60)
    print(f"  Cells: {num_cells}")
    print(f"  UEs: {num_ues}")
    print(f"  Area: {area:.0f} m")
    print(f"  Training: 100 episodes × 80 steps")
    print(f"  Evaluation: 10 seeds")
    print(f"  Estimated time: ~4.5 hours")
    print("=" * 60)
    print()

    lte_cfg = LTEConfig(
        num_cells=num_cells,
        num_ues=num_ues,
        area_m=area,
        cell_positions=positions,
    )
    dqn_cfg = DQNConfig(
        hidden_dim=128,
        num_gcn_layers=3,
        gamma=0.95,
        learning_rate=3e-4,
        batch_size=64,
        replay_capacity=100_000,
        train_every=4,
        target_update_every=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=80,
    )

    # --- TRAIN GNN-DQN ---
    print("Phase 1: Training GNN-DQN...")
    t0 = time.time()
    gnn_agent, gnn_history = train_gnn_dqn(
        lte_cfg,
        dqn_cfg,
        train_episodes=100,
        steps_per_episode=80,
        seed=42,
        verbose=True,
    )
    gnn_time = time.time() - t0
    print(f"\nGNN-DQN training done in {gnn_time/3600:.1f} hours")
    print(f"Parameters: {sum(p.numel() for p in gnn_agent.parameters()):,}")

    # Save model
    torch.save(gnn_agent.state_dict(), out_dir / "gnn_dqn_pokhara.pt")
    print(f"Model saved to {out_dir / 'gnn_dqn_pokhara.pt'}")

    # Save training history
    with open(out_dir / "gnn_history.json", "w") as f:
        json.dump(gnn_history, f, indent=2)

    # --- TRAIN FLAT-DQN (baseline) ---
    print("\n\nPhase 2: Training Flat-DQN (baseline)...")
    t1 = time.time()
    flat_agent, flat_history = train_flat_dqn(
        lte_cfg,
        dqn_cfg,
        train_episodes=100,
        steps_per_episode=80,
        seed=42,
        verbose=True,
    )
    flat_time = time.time() - t1
    print(f"\nFlat-DQN training done in {flat_time/3600:.1f} hours")

    torch.save(flat_agent.state_dict(), out_dir / "flat_dqn_pokhara.pt")

    with open(out_dir / "flat_history.json", "w") as f:
        json.dump(flat_history, f, indent=2)

    # --- EVALUATE ALL METHODS ---
    print("\n\nPhase 3: Evaluating all 6 methods (10 seeds)...")
    test_seeds = [42 + 10_000 + i * 37 for i in range(10)]
    rows = evaluate_policies(
        lte_cfg,
        default_policy_factories(gnn_agent=gnn_agent, flat_agent=flat_agent),
        steps=80,
        seeds=test_seeds,
    )
    rows = sorted(rows, key=lambda r: r["avg_ue_throughput_mbps"], reverse=True)
    write_summary_csv(rows, out_dir / "summary.csv")

    print()
    print(format_table(rows))

    gains = attach_improvement_vs_regular(rows)
    if gains:
        print(f"\n=== GNN-DQN vs Best Traditional ({gains['baseline']}) ===")
        print(f"  Avg throughput:  {gains['avg_throughput_gain_pct']:+.1f}%")
        print(f"  P5 throughput:   {gains['p5_throughput_gain_pct']:+.1f}%")
        print(f"  Load balance:    {gains['load_std_reduction_pct']:+.1f}% lower std")
        if gains.get("gnn_vs_flat_throughput_pct") is not None:
            print(f"\n=== GNN-DQN vs Flat-DQN (GNN value proof) ===")
            print(f"  Throughput:      {gains['gnn_vs_flat_throughput_pct']:+.1f}%")
            print(f"  Load balance:    {gains['gnn_vs_flat_load_std_pct']:+.1f}% lower std")

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  TOTAL TIME: {total_time/3600:.1f} hours")
    print(f"  Results: {out_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
