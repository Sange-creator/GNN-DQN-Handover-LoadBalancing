"""Full multi-scenario training: GNN-DQN learns from ALL environment types.

The model trains on a random mix of:
  - Dense urban (Lakeside/Kathmandu)
  - Highway (fast mobility)
  - Suburban (medium)
  - Sparse rural (hills)
  - Overloaded events (festivals)
  - Real Pokhara topology

Then tested on UNSEEN scenarios (Kathmandu, Dharan, unknown grid).

Run with:
    caffeinate -i python3 -u run_full_training.py

Expected time: ~4-6 hours on M1 Pro.
"""
from __future__ import annotations

import copy
import time
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from handover_gnn_dqn.gnn_dqn import DQNConfig, GnnDQNAgent, ReplayBuffer, _train_step
from handover_gnn_dqn.flat_dqn import FlatDQNAgent, _flat_train_step
from handover_gnn_dqn.simulator import LTEConfig, CellularNetworkEnv, adjacency_to_edge_index
from handover_gnn_dqn.topology import build_adjacency_from_positions
from handover_gnn_dqn.scenarios import get_training_scenarios, get_test_scenarios, Scenario
from handover_gnn_dqn.experiment import (
    default_policy_factories,
    evaluate_policies,
    format_table,
    write_summary_csv,
    attach_improvement_vs_regular,
)


def make_env_from_scenario(scenario: Scenario) -> CellularNetworkEnv:
    """Create an environment from a scenario definition."""
    cfg = LTEConfig(
        num_cells=scenario.num_cells,
        num_ues=scenario.num_ues,
        area_m=scenario.area_m,
        cell_positions=scenario.cell_positions,
        min_speed_mps=scenario.min_speed_mps,
        max_speed_mps=scenario.max_speed_mps,
    )
    return CellularNetworkEnv(cfg)


def train_multi_scenario(
    scenarios: List[Scenario],
    dqn_cfg: DQNConfig,
    total_episodes: int = 120,
    steps_per_episode: int = 80,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Train GNN-DQN across multiple scenarios (random cycling)."""

    rng = np.random.default_rng(seed)
    history: List[dict] = []
    replay = ReplayBuffer(dqn_cfg.replay_capacity)
    decisions = 0

    # Use the largest cell count to set model size
    max_cells = max(s.num_cells for s in scenarios)

    # Initialize agent with max cell count
    sample_env = make_env_from_scenario(scenarios[0])
    feature_dim = sample_env.feature_dim

    agent = GnnDQNAgent(max_cells, feature_dim, dqn_cfg, seed=seed)
    target_net = copy.deepcopy(agent)
    target_net.load_state_dict(agent.state_dict())
    optimizer = torch.optim.Adam(agent.parameters(), lr=dqn_cfg.learning_rate)
    best_reward = float("-inf")
    best_state = copy.deepcopy(agent.state_dict())
    best_episode = 0

    print(f"  Model: {sum(p.numel() for p in agent.parameters()):,} parameters")
    print(f"  Max cells: {max_cells}, Features: {feature_dim}")
    print(f"  Scenarios: {[s.name for s in scenarios]}")
    print()

    for episode in range(total_episodes):
        # Pick a random scenario each episode
        scenario = scenarios[rng.integers(len(scenarios))]
        env = make_env_from_scenario(scenario)

        # Rebuild agent for this scenario's cell count if needed
        num_cells = scenario.num_cells
        env.reset(seed + 101 * episode)

        edge_index, edge_weight = env.edge_data
        losses: List[float] = []
        episode_reward = 0.0

        frac = min(episode / max(dqn_cfg.epsilon_decay_episodes, 1), 1.0)
        epsilon = dqn_cfg.epsilon_start + frac * (dqn_cfg.epsilon_end - dqn_cfg.epsilon_start)

        # Build padded edge_index / edge_weight once per episode (graph is static
            # within an episode). Padded nodes are isolated (no self-loops, no edges)
            # so the GCN's internal normalization leaves their embeddings as zeros
            # rather than mixing them into real nodes.
        if num_cells < max_cells:
            ei_full = edge_index
            ew_full = edge_weight
        else:
            ei_full = edge_index
            ew_full = edge_weight

        def pad_state(s: np.ndarray) -> np.ndarray:
            if num_cells >= max_cells:
                return s
            pad = np.zeros((max_cells - num_cells, feature_dim), dtype=np.float32)
            return np.vstack([s, pad])

        def pad_mask(m: np.ndarray) -> np.ndarray:
            if num_cells >= max_cells:
                return m.astype(bool)
            out = np.zeros(max_cells, dtype=bool)
            out[:num_cells] = m
            return out

        for _step in range(steps_per_episode):
            env.advance_mobility()
            for ue_idx in rng.permutation(env.cfg.num_ues):
                state = env.build_state(int(ue_idx))
                state_padded = pad_state(state)
                state_t = torch.from_numpy(state_padded).float()

                # Mask: padded slots are invalid. epsilon-greedy and the Q-argmax
                # both respect this so padded nodes can never be chosen.
                valid = env.valid_actions(int(ue_idx))
                valid_padded = pad_mask(valid)

                action = agent.act(state_t, ei_full, ew_full,
                                   epsilon=epsilon, valid_mask=valid_padded, rng=rng)
                # With the mask above, action must already be a real cell index,
                # but keep a defensive assert so a bug surfaces instead of
                # silently remapping to cell (num_cells-1).
                assert 0 <= action < num_cells, (
                    f"padded action {action} leaked through mask "
                    f"(num_cells={num_cells}, max_cells={max_cells})"
                )

                next_state, reward, done, _info = env.step_user_action(int(ue_idx), action)
                next_state_padded = pad_state(next_state)
                next_valid = env.valid_actions(int(ue_idx))
                next_valid_padded = pad_mask(next_valid)

                replay.add((state_padded, action, reward, next_state_padded, done, next_valid_padded))
                episode_reward += reward

                if len(replay) >= dqn_cfg.batch_size and decisions % dqn_cfg.train_every == 0:
                    batch = replay.sample(rng, dqn_cfg.batch_size)
                    losses.append(_train_step(agent, target_net, batch, ei_full, ew_full, optimizer, dqn_cfg))
                if decisions % dqn_cfg.target_update_every == 0:
                    target_net.load_state_dict(agent.state_dict())
                decisions += 1

        metrics = env.metrics()
        metrics["episode"] = float(episode + 1)
        metrics["epsilon"] = float(epsilon)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["episode_reward"] = float(episode_reward)
        metrics["scenario"] = scenario.name
        history.append(metrics)

        if verbose and (episode + 1) % max(1, total_episodes // 20) == 0:
            print(
                f"[{episode + 1:04d}/{total_episodes}] "
                f"{scenario.name:16s} "
                f"eps={epsilon:.3f} "
                f"thr={metrics['avg_ue_throughput_mbps']:.2f} "
                f"load_std={metrics['load_std']:.2f} "
                f"loss={metrics['loss']:.4f}"
            )

        # Track best-performing episode (after initial exploration).
        if episode >= max(5, total_episodes // 10) and episode_reward > best_reward:
            best_reward = episode_reward
            best_state = copy.deepcopy(agent.state_dict())
            best_episode = episode + 1

    # Restore best checkpoint.
    if best_episode > 0:
        agent.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best checkpoint from episode {best_episode} "
                  f"(reward={best_reward:.1f})")
    target_net.load_state_dict(agent.state_dict())
    return agent, history, max_cells


def main():
    out_dir = Path("results/full_training")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  GNN-DQN MULTI-SCENARIO TRAINING")
    print("  Trains on: dense urban, highway, suburban, rural, events, Pokhara")
    print("  Tests on: Kathmandu, Dharan, unknown grid (never seen)")
    print("=" * 65)
    print()

    train_scenarios = get_training_scenarios(seed=42)
    test_scenarios = get_test_scenarios(seed=99)

    print("TRAINING SCENARIOS:")
    for s in train_scenarios:
        print(f"  {s.name:20s} | {s.num_cells} cells | {s.num_ues} UEs | {s.description}")
    print()
    print("TEST SCENARIOS (unseen):")
    for s in test_scenarios:
        print(f"  {s.name:20s} | {s.num_cells} cells | {s.num_ues} UEs | {s.description}")
    print()

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
        epsilon_decay_episodes=84,  # 70% of 120 total episodes
    )

    # --- TRAIN ---
    print("Phase 1: Multi-scenario GNN-DQN Training (120 episodes)...")
    t0 = time.time()
    gnn_agent, gnn_history, max_cells = train_multi_scenario(
        train_scenarios, dqn_cfg,
        total_episodes=120,
        steps_per_episode=80,
        seed=42,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"\nTraining done in {train_time/3600:.1f} hours")

    torch.save(gnn_agent.state_dict(), out_dir / "gnn_dqn_multiscenario.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(gnn_history, f, indent=2)

    # --- EVALUATE ON TRAINING SCENARIOS ---
    print("\n\nPhase 2: Evaluating on training scenarios (5 seeds each)...")
    for scenario in train_scenarios:
        cfg = LTEConfig(
            num_cells=scenario.num_cells,
            num_ues=scenario.num_ues,
            area_m=scenario.area_m,
            cell_positions=scenario.cell_positions,
            min_speed_mps=scenario.min_speed_mps,
            max_speed_mps=scenario.max_speed_mps,
        )
        seeds = [42 + 10_000 + i * 37 for i in range(5)]
        rows = evaluate_policies(
            cfg,
            default_policy_factories(gnn_agent=gnn_agent),
            steps=80,
            seeds=seeds,
        )
        rows = sorted(rows, key=lambda r: r["avg_ue_throughput_mbps"], reverse=True)
        write_summary_csv(rows, out_dir / f"results_{scenario.name}.csv")
        print(f"\n  === {scenario.name} ===")
        print(format_table(rows))

    # --- EVALUATE ON TEST SCENARIOS (GENERALIZATION) ---
    print("\n\nPhase 3: Generalization test (unseen scenarios)...")
    for scenario in test_scenarios:
        cfg = LTEConfig(
            num_cells=scenario.num_cells,
            num_ues=scenario.num_ues,
            area_m=scenario.area_m,
            cell_positions=scenario.cell_positions,
            min_speed_mps=scenario.min_speed_mps,
            max_speed_mps=scenario.max_speed_mps,
        )
        seeds = [99 + 10_000 + i * 37 for i in range(5)]
        rows = evaluate_policies(
            cfg,
            default_policy_factories(gnn_agent=gnn_agent),
            steps=80,
            seeds=seeds,
        )
        rows = sorted(rows, key=lambda r: r["avg_ue_throughput_mbps"], reverse=True)
        write_summary_csv(rows, out_dir / f"generalization_{scenario.name}.csv")
        print(f"\n  === {scenario.name} (UNSEEN) ===")
        print(format_table(rows))

    total_time = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  TOTAL TIME: {total_time/3600:.1f} hours")
    print(f"  Results: {out_dir}/")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
