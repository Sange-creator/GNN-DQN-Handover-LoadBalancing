"""SON-in-the-loop training: agent learns preferences that work THROUGH SON.

Key difference from standard training:
- Agent outputs per-UE cell preferences
- SON aggregates preferences → CIO/TTT adjustments
- A3 event decides actual handovers based on CIO-adjusted RSRP
- Reward is network-level (throughput, fairness, pp) AFTER SON+A3 execute
- Agent learns what preferences make SON produce good CIO decisions

This closes the training-deployment gap that caused previous runs to fail:
- Old: agent learns direct HO (+40% on overloaded) but SON can't translate
- New: agent learns SON-compatible preferences that produce real gains
"""
from __future__ import annotations

import sys, json, time, copy
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from handover_gnn_dqn.env import CellularNetworkEnv, LTEConfig
from handover_gnn_dqn.models.gnn_dqn import GnnDQNAgent, DQNConfig
from handover_gnn_dqn.son import SONConfig
from handover_gnn_dqn.son.controller import SONController
from handover_gnn_dqn.rl.training import (
    load_gnn_checkpoint, make_env_from_scenario, son_config_from_dict,
    dqn_config_from_dict,
)
from handover_gnn_dqn.topology.scenarios import get_training_scenarios, get_test_scenarios
from handover_gnn_dqn.metrics import default_policy_factories, evaluate_policies

DEVICE = torch.device("cpu")


@dataclass
class SONLoopConfig:
    episodes: int = 600
    steps_per_episode: int = 80
    son_update_interval: int = 4
    num_ues_cap: int = 80
    epsilon_start: float = 0.25
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 400
    lr: float = 3e-4
    lr_min: float = 1e-6
    batch_size: int = 256
    replay_capacity: int = 200_000
    gamma: float = 0.97
    tau: float = 0.005
    grad_clip: float = 1.0
    train_every_steps: int = 4
    target_update_every: int = 500
    validate_every: int = 50
    validation_seeds: int = 3
    validation_steps: int = 60
    checkpoint_every: int = 50
    # Reward weights
    thr_weight: float = 3.0
    p5_weight: float = 2.0
    fairness_weight: float = 1.5
    load_std_penalty: float = -2.0
    pp_penalty: float = -20.0  # harsh above 0.01
    ho_cost: float = -0.3
    outage_penalty: float = -5.0
    late_ho_penalty: float = -1.0
    # SON config for training
    cio_min_db: float = -6.0
    cio_max_db: float = 6.0
    max_cio_step_db: float = 3.0
    base_a3_offset_db: float = 3.0
    base_ttt_steps: int = 2
    preference_threshold: float = 0.08
    rollback_throughput_drop_frac: float = 0.40
    rollback_pingpong_floor: float = 0.015


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, rng: np.random.Generator, batch_size: int):
        indices = rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


def vectorized_a3_decide(
    rsrp: np.ndarray,
    serving: np.ndarray,
    cio_db: np.ndarray,
    valid_mask: np.ndarray,
    base_offset: float,
    ttt_steps: np.ndarray,
    candidates: np.ndarray,
    counters: np.ndarray,
) -> np.ndarray:
    """Vectorized A3 event decision for all UEs."""
    num_ues = rsrp.shape[0]
    adjusted = rsrp.copy()
    for u in range(num_ues):
        s = serving[u]
        adjusted[u] += cio_db[s]
        adjusted[u, s] -= cio_db[s, s]  # serving cell has no CIO to itself
    adjusted[~valid_mask] = -1e9
    best = np.argmax(adjusted, axis=1)

    serving_rsrp = rsrp[np.arange(num_ues), serving]
    threshold = serving_rsrp + base_offset

    targets = serving.copy()
    for u in range(num_ues):
        if best[u] != serving[u] and adjusted[u, best[u]] > threshold[u]:
            if candidates[u] == best[u]:
                counters[u] += 1
            else:
                candidates[u] = best[u]
                counters[u] = 1
            ttt = int(ttt_steps[serving[u], best[u]])
            if counters[u] >= ttt:
                targets[u] = best[u]
                counters[u] = 0
        else:
            candidates[u] = -1
            counters[u] = 0
    return targets


def compute_network_reward(
    env: CellularNetworkEnv,
    prev_metrics: dict,
    cfg: SONLoopConfig,
    prev_total_ho: int,
    prev_pp_ho: int,
) -> float:
    m = env.metrics()
    thr = m["avg_ue_throughput_mbps"]
    p5 = m["p5_ue_throughput_mbps"]
    jain = m["jain_load_fairness"]
    load_std = m["load_std"]
    outage = m["outage_rate"]
    late_ho_rate = m.get("late_ho_rate", 0.0)

    # Ping-pong: only penalize above operator KPI
    total_ho = env.total_handovers
    pp_ho = env.pingpong_handovers
    cycle_ho = total_ho - prev_total_ho
    cycle_pp = pp_ho - prev_pp_ho
    pp_rate = cycle_pp / max(cycle_ho, 1)
    pp_excess = max(pp_rate - 0.01, 0.0)

    # HO cost: penalize excessive handovers
    ho_rate = cycle_ho / max(env.cfg.num_ues, 1)

    # Throughput improvement vs previous step
    prev_thr = prev_metrics.get("avg_ue_throughput_mbps", thr)
    thr_delta = (thr - prev_thr) / max(prev_thr, 0.1)

    reward = (
        cfg.thr_weight * thr
        + cfg.p5_weight * p5
        + cfg.fairness_weight * jain
        + cfg.load_std_penalty * load_std
        + cfg.pp_penalty * pp_excess
        + cfg.ho_cost * ho_rate
        + cfg.outage_penalty * outage
        + cfg.late_ho_penalty * late_ho_rate
        + 1.0 * np.clip(thr_delta, -0.5, 0.5)
    )
    return float(reward), m


def train_step(
    agent: GnnDQNAgent,
    target_net: GnnDQNAgent,
    batch: list,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    optimizer: optim.Optimizer,
    gamma: float,
    grad_clip: float,
    num_cells: int,
) -> float:
    states, actions, rewards, next_states, dones, valid_masks = zip(*batch)
    batch_size = len(states)

    states_t = torch.from_numpy(np.array(states)).float().to(DEVICE)
    next_states_t = torch.from_numpy(np.array(next_states)).float().to(DEVICE)
    actions_t = torch.tensor(actions, dtype=torch.long, device=DEVICE)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
    valid_t = torch.from_numpy(np.array(valid_masks)).bool().to(DEVICE)

    # Expand edge_index for batch
    ei_batch = []
    ew_batch = []
    for i in range(batch_size):
        ei_batch.append(edge_index + i * num_cells)
        ew_batch.append(edge_weight)
    ei_batched = torch.cat(ei_batch, dim=1).to(DEVICE)
    ew_batched = torch.cat(ew_batch).to(DEVICE)

    x_flat = states_t.view(batch_size * num_cells, -1)
    q_all = agent.forward(x_flat, ei_batched, ew_batched,
                          batch_size=batch_size, nodes_per_graph=num_cells)
    q_values = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        nx_flat = next_states_t.view(batch_size * num_cells, -1)
        # Double DQN
        q_next_online = agent.forward(nx_flat, ei_batched, ew_batched,
                                      batch_size=batch_size, nodes_per_graph=num_cells)
        q_next_online[~valid_t] = -1e9
        best_actions = q_next_online.argmax(dim=1)
        q_next_target = target_net.forward(nx_flat, ei_batched, ew_batched,
                                           batch_size=batch_size, nodes_per_graph=num_cells)
        q_next = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        target = rewards_t + gamma * q_next * (1 - dones_t)

    loss = F.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
    optimizer.step()
    return float(loss)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def run_son_loop_training(config_path: str, resume_path: str | None = None):
    with open(config_path) as f:
        raw_cfg = json.load(f)

    cfg = SONLoopConfig(**{k: v for k, v in raw_cfg.items() if hasattr(SONLoopConfig, k)})
    out_dir = Path(raw_cfg.get("out_dir", "results/runs/son_loop"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    seed = raw_cfg.get("seed", 42)
    rng = np.random.default_rng(seed)

    son_config = SONConfig(
        update_interval_steps=cfg.son_update_interval,
        cio_min_db=cfg.cio_min_db,
        cio_max_db=cfg.cio_max_db,
        max_cio_step_db=cfg.max_cio_step_db,
        base_a3_offset_db=cfg.base_a3_offset_db,
        base_ttt_steps=cfg.base_ttt_steps,
        preference_threshold=cfg.preference_threshold,
        rollback_throughput_drop_frac=cfg.rollback_throughput_drop_frac,
        rollback_pingpong_floor=cfg.rollback_pingpong_floor,
    )

    # Load scenarios
    all_scenarios = get_training_scenarios(seed)
    scenario_names = raw_cfg.get("train_scenarios", [s.name for s in all_scenarios])
    scenarios = [s for s in all_scenarios if s.name in scenario_names]
    val_scenario_names = raw_cfg.get("validation_scenarios", ["dense_urban", "highway", "overloaded_event", "suburban"])
    val_scenarios = [s for s in all_scenarios if s.name in val_scenario_names]

    # Scenario weights
    weights_raw = raw_cfg.get("scenario_sampling_weights", {})
    if weights_raw:
        w = np.array([weights_raw.get(s.name, 1.0) for s in scenarios])
        sampling_probs = w / w.sum()
    else:
        sampling_probs = None

    # Model setup
    max_cells = max(s.num_cells for s in scenarios)
    feature_dim = 11  # ue_only

    dqn_cfg = dqn_config_from_dict(raw_cfg.get("dqn", {
        "hidden_dim": 256, "dropout": 0.06, "dueling": True, "double_dqn": True,
    }))

    if resume_path:
        agent, metadata, payload = load_gnn_checkpoint(Path(resume_path), strict_metadata=False)
        max_cells = int(metadata["max_cells"])
        feature_dim = int(metadata["feature_dim"])
        print(f"Resumed from {resume_path} (max_cells={max_cells}, features={feature_dim})")
    else:
        agent = GnnDQNAgent(max_cells, feature_dim, dqn_cfg, seed=seed)
        print(f"Fresh model (max_cells={max_cells}, features={feature_dim})")

    agent.to(DEVICE)
    target_net = copy.deepcopy(agent)
    target_net.eval()

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.episodes * cfg.steps_per_episode // cfg.train_every_steps, eta_min=cfg.lr_min
    )

    replay = ReplayBuffer(cfg.replay_capacity)
    total_decisions = 0
    best_val_score = -1e9
    history = []

    log_path = out_dir.parent / f"{out_dir.name}_train.log"

    print(f"=== SON-in-the-Loop Training ===")
    print(f"Episodes: {cfg.episodes}, Steps: {cfg.steps_per_episode}")
    print(f"SON update interval: {cfg.son_update_interval} steps")
    print(f"Scenarios: {[s.name for s in scenarios]}")
    print(f"Model params: {sum(p.numel() for p in agent.parameters()):,}")
    print(f"Output: {out_dir}")
    sys.stdout.flush()

    with open(log_path, "w") as log_f:
        log_f.write(f"=== SON-in-the-Loop Training ===\n")
        log_f.write(f"Episodes: {cfg.episodes}, Steps: {cfg.steps_per_episode}, SON interval: {cfg.son_update_interval}\n")
        log_f.write(f"Scenarios: {[s.name for s in scenarios]}\n")
        log_f.write(f"Model params: {sum(p.numel() for p in agent.parameters()):,}\n")
        log_f.flush()

        for episode in range(cfg.episodes):
            # Epsilon
            frac = min(episode / max(cfg.epsilon_decay_episodes, 1), 1.0)
            epsilon = cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

            # Pick scenario
            if sampling_probs is not None:
                sc_idx = int(rng.choice(len(scenarios), p=sampling_probs))
            else:
                sc_idx = episode % len(scenarios)
            scenario = scenarios[sc_idx]

            num_ues = min(cfg.num_ues_cap, scenario.num_ues)
            env = make_env_from_scenario(scenario, feature_mode="ue_only", prb_available=False, num_ues=num_ues)
            env.reset(seed + 1000 * episode)
            num_cells = env.cfg.num_cells

            # Edge data (cached) - may already be tensors
            ei_raw, ew_raw = env.edge_data
            edge_index = ei_raw if isinstance(ei_raw, torch.Tensor) else torch.from_numpy(ei_raw).long()
            edge_weight = ew_raw if isinstance(ew_raw, torch.Tensor) else torch.from_numpy(ew_raw).float()

            # SON controller for this episode
            son = SONController(agent, son_config)
            son.reset(env)

            # A3 state
            candidates = np.full(num_ues, -1, dtype=int)
            counters = np.zeros(num_ues, dtype=int)

            prev_metrics = env.metrics()
            prev_total_ho = 0
            prev_pp_ho = 0
            episode_reward = 0.0
            episode_losses = []

            for step in range(cfg.steps_per_episode):
                env.advance_mobility()

                # Agent outputs preferences for all UEs
                states = env.build_all_states()  # (num_ues, num_cells, feat_dim)
                valid = env.valid_actions_all()  # (num_ues, num_cells)

                # Pad if needed
                padded_states = np.zeros((num_ues, max_cells, feature_dim), dtype=np.float32)
                padded_valid = np.zeros((num_ues, max_cells), dtype=bool)
                padded_states[:, :num_cells, :] = states
                padded_valid[:, :num_cells] = valid

                actions = agent.act_batch(
                    padded_states, edge_index, edge_weight,
                    padded_valid, epsilon, rng
                )

                # SON aggregates preferences and updates CIO
                if step % cfg.son_update_interval == 0 and step > 0:
                    son.maybe_update(env)

                # A3 event: decide handovers using CIO-adjusted RSRP
                rsrp = env.rsrp_matrix()
                targets = vectorized_a3_decide(
                    rsrp, env.serving, son.cio_db, valid,
                    son_config.base_a3_offset_db,
                    son.ttt_steps, candidates, counters,
                )

                # Execute handovers
                for ue_idx in range(num_ues):
                    if targets[ue_idx] != env.serving[ue_idx]:
                        env.step_user_action(ue_idx, int(targets[ue_idx]))
                    else:
                        env.step_user_action(ue_idx, int(env.serving[ue_idx]))

                # Compute network-level reward
                reward, current_metrics = compute_network_reward(
                    env, prev_metrics, cfg, prev_total_ho, prev_pp_ho
                )
                prev_metrics = current_metrics
                prev_total_ho = env.total_handovers
                prev_pp_ho = env.pingpong_handovers
                episode_reward += reward

                # Store transitions for ALL UEs (shared team reward)
                next_states = env.build_all_states()
                padded_next = np.zeros((num_ues, max_cells, feature_dim), dtype=np.float32)
                padded_next[:, :num_cells, :] = next_states
                next_valid = env.valid_actions_all()
                padded_next_valid = np.zeros((num_ues, max_cells), dtype=bool)
                padded_next_valid[:, :num_cells] = next_valid

                per_ue_reward = reward / num_ues
                done = float(step == cfg.steps_per_episode - 1)

                for ue_idx in range(num_ues):
                    replay.add((
                        padded_states[ue_idx],
                        int(actions[ue_idx]),
                        per_ue_reward,
                        padded_next[ue_idx],
                        done,
                        padded_next_valid[ue_idx],
                    ))
                    total_decisions += 1

                # Train
                if len(replay) >= cfg.batch_size and total_decisions % cfg.train_every_steps == 0:
                    batch = replay.sample(rng, cfg.batch_size)
                    loss = train_step(
                        agent, target_net, batch,
                        edge_index, edge_weight, optimizer,
                        cfg.gamma, cfg.grad_clip, max_cells,
                    )
                    episode_losses.append(loss)
                    scheduler.step()
                    soft_update(target_net, agent, cfg.tau)

            # Episode summary
            final_m = env.metrics()
            pp_rate = env.pingpong_handovers / max(env.total_handovers, 1)
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0

            log_line = (
                f"[{episode+1:04d}/{cfg.episodes}] {scenario.name:<22} "
                f"eps={epsilon:.3f} thr={final_m['avg_ue_throughput_mbps']:.3f} "
                f"p5={final_m['p5_ue_throughput_mbps']:.3f} "
                f"load_std={final_m['load_std']:.3f} "
                f"jain={final_m['jain_load_fairness']:.3f} "
                f"pp={pp_rate:.3f} "
                f"late_ho={final_m.get('late_ho_rate', 0):.3f} "
                f"reward={episode_reward:.1f} loss={avg_loss:.4f}"
            )

            if episode % 5 == 4:
                print(log_line)
                sys.stdout.flush()
            log_f.write(log_line + "\n")
            log_f.flush()

            # Validation
            if (episode + 1) % cfg.validate_every == 0:
                agent.eval()
                val_scores = []
                val_margins = []
                for vs in val_scenarios:
                    venv = make_env_from_scenario(vs, feature_mode="ue_only", prb_available=False,
                                                  num_ues=min(80, vs.num_ues))
                    policies = default_policy_factories(gnn_agent=agent, son_config=son_config, include_true_prb=False)
                    rows = evaluate_policies(venv.cfg, policies, steps=cfg.validation_steps,
                                             seeds=list(range(cfg.validation_seeds)))
                    by_method = {r["method"]: r for r in rows}
                    son_thr = by_method.get("son_gnn_dqn", {}).get("avg_ue_throughput_mbps", 0)
                    a3_thr = by_method.get("a3_ttt", {}).get("avg_ue_throughput_mbps", 0)
                    margin = ((son_thr - a3_thr) / max(a3_thr, 1e-6)) * 100
                    val_margins.append(margin)
                    val_scores.append(son_thr)

                avg_margin = np.mean(val_margins)
                min_margin = np.min(val_margins)
                val_score = np.mean(val_scores)
                is_best = val_score > best_val_score

                if is_best:
                    best_val_score = val_score
                    torch.save({
                        "state_dict": agent.state_dict(),
                        "metadata": {
                            "max_cells": max_cells,
                            "feature_dim": feature_dim,
                            "model_class": "GnnDQNAgent",
                            "feature_profile": "ue_only",
                            "episode": episode + 1,
                            "config": raw_cfg,
                        },
                    }, out_dir / "checkpoints" / "best.pt")

                val_line = (
                    f"  [VAL ep={episode+1:04d}] avg_margin={avg_margin:+.2f}% "
                    f"min_margin={min_margin:+.2f}% "
                    f"{'★ BEST' if is_best else ''}"
                    f" per_scenario={[f'{n}:{m:+.1f}%' for n, m in zip(val_scenario_names, val_margins)]}"
                )
                print(val_line)
                sys.stdout.flush()
                log_f.write(val_line + "\n")
                log_f.flush()
                agent.train()

            # Checkpoint
            if (episode + 1) % cfg.checkpoint_every == 0:
                torch.save({
                    "state_dict": agent.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metadata": {
                        "max_cells": max_cells,
                        "feature_dim": feature_dim,
                        "model_class": "GnnDQNAgent",
                        "feature_profile": "ue_only",
                        "episode": episode + 1,
                        "config": raw_cfg,
                    },
                    "training_state": {"episode_completed": episode + 1},
                }, out_dir / "checkpoints" / f"ep{episode+1:04d}.pt")

    print(f"\nTraining complete. Best val score: {best_val_score:.3f}")
    print(f"Best checkpoint: {out_dir / 'checkpoints' / 'best.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    run_son_loop_training(args.config, args.resume)
