"""Flat DQN baseline: standard DQN without graph structure.

This serves as an ablation baseline to prove that the GNN component
adds value. Same features, same training, but NO graph convolution —
just a flat MLP processing the concatenated cell features.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_dqn import DQNConfig, ReplayBuffer, Transition
from ..env.simulator import CellularNetworkEnv, LTEConfig


class FlatDQNAgent(nn.Module):
    """Standard DQN with MLP (no graph structure)."""

    def __init__(self, num_cells: int, feature_dim: int, cfg: DQNConfig, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_cells = num_cells
        self.feature_dim = feature_dim
        self.cfg = cfg

        input_dim = num_cells * feature_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(),
        )

        if cfg.dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_cells),
            )
        else:
            self.q_head = nn.Linear(64, num_cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1) if x.dim() == 2 else x
        h = self.network(flat)

        if self.cfg.dueling:
            value = self.value_stream(h)
            advantage = self.advantage_stream(h)
            q = value + (advantage - advantage.mean())
        else:
            q = self.q_head(h)

        return q

    def act(
        self,
        x: torch.Tensor,
        epsilon: float = 0.0,
        valid_mask: torch.Tensor | np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        if rng is None:
            rng = np.random.default_rng()
        if valid_mask is None:
            valid_mask = torch.ones(self.num_cells, dtype=torch.bool)
        elif isinstance(valid_mask, np.ndarray):
            valid_mask = torch.from_numpy(valid_mask)

        valid_actions = valid_mask.nonzero(as_tuple=True)[0].numpy()
        if rng.random() < epsilon:
            return int(rng.choice(valid_actions))

        with torch.no_grad():
            was_training = self.training
            self.eval()
            q = self.forward(x).clone()
            if was_training:
                self.train()
            q[~valid_mask] = float("-inf")
            return int(q.argmax().item())


def _flat_train_step(
    agent: FlatDQNAgent,
    target_net: FlatDQNAgent,
    batch: List[Transition],
    optimizer: torch.optim.Optimizer,
    cfg: DQNConfig,
) -> float:
    states = torch.stack([torch.from_numpy(t[0]).float().reshape(-1) for t in batch])
    next_states = torch.stack([torch.from_numpy(t[3]).float().reshape(-1) for t in batch])
    actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
    rewards = torch.tensor([t[2] for t in batch], dtype=torch.float)
    dones = torch.tensor([float(t[4]) for t in batch], dtype=torch.float)

    next_masks = []
    for t in batch:
        if len(t) >= 6 and t[5] is not None:
            next_masks.append(torch.from_numpy(np.asarray(t[5], dtype=bool)))
        else:
            next_masks.append(torch.ones(agent.num_cells, dtype=torch.bool))
    next_mask = torch.stack(next_masks)

    agent_was_training = agent.training
    target_was_training = target_net.training
    agent.eval()
    target_net.eval()
    with torch.no_grad():
        q_next_target = torch.stack([target_net(ns) for ns in next_states])
        if cfg.double_dqn:
            q_next_online = torch.stack([agent(ns) for ns in next_states])
            q_next_online = q_next_online.masked_fill(~next_mask, float("-inf"))
            next_actions = q_next_online.argmax(dim=1)
            q_max = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            q_next_masked = q_next_target.masked_fill(~next_mask, float("-inf"))
            q_max = q_next_masked.max(dim=1).values
        q_max = torch.where(torch.isfinite(q_max), q_max, torch.zeros_like(q_max))
        targets = rewards + cfg.gamma * q_max * (1.0 - dones)
    if agent_was_training:
        agent.train()
    if target_was_training:
        target_net.train()

    q_pred = torch.stack([agent(s)[a] for s, a in zip(states, actions)])

    loss = F.smooth_l1_loss(q_pred, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
    optimizer.step()
    return loss.item()


def train_flat_dqn(
    lte_cfg: LTEConfig,
    dqn_cfg: DQNConfig,
    train_episodes: int,
    steps_per_episode: int,
    seed: int = 7,
    verbose: bool = True,
) -> Tuple[FlatDQNAgent, List[dict]]:
    env = CellularNetworkEnv(lte_cfg)
    agent = FlatDQNAgent(lte_cfg.num_cells, env.feature_dim, dqn_cfg, seed=seed)
    target_net = copy.deepcopy(agent)
    target_net.load_state_dict(agent.state_dict())
    optimizer = torch.optim.Adam(agent.parameters(), lr=dqn_cfg.learning_rate)
    replay = ReplayBuffer(dqn_cfg.replay_capacity)
    rng = np.random.default_rng(seed)
    history: List[dict] = []
    decisions = 0
    best_reward = float("-inf")
    best_state = copy.deepcopy(agent.state_dict())
    best_episode = 0

    for episode in range(train_episodes):
        env.reset(seed + 101 * episode)
        losses: List[float] = []
        episode_reward = 0.0

        frac = min(episode / max(dqn_cfg.epsilon_decay_episodes, 1), 1.0)
        epsilon = dqn_cfg.epsilon_start + frac * (dqn_cfg.epsilon_end - dqn_cfg.epsilon_start)

        for _step in range(steps_per_episode):
            env.advance_mobility()
            for ue_idx in rng.permutation(lte_cfg.num_ues):
                state = env.build_state(int(ue_idx))
                state_t = torch.from_numpy(state).float()
                valid = env.valid_actions(int(ue_idx))
                action = agent.act(
                    state_t,
                    epsilon=epsilon,
                    valid_mask=valid,
                    rng=rng,
                )
                next_state, reward, done, _info = env.step_user_action(int(ue_idx), action)
                next_valid = env.valid_actions(int(ue_idx))
                replay.add((state, action, reward, next_state, done, next_valid))
                episode_reward += reward

                if len(replay) >= dqn_cfg.batch_size and decisions % dqn_cfg.train_every == 0:
                    batch = replay.sample(rng, dqn_cfg.batch_size)
                    losses.append(
                        _flat_train_step(agent, target_net, batch, optimizer, dqn_cfg)
                    )
                if decisions % dqn_cfg.target_update_every == 0:
                    target_net.load_state_dict(agent.state_dict())
                decisions += 1

        metrics = env.metrics()
        metrics["episode"] = float(episode + 1)
        metrics["epsilon"] = float(epsilon)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["episode_reward"] = float(episode_reward)
        history.append(metrics)

        if episode >= max(5, train_episodes // 10) and episode_reward > best_reward:
            best_reward = episode_reward
            best_state = copy.deepcopy(agent.state_dict())
            best_episode = episode + 1

        if verbose and (episode + 1) % max(1, train_episodes // 20) == 0:
            print(
                f"[flat-DQN {episode + 1:04d}/{train_episodes}] "
                f"eps={epsilon:.3f} "
                f"avg_thr={metrics['avg_ue_throughput_mbps']:.3f} "
                f"load_std={metrics['load_std']:.3f} "
                f"loss={metrics['loss']:.4f}"
            )

    if best_episode > 0:
        agent.load_state_dict(best_state)
        if verbose:
            print(f"  Restored flat-DQN best checkpoint from episode {best_episode} "
                  f"(reward={best_reward:.1f})")
    target_net.load_state_dict(agent.state_dict())
    return agent, history
