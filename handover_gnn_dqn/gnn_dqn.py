from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .simulator import CellularNetworkEnv, LTEConfig

# Transition: (state, action, reward, next_state, done, next_valid_mask)
# next_valid_mask is a boolean array over the model's num_cells action space,
# including any padded (inactive) nodes for multi-scenario training.
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]

# MPS has too much overhead for small per-UE forward passes. CPU is faster here.
DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class DQNConfig:
    hidden_dim: int = 128
    num_gcn_layers: int = 3
    dropout: float = 0.1
    gamma: float = 0.95
    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    train_every: int = 4
    target_update_every: int = 500
    grad_clip: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 300  # ~70% of total_episodes; override per run
    dueling: bool = True
    double_dqn: bool = True


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.items: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: Transition) -> None:
        self.items.append(item)

    def sample(self, rng: np.random.Generator, batch_size: int) -> List[Transition]:
        idx = rng.choice(len(self.items), size=batch_size, replace=False)
        items = list(self.items)
        return [items[int(i)] for i in idx]


class GnnDQNAgent(nn.Module):
    """3-layer GCN + Dueling Q-head for per-UE handover decisions."""

    def __init__(self, num_cells: int, feature_dim: int, cfg: DQNConfig, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_cells = num_cells
        self.feature_dim = feature_dim
        self.cfg = cfg

        self.gcn1 = GCNConv(feature_dim, cfg.hidden_dim)
        self.gcn2 = GCNConv(cfg.hidden_dim, cfg.hidden_dim)
        self.gcn3 = GCNConv(cfg.hidden_dim, 64)

        if cfg.dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        else:
            self.q_head = nn.Linear(64, 1)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        h = self.dropout(h)
        h = F.relu(self.gcn2(h, edge_index, edge_weight))
        h = self.dropout(h)
        h = F.relu(self.gcn3(h, edge_index, edge_weight))

        if self.cfg.dueling:
            value = self.value_stream(h).squeeze(-1)
            advantage = self.advantage_stream(h).squeeze(-1)
            q = value + (advantage - advantage.mean())
        else:
            q = self.q_head(h).squeeze(-1)
        return q

    def act(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
            epsilon: float = 0.0, valid_mask: torch.Tensor | np.ndarray | None = None,
            rng: np.random.Generator | None = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        if valid_mask is None:
            valid_mask = torch.ones(self.num_cells, dtype=torch.bool)
        elif isinstance(valid_mask, np.ndarray):
            valid_mask = torch.from_numpy(valid_mask)

        valid_actions = valid_mask.nonzero(as_tuple=True)[0].numpy()
        if len(valid_actions) == 0:
            # Safety: if caller passed an all-false mask, fall back to argmax on all cells.
            valid_mask = torch.ones(self.num_cells, dtype=torch.bool)
            valid_actions = np.arange(self.num_cells)

        if rng.random() < epsilon:
            return int(rng.choice(valid_actions))

        with torch.no_grad():
            self.eval()
            q = self.forward(x.to(DEVICE), edge_index.to(DEVICE), edge_weight.to(DEVICE)).cpu()
            self.train()
            q = q.clone()
            q[~valid_mask] = float("-inf")
            return int(q.argmax().item())


def _train_step(
    agent: GnnDQNAgent,
    target_net: GnnDQNAgent,
    batch: List[Transition],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cfg: DQNConfig,
) -> float:
    states = torch.stack([torch.from_numpy(s).float() for s, *_ in batch]).to(DEVICE)
    next_states = torch.stack([torch.from_numpy(t[3]).float() for t in batch]).to(DEVICE)
    actions = [t[1] for t in batch]
    rewards = torch.tensor([t[2] for t in batch], dtype=torch.float, device=DEVICE)
    dones = torch.tensor([float(t[4]) for t in batch], dtype=torch.float, device=DEVICE)

    # Next-state valid-action masks. If absent (older transitions), default to all-valid.
    next_masks = []
    for t in batch:
        if len(t) >= 6 and t[5] is not None:
            next_masks.append(torch.from_numpy(np.asarray(t[5], dtype=bool)))
        else:
            next_masks.append(torch.ones(agent.num_cells, dtype=torch.bool))
    next_mask = torch.stack(next_masks).to(DEVICE)  # (B, num_cells)

    ei = edge_index.to(DEVICE)
    ew = edge_weight.to(DEVICE)

    with torch.no_grad():
        # Target net Q at next states (for all cells).
        q_next_target = torch.stack([target_net(ns, ei, ew) for ns in next_states])

        if cfg.double_dqn:
            # Double-DQN: select action with online net, evaluate with target net.
            q_next_online = torch.stack([agent(ns, ei, ew) for ns in next_states])
            q_next_online = q_next_online.masked_fill(~next_mask, float("-inf"))
            next_actions = q_next_online.argmax(dim=1)
            q_max = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            q_next_masked = q_next_target.masked_fill(~next_mask, float("-inf"))
            q_max = q_next_masked.max(dim=1).values

        # Terminal / all-invalid safety: if a row has no valid action, q_max is -inf;
        # force it to 0 so target = reward only.
        q_max = torch.where(torch.isfinite(q_max), q_max, torch.zeros_like(q_max))
        targets = rewards + cfg.gamma * q_max * (1.0 - dones)

    q_pred = torch.stack([agent(s, ei, ew)[a] for s, a in zip(states, actions)])

    loss = F.smooth_l1_loss(q_pred, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
    optimizer.step()
    return loss.item()


def train_gnn_dqn(
    lte_cfg: LTEConfig,
    dqn_cfg: DQNConfig,
    train_episodes: int,
    steps_per_episode: int,
    seed: int = 7,
    verbose: bool = True,
) -> Tuple[GnnDQNAgent, List[dict]]:
    print(f"  Device: {DEVICE}")
    env = CellularNetworkEnv(lte_cfg)
    agent = GnnDQNAgent(lte_cfg.num_cells, env.feature_dim, dqn_cfg, seed=seed).to(DEVICE)
    target_net = copy.deepcopy(agent).to(DEVICE)
    target_net.load_state_dict(agent.state_dict())
    optimizer = torch.optim.Adam(agent.parameters(), lr=dqn_cfg.learning_rate)
    replay = ReplayBuffer(dqn_cfg.replay_capacity)
    rng = np.random.default_rng(seed)
    history: List[dict] = []
    decisions = 0
    best_reward = float("-inf")
    best_state = copy.deepcopy(agent.state_dict())
    best_episode = 0

    edge_index, edge_weight = env.edge_data
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

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
                state_t = torch.from_numpy(state).float().to(DEVICE)
                valid = env.valid_actions(int(ue_idx))
                action = agent.act(state_t, edge_index, edge_weight,
                                   epsilon=epsilon, valid_mask=valid, rng=rng)
                next_state, reward, done, _info = env.step_user_action(int(ue_idx), action)
                next_valid = env.valid_actions(int(ue_idx))
                replay.add((state, action, reward, next_state, done, next_valid))
                episode_reward += reward

                if len(replay) >= dqn_cfg.batch_size and decisions % dqn_cfg.train_every == 0:
                    batch = replay.sample(rng, dqn_cfg.batch_size)
                    losses.append(_train_step(agent, target_net, batch, edge_index, edge_weight, optimizer, dqn_cfg))
                if decisions % dqn_cfg.target_update_every == 0:
                    target_net.load_state_dict(agent.state_dict())
                decisions += 1

        metrics = env.metrics()
        metrics["episode"] = float(episode + 1)
        metrics["epsilon"] = float(epsilon)
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["episode_reward"] = float(episode_reward)
        history.append(metrics)

        # Track best-performing episode (after exploration has cooled a bit).
        if episode >= max(5, train_episodes // 10) and episode_reward > best_reward:
            best_reward = episode_reward
            best_state = copy.deepcopy(agent.state_dict())
            best_episode = episode + 1

        if verbose and (episode + 1) % max(1, train_episodes // 20) == 0:
            print(
                f"[{episode + 1:04d}/{train_episodes}] "
                f"eps={epsilon:.3f} "
                f"avg_thr={metrics['avg_ue_throughput_mbps']:.3f} "
                f"load_std={metrics['load_std']:.3f} "
                f"loss={metrics['loss']:.4f} "
                f"reward={episode_reward:.1f}"
            )

    # Restore best checkpoint (if one was saved).
    if best_episode > 0:
        agent.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best checkpoint from episode {best_episode} "
                  f"(reward={best_reward:.1f})")
    target_net.load_state_dict(agent.state_dict())
    agent = agent.cpu()
    return agent, history
