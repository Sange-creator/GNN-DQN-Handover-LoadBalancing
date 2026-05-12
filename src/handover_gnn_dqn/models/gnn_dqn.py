from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from ..env.simulator import CellularNetworkEnv, LTEConfig

# Transition: (state, action, reward, next_state, done, next_valid_mask)
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]

# MPS has too much overhead for small per-UE forward passes. CPU is faster here.
DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class DQNConfig:
    hidden_dim: int = 128
    num_gcn_layers: int = 3
    dropout: float = 0.1
    gamma: float = 0.97
    learning_rate: float = 1e-4
    batch_size: int = 128
    replay_capacity: int = 300_000
    train_every: int = 4
    target_update_every: int = 1000
    tau: float = 0.005
    grad_clip: float = 1.0
    lr_min: float = 1e-5
    weight_decay: float = 1e-5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    epsilon_decay_episodes: int = 350
    dueling: bool = True
    double_dqn: bool = True
    use_gat: bool = False
    gat_heads: int = 4
    n_step: int = 1
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0


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

    def state_dict(self) -> dict:
        return {
            "capacity": self.items.maxlen,
            "items": list(self.items),
        }

    def load_state_dict(self, state: dict) -> None:
        capacity = int(state.get("capacity") or self.items.maxlen or 0)
        self.items = deque(state.get("items", []), maxlen=capacity)


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay (PER).

    Rare important transitions (handovers, outages, load-balance wins) get
    replayed more often, dramatically improving sample efficiency.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.items: List[Transition] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.size = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(self, item: Transition) -> None:
        if self.size < self.capacity:
            self.items.append(item)
        else:
            self.items[self.position] = item
        self.priorities[self.position] = self._max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, batch_size: int, beta: float = 0.4
               ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        priorities = self.priorities[:self.size]
        probs = priorities / (priorities.sum() + 1e-10)
        indices = rng.choice(self.size, size=batch_size, replace=False, p=probs)

        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-10)

        batch = [self.items[int(i)] for i in indices]
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + 1e-6) ** self.alpha
            self.priorities[int(idx)] = priority
            self._max_priority = max(self._max_priority, priority)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "items": self.items[:self.size],
            "priorities": self.priorities[:self.size].tolist(),
            "position": self.position,
            "max_priority": self._max_priority,
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        items = state.get("items", [])
        self.items = list(items)
        self.size = len(self.items)
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        prios = state.get("priorities", [])
        self.priorities[:len(prios)] = prios
        self.position = int(state.get("position", self.size % self.capacity))
        self._max_priority = float(state.get("max_priority", 1.0))


class NStepBuffer:
    """Accumulates n-step returns for better credit assignment."""

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buffer: Deque[Transition] = deque(maxlen=n)

    def add(self, transition: Transition) -> Transition | None:
        self.buffer.append(transition)
        if len(self.buffer) < self.n:
            return None
        # Compute n-step return
        state = self.buffer[0][0]
        action = self.buffer[0][1]
        n_step_reward = 0.0
        for i, t in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * t[2]
        last = self.buffer[-1]
        return (state, action, n_step_reward, last[3], last[4], last[5])

    def flush(self) -> List[Transition]:
        """Flush remaining partial n-step transitions at episode end."""
        results = []
        while len(self.buffer) > 0:
            state = self.buffer[0][0]
            action = self.buffer[0][1]
            n_step_reward = 0.0
            for i, t in enumerate(self.buffer):
                n_step_reward += (self.gamma ** i) * t[2]
            last = self.buffer[-1]
            results.append((state, action, n_step_reward, last[3], last[4], last[5]))
            self.buffer.popleft()
        return results

    def reset(self) -> None:
        self.buffer.clear()


class GnnDQNAgent(nn.Module):
    """3-layer GNN (GCN or GAT) + Dueling Q-head for per-UE handover decisions."""

    def __init__(self, num_cells: int, feature_dim: int, cfg: DQNConfig, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_cells = num_cells
        self.feature_dim = feature_dim
        self.cfg = cfg

        embed_dim = 64

        if cfg.use_gat:
            heads = cfg.gat_heads
            self.conv1 = GATConv(feature_dim, cfg.hidden_dim // heads, heads=heads, concat=True)
            self.conv2 = GATConv(cfg.hidden_dim, cfg.hidden_dim // heads, heads=heads, concat=True)
            self.conv3 = GATConv(cfg.hidden_dim, embed_dim // heads, heads=heads, concat=True)
        else:
            self.conv1 = GCNConv(feature_dim, cfg.hidden_dim)
            self.conv2 = GCNConv(cfg.hidden_dim, cfg.hidden_dim)
            self.conv3 = GCNConv(cfg.hidden_dim, embed_dim)

        self.layer_norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)

        if cfg.dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            self.q_head = nn.Linear(embed_dim, 1)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        *,
        batch_size: int | None = None,
        nodes_per_graph: int | None = None,
    ) -> torch.Tensor:
        """Return per-node Q values.

        Single-graph inference is the default and always returns ``(N,)``.
        Batched padded training must opt in with ``batch_size`` and
        ``nodes_per_graph`` so an unseen 40-cell graph is never mistaken for
        two 20-cell training graphs.
        """
        if (batch_size is None) != (nodes_per_graph is None):
            raise ValueError("batch_size and nodes_per_graph must be provided together")

        if self.cfg.use_gat:
            h = F.elu(self.layer_norm1(self.conv1(x, edge_index)))
            h = self.dropout(h)
            h = F.elu(self.layer_norm2(self.conv2(h, edge_index)))
            h = self.dropout(h)
            h = F.elu(self.layer_norm3(self.conv3(h, edge_index)))
        else:
            h = F.relu(self.layer_norm1(self.conv1(x, edge_index, edge_weight)))
            h = self.dropout(h)
            h = F.relu(self.layer_norm2(self.conv2(h, edge_index, edge_weight)))
            h = self.dropout(h)
            h = F.relu(self.layer_norm3(self.conv3(h, edge_index, edge_weight)))

        if self.cfg.dueling:
            advantage = self.advantage_stream(h).squeeze(-1)

            if batch_size is not None and nodes_per_graph is not None:
                expected = batch_size * nodes_per_graph
                if advantage.shape[0] != expected:
                    raise ValueError(
                        f"Batched GNN input has {advantage.shape[0]} nodes, expected {expected} "
                        f"({batch_size} x {nodes_per_graph})"
                    )
                h_batched = h.view(batch_size, nodes_per_graph, -1)
                h_pooled = h_batched.mean(dim=1)
                value = self.value_stream(h_pooled)
                advantage = advantage.view(batch_size, nodes_per_graph)
                q = value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                h_pooled = h.mean(dim=0, keepdim=True)
                value = self.value_stream(h_pooled).squeeze()
                q = value + (advantage - advantage.mean())
        else:
            q = self.q_head(h).squeeze(-1)
            if batch_size is not None and nodes_per_graph is not None:
                expected = batch_size * nodes_per_graph
                if q.shape[0] != expected:
                    raise ValueError(
                        f"Batched GNN input has {q.shape[0]} nodes, expected {expected} "
                        f"({batch_size} x {nodes_per_graph})"
                    )
                q = q.view(batch_size, nodes_per_graph)
        return q

    def act(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
            epsilon: float = 0.0, valid_mask: torch.Tensor | np.ndarray | None = None,
            rng: np.random.Generator | None = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[0], dtype=torch.bool)
        elif isinstance(valid_mask, np.ndarray):
            valid_mask = torch.from_numpy(valid_mask)
        valid_mask = valid_mask.to(dtype=torch.bool).cpu()

        valid_actions = valid_mask.nonzero(as_tuple=True)[0].numpy()
        if len(valid_actions) == 0:
            valid_mask = torch.ones(x.shape[0], dtype=torch.bool)
            valid_actions = np.arange(x.shape[0])

        if rng.random() < epsilon:
            return int(rng.choice(valid_actions))

        with torch.no_grad():
            was_training = self.training
            self.eval()
            q = self.forward(x.to(DEVICE), edge_index.to(DEVICE), edge_weight.to(DEVICE)).cpu()
            if was_training:
                self.train()
            q = q.clone()
            if q.ndim != 1:
                raise ValueError(f"act() expects single-graph Q values, got shape {tuple(q.shape)}")
            if q.shape[0] != valid_mask.shape[0]:
                raise ValueError(
                    f"valid_mask length {valid_mask.shape[0]} does not match Q length {q.shape[0]}"
                )
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
    importance_weights: np.ndarray | None = None,
) -> Tuple[float, np.ndarray]:
    """Returns (loss, td_errors) — td_errors used for PER priority updates."""
    batch_size = len(batch)
    num_cells = agent.num_cells

    # Flatten states for a single GNN forward pass: (B*N, D)
    states_list = [torch.from_numpy(s).float() for s, *_ in batch]
    next_states_list = [torch.from_numpy(t[3]).float() for t in batch]

    states = torch.cat(states_list, dim=0).to(DEVICE)
    next_states = torch.cat(next_states_list, dim=0).to(DEVICE)

    actions = torch.tensor([t[1] for t in batch], dtype=torch.long, device=DEVICE)
    rewards = torch.tensor([t[2] for t in batch], dtype=torch.float, device=DEVICE)
    dones = torch.tensor([float(t[4]) for t in batch], dtype=torch.float, device=DEVICE)

    if importance_weights is not None:
        is_weights = torch.from_numpy(importance_weights).float().to(DEVICE)
    else:
        is_weights = torch.ones(batch_size, device=DEVICE)

    next_masks = []
    for t in batch:
        if len(t) >= 6 and t[5] is not None:
            next_masks.append(torch.from_numpy(np.asarray(t[5], dtype=bool)))
        else:
            next_masks.append(torch.ones(num_cells, dtype=torch.bool))
    next_mask = torch.stack(next_masks).to(DEVICE)  # (B, num_cells)

    # Replicate edge_index and edge_weight for the batch
    ei = edge_index.to(DEVICE)
    ew = edge_weight.to(DEVICE)

    ei_batch = torch.cat([ei + i * num_cells for i in range(batch_size)], dim=1)
    ew_batch = ew.repeat(batch_size)

    agent_was_training = agent.training
    target_was_training = target_net.training
    target_net.eval()
    agent.eval()
    with torch.no_grad():
        # Target net Q: (B, num_cells)
        q_next_target = target_net(
            next_states,
            ei_batch,
            ew_batch,
            batch_size=batch_size,
            nodes_per_graph=num_cells,
        )

        if cfg.double_dqn:
            # Online net Q for action selection: (B, num_cells)
            q_next_online = agent(
                next_states,
                ei_batch,
                ew_batch,
                batch_size=batch_size,
                nodes_per_graph=num_cells,
            )
            q_next_online = q_next_online.masked_fill(~next_mask, float("-inf"))
            next_actions = q_next_online.argmax(dim=1)
            q_max = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            q_next_masked = q_next_target.masked_fill(~next_mask, float("-inf"))
            q_max = q_next_masked.max(dim=1).values

        q_max = torch.where(torch.isfinite(q_max), q_max, torch.zeros_like(q_max))
        gamma_n = cfg.gamma ** cfg.n_step
        targets = rewards + gamma_n * q_max * (1.0 - dones)
    if agent_was_training:
        agent.train()
    if target_was_training:
        target_net.train()

    # Current Q values: (B, num_cells)
    q_pred_all = agent(
        states,
        ei_batch,
        ew_batch,
        batch_size=batch_size,
        nodes_per_graph=num_cells,
    )
    q_pred = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    td_errors = (q_pred - targets).detach().cpu().numpy()

    # Weighted Huber loss for PER
    element_loss = F.smooth_l1_loss(q_pred, targets, reduction='none')
    loss = (is_weights * element_loss).mean()

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite DQN loss encountered")
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
    optimizer.step()
    return loss.item(), td_errors


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
                    loss_val, _td_errors = _train_step(
                        agent, target_net, batch, edge_index, edge_weight, optimizer, dqn_cfg
                    )
                    losses.append(loss_val)
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

        if verbose and (episode + 1) % 5 == 0:
            print(
                f"[{episode + 1:04d}/{train_episodes}] "
                f"eps={epsilon:.3f} "
                f"avg_thr={metrics['avg_ue_throughput_mbps']:.3f} "
                f"load_std={metrics['load_std']:.3f} "
                f"loss={metrics['loss']:.4f} "
                f"reward={episode_reward:.1f}",
                flush=True
            )

    if best_episode > 0:
        agent.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best checkpoint from episode {best_episode} "
                  f"(reward={best_reward:.1f})")
    target_net.load_state_dict(agent.state_dict())
    agent = agent.cpu()
    return agent, history
