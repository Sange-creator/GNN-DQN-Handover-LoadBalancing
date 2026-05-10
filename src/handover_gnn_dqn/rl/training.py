from __future__ import annotations

import copy
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from torch import nn

from ..env import CellularNetworkEnv, LTEConfig
from ..metrics import default_policy_factories, evaluate_policies, write_summary_csv
from ..models import DQNConfig, GnnDQNAgent, ReplayBuffer
from ..models.gnn_dqn import Transition, _train_step
from ..topology import Scenario

MODEL_VERSION = "gnn_dqn_v3_graph_value_head"
REWARD_VERSION = "throughput_fairness_pingpong_v3"


class ScenarioReplayBuffer:
    """Per-scenario replay buffers to keep graph topology and transitions aligned."""

    def __init__(self, capacity_per_scenario: int):
        self.capacity_per_scenario = capacity_per_scenario
        self.buffers: dict[str, ReplayBuffer] = {}

    def add(self, scenario_name: str, item: Transition) -> None:
        if scenario_name not in self.buffers:
            self.buffers[scenario_name] = ReplayBuffer(self.capacity_per_scenario)
        self.buffers[scenario_name].add(item)

    def can_sample(self, scenario_name: str, batch_size: int) -> bool:
        buf = self.buffers.get(scenario_name)
        return buf is not None and len(buf) >= batch_size

    def sample(self, scenario_name: str, rng: np.random.Generator, batch_size: int) -> List[Transition]:
        return self.buffers[scenario_name].sample(rng, batch_size)

    def state_dict(self) -> dict:
        return {
            "capacity_per_scenario": self.capacity_per_scenario,
            "buffers": {name: buf.state_dict() for name, buf in self.buffers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity_per_scenario = int(
            state.get("capacity_per_scenario", self.capacity_per_scenario)
        )
        self.buffers = {}
        for name, buf_state in state.get("buffers", {}).items():
            buf = ReplayBuffer(int(buf_state.get("capacity", self.capacity_per_scenario)))
            buf.load_state_dict(buf_state)
            self.buffers[name] = buf


def make_env_from_scenario(scenario: Scenario, feature_mode: str = "ue_only", prb_available: bool = True) -> CellularNetworkEnv:
    cfg = LTEConfig(
        num_cells=scenario.num_cells,
        num_ues=scenario.num_ues,
        area_m=scenario.area_m,
        cell_positions=scenario.cell_positions,
        min_speed_mps=scenario.min_speed_mps,
        max_speed_mps=scenario.max_speed_mps,
        shadow_sigma_db=scenario.shadow_sigma_db,
        min_demand_mbps=scenario.min_demand_mbps,
        max_demand_mbps=scenario.max_demand_mbps,
        mobility_model=scenario.mobility_model,
        feature_mode=feature_mode,
        prb_available=prb_available,
    )
    return CellularNetworkEnv(cfg)


def training_validation_score(metrics: dict[str, float]) -> float:
    """Single scalar used only for best-checkpoint selection.

    Weights tuned to select policies that beat A3/TTT: high throughput,
    strong load fairness, low handover rate (A3 does ~12/k decisions).
    """
    return float(
        2.0 * metrics.get("p5_ue_throughput_mbps", 0.0)
        + 1.0 * metrics.get("avg_ue_throughput_mbps", 0.0)
        + 1.2 * metrics.get("jain_load_fairness", 0.0)
        - 0.5 * metrics.get("load_std", 0.0)
        - 2.0 * metrics.get("outage_rate", 0.0)
        - 1.5 * metrics.get("pingpong_rate", 0.0)
        - 0.005 * metrics.get("handovers_per_1000_decisions", 0.0)
    )


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def train_multi_scenario(
    scenarios: List[Scenario],
    dqn_cfg: DQNConfig,
    total_episodes: int,
    steps_per_episode: int,
    feature_mode: str = "ue_only",
    prb_available: bool = True,
    seed: int = 42,
    verbose: bool = True,
    resume_payload: dict | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every_episodes: int = 0,
    checkpoint_include_replay: bool = False,
    checkpoint_config: dict | None = None,
    cwd: Path | None = None,
) -> tuple[GnnDQNAgent, list[dict], int]:
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    per_scenario_cap = max(dqn_cfg.replay_capacity // max(len(scenarios), 1), dqn_cfg.batch_size)
    replay = ScenarioReplayBuffer(per_scenario_cap)
    decisions = 0
    start_episode = 0

    max_cells = max(s.num_cells for s in scenarios)
    sample_env = make_env_from_scenario(scenarios[0], feature_mode=feature_mode, prb_available=prb_available)
    feature_dim = sample_env.feature_dim

    agent = GnnDQNAgent(max_cells, feature_dim, dqn_cfg, seed=seed)
    target_net = copy.deepcopy(agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=dqn_cfg.learning_rate, weight_decay=dqn_cfg.weight_decay)
    best_score = float("-inf")
    best_state = copy.deepcopy(agent.state_dict())
    best_episode = 0
    scenario_edge_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    avg_ues = sum(s.num_ues for s in scenarios) / max(len(scenarios), 1)
    est_grad_steps = max(int(steps_per_episode * avg_ues * total_episodes) // dqn_cfg.train_every, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=est_grad_steps, eta_min=dqn_cfg.lr_min)

    if resume_payload is not None:
        agent.load_state_dict(resume_payload["state_dict"])
        target_net.load_state_dict(resume_payload.get("target_state_dict", resume_payload["state_dict"]))
        if resume_payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if resume_payload.get("replay_state") is not None:
            replay.load_state_dict(resume_payload["replay_state"])

        training_state = resume_payload.get("training_state") or resume_payload.get("metadata", {}).get("training_state", {})
        decisions = int(training_state.get("decisions", 0))
        start_episode = int(training_state.get("episode_completed", 0))
        best_score = float(training_state.get("best_score", float("-inf")))
        best_episode = int(training_state.get("best_episode", 0))
        if resume_payload.get("best_state_dict") is not None:
            best_state = copy.deepcopy(resume_payload["best_state_dict"])
        else:
            best_state = copy.deepcopy(agent.state_dict())
        if training_state.get("rng_state") is not None:
            rng.bit_generator.state = training_state["rng_state"]
        if training_state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
        history = list(resume_payload.get("history", []))

    if verbose:
        print(f"  Model parameters: {sum(p.numel() for p in agent.parameters()):,}")
        print(f"  Max training cells: {max_cells}, features: {feature_dim}, profile: {feature_mode}")
        print(f"  Replay: {per_scenario_cap:,} transitions/scenario")
        if start_episode:
            print(f"  Resuming from episode {start_episode}/{total_episodes}")

    scenario_order = np.arange(len(scenarios))
    for episode in range(start_episode, total_episodes):
        offset = episode - start_episode
        if offset % len(scenarios) == 0:
            scenario_order = rng.permutation(len(scenarios))
        scenario = scenarios[int(scenario_order[offset % len(scenarios)])]
        env = make_env_from_scenario(scenario, feature_mode=feature_mode, prb_available=prb_available)
        env.reset(seed + 101 * episode)
        num_cells = scenario.num_cells

        if scenario.name not in scenario_edge_cache:
            scenario_edge_cache[scenario.name] = env.edge_data
        edge_index, edge_weight = scenario_edge_cache[scenario.name]

        def pad_state(state: np.ndarray) -> np.ndarray:
            if num_cells >= max_cells:
                return state
            return np.vstack(
                [state, np.zeros((max_cells - num_cells, feature_dim), dtype=np.float32)]
            )

        def pad_mask(mask: np.ndarray) -> np.ndarray:
            if num_cells >= max_cells:
                return mask.astype(bool)
            out = np.zeros(max_cells, dtype=bool)
            out[:num_cells] = mask
            return out

        losses: list[float] = []
        episode_reward = 0.0
        invalid_actions = 0
        nan_state_count = 0
        frac = min(episode / max(dqn_cfg.epsilon_decay_episodes, 1), 1.0)
        epsilon = dqn_cfg.epsilon_start + frac * (dqn_cfg.epsilon_end - dqn_cfg.epsilon_start)

        for _step in range(steps_per_episode):
            env.advance_mobility()
            for ue_idx in rng.permutation(env.cfg.num_ues):
                ue_idx = int(ue_idx)
                state = pad_state(env.build_state(ue_idx))
                if not np.isfinite(state).all():
                    nan_state_count += 1
                    raise RuntimeError(f"Non-finite state in scenario {scenario.name}, episode {episode + 1}")
                valid = pad_mask(env.valid_actions(ue_idx))
                action = agent.act(
                    torch.from_numpy(state).float(),
                    edge_index,
                    edge_weight,
                    epsilon=epsilon,
                    valid_mask=valid,
                    rng=rng,
                )
                if not 0 <= action < num_cells:
                    invalid_actions += 1
                    raise RuntimeError(
                        f"Invalid padded action {action} for scenario {scenario.name} with {num_cells} cells"
                    )

                next_state, reward, done, _info = env.step_user_action(ue_idx, action)
                next_state_padded = pad_state(next_state)
                next_valid = pad_mask(env.valid_actions(ue_idx))
                replay.add(scenario.name, (state, action, reward, next_state_padded, done, next_valid))
                episode_reward += reward

                if replay.can_sample(scenario.name, dqn_cfg.batch_size) and decisions % dqn_cfg.train_every == 0:
                    batch = replay.sample(scenario.name, rng, dqn_cfg.batch_size)
                    losses.append(_train_step(agent, target_net, batch, edge_index, edge_weight, optimizer, dqn_cfg))
                    scheduler.step()
                    if dqn_cfg.tau > 0:
                        _soft_update(target_net, agent, dqn_cfg.tau)
                if dqn_cfg.tau == 0 and decisions % dqn_cfg.target_update_every == 0:
                    target_net.load_state_dict(agent.state_dict())
                decisions += 1

        metrics = env.metrics()
        episode_decisions = max(steps_per_episode * env.cfg.num_ues, 1)
        metrics["handovers_per_1000_decisions"] = float(1000.0 * env.total_handovers / episode_decisions)
        metrics["pingpong_rate"] = float(env.pingpong_handovers / max(env.total_handovers, 1))
        metrics["weak_target_ho_rate"] = float(env.weak_target_handovers / max(env.total_handovers, 1))
        metrics.update(
            {
                "episode": float(episode + 1),
                "epsilon": float(epsilon),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "episode_reward": float(episode_reward),
                "validation_score": training_validation_score(metrics),
                "scenario": scenario.name,
                "feature_mode": feature_mode,
                "invalid_action_count": float(invalid_actions),
                "nan_state_count": float(nan_state_count),
            }
        )
        history.append(metrics)

        validation_score = float(metrics["validation_score"])
        if episode >= max(5, total_episodes // 10) and validation_score > best_score:
            best_score = validation_score
            best_state = copy.deepcopy(agent.state_dict())
            best_episode = episode + 1

        if verbose and (episode + 1) % max(1, total_episodes // 20) == 0:
            print(
                f"[{episode + 1:04d}/{total_episodes}] {scenario.name:18s} "
                f"eps={epsilon:.3f} thr={metrics['avg_ue_throughput_mbps']:.3f} "
                f"p5={metrics['p5_ue_throughput_mbps']:.3f} "
                f"load_std={metrics['load_std']:.3f} "
                f"pp={metrics['pingpong_rate']:.3f} score={metrics['validation_score']:.3f} "
                f"loss={metrics['loss']:.4f}",
                flush=True,
            )

        if checkpoint_dir is not None and checkpoint_every_episodes > 0:
            if (episode + 1) % checkpoint_every_episodes == 0 or (episode + 1) == total_episodes:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    agent,
                    checkpoint_dir / f"resume_ep{episode + 1:04d}.pt",
                    config=checkpoint_config or {},
                    feature_dim=feature_dim,
                    max_cells=max_cells,
                    scenario_names=[s.name for s in scenarios],
                    history=history,
                    cwd=cwd or Path.cwd(),
                    target_net=target_net,
                    optimizer=optimizer,
                    replay=replay if checkpoint_include_replay else None,
                    training_state={
                        "episode_completed": episode + 1,
                        "decisions": decisions,
                        "best_episode": best_episode,
                        "best_score": best_score,
                        "rng_state": rng.bit_generator.state,
                        "replay_included": checkpoint_include_replay,
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    best_state_dict=best_state,
                    checkpoint_kind="resume",
                )

    if best_episode:
        agent.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best checkpoint from episode {best_episode} (score={best_score:.3f})")
    return agent, history, max_cells


def git_commit_hash(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return "unknown"


def save_checkpoint(
    agent: GnnDQNAgent,
    path: Path,
    *,
    config: dict,
    feature_dim: int,
    max_cells: int,
    scenario_names: Iterable[str],
    history: list[dict],
    cwd: Path,
    target_net: GnnDQNAgent | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    replay: ScenarioReplayBuffer | None = None,
    training_state: dict | None = None,
    best_state_dict: dict | None = None,
    checkpoint_kind: str = "model",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_profile = config.get("feature_mode", "ue_only")
    dqn_cfg = config.get("dqn", {})
    payload = {
        "state_dict": agent.state_dict(),
        "target_state_dict": target_net.state_dict() if target_net is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "replay_state": replay.state_dict() if replay is not None else None,
        "training_state": training_state,
        "best_state_dict": best_state_dict,
        "history": history,
        "metadata": {
            "checkpoint_kind": checkpoint_kind,
            "model_version": MODEL_VERSION,
            "reward_version": REWARD_VERSION,
            "model_class": agent.__class__.__name__,
            "config": config,
            "dqn_config": dqn_cfg,
            "feature_profile": feature_profile,
            "prb_available": bool(config.get("prb_available", feature_profile != "ue_only")),
            "feature_dim": feature_dim,
            "max_cells": max_cells,
            "scenario_names": list(scenario_names),
            "git_commit": git_commit_hash(cwd),
            "history_last": history[-1] if history else None,
            "best_episode": training_state.get("best_episode") if training_state else None,
            "best_score": training_state.get("best_score") if training_state else None,
            "training_state": training_state,
            "saved_at_unix": time.time(),
        },
    }
    torch.save(payload, path)


def load_checkpoint_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload or "metadata" not in payload:
        raise ValueError(f"{path} is not a compatible GNN-DQN checkpoint")
    return payload


def validate_checkpoint_metadata(
    metadata: dict,
    *,
    expected_feature_profile: str | None = None,
    expected_feature_dim: int | None = None,
    expected_model_version: str | None = MODEL_VERSION,
) -> None:
    problems: list[str] = []
    if metadata.get("model_class") != "GnnDQNAgent":
        problems.append(f"model_class={metadata.get('model_class')!r}")
    if expected_model_version is not None and metadata.get("model_version") != expected_model_version:
        problems.append(
            f"model_version={metadata.get('model_version')!r}, expected {expected_model_version!r}"
        )
    if expected_feature_profile is not None and metadata.get("feature_profile") != expected_feature_profile:
        problems.append(
            f"feature_profile={metadata.get('feature_profile')!r}, expected {expected_feature_profile!r}"
        )
    if expected_feature_dim is not None and int(metadata.get("feature_dim", -1)) != expected_feature_dim:
        problems.append(
            f"feature_dim={metadata.get('feature_dim')!r}, expected {expected_feature_dim}"
        )
    if problems:
        raise ValueError("Incompatible checkpoint: " + "; ".join(problems))


def load_gnn_checkpoint(
    path: Path,
    *,
    expected_feature_profile: str | None = None,
    expected_feature_dim: int | None = None,
    strict_metadata: bool = True,
) -> tuple[GnnDQNAgent, dict, dict]:
    payload = load_checkpoint_payload(path)
    metadata = payload["metadata"]
    if strict_metadata:
        validate_checkpoint_metadata(
            metadata,
            expected_feature_profile=expected_feature_profile,
            expected_feature_dim=expected_feature_dim,
        )
    cfg = metadata.get("config", {})
    dqn_cfg = DQNConfig(**cfg.get("dqn", {}))
    agent = GnnDQNAgent(int(metadata["max_cells"]), int(metadata["feature_dim"]), dqn_cfg)
    agent.load_state_dict(payload["state_dict"])
    agent.eval()
    return agent, metadata, payload


def write_history(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))


def evaluate_and_write(
    scenarios: List[Scenario],
    agent: GnnDQNAgent,
    out_dir: Path,
    *,
    feature_mode: str,
    prb_available: bool,
    steps: int,
    seeds: Iterable[int],
) -> None:
    for scenario in scenarios:
        env = make_env_from_scenario(scenario, feature_mode=feature_mode, prb_available=prb_available)
        rows = evaluate_policies(
            env.cfg,
            default_policy_factories(gnn_agent=agent),
            steps=steps,
            seeds=seeds,
        )
        rows = sorted(rows, key=lambda r: r["avg_ue_throughput_mbps"], reverse=True)
        write_summary_csv(rows, out_dir / f"{scenario.name}.csv")
