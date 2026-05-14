from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from torch import nn

from ..env import CellularNetworkEnv, LTEConfig
from ..metrics import default_policy_factories, evaluate_policies, write_summary_csv
from ..models import DQNConfig, FlatDQNAgent, GnnDQNAgent, NStepBuffer, PrioritizedReplayBuffer, ReplayBuffer
from ..models.flat_dqn import _flat_train_step
from ..models.gnn_dqn import Transition, _train_step
from ..policies import A3HandoverPolicy, LoadAwarePolicy, StrongestRsrpPolicy
from ..son import SONConfig
from ..topology import Scenario

MODEL_VERSION = "gnn_dqn_v4_layernorm_per_nstep"
REWARD_VERSION = "speed_aware_proactive_v5_son_margin"

torch.set_num_threads(int(os.environ.get("GNN_DQN_NUM_THREADS", "6")))


def dqn_config_from_dict(values: dict) -> DQNConfig:
    valid = {f.name for f in fields(DQNConfig)}
    return DQNConfig(**{k: v for k, v in values.items() if k in valid})


def son_config_from_dict(values: dict | None) -> SONConfig:
    if not values:
        return SONConfig()
    if "son_config" in values and isinstance(values["son_config"], dict):
        values = values["son_config"]
    valid = {f.name for f in fields(SONConfig)}
    return SONConfig(**{k: v for k, v in values.items() if k in valid})


def son_config_to_dict(config: SONConfig) -> dict:
    return asdict(config)


class ScenarioReplayBuffer:
    """Per-scenario replay buffers with optional PER to keep graph topology aligned."""

    def __init__(self, capacity_per_scenario: int, use_per: bool = False, per_alpha: float = 0.6):
        self.capacity_per_scenario = capacity_per_scenario
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.buffers: dict[str, ReplayBuffer | PrioritizedReplayBuffer] = {}

    def add(self, scenario_name: str, item: Transition) -> None:
        if scenario_name not in self.buffers:
            if self.use_per:
                self.buffers[scenario_name] = PrioritizedReplayBuffer(
                    self.capacity_per_scenario, alpha=self.per_alpha
                )
            else:
                self.buffers[scenario_name] = ReplayBuffer(self.capacity_per_scenario)
        self.buffers[scenario_name].add(item)

    def can_sample(self, scenario_name: str, batch_size: int) -> bool:
        buf = self.buffers.get(scenario_name)
        return buf is not None and len(buf) >= batch_size

    def sample(self, scenario_name: str, rng: np.random.Generator, batch_size: int,
               beta: float = 0.4):
        buf = self.buffers[scenario_name]
        if self.use_per and isinstance(buf, PrioritizedReplayBuffer):
            return buf.sample(rng, batch_size, beta=beta)
        return buf.sample(rng, batch_size), None, None

    def update_priorities(self, scenario_name: str, indices: np.ndarray, td_errors: np.ndarray) -> None:
        if self.use_per and scenario_name in self.buffers:
            buf = self.buffers[scenario_name]
            if isinstance(buf, PrioritizedReplayBuffer):
                buf.update_priorities(indices, td_errors)

    def state_dict(self) -> dict:
        return {
            "capacity_per_scenario": self.capacity_per_scenario,
            "use_per": self.use_per,
            "buffers": {name: buf.state_dict() for name, buf in self.buffers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity_per_scenario = int(
            state.get("capacity_per_scenario", self.capacity_per_scenario)
        )
        self.use_per = bool(state.get("use_per", self.use_per))
        self.buffers = {}
        for name, buf_state in state.get("buffers", {}).items():
            cap = int(buf_state.get("capacity", self.capacity_per_scenario))
            if self.use_per:
                buf = PrioritizedReplayBuffer(cap, alpha=self.per_alpha)
            else:
                buf = ReplayBuffer(cap)
            buf.load_state_dict(buf_state)
            self.buffers[name] = buf


def make_env_from_scenario(
    scenario: Scenario,
    feature_mode: str = "ue_only",
    prb_available: bool = True,
    num_ues: int | None = None,
) -> CellularNetworkEnv:
    cfg = LTEConfig(
        num_cells=scenario.num_cells,
        num_ues=int(num_ues) if num_ues is not None else scenario.num_ues,
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


def _pct_gain(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline) / max(abs(baseline), 1e-9)


def score_son_against_baselines(rows: list[dict[str, float]]) -> dict[str, float]:
    """Score validation results by deployable SON margin, not raw GNN reward."""
    by_method = {row["method"]: row for row in rows}
    if "son_gnn_dqn" not in by_method:
        raise ValueError("son_gnn_dqn row is required for SON validation scoring")
    son = by_method["son_gnn_dqn"]
    classical_names = ["no_handover", "strongest_rsrp", "a3_ttt", "load_aware"]
    classical_rows = [by_method[name] for name in classical_names if name in by_method]
    if not classical_rows:
        raise ValueError("At least one classical baseline row is required")

    best_avg = max(classical_rows, key=lambda r: r.get("avg_ue_throughput_mbps", 0.0))
    best_p5 = max(classical_rows, key=lambda r: r.get("p5_ue_throughput_mbps", 0.0))
    best_load = min(classical_rows, key=lambda r: r.get("load_std", float("inf")))
    best_pingpong = min(classical_rows, key=lambda r: r.get("pingpong_rate", float("inf")))
    best_outage = min(classical_rows, key=lambda r: r.get("outage_rate", float("inf")))

    avg_margin = _pct_gain(
        son.get("avg_ue_throughput_mbps", 0.0), best_avg.get("avg_ue_throughput_mbps", 0.0)
    )
    p5_margin = _pct_gain(
        son.get("p5_ue_throughput_mbps", 0.0), best_p5.get("p5_ue_throughput_mbps", 0.0)
    )
    load_std_reduction = _pct_gain(
        best_load.get("load_std", 0.0), son.get("load_std", 0.0)
    )
    pingpong_reduction = _pct_gain(
        best_pingpong.get("pingpong_rate", 0.0), son.get("pingpong_rate", 0.0)
    )
    outage_delta = son.get("outage_rate", 0.0) - best_outage.get("outage_rate", 0.0)
    random_row = by_method.get("random_valid")
    random_margin = (
        _pct_gain(
            son.get("avg_ue_throughput_mbps", 0.0),
            random_row.get("avg_ue_throughput_mbps", 0.0),
        )
        if random_row is not None
        else 0.0
    )

    score = (
        training_validation_score(son)
        + 0.08 * avg_margin
        + 0.10 * p5_margin
        + 0.04 * load_std_reduction
        + 0.03 * pingpong_reduction
        - 12.0 * max(outage_delta, 0.0)
        - 0.05 * max(-random_margin, 0.0)
    )
    return {
        "holdout_validation_score": float(score),
        "son_avg_margin_pct": float(avg_margin),
        "son_p5_margin_pct": float(p5_margin),
        "son_load_std_reduction_pct": float(load_std_reduction),
        "son_pingpong_reduction_pct": float(pingpong_reduction),
        "son_outage_delta": float(outage_delta),
        "son_random_avg_margin_pct": float(random_margin),
        "best_classical_avg_ue_throughput_mbps": float(best_avg.get("avg_ue_throughput_mbps", 0.0)),
        "best_classical_p5_ue_throughput_mbps": float(best_p5.get("p5_ue_throughput_mbps", 0.0)),
        "best_classical_load_std": float(best_load.get("load_std", 0.0)),
        "best_classical_pingpong_rate": float(best_pingpong.get("pingpong_rate", 0.0)),
    }


def training_validation_score(metrics: dict[str, float]) -> float:
    """Single scalar used only for best-checkpoint selection.

    Weights tuned to select policies that decisively beat A3/TTT:
    - Heavy P5 weight (protects worst-case users — Nepal's biggest complaint)
    - Load fairness (prevents cell congestion hotspots)
    - Low pingpong (stability matters for voice calls)
    - Moderate HO rate penalty (don't over-penalize necessary highway HOs)
    """
    return float(
        2.5 * metrics.get("p5_ue_throughput_mbps", 0.0)
        + 1.5 * metrics.get("avg_ue_throughput_mbps", 0.0)
        + 1.5 * metrics.get("jain_load_fairness", 0.0)
        - 0.8 * metrics.get("load_std", 0.0)
        - 3.0 * metrics.get("outage_rate", 0.0)
        - 2.0 * metrics.get("pingpong_rate", 0.0)
        - 0.003 * metrics.get("handovers_per_1000_decisions", 0.0)
    )


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def steps_for_episode(
    default_steps: int,
    episode_index: int,
    curriculum: dict | None = None,
) -> int:
    """Return the configured RL step count for a zero-based episode index."""
    if not curriculum:
        return int(default_steps)
    phase1_episodes = int(curriculum.get("phase1_episodes", 0))
    phase2_episodes = int(curriculum.get("phase2_episodes", 0))
    if phase1_episodes > 0 and episode_index < phase1_episodes:
        return int(curriculum.get("phase1_steps", default_steps))
    if phase2_episodes > 0 and episode_index < phase2_episodes:
        return int(curriculum.get("phase2_steps", default_steps))
    return int(curriculum.get("phase3_steps", default_steps))


def capped_num_ues(scenario: Scenario, scenario_ue_caps: dict[str, int] | None = None) -> int:
    """Return the per-episode training UE count for a scenario."""
    if not scenario_ue_caps or scenario.name not in scenario_ue_caps:
        return int(scenario.num_ues)
    return max(1, min(int(scenario_ue_caps[scenario.name]), int(scenario.num_ues)))


def _run_validation_pass(
    agent: GnnDQNAgent,
    val_scenarios: List[Scenario],
    seeds: list[int],
    steps: int,
    feature_mode: str,
    prb_available: bool,
    *,
    son_config: SONConfig | None = None,
    validation_ue_cap: int | None = None,
    validation_steps_override: int | None = None,
) -> dict[str, float]:
    """Held-out validation by SON margin against the classical baseline pack."""

    was_training = agent.training
    agent.eval()
    try:
        per_scenario_scores: list[float] = []
        aggregated: dict[str, list[float]] = {}
        method_aggregated: dict[str, dict[str, list[float]]] = {}
        for scenario in val_scenarios:
            ue_cap = (
                min(int(validation_ue_cap), scenario.num_ues)
                if validation_ue_cap is not None and validation_ue_cap > 0
                else None
            )
            env = make_env_from_scenario(
                scenario, feature_mode=feature_mode, prb_available=prb_available, num_ues=ue_cap
            )
            rows = evaluate_policies(
                env.cfg,
                default_policy_factories(gnn_agent=agent, son_config=son_config, include_true_prb=False),
                steps=int(validation_steps_override or steps),
                seeds=seeds,
            )
            score_result = score_son_against_baselines(rows)
            per_scenario_scores.append(score_result["holdout_validation_score"])
            for k, v in score_result.items():
                aggregated.setdefault(k, []).append(float(v))
            for row in rows:
                bucket = method_aggregated.setdefault(row["method"], {})
                for k, v in row.items():
                    if k != "method" and isinstance(v, (int, float, np.floating)):
                        bucket.setdefault(k, []).append(float(v))
    finally:
        if was_training:
            agent.train()

    result: dict[str, float] = {k: float(np.mean(vs)) for k, vs in aggregated.items()}
    son_bucket = method_aggregated.get("son_gnn_dqn", {})
    for k, vs in son_bucket.items():
        result[k] = float(np.mean(vs))
    result["holdout_validation_score"] = float(np.mean(per_scenario_scores))
    result["holdout_validation_score_std"] = float(np.std(per_scenario_scores))
    return result


def _teacher_for_scenario(scenario_name: str):
    if scenario_name in {"highway", "highway_fast", "sparse_rural"}:
        return StrongestRsrpPolicy(hysteresis_db=2.0)
    if scenario_name in {"dense_urban", "overloaded_event", "real_pokhara", "pokhara_dense_peakhour"}:
        return LoadAwarePolicy(load_weight=0.48, handover_cost=0.04)
    return A3HandoverPolicy(offset_db=3.0, time_to_trigger=3)


def _batched_q_logits(
    agent: GnnDQNAgent,
    states: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_cells, feature_dim = states.shape
    states_t = torch.from_numpy(states).float().reshape(batch_size * num_cells, feature_dim)
    ei_batch = torch.cat([edge_index + i * num_cells for i in range(batch_size)], dim=1)
    ew_batch = edge_weight.repeat(batch_size)
    return agent(states_t, ei_batch, ew_batch, batch_size=batch_size, nodes_per_graph=num_cells)


def _run_behavioral_cloning(
    agent: GnnDQNAgent,
    scenarios: List[Scenario],
    dqn_cfg: DQNConfig,
    *,
    episodes: int,
    steps_per_episode: int,
    feature_mode: str,
    prb_available: bool,
    seed: int,
    scenario_sampling_weights: dict[str, float] | None = None,
    scenario_ue_caps: dict[str, int] | None = None,
    verbose: bool = True,
) -> list[dict]:
    if episodes <= 0:
        return []

    rng = np.random.default_rng(seed + 13_000)
    optimizer = torch.optim.Adam(agent.parameters(), lr=dqn_cfg.learning_rate, weight_decay=dqn_cfg.weight_decay)
    max_cells = agent.num_cells
    feature_dim = agent.feature_dim
    sampling_probs: np.ndarray | None = None
    if scenario_sampling_weights:
        weights = np.array(
            [max(float(scenario_sampling_weights.get(s.name, 1.0)), 0.0) for s in scenarios],
            dtype=float,
        )
        if float(weights.sum()) > 0.0:
            sampling_probs = weights / weights.sum()
    scenario_order = np.arange(len(scenarios))
    history: list[dict] = []

    for episode in range(episodes):
        if sampling_probs is not None:
            scenario = scenarios[int(rng.choice(len(scenarios), p=sampling_probs))]
        else:
            if episode % len(scenarios) == 0:
                scenario_order = rng.permutation(len(scenarios))
            scenario = scenarios[int(scenario_order[episode % len(scenarios)])]
        env = make_env_from_scenario(
            scenario,
            feature_mode=feature_mode,
            prb_available=prb_available,
            num_ues=capped_num_ues(scenario, scenario_ue_caps),
        )
        env.reset(seed + 17_000 + 101 * episode)
        teacher = _teacher_for_scenario(scenario.name)
        teacher.reset(env)
        edge_index, edge_weight = env.edge_data
        losses: list[float] = []
        correct = 0
        total = 0
        repaired_teacher_targets = 0

        for _step in range(steps_per_episode):
            env.advance_mobility()
            states_now = env.build_all_states()
            valid_now = env.valid_actions_all()
            states = np.zeros((env.cfg.num_ues, max_cells, feature_dim), dtype=np.float32)
            valid = np.zeros((env.cfg.num_ues, max_cells), dtype=bool)
            states[:, : env.cfg.num_cells, :] = states_now
            valid[:, : env.cfg.num_cells] = valid_now

            ue_order = rng.permutation(env.cfg.num_ues).astype(int)
            targets = np.array([teacher.select(env, int(ue_idx)) for ue_idx in ue_order], dtype=np.int64)
            target_valid = valid_now[ue_order, targets]
            if not np.all(target_valid):
                bad = ~target_valid
                repaired_teacher_targets += int(np.sum(bad))
                targets[bad] = env.serving[ue_order[bad]]
            logits = _batched_q_logits(agent, states[ue_order], edge_index, edge_weight)
            mask = torch.from_numpy(valid[ue_order]).bool()
            logits = logits.masked_fill(~mask, -1e9)
            target_t = torch.from_numpy(targets).long()
            loss = torch.nn.functional.cross_entropy(logits, target_t)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite behavioral cloning loss encountered")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), dqn_cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            correct += int(np.sum(pred == targets))
            total += int(len(targets))

            for ue_idx, action in zip(ue_order, targets):
                env.step_user_action(int(ue_idx), int(action))

        metrics = env.metrics()
        metrics.update(
            {
                "episode": float(episode + 1),
                "phase": "behavioral_cloning",
                "scenario": scenario.name,
                "bc_loss": float(np.mean(losses)) if losses else 0.0,
                "bc_accuracy": float(correct / max(total, 1)),
                "bc_repaired_teacher_targets": float(repaired_teacher_targets),
                "num_ues": float(env.cfg.num_ues),
                "feature_mode": feature_mode,
            }
        )
        history.append(metrics)
        if verbose and (episode + 1) % max(1, episodes // 10) == 0:
            print(
                f"[BC {episode + 1:04d}/{episodes}] {scenario.name:18s} "
                f"loss={metrics['bc_loss']:.4f} acc={metrics['bc_accuracy']:.3f} "
                f"thr={metrics['avg_ue_throughput_mbps']:.3f}",
                flush=True,
            )
    return history


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
    validation_scenarios: List[Scenario] | None = None,
    validation_seeds: list[int] | None = None,
    validate_every_episodes: int = 0,
    validation_steps: int = 0,
    validation_ue_cap: int | None = None,
    validation_steps_override: int | None = None,
    skip_validation_epsilon_above: float | None = None,
    early_stopping_min_episodes: int = 0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    scenario_sampling_weights: dict[str, float] | None = None,
    son_config: SONConfig | None = None,
    behavioral_clone_episodes: int = 0,
    log_every_episodes: int | None = None,
    steps_per_episode_curriculum: dict | None = None,
    scenario_ue_caps: dict[str, int] | None = None,
) -> tuple[GnnDQNAgent, list[dict], int]:
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    per_scenario_cap = max(dqn_cfg.replay_capacity // max(len(scenarios), 1), dqn_cfg.batch_size)
    use_per = dqn_cfg.per_alpha > 0.0
    replay = ScenarioReplayBuffer(per_scenario_cap, use_per=use_per, per_alpha=dqn_cfg.per_alpha)
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
    validation_non_improve_count = 0
    early_stopped = False
    scenario_edge_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    sampling_probs: np.ndarray | None = None
    if scenario_sampling_weights:
        weights = np.array(
            [max(float(scenario_sampling_weights.get(s.name, 1.0)), 0.0) for s in scenarios],
            dtype=float,
        )
        if float(weights.sum()) > 0.0:
            sampling_probs = weights / weights.sum()

    avg_ues = sum(capped_num_ues(s, scenario_ue_caps) for s in scenarios) / max(len(scenarios), 1)
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
        if steps_per_episode_curriculum:
            print(f"  Step curriculum: {steps_per_episode_curriculum}")
        if scenario_ue_caps:
            active_caps = {
                s.name: capped_num_ues(s, scenario_ue_caps)
                for s in scenarios
                if capped_num_ues(s, scenario_ue_caps) != s.num_ues
            }
            if active_caps:
                print(f"  Training UE caps: {active_caps}")
        if start_episode:
            print(f"  Resuming from episode {start_episode}/{total_episodes}")

    if behavioral_clone_episodes > 0 and start_episode == 0:
        bc_history = _run_behavioral_cloning(
            agent,
            scenarios,
            dqn_cfg,
            episodes=behavioral_clone_episodes,
            steps_per_episode=steps_per_episode,
            feature_mode=feature_mode,
            prb_available=prb_available,
            seed=seed,
            scenario_sampling_weights=scenario_sampling_weights,
            scenario_ue_caps=scenario_ue_caps,
            verbose=verbose,
        )
        history.extend(bc_history)
        target_net.load_state_dict(agent.state_dict())
        if checkpoint_dir is not None:
            save_checkpoint(
                agent,
                checkpoint_dir.parent / "bc_warmstart.pt",
                config=checkpoint_config or {},
                feature_dim=feature_dim,
                max_cells=max_cells,
                scenario_names=[s.name for s in scenarios],
                history=bc_history,
                cwd=cwd or Path.cwd(),
                target_net=target_net,
                optimizer=optimizer,
                training_state={
                    "behavioral_clone_episodes": behavioral_clone_episodes,
                    "phase": "behavioral_cloning",
                },
                checkpoint_kind="bc_warmstart",
            )

    scenario_order = np.arange(len(scenarios))
    for episode in range(start_episode, total_episodes):
        offset = episode - start_episode
        if sampling_probs is not None:
            scenario = scenarios[int(rng.choice(len(scenarios), p=sampling_probs))]
        else:
            if offset % len(scenarios) == 0:
                scenario_order = rng.permutation(len(scenarios))
            scenario = scenarios[int(scenario_order[offset % len(scenarios)])]
        env = make_env_from_scenario(
            scenario,
            feature_mode=feature_mode,
            prb_available=prb_available,
            num_ues=capped_num_ues(scenario, scenario_ue_caps),
        )
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
        steps_this_episode = steps_for_episode(
            steps_per_episode,
            episode,
            steps_per_episode_curriculum,
        )

        # PER beta annealing: starts low, reaches 1.0 by end of training
        beta_frac = min(episode / max(total_episodes - 1, 1), 1.0)
        per_beta = dqn_cfg.per_beta_start + beta_frac * (dqn_cfg.per_beta_end - dqn_cfg.per_beta_start)

        # N-step buffers per UE for this episode
        n_step_buffers: dict[int, NStepBuffer] = {}
        if dqn_cfg.n_step > 1:
            for ue in range(env.cfg.num_ues):
                n_step_buffers[ue] = NStepBuffer(dqn_cfg.n_step, dqn_cfg.gamma)

        for _step in range(steps_this_episode):
            env.advance_mobility()

            # --- Batched inference: build all states, one forward pass ---
            ue_order = rng.permutation(env.cfg.num_ues).astype(int)
            states_now = env.build_all_states()
            valid_now = env.valid_actions_all()
            if not np.isfinite(states_now).all():
                nan_state_count += 1
                raise RuntimeError(f"Non-finite state in scenario {scenario.name}, episode {episode + 1}")
            all_states = np.zeros((env.cfg.num_ues, max_cells, feature_dim), dtype=np.float32)
            all_valid = np.zeros((env.cfg.num_ues, max_cells), dtype=bool)
            all_states[:, :num_cells, :] = states_now
            all_valid[:, :num_cells] = valid_now

            actions = agent.act_batch(
                all_states[ue_order], edge_index, edge_weight,
                all_valid[ue_order], epsilon, rng,
            )

            # --- Sequential env interaction (preserves mutation semantics) ---
            for i, ue_idx in enumerate(ue_order):
                action = int(actions[i])
                if not 0 <= action < num_cells:
                    invalid_actions += 1
                    raise RuntimeError(
                        f"Invalid padded action {action} for scenario {scenario.name} with {num_cells} cells"
                    )

                state = all_states[ue_idx]
                next_state, reward, done, _info = env.step_user_action(ue_idx, action)
                next_state_padded = pad_state(next_state)
                next_valid = pad_mask(env.valid_actions(ue_idx))
                episode_reward += reward

                transition = (state, action, reward, next_state_padded, done, next_valid)
                if dqn_cfg.n_step > 1:
                    nstep_t = n_step_buffers[ue_idx].add(transition)
                    if nstep_t is not None:
                        replay.add(scenario.name, nstep_t)
                else:
                    replay.add(scenario.name, transition)

                if replay.can_sample(scenario.name, dqn_cfg.batch_size) and decisions % dqn_cfg.train_every == 0:
                    batch, per_indices, per_weights = replay.sample(
                        scenario.name, rng, dqn_cfg.batch_size, beta=per_beta
                    )
                    loss_val, td_errors = _train_step(
                        agent, target_net, batch, edge_index, edge_weight,
                        optimizer, dqn_cfg,
                        importance_weights=per_weights,
                    )
                    losses.append(loss_val)
                    if per_indices is not None:
                        replay.update_priorities(scenario.name, per_indices, td_errors)
                    scheduler.step()
                    if dqn_cfg.tau > 0:
                        _soft_update(target_net, agent, dqn_cfg.tau)
                if dqn_cfg.tau == 0 and decisions % dqn_cfg.target_update_every == 0:
                    target_net.load_state_dict(agent.state_dict())
                decisions += 1

        # Flush remaining n-step transitions at episode end
        if dqn_cfg.n_step > 1:
            for ue in range(env.cfg.num_ues):
                for t in n_step_buffers[ue].flush():
                    replay.add(scenario.name, t)

        metrics = env.metrics()
        episode_decisions = max(steps_this_episode * env.cfg.num_ues, 1)
        metrics["handovers_per_1000_decisions"] = float(1000.0 * env.total_handovers / episode_decisions)
        metrics["pingpong_rate"] = float(env.pingpong_handovers / max(env.total_handovers, 1))
        metrics["weak_target_ho_rate"] = float(env.weak_target_handovers / max(env.total_handovers, 1))
        metrics.update(
            {
                "episode": float(episode + 1),
                "epsilon": float(epsilon),
                "steps_this_episode": float(steps_this_episode),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "episode_reward": float(episode_reward),
                "validation_score": training_validation_score(metrics),
                "scenario": scenario.name,
                "feature_mode": feature_mode,
                "num_ues": float(env.cfg.num_ues),
                "invalid_action_count": float(invalid_actions),
                "nan_state_count": float(nan_state_count),
            }
        )
        history.append(metrics)

        # ---- Periodic held-out validation pass (epsilon=0, eval mode) ----
        # Run validation every N episodes when configured. The validation score
        # is a much more reliable best-checkpoint signal than the training-time
        # 'validation_score' field above (which is biased by active exploration).
        validation_enabled = (
            validation_scenarios
            and validation_seeds
            and validate_every_episodes > 0
            and validation_steps > 0
        )
        ran_validation_this_episode = False
        skipped_warmup_validation = False
        if validation_enabled and (
            (episode + 1) % validate_every_episodes == 0
            or (episode + 1) == total_episodes
        ):
            skipped_warmup_validation = (
                skip_validation_epsilon_above is not None
                and epsilon > float(skip_validation_epsilon_above)
                and (episode + 1) != total_episodes
            )
            if skipped_warmup_validation:
                metrics["val_skipped_warmup"] = 1.0
            else:
                val_result = _run_validation_pass(
                    agent,
                    validation_scenarios,
                    validation_seeds,
                    steps=validation_steps,
                    feature_mode=feature_mode,
                    prb_available=prb_available,
                    son_config=son_config,
                    validation_ue_cap=validation_ue_cap,
                    validation_steps_override=validation_steps_override,
                )
                # Store with a 'val_' prefix so we don't collide with training-episode metrics
                for k, v in val_result.items():
                    metrics[f"val_{k}"] = v
                ran_validation_this_episode = True
                if verbose:
                    print(
                        f"  [VAL ep={episode + 1:04d}] "
                        f"son_score={val_result['holdout_validation_score']:+.3f} "
                        f"(best={best_score:+.3f}) "
                        f"son_avg_margin={val_result.get('son_avg_margin_pct', 0.0):+.2f}% "
                        f"son_p5_margin={val_result.get('son_p5_margin_pct', 0.0):+.2f}% "
                        f"pp_red={val_result.get('son_pingpong_reduction_pct', 0.0):+.2f}%",
                        flush=True,
                    )

        # ---- Best-checkpoint selection ----
        # Prefer the held-out validation score when validation has been run at least once.
        # Otherwise fall back to the (less reliable) training-metric score.
        if validation_enabled and ran_validation_this_episode:
            candidate_score = float(metrics["val_holdout_validation_score"])
        elif validation_enabled:
            # Skip best-checkpoint update on non-validation episodes when validation is on,
            # so the choice is made strictly from honest validation evidence.
            candidate_score = float("-inf")
        else:
            candidate_score = float(metrics["validation_score"])

        can_select_best = episode >= max(5, total_episodes // 10)
        improved = can_select_best and candidate_score > best_score + float(early_stopping_min_delta)
        if improved:
            best_score = candidate_score
            best_state = copy.deepcopy(agent.state_dict())
            best_episode = episode + 1
            validation_non_improve_count = 0
        elif validation_enabled and ran_validation_this_episode and can_select_best:
            validation_non_improve_count += 1

        if (
            validation_enabled
            and early_stopping_patience > 0
            and (episode + 1) >= int(early_stopping_min_episodes)
            and validation_non_improve_count >= int(early_stopping_patience)
        ):
            early_stopped = True
            if verbose:
                print(
                    f"  [EARLY STOP] episode={episode + 1} "
                    f"non_improve_validations={validation_non_improve_count} "
                    f"best_episode={best_episode} best_score={best_score:+.3f}",
                    flush=True,
                )
            break

        progress_interval = max(
            1,
            int(log_every_episodes)
            if log_every_episodes is not None and int(log_every_episodes) > 0
            else total_episodes // 20,
        )
        if verbose and (episode + 1) % progress_interval == 0:
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
                        "early_stopped": early_stopped,
                    },
                    best_state_dict=best_state,
                    checkpoint_kind="resume",
                )

    if best_episode:
        agent.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best checkpoint from episode {best_episode} (score={best_score:.3f})")
    return agent, history, max_cells


def train_flat_multi_scenario(
    scenarios: List[Scenario],
    dqn_cfg: DQNConfig,
    total_episodes: int,
    steps_per_episode: int,
    feature_mode: str = "ue_only",
    prb_available: bool = True,
    seed: int = 42,
    verbose: bool = True,
    scenario_sampling_weights: dict[str, float] | None = None,
) -> tuple[FlatDQNAgent, list[dict], int]:
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    replay = ReplayBuffer(dqn_cfg.replay_capacity)
    decisions = 0
    max_cells = max(s.num_cells for s in scenarios)
    sample_env = make_env_from_scenario(scenarios[0], feature_mode=feature_mode, prb_available=prb_available)
    feature_dim = sample_env.feature_dim
    agent = FlatDQNAgent(max_cells, feature_dim, dqn_cfg, seed=seed)
    target_net = copy.deepcopy(agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=dqn_cfg.learning_rate, weight_decay=dqn_cfg.weight_decay)
    best_score = float("-inf")
    best_state = copy.deepcopy(agent.state_dict())
    best_episode = 0
    sampling_probs: np.ndarray | None = None
    if scenario_sampling_weights:
        weights = np.array(
            [max(float(scenario_sampling_weights.get(s.name, 1.0)), 0.0) for s in scenarios],
            dtype=float,
        )
        if float(weights.sum()) > 0.0:
            sampling_probs = weights / weights.sum()
    scenario_order = np.arange(len(scenarios))

    def pad_state(state: np.ndarray, num_cells: int) -> np.ndarray:
        if num_cells >= max_cells:
            return state[:max_cells]
        out = np.zeros((max_cells, feature_dim), dtype=np.float32)
        out[:num_cells] = state
        return out

    def pad_mask(mask: np.ndarray, num_cells: int) -> np.ndarray:
        if num_cells >= max_cells:
            return mask[:max_cells].astype(bool)
        out = np.zeros(max_cells, dtype=bool)
        out[:num_cells] = mask
        return out

    if verbose:
        print(f"  Flat-DQN parameters: {sum(p.numel() for p in agent.parameters()):,}")
        print(f"  Flat-DQN max cells: {max_cells}, features: {feature_dim}")

    for episode in range(total_episodes):
        if sampling_probs is not None:
            scenario = scenarios[int(rng.choice(len(scenarios), p=sampling_probs))]
        else:
            if episode % len(scenarios) == 0:
                scenario_order = rng.permutation(len(scenarios))
            scenario = scenarios[int(scenario_order[episode % len(scenarios)])]
        env = make_env_from_scenario(scenario, feature_mode=feature_mode, prb_available=prb_available)
        env.reset(seed + 31_000 + 101 * episode)
        losses: list[float] = []
        episode_reward = 0.0
        frac = min(episode / max(dqn_cfg.epsilon_decay_episodes, 1), 1.0)
        epsilon = dqn_cfg.epsilon_start + frac * (dqn_cfg.epsilon_end - dqn_cfg.epsilon_start)

        for _step in range(steps_per_episode):
            env.advance_mobility()
            for ue_idx in rng.permutation(env.cfg.num_ues):
                ue = int(ue_idx)
                state = pad_state(env.build_state(ue), env.cfg.num_cells)
                valid = pad_mask(env.valid_actions(ue), env.cfg.num_cells)
                action = agent.act(torch.from_numpy(state).float(), epsilon=epsilon, valid_mask=valid, rng=rng)
                next_state, reward, done, _info = env.step_user_action(ue, int(action))
                next_state = pad_state(next_state, env.cfg.num_cells)
                next_valid = pad_mask(env.valid_actions(ue), env.cfg.num_cells)
                replay.add((state, int(action), reward, next_state, done, next_valid))
                episode_reward += reward
                if len(replay) >= dqn_cfg.batch_size and decisions % dqn_cfg.train_every == 0:
                    batch = replay.sample(rng, dqn_cfg.batch_size)
                    losses.append(_flat_train_step(agent, target_net, batch, optimizer, dqn_cfg))
                if decisions % dqn_cfg.target_update_every == 0:
                    target_net.load_state_dict(agent.state_dict())
                decisions += 1

        metrics = env.metrics()
        episode_decisions = max(steps_per_episode * env.cfg.num_ues, 1)
        metrics.update(
            {
                "episode": float(episode + 1),
                "epsilon": float(epsilon),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "episode_reward": float(episode_reward),
                "validation_score": training_validation_score(metrics),
                "scenario": scenario.name,
                "feature_mode": feature_mode,
                "handovers_per_1000_decisions": float(1000.0 * env.total_handovers / episode_decisions),
                "pingpong_rate": float(env.pingpong_handovers / max(env.total_handovers, 1)),
                "weak_target_ho_rate": float(env.weak_target_handovers / max(env.total_handovers, 1)),
            }
        )
        history.append(metrics)
        candidate_score = float(metrics["validation_score"])
        if episode >= max(5, total_episodes // 10) and candidate_score > best_score:
            best_score = candidate_score
            best_state = copy.deepcopy(agent.state_dict())
            best_episode = episode + 1
        if verbose and (episode + 1) % max(1, total_episodes // 20) == 0:
            print(
                f"[flat-DQN {episode + 1:04d}/{total_episodes}] {scenario.name:18s} "
                f"eps={epsilon:.3f} thr={metrics['avg_ue_throughput_mbps']:.3f} "
                f"p5={metrics['p5_ue_throughput_mbps']:.3f} "
                f"loss={metrics['loss']:.4f}",
                flush=True,
            )

    if best_episode:
        agent.load_state_dict(best_state)
        if verbose:
            print(f"  Restored flat-DQN best checkpoint from episode {best_episode} (score={best_score:.3f})")
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
    dqn_cfg = dqn_config_from_dict(cfg.get("dqn", {}))
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
    flat_agent: FlatDQNAgent | None = None,
    son_config: SONConfig | None = None,
) -> None:
    for scenario in scenarios:
        env = make_env_from_scenario(scenario, feature_mode=feature_mode, prb_available=prb_available)
        rows = evaluate_policies(
            env.cfg,
            default_policy_factories(gnn_agent=agent, flat_agent=flat_agent, son_config=son_config),
            steps=steps,
            seeds=seeds,
        )
        rows = sorted(rows, key=lambda r: r["avg_ue_throughput_mbps"], reverse=True)
        write_summary_csv(rows, out_dir / f"{scenario.name}.csv")
