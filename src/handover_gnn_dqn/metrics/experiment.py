from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np

from ..models.flat_dqn import FlatDQNAgent
from ..policies.policies import (
    A3HandoverPolicy,
    GnnDqnPolicy,
    LoadAwarePolicy,
    NoHandoverPolicy,
    RandomValidPolicy,
    SONTunedA3Policy,
    StrongestRsrpPolicy,
)
from ..son import SONConfig
from ..env.simulator import CellularNetworkEnv, LTEConfig

import torch

PolicyFactory = Callable[[], object]


class FlatDqnPolicy:
    name = "flat_dqn"

    def __init__(self, agent: FlatDQNAgent, epsilon: float = 0.0):
        self.agent = agent
        self.epsilon = epsilon

    def reset(self, env: CellularNetworkEnv) -> None:
        pass

    def select(self, env: CellularNetworkEnv, ue_idx: int) -> int:
        state = env.build_state(ue_idx)
        valid = env.valid_actions(ue_idx)
        if env.cfg.num_cells < self.agent.num_cells:
            padded = np.zeros((self.agent.num_cells, env.feature_dim), dtype=np.float32)
            padded[: env.cfg.num_cells] = state
            valid_padded = np.zeros(self.agent.num_cells, dtype=bool)
            valid_padded[: env.cfg.num_cells] = valid
            state = padded
            valid = valid_padded
        elif env.cfg.num_cells > self.agent.num_cells:
            state = state[: self.agent.num_cells]
            valid = valid[: self.agent.num_cells].copy()
            if int(env.serving[ue_idx]) >= self.agent.num_cells and not valid.any():
                valid[:] = True
        state_t = torch.from_numpy(state).float()
        return self.agent.act(
            state_t,
            epsilon=self.epsilon,
            valid_mask=valid,
        )


def run_policy_episode(env: CellularNetworkEnv, policy, steps: int, seed: int) -> Dict[str, float]:
    if steps <= 0:
        raise ValueError("steps must be positive")

    rng = np.random.default_rng(seed)
    policy.reset(env)
    step_metrics: List[Dict[str, float]] = []

    for _ in range(steps):
        env.advance_mobility()
        for ue_idx in rng.permutation(env.cfg.num_ues):
            action = policy.select(env, int(ue_idx))
            env.step_user_action(int(ue_idx), action)
        step_metrics.append(env.metrics())

    result = {
        key: float(np.mean([m[key] for m in step_metrics]))
        for key in step_metrics[0].keys()
    }
    decisions = max(steps * env.cfg.num_ues, 1)
    result["handovers_per_1000_decisions"] = 1000.0 * env.total_handovers / decisions
    result["pingpong_rate"] = env.pingpong_handovers / max(env.total_handovers, 1)
    result["weak_target_ho_rate"] = env.weak_target_handovers / max(env.total_handovers, 1)
    if hasattr(policy, "son_metrics"):
        result.update(policy.son_metrics())
    return result


def evaluate_policies(
    lte_cfg: LTEConfig,
    policy_factories: Dict[str, PolicyFactory],
    steps: int,
    seeds: Iterable[int],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    seeds_list = list(seeds)
    if not seeds_list:
        raise ValueError("seeds must contain at least one seed")
    if steps <= 0:
        raise ValueError("steps must be positive")

    for name, make_policy in policy_factories.items():
        episode_rows = []
        for seed in seeds_list:
            env = CellularNetworkEnv(lte_cfg)
            env.reset(seed)
            episode_rows.append(run_policy_episode(env, make_policy(), steps=steps, seed=seed + 999))

        row = {"method": name}
        for key in episode_rows[0].keys():
            values = [m[key] for m in episode_rows]
            row[key] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
            row[f"{key}_ci95"] = float(1.96 * np.std(values) / np.sqrt(len(values)))
        rows.append(row)
    return rows


def default_policy_factories(
    gnn_agent=None,
    flat_agent=None,
    *,
    son_config: SONConfig | None = None,
    include_true_prb: bool = False,
) -> Dict[str, PolicyFactory]:
    policies: Dict[str, PolicyFactory] = {
        "no_handover": lambda: NoHandoverPolicy(),
        "random_valid": lambda: RandomValidPolicy(seed=1234),
        "strongest_rsrp": lambda: StrongestRsrpPolicy(hysteresis_db=2.0),
        "a3_ttt": lambda: A3HandoverPolicy(offset_db=3.0, time_to_trigger=3),
        "load_aware": lambda: LoadAwarePolicy(load_weight=0.48, handover_cost=0.04),
    }
    if flat_agent is not None:
        policies["flat_dqn"] = lambda: FlatDqnPolicy(flat_agent, epsilon=0.0)
    if gnn_agent is not None:
        policies["gnn_dqn"] = lambda: GnnDqnPolicy(gnn_agent, epsilon=0.0)
        policies["son_gnn_dqn"] = lambda: SONTunedA3Policy(gnn_agent, son_config)
        if include_true_prb:
            # Network-cooperative SON ablation: same model + SON layer, but the
            # load signal comes from true PRB (PM counters / E2 KPM) instead of
            # the RSRQ proxy. Keep this out of the main UE-only result tables.
            policies["son_gnn_dqn_true_prb"] = lambda: SONTunedA3Policy(
                gnn_agent, SONConfig(load_signal="true_prb")
            )
    return policies


def write_summary_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        raise ValueError("rows must contain at least one result")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_table(rows: List[Dict[str, float]]) -> str:
    columns = [
        ("method", "Method"),
        ("avg_ue_throughput_mbps", "Avg Mbps"),
        ("p5_ue_throughput_mbps", "P5 Mbps"),
        ("total_throughput_mbps", "Total Mbps"),
        ("load_std", "Load Std"),
        ("jain_load_fairness", "Jain"),
        ("outage_rate", "Outage"),
        ("overload_rate", "Overload"),
        ("handovers_per_1000_decisions", "HO/1000"),
        ("pingpong_rate", "Ping-pong"),
        ("son_update_count", "SON upd"),
    ]

    widths = []
    for key, title in columns:
        values = [
            str(row.get(key, "")) if key == "method" else f"{row.get(key, 0.0):.3f}"
            for row in rows
        ]
        widths.append(max(len(title), *(len(v) for v in values)))

    def render(values):
        return " | ".join(str(v).rjust(width) for v, width in zip(values, widths))

    header = render([title for _key, title in columns])
    sep = "-+-".join("-" * width for width in widths)
    body = []
    for row in rows:
        values = []
        for key, _title in columns:
            values.append(row.get(key, "") if key == "method" else f"{row.get(key, 0.0):.3f}")
        body.append(render(values))
    return "\n".join([header, sep, *body])


def attach_improvement_vs_regular(rows: List[Dict[str, float]]) -> Dict[str, float]:
    by_name = {row["method"]: row for row in rows}
    candidate = "son_gnn_dqn" if "son_gnn_dqn" in by_name else "gnn_dqn"
    if candidate not in by_name:
        return {}
    regular_names = ["strongest_rsrp", "a3_ttt"]
    regular = max(regular_names, key=lambda n: by_name[n]["avg_ue_throughput_mbps"])
    gnn = by_name[candidate]
    base = by_name[regular]
    pingpong_reduction = None
    if base["pingpong_rate"] > 1e-9:
        pingpong_reduction = 100.0 * (base["pingpong_rate"] - gnn["pingpong_rate"]) / base["pingpong_rate"]

    result = {
        "baseline": regular,
        "candidate": candidate,
        "avg_throughput_gain_pct": 100.0
        * (gnn["avg_ue_throughput_mbps"] - base["avg_ue_throughput_mbps"])
        / max(base["avg_ue_throughput_mbps"], 1e-9),
        "p5_throughput_gain_pct": 100.0
        * (gnn["p5_ue_throughput_mbps"] - base["p5_ue_throughput_mbps"])
        / max(base["p5_ue_throughput_mbps"], 1e-9),
        "load_std_reduction_pct": 100.0
        * (base["load_std"] - gnn["load_std"])
        / max(base["load_std"], 1e-9),
        "pingpong_reduction_pct": pingpong_reduction,
        "pingpong_delta": gnn["pingpong_rate"] - base["pingpong_rate"],
    }

    if "flat_dqn" in by_name:
        flat = by_name["flat_dqn"]
        result["gnn_vs_flat_throughput_pct"] = 100.0 * (
            gnn["avg_ue_throughput_mbps"] - flat["avg_ue_throughput_mbps"]
        ) / max(flat["avg_ue_throughput_mbps"], 1e-9)
        result["gnn_vs_flat_load_std_pct"] = 100.0 * (
            flat["load_std"] - gnn["load_std"]
        ) / max(flat["load_std"], 1e-9)

    return result
