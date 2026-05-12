from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import torch

from ..env import CellularNetworkEnv


@dataclass(frozen=True)
class SONConfig:
    """Safety-bounded SON parameter update policy.

    v2: Faster adaptation (1.0 dB steps, 12 updates/cycle, lower threshold)
    to exploit GNN-DQN's per-UE knowledge more aggressively while maintaining
    safety bounds.
    """

    update_interval_steps: int = 10
    cio_min_db: float = -6.0
    cio_max_db: float = 6.0
    max_cio_step_db: float = 1.0
    base_a3_offset_db: float = 3.0
    base_ttt_steps: int = 3
    max_ttt_steps: int = 8
    min_ttt_steps: int = 2
    preference_threshold: float = 0.12
    load_proxy_overload_threshold: float = 0.80
    max_updates_per_cycle: int = 12
    rollback_throughput_drop_frac: float = 0.12
    rollback_pingpong_increase_frac: float = 0.25
    rollback_pingpong_floor: float = 0.05
    ttt_decrease_threshold: float = 0.10
    ttt_cooldown_steps: int = 20
    # Where the SON gets its load signal from. "rsrq_proxy" is what a
    # phone-only deployment can compute from UE measurements; "true_prb"
    # is what a real eNB / near-RT RIC reads from PM counters. The trained
    # model is identical in both modes — only the SON's load lookup changes.
    load_signal: Literal["rsrq_proxy", "true_prb"] = "rsrq_proxy"


@dataclass(frozen=True)
class SONUpdate:
    source_cell: int
    target_cell: int
    delta_cio_db: float
    new_cio_db: float
    reason: str


class SONController:
    """Translate per-UE GNN-DQN preferences into SON-safe handover parameters.

    The controller is intentionally conservative: it periodically samples the
    GNN-DQN target preference for all UEs, aggregates those preferences by
    serving-target cell pair, and changes CIO by at most one bounded step per
    cycle. Standard A3 execution remains responsible for the actual handover.
    """

    def __init__(self, agent, config: SONConfig | None = None):
        self.agent = agent
        self.config = config or SONConfig()
        self.cio_db: np.ndarray | None = None
        self.ttt_steps: np.ndarray | None = None
        self.last_update_step = -10_000
        self._last_ttt_change_step = -10_000
        self.update_count = 0
        self.rollback_count = 0
        self.last_metrics: dict[str, float] | None = None
        self._previous_cio: np.ndarray | None = None
        self.updates: List[SONUpdate] = []

    def reset(self, env: CellularNetworkEnv) -> None:
        n = env.cfg.num_cells
        self.cio_db = np.zeros((n, n), dtype=float)
        self.ttt_steps = np.full((n, n), self.config.base_ttt_steps, dtype=int)
        self.last_update_step = -10_000
        self._last_ttt_change_step = -10_000
        self.update_count = 0
        self.rollback_count = 0
        self.last_metrics = None
        self._previous_cio = self.cio_db.copy()
        self.updates = []

    def maybe_update(self, env: CellularNetworkEnv) -> None:
        if self.cio_db is None:
            self.reset(env)
        if env.step_index - self.last_update_step < self.config.update_interval_steps:
            return
        self.update(env)

    def update(self, env: CellularNetworkEnv) -> list[SONUpdate]:
        assert self.cio_db is not None
        self._maybe_rollback(env)
        self._previous_cio = self.cio_db.copy()

        current_metrics = env.metrics()
        preferences = self._collect_preferences(env)
        served_counts = np.bincount(env.serving, minlength=env.cfg.num_cells).astype(float)

        # Load signal source: phone-only proxy vs. eNB/RIC true PRB from PM counters.
        if self.config.load_signal == "true_prb":
            cell_load = np.clip(env.cell_loads(), 0.0, 1.0)
            target_load = lambda t: float(cell_load[t])
            overload_reason = "target_prb_overloaded"
        else:
            rsrq_load_proxy = env.load_from_rsrq(env._compute_rsrq())

            def _served_load(t: int) -> float:
                served_mask = env.serving == t
                if not np.any(served_mask):
                    return 0.0
                return float(np.mean(rsrq_load_proxy[served_mask, t]))

            target_load = _served_load
            overload_reason = "target_load_proxy_high"

        candidates: list[tuple[float, SONUpdate]] = []
        for source in range(env.cfg.num_cells):
            if served_counts[source] <= 0:
                continue
            row = preferences[source]
            for target in range(env.cfg.num_cells):
                if target == source:
                    continue
                preference_share = row[target] / served_counts[source]
                t_load = target_load(target)
                if t_load >= self.config.load_proxy_overload_threshold:
                    delta = -self.config.max_cio_step_db
                    reason = overload_reason
                    priority = t_load
                elif preference_share >= self.config.preference_threshold:
                    delta = self.config.max_cio_step_db
                    reason = "gnn_preference"
                    priority = preference_share
                else:
                    continue

                new_cio = float(
                    np.clip(
                        self.cio_db[source, target] + delta,
                        self.config.cio_min_db,
                        self.config.cio_max_db,
                    )
                )
                if abs(new_cio - self.cio_db[source, target]) < 1e-9:
                    continue
                candidates.append(
                    (
                        priority,
                        SONUpdate(
                            source_cell=source,
                            target_cell=target,
                            delta_cio_db=float(new_cio - self.cio_db[source, target]),
                            new_cio_db=new_cio,
                            reason=reason,
                        ),
                    )
                )

        candidates.sort(key=lambda item: item[0], reverse=True)
        applied = [item[1] for item in candidates[: self.config.max_updates_per_cycle]]
        for update in applied:
            self.cio_db[update.source_cell, update.target_cell] = update.new_cio_db

        pingpong_rate = env.pingpong_handovers / max(env.total_handovers, 1)
        cooldown_elapsed = env.step_index - self._last_ttt_change_step >= self.config.ttt_cooldown_steps
        if self.ttt_steps is not None and cooldown_elapsed:
            if pingpong_rate > 0.25:
                next_ttt = np.minimum(self.ttt_steps + 1, self.config.max_ttt_steps)
                if np.any(next_ttt != self.ttt_steps):
                    self.ttt_steps[:] = next_ttt
                    self._last_ttt_change_step = env.step_index
            elif pingpong_rate < self.config.ttt_decrease_threshold:
                next_ttt = np.maximum(self.ttt_steps - 1, self.config.min_ttt_steps)
                if np.any(next_ttt != self.ttt_steps):
                    self.ttt_steps[:] = next_ttt
                    self._last_ttt_change_step = env.step_index

        self.updates.extend(applied)
        self.update_count += len(applied)
        self.last_update_step = env.step_index
        self.last_metrics = current_metrics | {"pingpong_rate": float(pingpong_rate)}
        return applied

    def cio(self, source_cell: int, target_cell: int) -> float:
        if self.cio_db is None:
            return 0.0
        return float(self.cio_db[source_cell, target_cell])

    def ttt(self, source_cell: int, target_cell: int) -> int:
        if self.ttt_steps is None:
            return self.config.base_ttt_steps
        return int(self.ttt_steps[source_cell, target_cell])

    def metrics(self) -> dict[str, float]:
        cio = self.cio_db if self.cio_db is not None else np.zeros((1, 1))
        return {
            "son_update_count": float(self.update_count),
            "son_avg_abs_cio_db": float(np.mean(np.abs(cio))),
            "son_max_abs_cio_db": float(np.max(np.abs(cio))),
            "son_rollback_count": float(self.rollback_count),
        }

    def _collect_preferences(self, env: CellularNetworkEnv) -> np.ndarray:
        preferences = np.zeros((env.cfg.num_cells, env.cfg.num_cells), dtype=float)
        edge_index, edge_weight = env.edge_data
        for ue_idx in range(env.cfg.num_ues):
            source = int(env.serving[ue_idx])
            state = torch.from_numpy(env.build_state(ue_idx)).float()
            target = int(
                self.agent.act(
                    state,
                    edge_index,
                    edge_weight,
                    epsilon=0.0,
                    valid_mask=env.valid_actions(ue_idx),
                )
            )
            preferences[source, target] += 1.0
        return preferences

    def _maybe_rollback(self, env: CellularNetworkEnv) -> None:
        if self.last_metrics is None or self._previous_cio is None or self.cio_db is None:
            return
        metrics = env.metrics()
        previous_thr = self.last_metrics.get("avg_ue_throughput_mbps", 0.0)
        current_thr = metrics["avg_ue_throughput_mbps"]
        previous_pingpong = self.last_metrics.get("pingpong_rate", 0.0)
        current_pingpong = env.pingpong_handovers / max(env.total_handovers, 1)
        throughput_bad = (
            previous_thr > 1e-9
            and current_thr < previous_thr * (1.0 - self.config.rollback_throughput_drop_frac)
        )
        pingpong_bad = (
            current_pingpong > self.config.rollback_pingpong_floor
            and current_pingpong > previous_pingpong * (
                1.0 + self.config.rollback_pingpong_increase_frac
            ) + 1e-9
        )
        if throughput_bad or pingpong_bad:
            self.cio_db[:] = self._previous_cio
            self.rollback_count += 1
