from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


def adjacency_to_edge_index(adj: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense adjacency matrix to PyG edge_index + edge_weight."""
    rows, cols = np.nonzero(adj)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(adj[rows, cols], dtype=torch.float)
    return edge_index, edge_weight


@dataclass(frozen=True)
class LTEConfig:
    num_cells: int = 9
    num_ues: int = 54
    area_m: float = 1500.0
    decision_interval_s: float = 1.0
    # Realistic LTE 10 MHz per-cell capacity (MIMO 2x2, ~15 bps/Hz peak).
    # Raised from 72 Mbps so the simulator isn't permanently in overload.
    cell_capacity_mbps: float = 150.0
    bandwidth_mhz: float = 10.0
    num_prbs: int = 50
    tx_power_dbm: float = 43.0
    shadow_sigma_db: float = 6.0
    min_rsrp_dbm: float = -112.0
    noise_floor_dbm: float = -174.0 + 10 * 1.0  # -174 + 10*log10(BW in Hz) approx
    pingpong_window_steps: int = 6
    handover_interruption_frac: float = 0.18
    graph_neighbors: int = 4
    min_speed_mps: float = 1.0
    max_speed_mps: float = 17.0
    # Lowered demand range (was 5-15) so realistic total load ~ 50-80% of cap.
    min_demand_mbps: float = 2.0
    max_demand_mbps: float = 8.0
    rsrp_noise_std_db: float = 1.0
    rsrq_noise_std_db: float = 0.5
    # Raised from 0.08 -> 0.25 so weak-signal UEs don't consume 12.5x their demand.
    quality_floor: float = 0.25
    cell_positions: np.ndarray | None = None
    # Feature mode: "ue_only" (10 features, phone-measurable) or
    # "full" (13 features, with 3 eNB-side features from E2/SON interface).
    feature_mode: str = "ue_only"


class CellularNetworkEnv:
    """LTE handover simulator with UE-observable features and RSRQ load proxy.

    State features are designed to be measurable by a real UE:
    - RSRP (reference signal received power)
    - RSRQ (reference signal received quality → load proxy)
    - Signal trends (improving/degrading)
    - Serving cell indicator
    - UE mobility context

    A single per-step channel snapshot is cached so that build_state(),
    valid_actions(), cell_loads(), user_throughputs() and the reward all
    read from the same realization (no stochastic mismatch inside a step).
    """

    def __init__(self, config: LTEConfig):
        self.cfg = config
        if config.cell_positions is not None:
            self.cell_pos = np.array(config.cell_positions, dtype=float)
        else:
            self.cell_pos = self._make_cell_positions()
        # Raw (non-normalized) adjacency - GCNConv will normalize internally.
        self.adjacency = self._make_graph()
        self.rng = np.random.default_rng(0)
        self.reset(0)

    @property
    def feature_dim(self) -> int:
        return 10 if self.cfg.feature_mode == "ue_only" else 13

    @property
    def edge_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "_edge_data"):
            self._edge_data = adjacency_to_edge_index(self.adjacency)
        return self._edge_data

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------
    def _refresh_snapshot(self, reroll_noise: bool = True) -> None:
        """Sample one coherent per-step snapshot of the channel.

        All downstream accessors (build_state / cell_loads / user_throughputs /
        valid_actions / reward) read from this cache. Prevents the bug where
        different callers observe independent Gaussian noise realizations
        within the same decision step.
        """
        d_km = np.maximum(self.ue_cell_distances() / 1000.0, 0.035)
        path_loss_db = 128.1 + 37.6 * np.log10(d_km)
        rsrp_mean = self.cfg.tx_power_dbm - path_loss_db + self.shadowing

        if reroll_noise or not hasattr(self, "_rsrp_noise"):
            self._rsrp_noise = self.rng.normal(
                0.0, self.cfg.rsrp_noise_std_db, size=rsrp_mean.shape
            )
        rsrp = rsrp_mean + self._rsrp_noise
        self._rsrp = rsrp

        # Compute loads first (needed for RSRQ interference term).
        qualities = self.quality_from_rsrp(
            rsrp[np.arange(self.cfg.num_ues), self.serving]
        )
        pressure = self.demands / np.maximum(qualities, self.cfg.quality_floor)
        self._loads = np.bincount(
            self.serving, weights=pressure, minlength=self.cfg.num_cells
        ) / self.cfg.cell_capacity_mbps

        if reroll_noise or not hasattr(self, "_rsrq_noise"):
            self._rsrq_noise = self.rng.normal(
                0.0, self.cfg.rsrq_noise_std_db, size=rsrp.shape
            )
        self._rsrq = self._rsrq_from_rsrp(rsrp, self._loads, reroll_noise=reroll_noise)

        # User throughput: share cell capacity proportional to demand.
        load_factor = np.maximum(self._loads[self.serving], 1.0)
        throughput = self.demands / load_factor
        outage = rsrp[np.arange(self.cfg.num_ues), self.serving] < self.cfg.min_rsrp_dbm
        throughput = np.where(outage, throughput * 0.10, throughput)
        interrupted = self.last_handover_step == self.step_index
        throughput = np.where(
            interrupted,
            throughput * (1.0 - self.cfg.handover_interruption_frac),
            throughput,
        )
        self._throughputs = throughput

    def _rsrq_from_rsrp(self, rsrp: np.ndarray, loads: np.ndarray, reroll_noise: bool = True) -> np.ndarray:
        """Vectorized RSRQ computation from a given rsrp snapshot + loads."""
        cfg = self.cfg
        rsrp_linear = 10.0 ** (rsrp / 10.0)

        # Per (UE, cell_j) path loss to every cell - vectorized.
        delta = self.ue_pos[:, None, :] - self.cell_pos[None, :, :]
        d_km = np.maximum(np.linalg.norm(delta, axis=2) / 1000.0, 0.035)
        pl = 128.1 + 37.6 * np.log10(d_km)
        intf_power_all = 10.0 ** ((cfg.tx_power_dbm - pl) / 10.0)  # (U, C)

        weighted = intf_power_all * loads[None, :]  # (U, C), weighted per source cell
        total = weighted.sum(axis=1, keepdims=True)  # (U, 1), sum over all cells
        # Interference at target cell c = total - contribution of c itself.
        interference = total - weighted  # (U, C)

        noise_power = 10.0 ** (cfg.noise_floor_dbm / 10.0)
        rssi = rsrp_linear + interference + noise_power
        rsrq_linear = cfg.num_prbs * rsrp_linear / np.maximum(rssi, 1e-20)
        rsrq_db = 10.0 * np.log10(np.maximum(rsrq_linear, 1e-20))
        rsrq_db = np.clip(rsrq_db, -20.0, -3.0)
        return rsrq_db + self._rsrq_noise

    # ------------------------------------------------------------------
    # Setup / reset
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        cfg = self.cfg

        self.ue_pos = self.rng.uniform(0.0, cfg.area_m, size=(cfg.num_ues, 2))
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=cfg.num_ues)
        speeds = self.rng.uniform(cfg.min_speed_mps, cfg.max_speed_mps, size=cfg.num_ues)
        self.ue_vel = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, None]
        self.ue_speed = speeds
        self.demands = self.rng.uniform(cfg.min_demand_mbps, cfg.max_demand_mbps, size=cfg.num_ues)
        self.shadowing = self.rng.normal(0.0, cfg.shadow_sigma_db, size=(cfg.num_ues, cfg.num_cells))

        self.step_index = 0
        self.total_handovers = 0
        self.pingpong_handovers = 0
        self.weak_target_handovers = 0
        self.last_handover_step = np.full(cfg.num_ues, -10_000, dtype=int)
        self.previous_cell = np.full(cfg.num_ues, -1, dtype=int)

        # Bootstrap serving cells from an initial (uncached) RSRP draw.
        d_km = np.maximum(self.ue_cell_distances() / 1000.0, 0.035)
        init_rsrp = cfg.tx_power_dbm - (128.1 + 37.6 * np.log10(d_km)) + self.shadowing
        self.serving = np.argmax(init_rsrp, axis=1).astype(int)
        self.previous_cell[:] = self.serving

        # Previous-step arrays for trend features.
        self._prev_rsrp = init_rsrp.copy()
        self._prev_rsrq_valid = False

        # Build first coherent snapshot.
        self._refresh_snapshot()
        self._prev_rsrp = self._rsrp.copy()

    def _make_cell_positions(self) -> np.ndarray:
        cfg = self.cfg
        side = int(np.ceil(np.sqrt(cfg.num_cells)))
        xs = np.linspace(0.18 * cfg.area_m, 0.82 * cfg.area_m, side)
        ys = np.linspace(0.18 * cfg.area_m, 0.82 * cfg.area_m, side)
        pts = np.array([(x, y) for y in ys for x in xs], dtype=float)
        center = np.array([cfg.area_m / 2.0, cfg.area_m / 2.0])
        order = np.argsort(np.linalg.norm(pts - center, axis=1))
        return pts[order[: cfg.num_cells]]

    def _make_graph(self) -> np.ndarray:
        """Return raw (un-normalized, no self-loops) weighted adjacency.

        GCNConv(add_self_loops=True) will add self-loops and symmetric
        degree-normalization internally - we must not do it here or
        self-loops get double-counted and the renormalization is wrong.
        """
        cfg = self.cfg
        d = self.cell_distance_matrix()
        scale = max(np.median(d[d > 0]) / 2.0, 1.0)
        w = np.exp(-d / scale)
        np.fill_diagonal(w, 0.0)

        adj = np.zeros((cfg.num_cells, cfg.num_cells))
        for i in range(cfg.num_cells):
            nn_idx = np.argsort(d[i])[1: cfg.graph_neighbors + 1]
            adj[i, nn_idx] = w[i, nn_idx]
        # Symmetrize (undirected neighbour graph).
        adj = np.maximum(adj, adj.T)
        return adj

    def cell_distance_matrix(self) -> np.ndarray:
        delta = self.cell_pos[:, None, :] - self.cell_pos[None, :, :]
        return np.linalg.norm(delta, axis=2)

    def ue_cell_distances(self) -> np.ndarray:
        delta = self.ue_pos[:, None, :] - self.cell_pos[None, :, :]
        return np.linalg.norm(delta, axis=2)

    # ------------------------------------------------------------------
    # Snapshot-backed accessors (no re-sampling)
    # ------------------------------------------------------------------
    def rsrp_matrix(self) -> np.ndarray:
        """Current step's cached RSRP snapshot."""
        return self._rsrp

    def _compute_rsrq(self) -> np.ndarray:
        return self._rsrq

    def cell_loads(self) -> np.ndarray:
        return self._loads

    def user_throughputs(self) -> np.ndarray:
        return self._throughputs

    def cell_user_counts(self) -> np.ndarray:
        return np.bincount(self.serving, minlength=self.cfg.num_cells).astype(float)

    @staticmethod
    def quality_from_rsrp(rsrp_dbm: np.ndarray) -> np.ndarray:
        quality = 1.0 / (1.0 + np.exp(-(rsrp_dbm + 96.0) / 7.0))
        return np.clip(quality, 0.04, 1.0)

    @staticmethod
    def load_from_rsrq(rsrq_db: np.ndarray) -> np.ndarray:
        """Estimate cell load from RSRQ (UE-observable proxy).

        RSRQ range: -3 dB (unloaded) to -20 dB (fully loaded).
        Maps to estimated load in [0, 1].
        """
        return np.clip((-3.0 - rsrq_db) / 17.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Time step
    # ------------------------------------------------------------------
    def advance_mobility(self) -> None:
        cfg = self.cfg
        self.step_index += 1

        # Remember previous step's snapshot for trend feature.
        self._prev_rsrp = self._rsrp.copy()

        # Move UEs.
        self.ue_pos += self.ue_vel * cfg.decision_interval_s
        for dim in range(2):
            low = self.ue_pos[:, dim] < 0.0
            high = self.ue_pos[:, dim] > cfg.area_m
            self.ue_pos[low, dim] *= -1.0
            self.ue_pos[high, dim] = 2.0 * cfg.area_m - self.ue_pos[high, dim]
            self.ue_vel[low | high, dim] *= -1.0

        # AR(1) shadowing update.
        noise = self.rng.normal(0.0, 0.45, size=self.shadowing.shape)
        self.shadowing = 0.985 * self.shadowing + noise

        # New coherent snapshot for this step.
        self._refresh_snapshot()

    def valid_actions(self, ue_idx: int) -> np.ndarray:
        rsrp = self._rsrp[ue_idx]
        valid = rsrp >= self.cfg.min_rsrp_dbm
        valid[self.serving[ue_idx]] = True
        if not np.any(valid):
            valid[:] = True
        return valid

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def build_state(self, ue_idx: int) -> np.ndarray:
        """Build per-cell-node feature matrix.

        UE-observable features (from phone measurements), always present:
        1. rsrp_norm: Normalized RSRP to this cell
        2. rsrq_norm: Normalized RSRQ (load proxy)
        3. rsrp_delta: RSRP(cell) - RSRP(serving) normalized
        4. rsrq_delta: RSRQ(cell) - RSRQ(serving) normalized
        5. rsrp_trend: RSRP change since last step (improving/degrading)
        6. is_serving: Binary indicator
        7. signal_usable: Whether RSRP > threshold
        8. ue_speed_class: Normalized speed (0=static, 1=fast)
        9. time_since_ho: Normalized time since last handover
        10. was_previous_serving: 1 if UE was previously on this cell

        Network-side features (only present when feature_mode='full';
        these come from E2/SON in O-RAN deployment, NOT available to a
        phone-only deployment):
        11. prb_utilization: Actual PRB load (0-1)
        12. connected_ue_count: Normalized number of UEs on this cell
        13. cell_throughput_norm: Average throughput per cell (normalized)
        """
        cfg = self.cfg
        rsrp = self._rsrp[ue_idx]
        rsrq = self._rsrq[ue_idx]
        current = self.serving[ue_idx]

        rsrp_norm = np.clip((rsrp - (-120.0)) / 55.0, 0.0, 1.0)
        rsrq_norm = np.clip((rsrq - (-20.0)) / 17.0, 0.0, 1.0)

        serving_rsrp = rsrp[current]
        rsrp_delta = np.clip((rsrp - serving_rsrp + 20.0) / 40.0, 0.0, 1.0)

        serving_rsrq = rsrq[current]
        rsrq_delta = np.clip((rsrq - serving_rsrq + 10.0) / 20.0, 0.0, 1.0)

        prev_rsrp = self._prev_rsrp[ue_idx] if hasattr(self, '_prev_rsrp') else rsrp
        rsrp_trend = np.clip((rsrp - prev_rsrp + 5.0) / 10.0, 0.0, 1.0)

        is_serving = np.zeros(cfg.num_cells)
        is_serving[current] = 1.0

        signal_usable = (rsrp >= cfg.min_rsrp_dbm).astype(float)

        speed_norm = np.clip(self.ue_speed[ue_idx] / cfg.max_speed_mps, 0.0, 1.0)
        ue_speed_class = np.full(cfg.num_cells, speed_norm)

        steps_since_ho = self.step_index - self.last_handover_step[ue_idx]
        time_norm = np.clip(steps_since_ho / 20.0, 0.0, 1.0)
        time_since_ho = np.full(cfg.num_cells, time_norm)

        was_previous = np.zeros(cfg.num_cells)
        prev_cell = self.previous_cell[ue_idx]
        if prev_cell >= 0:
            was_previous[prev_cell] = 1.0

        cols = [
            rsrp_norm,
            rsrq_norm,
            rsrp_delta,
            rsrq_delta,
            rsrp_trend,
            is_serving,
            signal_usable,
            ue_speed_class,
            time_since_ho,
            was_previous,
        ]

        if cfg.feature_mode == "full":
            # Network-side features (available in xApp/SON deployment only).
            prb_utilization = np.clip(self._loads, 0.0, 1.0)

            ue_counts = self.cell_user_counts()
            max_ues_per_cell = max(cfg.num_ues / cfg.num_cells * 3, 1.0)
            connected_ue_count = np.clip(ue_counts / max_ues_per_cell, 0.0, 1.0)

            cell_thr = np.zeros(cfg.num_cells)
            for c in range(cfg.num_cells):
                mask = self.serving == c
                if mask.any():
                    cell_thr[c] = self._throughputs[mask].mean()
            cell_throughput_norm = np.clip(cell_thr / cfg.cell_capacity_mbps, 0.0, 1.0)

            cols.extend([prb_utilization, connected_ue_count, cell_throughput_norm])

        return np.column_stack(cols).astype(np.float32)

    # ------------------------------------------------------------------
    # Action / reward
    # ------------------------------------------------------------------
    def step_user_action(self, ue_idx: int, target_cell: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        target_cell = int(np.clip(target_cell, 0, self.cfg.num_cells - 1))
        old_cell = int(self.serving[ue_idx])
        handover = target_cell != old_cell

        if handover:
            rsrp_target = self._rsrp[ue_idx, target_cell]
            if rsrp_target < self.cfg.min_rsrp_dbm:
                self.weak_target_handovers += 1
            if (
                target_cell == self.previous_cell[ue_idx]
                and self.step_index - self.last_handover_step[ue_idx] <= self.cfg.pingpong_window_steps
            ):
                self.pingpong_handovers += 1
            self.previous_cell[ue_idx] = old_cell
            self.serving[ue_idx] = target_cell
            self.last_handover_step[ue_idx] = self.step_index
            self.total_handovers += 1

            # Serving changed → per-UE throughput / load changed slightly.
            # Refresh snapshot so the reward / next build_state see the
            # post-action state. Cheap: vectorized, O(UC).
            self._refresh_snapshot()

        reward = self.user_reward(ue_idx, old_cell, target_cell)
        info = {"handover": float(handover)}
        return self.build_state(ue_idx), reward, False, info

    def user_reward(self, ue_idx: int, old_cell: int, target_cell: int) -> float:
        """Multi-objective reward: throughput-dominant with strong handover penalty.

        The agent must only handover when the throughput gain clearly justifies
        the switching cost. Staying on a good cell is explicitly rewarded.
        """
        throughputs = self._throughputs
        loads = self._loads
        rsrp = self._rsrp[ue_idx, target_cell]

        satisfaction = np.clip(
            throughputs[ue_idx] / max(self.demands[ue_idx], 1e-6), 0.0, 1.2
        )
        thr_bonus = np.log2(1.0 + max(throughputs[ue_idx], 0.0)) / 4.0

        load_std = float(min(np.std(loads), 1.0))
        overload = float(np.mean(np.maximum(loads - 1.0, 0.0)))

        handover = float(old_cell != target_cell)

        # Escalating penalty for frequent handovers.
        steps_since_last = self.step_index - self.last_handover_step[ue_idx]
        ho_recency_penalty = 0.0
        if handover and steps_since_last < 10:
            ho_recency_penalty = 1.0 * (1.0 - steps_since_last / 10.0)

        pingpong = float(
            target_cell == self.previous_cell[ue_idx]
            and steps_since_last <= self.cfg.pingpong_window_steps
        ) if handover else 0.0
        outage = float(rsrp < self.cfg.min_rsrp_dbm)

        # Stay bonus: reward for not switching when signal is adequate.
        stay_bonus = 0.0
        if not handover and rsrp >= self.cfg.min_rsrp_dbm + 6.0:
            stay_bonus = 0.5

        return float(
            3.0 * satisfaction
            + 1.0 * thr_bonus
            + stay_bonus
            - 0.4 * load_std
            - 0.3 * overload
            - 2.0 * handover
            - ho_recency_penalty
            - 5.0 * pingpong
            - 1.5 * outage
        )

    def metrics(self) -> Dict[str, float]:
        throughputs = self._throughputs
        loads = self._loads
        serving_rsrp = self._rsrp[np.arange(self.cfg.num_ues), self.serving]
        load_sum_sq = np.sum(loads * loads)
        jain = (np.sum(loads) ** 2) / (self.cfg.num_cells * load_sum_sq + 1e-9)

        return {
            "avg_ue_throughput_mbps": float(np.mean(throughputs)),
            "p5_ue_throughput_mbps": float(np.percentile(throughputs, 5)),
            "total_throughput_mbps": float(np.sum(throughputs)),
            "avg_cell_load": float(np.mean(loads)),
            "load_std": float(np.std(loads)),
            "jain_load_fairness": float(jain),
            "outage_rate": float(np.mean(serving_rsrp < self.cfg.min_rsrp_dbm)),
            "overload_rate": float(np.mean(loads > 1.0)),
        }
