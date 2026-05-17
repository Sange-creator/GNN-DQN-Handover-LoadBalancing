from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np
import torch


class FeatureProfile(str, Enum):
    """Measurement contracts for the handover controller."""

    UE_ONLY = "ue_only"
    ORAN_E2 = "oran_e2"


def normalize_positions_to_area(positions: np.ndarray, area_m: float, margin_frac: float = 0.06) -> np.ndarray:
    """Shift/scale local meter coordinates into the simulator UE area.

    OpenCellID and synthetic generators often produce coordinates centered on
    (0, 0), while the simulator samples UEs inside [0, area_m]^2. Keeping those
    frames mismatched creates artificial outage, especially on highway/rural
    scenarios. This preserves relative geometry and only scales if needed.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.size == 0:
        return pos.copy()
    if (
        np.all(pos[:, 0] >= 0.0)
        and np.all(pos[:, 1] >= 0.0)
        and np.all(pos[:, 0] <= area_m)
        and np.all(pos[:, 1] <= area_m)
    ):
        return pos.copy()

    pmin = pos.min(axis=0)
    pmax = pos.max(axis=0)
    span = np.maximum(pmax - pmin, 1.0)
    margin = max(25.0, margin_frac * area_m)
    available = max(area_m - 2.0 * margin, 1.0)
    scale = min(1.0, available / float(span.max()))
    return (pos - pmin) * scale + margin


def feature_profile_from_value(value: str | FeatureProfile) -> FeatureProfile:
    if isinstance(value, FeatureProfile):
        return value
    normalized = str(value).lower()
    if normalized in {"full", "oran", "oran_e2", "e2"}:
        return FeatureProfile.ORAN_E2
    if normalized in {"ue", "ue_only", "drive_test"}:
        return FeatureProfile.UE_ONLY
    raise ValueError(f"Unknown feature profile: {value!r}")


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
    # 2026-05-13: dropped from 150 Mbps to 50 Mbps to push scenarios into a
    # congested regime where handover policy choice actually matters. At 150
    # Mbps most training scenarios ran at 20-65% utilization, so the
    # demand-capped throughput model gave every policy identical avg_throughput
    # (every UE got its full demand regardless of cell). At 50 Mbps the same
    # scenarios load 60-200% and load-balancing decisions translate into
    # measurable throughput deltas.
    #
    # The reward function uses satisfaction = throughput/demand, which is
    # self-normalizing — no reward weights need tuning for this change.
    # Absolute Mbps numbers in the eval CSVs will be ~3x lower than prior runs;
    # relative rankings between policies are the result.
    cell_capacity_mbps: float = 45.0
    bandwidth_mhz: float = 10.0
    num_prbs: int = 50
    tx_power_dbm: float = 43.0
    shadow_sigma_db: float = 6.0
    min_rsrp_dbm: float = -112.0
    noise_floor_dbm: float = -104.0  # -174 + 10*log10(10e6) for 10 MHz LTE
    pingpong_window_steps: int = 5
    handover_interruption_frac: float = 0.15
    graph_neighbors: int = 5
    min_speed_mps: float = 1.0
    max_speed_mps: float = 17.0
    # Raised demand range to ensure congestion matters.
    min_demand_mbps: float = 3.0
    max_demand_mbps: float = 14.0
    rsrp_noise_std_db: float = 1.0
    rsrq_noise_std_db: float = 0.5
    # Raised from 0.08 -> 0.25 so weak-signal UEs don't consume 12.5x their demand.
    quality_floor: float = 0.25
    cell_positions: np.ndarray | None = None
    # Feature mode: "ue_only" for phone/drive-test features, or "oran_e2"
    # for network-assisted features with true PRB/counter availability.
    feature_mode: str = "ue_only"
    prb_available: bool = True
    mobility_model: str = "random"
    road_width_m: float = 160.0
    event_cluster_std_frac: float = 0.12
    pingpong_penalty_weight: float = 6.0


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
        self.feature_profile = feature_profile_from_value(config.feature_mode)
        if config.cell_positions is not None:
            self.cell_pos = normalize_positions_to_area(config.cell_positions, config.area_m)
        else:
            self.cell_pos = self._make_cell_positions()
        # Raw (non-normalized) adjacency - GCNConv will normalize internally.
        self.adjacency = self._make_graph()
        self.rng = np.random.default_rng(0)
        self.reset(0)

    @property
    def feature_dim(self) -> int:
        return 11 if self.feature_profile == FeatureProfile.UE_ONLY else 15

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

        self.ue_pos, angles = self._initial_ue_positions_and_angles()
        speeds = self.rng.uniform(cfg.min_speed_mps, cfg.max_speed_mps, size=cfg.num_ues)
        if cfg.mobility_model == "highway":
            directions = self.rng.choice([-1.0, 1.0], size=cfg.num_ues)
            angles = np.where(directions > 0.0, 0.0, np.pi)
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
        # Sticky-cell tracking: counts steps each UE stays on a cell whose RSRP
        # is materially below the best available neighbor (3 dB), and flags
        # handovers that fire only after sustained degradation (late HOs).
        self.steps_below_margin = np.zeros(cfg.num_ues, dtype=int)
        self.steps_below_margin_total = 0
        self.late_ho_count = 0

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

    def _initial_ue_positions_and_angles(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        if cfg.mobility_model == "highway":
            x_min = max(0.0, self.cell_pos[:, 0].min() - 0.08 * cfg.area_m)
            x_max = min(cfg.area_m, self.cell_pos[:, 0].max() + 0.08 * cfg.area_m)
            road_y = float(np.median(self.cell_pos[:, 1]))
            x = self.rng.uniform(x_min, x_max, size=cfg.num_ues)
            y = road_y + self.rng.normal(0.0, cfg.road_width_m / 3.0, size=cfg.num_ues)
            y = np.clip(y, 0.0, cfg.area_m)
            angles = self.rng.choice([0.0, np.pi], size=cfg.num_ues)
            return np.column_stack((x, y)), angles

        if cfg.mobility_model == "event":
            center = self.cell_pos.mean(axis=0)
            std = max(cfg.area_m * cfg.event_cluster_std_frac, 25.0)
            pos = self.rng.normal(center, std, size=(cfg.num_ues, 2))
            pos = np.clip(pos, 0.0, cfg.area_m)
            angles = self.rng.uniform(0.0, 2.0 * np.pi, size=cfg.num_ues)
            return pos, angles

        pos = self.rng.uniform(0.0, cfg.area_m, size=(cfg.num_ues, 2))
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=cfg.num_ues)
        return pos, angles

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
        if cfg.num_cells <= 1:
            return np.zeros((cfg.num_cells, cfg.num_cells), dtype=float)

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
        if cfg.mobility_model == "highway":
            road_y = float(np.median(self.cell_pos[:, 1]))
            self.ue_pos[:, 1] = np.clip(
                self.ue_pos[:, 1],
                max(0.0, road_y - cfg.road_width_m),
                min(cfg.area_m, road_y + cfg.road_width_m),
            )
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

    def valid_actions_all(self) -> np.ndarray:
        """Return valid action masks for every UE as ``(num_ues, num_cells)``."""
        valid = self._rsrp >= self.cfg.min_rsrp_dbm
        valid[np.arange(self.cfg.num_ues), self.serving] = True
        empty = ~valid.any(axis=1)
        if np.any(empty):
            valid[empty, :] = True
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
        11. load_proxy_rsrq: RSRQ-derived load estimate, not true PRB

        Network-side features (only present when feature_mode='oran_e2';
        these come from E2/SON in O-RAN deployment, NOT available to a
        phone-only deployment):
        12. prb_utilization: Actual PRB load (0-1), zero if unavailable
        13. prb_available: Availability mask for true PRB counters
        14. connected_ue_count: Normalized number of UEs on this cell
        15. cell_throughput_norm: Average throughput per cell (normalized)
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
        load_proxy_rsrq = self.load_from_rsrq(rsrq)

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
            load_proxy_rsrq,
        ]

        if self.feature_profile == FeatureProfile.ORAN_E2:
            # Network-side features (available in xApp/SON deployment only).
            if cfg.prb_available:
                prb_utilization = np.clip(self._loads, 0.0, 1.0)
                prb_available = np.ones(cfg.num_cells)
            else:
                prb_utilization = np.zeros(cfg.num_cells)
                prb_available = np.zeros(cfg.num_cells)

            ue_counts = self.cell_user_counts()
            max_ues_per_cell = max(cfg.num_ues / cfg.num_cells * 3, 1.0)
            connected_ue_count = np.clip(ue_counts / max_ues_per_cell, 0.0, 1.0)

            cell_thr = np.zeros(cfg.num_cells)
            for c in range(cfg.num_cells):
                mask = self.serving == c
                if mask.any():
                    cell_thr[c] = self._throughputs[mask].mean()
            cell_throughput_norm = np.clip(cell_thr / cfg.cell_capacity_mbps, 0.0, 1.0)

            cols.extend([prb_utilization, prb_available, connected_ue_count, cell_throughput_norm])

        return np.column_stack(cols).astype(np.float32)

    def build_all_states(self) -> np.ndarray:
        """Vectorized version of :meth:`build_state` for all UEs.

        Returns:
            ``(num_ues, num_cells, feature_dim)`` float32 tensor-equivalent array.
        """
        cfg = self.cfg
        u = cfg.num_ues
        c = cfg.num_cells
        rows = np.arange(u)

        rsrp = self._rsrp
        rsrq = self._rsrq
        current = self.serving

        rsrp_norm = np.clip((rsrp - (-120.0)) / 55.0, 0.0, 1.0)
        rsrq_norm = np.clip((rsrq - (-20.0)) / 17.0, 0.0, 1.0)

        serving_rsrp = rsrp[rows, current][:, None]
        rsrp_delta = np.clip((rsrp - serving_rsrp + 20.0) / 40.0, 0.0, 1.0)

        serving_rsrq = rsrq[rows, current][:, None]
        rsrq_delta = np.clip((rsrq - serving_rsrq + 10.0) / 20.0, 0.0, 1.0)

        prev_rsrp = self._prev_rsrp if hasattr(self, "_prev_rsrp") else rsrp
        rsrp_trend = np.clip((rsrp - prev_rsrp + 5.0) / 10.0, 0.0, 1.0)

        is_serving = np.zeros((u, c))
        is_serving[rows, current] = 1.0

        signal_usable = (rsrp >= cfg.min_rsrp_dbm).astype(float)

        speed_norm = np.clip(self.ue_speed / cfg.max_speed_mps, 0.0, 1.0)
        ue_speed_class = np.broadcast_to(speed_norm[:, None], (u, c))

        steps_since_ho = self.step_index - self.last_handover_step
        time_norm = np.clip(steps_since_ho / 20.0, 0.0, 1.0)
        time_since_ho = np.broadcast_to(time_norm[:, None], (u, c))

        was_previous = np.zeros((u, c))
        previous_mask = self.previous_cell >= 0
        if np.any(previous_mask):
            was_previous[rows[previous_mask], self.previous_cell[previous_mask]] = 1.0

        load_proxy_rsrq = self.load_from_rsrq(rsrq)

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
            load_proxy_rsrq,
        ]

        if self.feature_profile == FeatureProfile.ORAN_E2:
            if cfg.prb_available:
                prb_utilization = np.clip(self._loads, 0.0, 1.0)
                prb_available = np.ones(c)
            else:
                prb_utilization = np.zeros(c)
                prb_available = np.zeros(c)

            ue_counts = self.cell_user_counts()
            max_ues_per_cell = max(cfg.num_ues / cfg.num_cells * 3, 1.0)
            connected_ue_count = np.clip(ue_counts / max_ues_per_cell, 0.0, 1.0)

            cell_thr = np.zeros(c)
            for cell in range(c):
                mask = self.serving == cell
                if mask.any():
                    cell_thr[cell] = self._throughputs[mask].mean()
            cell_throughput_norm = np.clip(cell_thr / cfg.cell_capacity_mbps, 0.0, 1.0)

            cols.extend(
                [
                    np.broadcast_to(prb_utilization, (u, c)),
                    np.broadcast_to(prb_available, (u, c)),
                    np.broadcast_to(connected_ue_count, (u, c)),
                    np.broadcast_to(cell_throughput_norm, (u, c)),
                ]
            )

        return np.stack(cols, axis=2).astype(np.float32)

    # ------------------------------------------------------------------
    # Action / reward
    # ------------------------------------------------------------------
    def step_user_action(self, ue_idx: int, target_cell: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        target_cell = int(np.clip(target_cell, 0, self.cfg.num_cells - 1))
        old_cell = int(self.serving[ue_idx])
        pre_action_rsrp = self._rsrp.copy()
        pre_action_loads = self._loads.copy()
        pre_action_throughputs = self._throughputs.copy()
        handover = target_cell != old_cell
        previous_before_action = int(self.previous_cell[ue_idx])
        last_ho_before_action = int(self.last_handover_step[ue_idx])
        steps_since_last = self.step_index - last_ho_before_action
        
        # Speed-adaptive ping-pong window (Bug 5 fix)
        speed_ratio = self.ue_speed[ue_idx] / max(self.cfg.max_speed_mps, 1.0)
        adaptive_window = max(4, int(12 * (1.0 - speed_ratio)))
        
        pingpong = bool(
            handover
            and target_cell == previous_before_action
            and steps_since_last <= adaptive_window
        )

        # Sticky-cell tracking: did the UE stay on a worse cell than the best
        # available neighbor for too long? margin > 3 dB means the serving cell
        # is materially worse than an alternative.
        rsrp_now = pre_action_rsrp[ue_idx]
        serving_rsrp = rsrp_now[old_cell]
        neighbor_mask = np.ones_like(rsrp_now, dtype=bool)
        neighbor_mask[old_cell] = False
        best_neighbor_rsrp = float(np.max(rsrp_now[neighbor_mask])) if neighbor_mask.any() else serving_rsrp
        below_margin = bool(serving_rsrp < best_neighbor_rsrp - 3.0)
        if below_margin:
            self.steps_below_margin[ue_idx] += 1
            self.steps_below_margin_total += 1
        else:
            # Handover after sustained degradation = late HO (sticky cell symptom)
            if handover and self.steps_below_margin[ue_idx] >= 5:
                self.late_ho_count += 1
            self.steps_below_margin[ue_idx] = 0

        if handover:
            rsrp_target = self._rsrp[ue_idx, target_cell]
            if rsrp_target < self.cfg.min_rsrp_dbm:
                self.weak_target_handovers += 1
            if pingpong:
                self.pingpong_handovers += 1
            self.previous_cell[ue_idx] = old_cell
            self.serving[ue_idx] = target_cell
            self.last_handover_step[ue_idx] = self.step_index
            self.total_handovers += 1

            # Serving changed → per-UE throughput / load changed slightly.
            # Refresh snapshot so the reward / next build_state see the
            # post-action state. Keep the same measurement-noise realization;
            # otherwise an action is rewarded/punished partly for a new random
            # channel draw that was not visible when the action was selected.
            self._refresh_snapshot(reroll_noise=False)

        reward = self.user_reward(
            ue_idx,
            old_cell,
            target_cell,
            steps_since_last=steps_since_last,
            pingpong=pingpong,
            pre_action_rsrp=pre_action_rsrp,
            pre_action_loads=pre_action_loads,
            pre_action_throughputs=pre_action_throughputs,
        )
        info = {
            "handover": float(handover),
            "pingpong": float(pingpong),
            "steps_since_last_handover": float(steps_since_last),
        }
        return self.build_state(ue_idx), reward, False, info

    def user_reward(
        self,
        ue_idx: int,
        old_cell: int,
        target_cell: int,
        *,
        steps_since_last: int | None = None,
        pingpong: bool = False,
        pre_action_rsrp: np.ndarray | None = None,
        pre_action_loads: np.ndarray | None = None,
        pre_action_throughputs: np.ndarray | None = None,
    ) -> float:
        """Speed-aware multi-objective reward designed to decisively beat A3/TTT.

        Key innovations over v3:
        - Speed-conditioned HO penalty: fast UEs pay less for necessary HOs
        - Proactive HO bonus: reward handover BEFORE signal drops below threshold
        - P5 throughput protection: penalize being in the tail
        - Load-gain scaling: stronger incentive to escape congested cells
        - Outage severity: cumulative penalty based on how far below threshold
        """
        throughputs = self._throughputs
        loads = self._loads
        rsrp_target = self._rsrp[ue_idx, target_cell]
        rsrp_serving = self._rsrp[ue_idx, old_cell]
        before_rsrp = pre_action_rsrp if pre_action_rsrp is not None else self._rsrp
        before_loads = pre_action_loads if pre_action_loads is not None else self._loads
        before_throughputs = (
            pre_action_throughputs
            if pre_action_throughputs is not None
            else self._throughputs
        )
        before_serving_rsrp = before_rsrp[ue_idx, old_cell]
        before_target_rsrp = before_rsrp[ue_idx, target_cell]

        # --- Throughput satisfaction ---
        satisfaction = np.clip(
            throughputs[ue_idx] / max(self.demands[ue_idx], 1e-6), 0.0, 1.2
        )
        thr_bonus = np.log2(1.0 + max(throughputs[ue_idx], 0.0)) / 4.0
        thr_delta = (
            throughputs[ue_idx] - before_throughputs[ue_idx]
        ) / max(self.demands[ue_idx], 1e-6)

        # P5 protection: penalize being in the bottom 20% of throughput
        all_thr = throughputs
        p20_thr = float(np.percentile(all_thr, 20))
        pre_p5_thr = float(np.percentile(before_throughputs, 5))
        post_p5_thr = float(np.percentile(throughputs, 5))
        p5_delta = (post_p5_thr - pre_p5_thr) / max(pre_p5_thr, 1e-6)
        tail_penalty = 0.0
        if throughputs[ue_idx] < p20_thr and p20_thr > 0.1:
            tail_penalty = 0.5 * (1.0 - throughputs[ue_idx] / p20_thr)

        # --- Load balancing ---
        load_std = float(min(np.std(loads), 1.0))
        pre_load_std = float(min(np.std(before_loads), 1.0))
        load_std_improvement = pre_load_std - load_std
        overload = float(np.sum(np.maximum(loads - 1.0, 0.0)))
        pre_overload = float(np.sum(np.maximum(before_loads - 1.0, 0.0)))
        overload_improvement = pre_overload - overload
        load_sum_sq = np.sum(loads * loads)
        jain_fairness = float((np.sum(loads) ** 2) / (self.cfg.num_cells * load_sum_sq + 1e-9))

        # --- Speed-aware handover cost ---
        handover = float(old_cell != target_cell)
        speed = self.ue_speed[ue_idx]
        speed_ratio = speed / max(self.cfg.max_speed_mps, 1.0)

        # Fast UEs (highway): HO penalty reduced aggressively for proactive
        # mobility. Slow UEs remain moderately conservative but still allow
        # load-balancing HOs that A3 cannot perform.
        ho_cost_base = np.clip(2.2 - 1.6 * (speed_ratio ** 1.2), 0.5, 0.8)
        ho_cost = ho_cost_base * handover

        if steps_since_last is None:
            steps_since_last = self.step_index - self.last_handover_step[ue_idx]

        # Recency penalty also scaled by speed
        ho_recency_penalty = 0.0
        recency_window = max(3, int(8 * (1.0 - speed_ratio)))
        if handover and steps_since_last < recency_window:
            ho_recency_penalty = 1.2 * (1.0 - steps_since_last / recency_window)

        pingpong_penalty = float(pingpong)

        # --- Signal quality and proactive HO ---
        margin_db = rsrp_target - self.cfg.min_rsrp_dbm
        outage = float(rsrp_target < self.cfg.min_rsrp_dbm)
        # Severity: how far below threshold (deeper outage = worse)
        outage_severity = 0.0
        if outage:
            outage_severity = min(abs(margin_db) / 10.0, 1.0)

        # Proactive HO bonus: reward moving to a stronger cell BEFORE outage
        proactive_bonus = 0.0
        if handover and not outage:
            serving_margin = before_serving_rsrp - self.cfg.min_rsrp_dbm
            target_margin = before_target_rsrp - self.cfg.min_rsrp_dbm
            # Bonus if leaving a cell with low margin for one with higher margin
            if serving_margin < 10.0 and target_margin > serving_margin + 3.0:
                proactive_bonus = 0.9 * min((target_margin - serving_margin) / 10.0, 1.0)
            if (
                self.ue_speed[ue_idx] > 0.70 * max(self.cfg.max_speed_mps, 1.0)
                and target_margin > serving_margin + 1.5
                and before_serving_rsrp < self.cfg.min_rsrp_dbm + 14.0
            ):
                proactive_bonus += 0.5

        # Stay bonus: only for genuinely strong signal + low load
        stay_bonus = 0.0
        if not handover and rsrp_target >= self.cfg.min_rsrp_dbm + 6.0:
            serving_load = loads[old_cell]
            if serving_load < 0.85:
                stay_bonus = 0.0
            else:
                stay_bonus = 0.0

        # --- Load-aware HO incentive (key differentiator vs A3) ---
        target_load = loads[target_cell] if handover else loads[old_cell]
        source_load = before_loads[old_cell]
        load_gain = 0.0
        if handover:
            load_diff = before_loads[old_cell] - before_loads[target_cell]
            if load_diff > 0.03:
                load_gain = 5.0 * min(load_diff, 0.6)
            elif load_diff < -0.15:
                load_gain = 0.8 * load_diff

        # Overload escape bonus: strong reward for leaving overloaded cell
        overload_escape = 0.0
        if handover and source_load > 0.85 and before_loads[target_cell] < 0.75:
            overload_escape = 4.0
        elif handover and source_load > 1.0 and before_loads[target_cell] < 0.9:
            overload_escape = 5.5

        return float(
            3.5 * satisfaction
            + 1.2 * thr_bonus
            + 1.8 * np.clip(thr_delta, -0.5, 0.8)
            + 1.2 * np.clip(p5_delta, -0.5, 0.8)
            + stay_bonus
            + 1.2 * jain_fairness
            + load_gain
            + proactive_bonus
            + overload_escape
            + 3.0 * np.clip(load_std_improvement, -0.25, 0.25)
            + 1.5 * np.clip(overload_improvement, -0.5, 0.5)
            - 1.8 * tail_penalty
            - 1.5 * load_std
            - 2.0 * overload
            - ho_cost
            - ho_recency_penalty
            - self.cfg.pingpong_penalty_weight * pingpong_penalty
            - 2.5 * outage
            - 1.8 * outage_severity
            - 0.4 * float(handover and target_load > 0.85)
        )

    def metrics(self) -> Dict[str, float]:
        throughputs = self._throughputs
        loads = self._loads
        serving_rsrp = self._rsrp[np.arange(self.cfg.num_ues), self.serving]
        load_sum_sq = np.sum(loads * loads)
        jain = (np.sum(loads) ** 2) / (self.cfg.num_cells * load_sum_sq + 1e-9)
        steps_so_far = max(self.step_index, 1)
        ue_steps = self.cfg.num_ues * steps_so_far

        return {
            "avg_ue_throughput_mbps": float(np.mean(throughputs)),
            "p5_ue_throughput_mbps": float(np.percentile(throughputs, 5)),
            "p10_ue_throughput_mbps": float(np.percentile(throughputs, 10)),
            "total_throughput_mbps": float(np.sum(throughputs)),
            "avg_cell_load": float(np.mean(loads)),
            "load_std": float(np.std(loads)),
            "load_gap": float((np.max(loads) - np.min(loads)) / max(np.mean(loads), 1e-6)),
            "jain_load_fairness": float(jain),
            "outage_rate": float(np.mean(serving_rsrp < self.cfg.min_rsrp_dbm)),
            "overload_rate": float(np.mean(loads > 1.0)),
            # Handover quality KPIs
            "ho_success_rate": float(1.0 - self.weak_target_handovers / max(self.total_handovers, 1)),
            "ho_interruption_fraction": float(self.total_handovers * self.cfg.handover_interruption_frac / ue_steps),
            "mean_time_between_handovers": float(ue_steps / max(self.total_handovers, 1)),
            # Sticky-cell KPIs
            "late_ho_rate": float(self.late_ho_count / max(self.total_handovers, 1)),
            "avg_steps_below_margin": float(self.steps_below_margin_total / ue_steps),
        }
