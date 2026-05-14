from __future__ import annotations

import numpy as np

from handover_gnn_dqn.env import CellularNetworkEnv, LTEConfig, normalize_positions_to_area
from handover_gnn_dqn.topology import build_adjacency_from_positions
from handover_gnn_dqn.topology import get_training_scenarios


def test_normalize_positions_to_area_places_cells_inside_ue_frame() -> None:
    positions = np.array([[-1000.0, -500.0], [0.0, 0.0], [1200.0, 800.0]])
    normalized = normalize_positions_to_area(positions, area_m=2500.0)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 2500.0


def test_env_normalizes_centered_topology_positions() -> None:
    positions = np.array([[-500.0, -200.0], [0.0, 0.0], [500.0, 200.0]])
    env = CellularNetworkEnv(LTEConfig(num_cells=3, num_ues=9, area_m=1400.0, cell_positions=positions))
    assert env.cell_pos.min() >= 0.0
    assert env.cell_pos.max() <= env.cfg.area_m
    assert env.ue_pos.min() >= 0.0
    assert env.ue_pos.max() <= env.cfg.area_m


def test_highway_mobility_stays_near_corridor() -> None:
    highway = [s for s in get_training_scenarios() if s.name == "highway"][0]
    env = CellularNetworkEnv(
        LTEConfig(
            num_cells=highway.num_cells,
            num_ues=highway.num_ues,
            area_m=highway.area_m,
            cell_positions=highway.cell_positions,
            min_speed_mps=highway.min_speed_mps,
            max_speed_mps=highway.max_speed_mps,
            mobility_model=highway.mobility_model,
        )
    )
    road_y = float(np.median(env.cell_pos[:, 1]))
    assert np.max(np.abs(env.ue_pos[:, 1] - road_y)) <= env.cfg.road_width_m
    for _ in range(5):
        env.advance_mobility()
    assert np.max(np.abs(env.ue_pos[:, 1] - road_y)) <= env.cfg.road_width_m + 1e-6


def test_oran_prb_missing_mask_is_explicit() -> None:
    env = CellularNetworkEnv(LTEConfig(feature_mode="oran_e2", prb_available=False))
    state = env.build_state(0)
    assert state.shape == (env.cfg.num_cells, 15)
    assert np.allclose(state[:, 11], 0.0)
    assert np.allclose(state[:, 12], 0.0)


def test_single_cell_topology_has_empty_graph_without_nan_warning() -> None:
    positions = np.array([[0.0, 0.0]])
    adjacency = build_adjacency_from_positions(positions)
    assert adjacency.shape == (1, 1)
    assert np.allclose(adjacency, 0.0)

    env = CellularNetworkEnv(
        LTEConfig(num_cells=1, num_ues=2, area_m=500.0, cell_positions=positions)
    )
    edge_index, edge_weight = env.edge_data
    assert env.adjacency.shape == (1, 1)
    assert edge_index.shape == (2, 0)
    assert edge_weight.shape == (0,)
    assert env.build_state(0).shape == (1, env.feature_dim)


def test_reward_pingpong_uses_pre_action_history() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=3, num_ues=3, area_m=900.0))
    ue_idx = 0
    env.serving[ue_idx] = 0
    env.previous_cell[ue_idx] = 1
    env.last_handover_step[ue_idx] = env.step_index - 1
    env._refresh_snapshot(reroll_noise=False)
    _next_state, _reward, _done, info = env.step_user_action(ue_idx, 1)
    assert info["pingpong"] == 1.0
    assert env.pingpong_handovers == 1


def test_handover_reward_uses_same_measurement_noise_snapshot() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=3, num_ues=3, area_m=900.0))
    env.reset(123)
    ue_idx = 0
    current = int(env.serving[ue_idx])
    valid_targets = np.flatnonzero(env.valid_actions(ue_idx))
    target = int(next(cell for cell in valid_targets if cell != current))
    rsrp_noise_before = env._rsrp_noise.copy()
    rsrq_noise_before = env._rsrq_noise.copy()

    env.step_user_action(ue_idx, target)

    assert np.allclose(env._rsrp_noise, rsrp_noise_before)
    assert np.allclose(env._rsrq_noise, rsrq_noise_before)


def test_reward_prefers_improving_load_balance_with_same_radio_snapshot() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=3, num_ues=6, area_m=900.0))
    env.reset(321)
    ue_idx = 0
    old_cell = 0
    good_target = 1
    bad_target = 2
    env.serving[ue_idx] = old_cell
    env._rsrp[ue_idx, [old_cell, good_target, bad_target]] = np.array([-96.0, -94.0, -94.0])
    env._throughputs[:] = 3.0
    env._loads[:] = np.array([0.90, 0.55, 1.10])
    before_rsrp = env._rsrp.copy()
    before_loads = np.array([1.20, 0.45, 1.15])
    before_throughputs = env._throughputs.copy()

    good_reward = env.user_reward(
        ue_idx,
        old_cell,
        good_target,
        pre_action_rsrp=before_rsrp,
        pre_action_loads=before_loads,
        pre_action_throughputs=before_throughputs,
    )
    bad_reward = env.user_reward(
        ue_idx,
        old_cell,
        bad_target,
        pre_action_rsrp=before_rsrp,
        pre_action_loads=before_loads,
        pre_action_throughputs=before_throughputs,
    )

    assert good_reward > bad_reward
