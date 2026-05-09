from __future__ import annotations

import numpy as np

from handover_gnn_dqn.env import CellularNetworkEnv, LTEConfig, normalize_positions_to_area
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
