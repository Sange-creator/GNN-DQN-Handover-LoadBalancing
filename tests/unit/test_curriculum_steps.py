from __future__ import annotations

import numpy as np

from handover_gnn_dqn.rl import capped_num_ues, steps_for_episode
from handover_gnn_dqn.topology import Scenario


def test_steps_for_episode_uses_default_without_curriculum() -> None:
    assert steps_for_episode(120, 0, None) == 120
    assert steps_for_episode(120, 300, {}) == 120


def test_steps_for_episode_uses_phase_boundaries() -> None:
    curriculum = {
        "phase1_episodes": 3,
        "phase1_steps": 10,
        "phase2_episodes": 6,
        "phase2_steps": 15,
        "phase3_steps": 20,
    }

    assert steps_for_episode(120, 0, curriculum) == 10
    assert steps_for_episode(120, 2, curriculum) == 10
    assert steps_for_episode(120, 3, curriculum) == 15
    assert steps_for_episode(120, 5, curriculum) == 15
    assert steps_for_episode(120, 6, curriculum) == 20


def test_capped_num_ues_keeps_scenario_bounds() -> None:
    scenario = Scenario(
        name="dense",
        num_cells=4,
        num_ues=120,
        area_m=1000.0,
        cell_positions=np.zeros((4, 2)),
        min_speed_mps=1.0,
        max_speed_mps=5.0,
        description="unit test",
    )

    assert capped_num_ues(scenario, None) == 120
    assert capped_num_ues(scenario, {"dense": 80}) == 80
    assert capped_num_ues(scenario, {"dense": 200}) == 120
    assert capped_num_ues(scenario, {"dense": 0}) == 1
