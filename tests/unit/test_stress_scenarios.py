"""Tests for the stress evaluation scenarios.

These scenarios are eval-only and should drive higher cell utilization than
the training scenarios. We don't want them to be confused with training
scenarios or accidentally bypass safety checks (e.g. zero UEs).
"""
from __future__ import annotations

import numpy as np
import pytest

from handover_gnn_dqn.env import LTEConfig
from handover_gnn_dqn.topology import (
    Scenario,
    get_stress_scenarios,
    get_test_scenarios,
    get_training_scenarios,
)

# Track the actual default so the test follows simulator changes automatically
# instead of silently going stale when capacity is tuned.
CELL_CAPACITY_MBPS = LTEConfig.__dataclass_fields__["cell_capacity_mbps"].default


def test_get_stress_scenarios_returns_list_of_scenarios() -> None:
    scenarios = get_stress_scenarios()
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 4, "Expected at least 4 stress scenarios"
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_stress_scenarios_have_unique_names() -> None:
    scenarios = get_stress_scenarios()
    names = [s.name for s in scenarios]
    assert len(names) == len(set(names)), f"Duplicate stress scenario names: {names}"
    for name in names:
        assert name.startswith("stress_"), (
            f"Stress scenario names must start with 'stress_' to avoid confusion "
            f"with training scenarios. Got: {name}"
        )


def test_stress_scenarios_do_not_collide_with_training_scenarios() -> None:
    train_names = {s.name for s in get_training_scenarios()}
    test_names = {s.name for s in get_test_scenarios()}
    stress_names = {s.name for s in get_stress_scenarios()}
    overlap_train = stress_names & train_names
    overlap_test = stress_names & test_names
    assert not overlap_train, f"Stress scenario name collides with training: {overlap_train}"
    assert not overlap_test, f"Stress scenario name collides with test: {overlap_test}"


def test_stress_scenarios_drive_real_congestion() -> None:
    """The whole point of stress scenarios is to push cells over capacity.

    Mean aggregate demand per cell should exceed cell capacity by ≥20%,
    otherwise these scenarios add no value vs the existing training set.
    """
    scenarios = get_stress_scenarios()
    for s in scenarios:
        avg_demand_per_ue = (s.min_demand_mbps + s.max_demand_mbps) / 2.0
        ues_per_cell = s.num_ues / s.num_cells
        agg_demand_per_cell = avg_demand_per_ue * ues_per_cell
        ratio = agg_demand_per_cell / CELL_CAPACITY_MBPS
        assert ratio >= 1.2, (
            f"{s.name}: aggregate demand/cell = {agg_demand_per_cell:.1f} Mbps "
            f"({ratio:.2f}× capacity) is not high enough to drive congestion."
        )


def test_stress_scenarios_have_more_ues_than_training_counterparts() -> None:
    """When a stress scenario shares a layout family with a training scenario
    (urban, event, pokhara, highway), the stress version must have strictly
    more UEs per cell — otherwise it's not actually stressing anything.
    """
    train_density = {
        s.name: s.num_ues / s.num_cells for s in get_training_scenarios()
    }
    pairs = [
        ("stress_dense_urban", "dense_urban"),
        ("stress_overload_event", "overloaded_event"),
        ("stress_pokhara_peakhour", "pokhara_dense_peakhour"),
        ("stress_highway_jam", "highway"),
    ]
    stress_density = {s.name: s.num_ues / s.num_cells for s in get_stress_scenarios()}
    for stress_name, train_name in pairs:
        assert stress_name in stress_density, f"Missing {stress_name}"
        assert train_name in train_density, f"Missing {train_name}"
        assert stress_density[stress_name] > train_density[train_name], (
            f"{stress_name} density ({stress_density[stress_name]:.1f}) "
            f"must exceed {train_name} ({train_density[train_name]:.1f})"
        )


def test_stress_scenarios_have_valid_layouts() -> None:
    scenarios = get_stress_scenarios()
    for s in scenarios:
        assert isinstance(s.cell_positions, np.ndarray)
        assert s.cell_positions.shape == (s.num_cells, 2), (
            f"{s.name}: cell_positions shape {s.cell_positions.shape} "
            f"!= ({s.num_cells}, 2)"
        )
        assert s.num_ues > 0
        assert s.area_m > 0
        assert s.min_speed_mps >= 0
        assert s.max_speed_mps >= s.min_speed_mps
        assert s.min_demand_mbps > 0
        assert s.max_demand_mbps >= s.min_demand_mbps


def test_stress_scenarios_seed_is_reproducible() -> None:
    """Two calls with the same seed produce identical layouts.

    Currently most stress scenarios use deterministic hex/highway grids that
    ignore the seed, but the contract is that same seed → same positions.
    """
    s1 = get_stress_scenarios(seed=137)
    s2 = get_stress_scenarios(seed=137)
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a.cell_positions, b.cell_positions)
        assert a.num_cells == b.num_cells
        assert a.num_ues == b.num_ues


def test_stress_scenarios_compatible_with_env_construction() -> None:
    """End-to-end smoke: each stress scenario can build a CellularNetworkEnv.

    This catches any regression where the Scenario dataclass adds a field
    that the env constructor doesn't know about.
    """
    from handover_gnn_dqn.rl.training import make_env_from_scenario

    scenarios = get_stress_scenarios()
    for s in scenarios:
        env = make_env_from_scenario(s, feature_mode="ue_only", prb_available=False)
        # Reset with a fixed seed and verify state shape.
        env.reset(seed=0)
        state = env.build_state(0)
        assert state.shape[0] == s.num_cells, (
            f"{s.name}: per-cell state rows {state.shape[0]} != num_cells {s.num_cells}"
        )
