from __future__ import annotations

import numpy as np

from handover_gnn_dqn.env import CellularNetworkEnv, LTEConfig
from handover_gnn_dqn.policies import SONTunedA3Policy
from handover_gnn_dqn.son import SONConfig, SONController


class _FixedAgent:
    def __init__(self, target: int):
        self.target = target

    def act(self, *_args, valid_mask=None, **_kwargs) -> int:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid[self.target]:
            return self.target
        return int(np.flatnonzero(valid)[0])


def test_son_controller_applies_bounded_cio_update() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=4, num_ues=16, area_m=1200.0))
    env.serving[:] = 0
    env._refresh_snapshot(reroll_noise=False)
    controller = SONController(
        _FixedAgent(target=1),
        SONConfig(update_interval_steps=1, max_cio_step_db=0.5, max_updates_per_cycle=2),
    )
    controller.reset(env)
    updates = controller.update(env)
    assert updates
    assert controller.cio(0, 1) <= 0.5
    assert controller.cio(0, 1) >= -0.5
    metrics = controller.metrics()
    assert metrics["son_update_count"] >= 1.0


def test_son_tuned_a3_policy_returns_valid_action_and_metrics() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=4, num_ues=8, area_m=1200.0))
    policy = SONTunedA3Policy(
        _FixedAgent(target=1),
        SONConfig(update_interval_steps=1, max_cio_step_db=0.5),
    )
    policy.reset(env)
    action = policy.select(env, 0)
    assert 0 <= action < env.cfg.num_cells
    assert "son_update_count" in policy.son_metrics()


def test_son_controller_true_prb_load_signal_uses_real_load() -> None:
    """SON in true_prb mode reads env.cell_loads() instead of the RSRQ proxy."""
    env = CellularNetworkEnv(LTEConfig(num_cells=4, num_ues=16, area_m=1200.0))
    env.serving[:] = 0
    # Force cell 1 to look heavily overloaded via the true-load signal.
    env._refresh_snapshot(reroll_noise=False)
    env._loads[1] = 0.95
    controller = SONController(
        _FixedAgent(target=1),
        SONConfig(
            update_interval_steps=1,
            max_cio_step_db=0.5,
            load_signal="true_prb",
            load_proxy_overload_threshold=0.85,
            preference_threshold=0.99,  # disable pull-toward path so only overload triggers
        ),
    )
    controller.reset(env)
    updates = controller.update(env)
    overload_updates = [u for u in updates if u.target_cell == 1]
    assert overload_updates, "expected SON to react to true-PRB overload on cell 1"
    assert all(u.delta_cio_db < 0 for u in overload_updates), "overload should push CIO negative"
    assert all(u.reason == "target_prb_overloaded" for u in overload_updates)


def test_ttt_cooldown_only_starts_when_ttt_changes() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=4, num_ues=8, area_m=1200.0))
    controller = SONController(
        _FixedAgent(target=0),
        SONConfig(
            update_interval_steps=1,
            ttt_decrease_threshold=0.10,
            ttt_cooldown_steps=30,
            max_updates_per_cycle=0,
        ),
    )
    controller.reset(env)
    assert controller.ttt_steps is not None
    # Set to min_ttt_steps so there's no room to decrease
    controller.ttt_steps[:] = controller.config.min_ttt_steps

    env.step_index = 1
    env.total_handovers = 10
    env.pingpong_handovers = 0
    controller.update(env)
    assert controller._last_ttt_change_step == -10_000

    env.step_index = 2
    env.pingpong_handovers = 10
    controller.update(env)
    assert controller.ttt(0, 1) == controller.config.min_ttt_steps + 1
    assert controller._last_ttt_change_step == 2
