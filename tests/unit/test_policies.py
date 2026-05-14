from __future__ import annotations

from handover_gnn_dqn.env import CellularNetworkEnv, LTEConfig
from handover_gnn_dqn.metrics import default_policy_factories
from handover_gnn_dqn.policies import A3HandoverPolicy, RandomValidPolicy, StrongestRsrpPolicy


def test_random_valid_policy_selects_valid_action() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=9, num_ues=12))
    policy = RandomValidPolicy(seed=11)
    policy.reset(env)
    action = policy.select(env, 0)
    assert env.valid_actions(0)[action]


def test_default_policies_include_random_baseline() -> None:
    factories = default_policy_factories()
    assert "random_valid" in factories


def test_signal_baselines_do_not_select_invalid_strongest_cell() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=4, num_ues=4, area_m=1200.0))
    env.reset(123)
    ue_idx = 0
    current = int(env.serving[ue_idx])
    invalid_target = (current + 1) % env.cfg.num_cells

    env._rsrp[ue_idx, current] = env.cfg.min_rsrp_dbm - 4.0
    env._rsrp[ue_idx, invalid_target] = env.cfg.min_rsrp_dbm - 20.0
    valid = env.valid_actions(ue_idx)
    env._rsrp[ue_idx, invalid_target] = env._rsrp[ue_idx, current] + 25.0
    assert not valid[invalid_target]

    for policy in (
        StrongestRsrpPolicy(hysteresis_db=0.0),
        A3HandoverPolicy(offset_db=0.0, time_to_trigger=1),
    ):
        policy.reset(env)
        action = policy.select(env, ue_idx)
        assert valid[action]
