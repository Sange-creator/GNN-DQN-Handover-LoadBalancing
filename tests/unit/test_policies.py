from __future__ import annotations

from handover_gnn_dqn.env import CellularNetworkEnv, LTEConfig
from handover_gnn_dqn.metrics import default_policy_factories
from handover_gnn_dqn.policies import RandomValidPolicy


def test_random_valid_policy_selects_valid_action() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=9, num_ues=12))
    policy = RandomValidPolicy(seed=11)
    policy.reset(env)
    action = policy.select(env, 0)
    assert env.valid_actions(0)[action]


def test_default_policies_include_random_baseline() -> None:
    factories = default_policy_factories()
    assert "random_valid" in factories
