from __future__ import annotations

from handover_gnn_dqn.models import DQNConfig
from handover_gnn_dqn.rl import train_multi_scenario
from handover_gnn_dqn.topology import get_training_scenarios


def test_short_ue_only_training_smoke() -> None:
    scenarios = get_training_scenarios()[:2]
    cfg = DQNConfig(hidden_dim=32, batch_size=16, replay_capacity=2000, epsilon_decay_episodes=2)
    agent, history, max_cells = train_multi_scenario(
        scenarios,
        cfg,
        total_episodes=2,
        steps_per_episode=2,
        feature_mode="ue_only",
        prb_available=False,
        seed=3,
        verbose=False,
    )
    assert max_cells == max(s.num_cells for s in scenarios)
    assert len(history) == 2
    assert agent.feature_dim == 11
