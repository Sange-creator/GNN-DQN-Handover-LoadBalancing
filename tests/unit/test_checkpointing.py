from __future__ import annotations

import pytest

from handover_gnn_dqn.env import CellularNetworkEnv, LTEConfig
from handover_gnn_dqn.models import DQNConfig, GnnDQNAgent
from handover_gnn_dqn.rl import load_gnn_checkpoint, save_checkpoint


def test_checkpoint_metadata_round_trip_and_compatibility(tmp_path) -> None:
    env = CellularNetworkEnv(LTEConfig(feature_mode="ue_only", prb_available=False))
    cfg = {
        "run_name": "unit",
        "feature_mode": "ue_only",
        "prb_available": False,
        "dqn": {"hidden_dim": 32},
    }
    agent = GnnDQNAgent(env.cfg.num_cells, env.feature_dim, DQNConfig(hidden_dim=32))
    checkpoint = tmp_path / "gnn.pt"

    save_checkpoint(
        agent,
        checkpoint,
        config=cfg,
        feature_dim=env.feature_dim,
        max_cells=env.cfg.num_cells,
        scenario_names=["unit"],
        history=[{"episode": 1.0, "validation_score": 0.1}],
        cwd=tmp_path,
        training_state={"episode_completed": 1, "best_episode": 1, "best_score": 0.1},
    )

    loaded, meta, payload = load_gnn_checkpoint(
        checkpoint,
        expected_feature_profile="ue_only",
        expected_feature_dim=env.feature_dim,
    )
    assert loaded.feature_dim == env.feature_dim
    assert meta["feature_profile"] == "ue_only"
    assert meta["reward_version"]
    assert payload["metadata"]["training_state"]["best_episode"] == 1

    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        load_gnn_checkpoint(
            checkpoint,
            expected_feature_profile="oran_e2",
            expected_feature_dim=15,
        )
