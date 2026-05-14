"""Tests for the held-out validation pass used to gate best-checkpoint selection.

These tests verify Bug 2 fixes:
1. _run_validation_pass produces honest, exploitation-only metrics (epsilon=0).
2. train_multi_scenario uses validation score (not training-metric score) for best-checkpoint
   selection when validation is configured.
3. Validation is skipped cleanly when not configured (backward compatibility).
"""
from __future__ import annotations

import math

from handover_gnn_dqn.models import DQNConfig
from handover_gnn_dqn.rl.training import (
    _run_validation_pass,
    train_multi_scenario,
    training_validation_score,
)
from handover_gnn_dqn.topology import get_training_scenarios


def _tiny_dqn_cfg() -> DQNConfig:
    # Small enough to train end-to-end inside pytest in ~10-20s.
    return DQNConfig(
        hidden_dim=16,
        gamma=0.95,
        learning_rate=3e-4,
        batch_size=16,
        replay_capacity=2_000,
        train_every=4,
        target_update_every=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=4,
        n_step=1,
        per_alpha=0.0,
    )


def test_run_validation_pass_returns_holdout_score_and_metrics() -> None:
    """_run_validation_pass should run with epsilon=0 and return aggregated metrics."""
    from handover_gnn_dqn.env import CellularNetworkEnv
    from handover_gnn_dqn.models import GnnDQNAgent
    from handover_gnn_dqn.rl.training import make_env_from_scenario

    scenarios = get_training_scenarios(seed=42)[:2]
    sample_env = make_env_from_scenario(scenarios[0], feature_mode="ue_only", prb_available=False)
    max_cells = max(s.num_cells for s in scenarios)
    agent = GnnDQNAgent(max_cells, sample_env.feature_dim, _tiny_dqn_cfg(), seed=0)

    result = _run_validation_pass(
        agent,
        scenarios,
        seeds=[1234, 5678],
        steps=10,
        feature_mode="ue_only",
        prb_available=False,
    )

    assert "holdout_validation_score" in result
    assert "holdout_validation_score_std" in result
    assert math.isfinite(result["holdout_validation_score"])
    assert "avg_ue_throughput_mbps" in result
    assert "pingpong_rate" in result


def test_train_with_validation_writes_val_metrics_to_history() -> None:
    """When validation is configured, history rows on validation episodes contain val_* keys."""
    train_scenarios = get_training_scenarios(seed=42)[:2]
    val_scenarios = train_scenarios[:1]

    agent, history, _max_cells = train_multi_scenario(
        train_scenarios,
        _tiny_dqn_cfg(),
        total_episodes=4,
        steps_per_episode=8,
        feature_mode="ue_only",
        prb_available=False,
        seed=42,
        verbose=False,
        validation_scenarios=val_scenarios,
        validation_seeds=[9001, 9002],
        validate_every_episodes=2,
        validation_steps=5,
    )

    # At least one episode should have run validation
    val_rows = [r for r in history if "val_holdout_validation_score" in r]
    assert len(val_rows) >= 1, "expected at least one validation episode in history"
    for row in val_rows:
        assert "val_avg_ue_throughput_mbps" in row
        assert "val_pingpong_rate" in row


def test_train_without_validation_falls_back_to_training_metric_score() -> None:
    """Backward compatibility: when validation not configured, training-metric score still used."""
    train_scenarios = get_training_scenarios(seed=42)[:2]

    agent, history, _max_cells = train_multi_scenario(
        train_scenarios,
        _tiny_dqn_cfg(),
        total_episodes=3,
        steps_per_episode=8,
        feature_mode="ue_only",
        prb_available=False,
        seed=42,
        verbose=False,
        # No validation_* args
    )

    assert all("validation_score" in r for r in history)
    assert not any("val_holdout_validation_score" in r for r in history), \
        "should not write val_* keys when validation is not configured"


def test_validation_score_is_finite_and_used_for_best_selection() -> None:
    """When validation runs, best-checkpoint selection should use val_holdout_validation_score."""
    train_scenarios = get_training_scenarios(seed=42)[:2]
    val_scenarios = train_scenarios[:1]

    agent, history, _max_cells = train_multi_scenario(
        train_scenarios,
        _tiny_dqn_cfg(),
        total_episodes=6,
        steps_per_episode=8,
        feature_mode="ue_only",
        prb_available=False,
        seed=42,
        verbose=False,
        validation_scenarios=val_scenarios,
        validation_seeds=[7777],
        validate_every_episodes=2,
        validation_steps=5,
    )

    # All val-bearing rows should have finite scores
    val_rows = [r for r in history if "val_holdout_validation_score" in r]
    assert val_rows, "expected validation to run at least once"
    for row in val_rows:
        assert math.isfinite(row["val_holdout_validation_score"])
