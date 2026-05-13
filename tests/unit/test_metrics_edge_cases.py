from __future__ import annotations

import pytest

from handover_gnn_dqn.env import LTEConfig
from handover_gnn_dqn.metrics import default_policy_factories, evaluate_policies, write_summary_csv


def test_evaluate_policies_rejects_empty_seed_list() -> None:
    with pytest.raises(ValueError, match="seeds"):
        evaluate_policies(
            LTEConfig(num_cells=3, num_ues=3),
            default_policy_factories(),
            steps=1,
            seeds=[],
        )


def test_evaluate_policies_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        evaluate_policies(
            LTEConfig(num_cells=3, num_ues=3),
            default_policy_factories(),
            steps=0,
            seeds=[1],
        )


def test_write_summary_csv_rejects_empty_rows(tmp_path) -> None:
    with pytest.raises(ValueError, match="rows"):
        write_summary_csv([], tmp_path / "empty.csv")
