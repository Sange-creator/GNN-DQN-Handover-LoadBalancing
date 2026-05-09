from __future__ import annotations

import pytest
import torch

from handover_gnn_dqn.env import LTEConfig, CellularNetworkEnv
from handover_gnn_dqn.models import DQNConfig, GnnDQNAgent
from handover_gnn_dqn.topology.scenarios import _hex_grid


@pytest.mark.parametrize("eval_cells", [19, 25, 40, 41])
def test_gnn_trained_on_20_cells_can_infer_variable_cell_graph(eval_cells: int) -> None:
    positions = _hex_grid(eval_cells, 350.0)
    env = CellularNetworkEnv(
        LTEConfig(num_cells=eval_cells, num_ues=50, area_m=3500.0, cell_positions=positions)
    )
    agent = GnnDQNAgent(num_cells=20, feature_dim=env.feature_dim, cfg=DQNConfig(hidden_dim=32))
    state = torch.from_numpy(env.build_state(0)).float()
    edge_index, edge_weight = env.edge_data
    q = agent(state, edge_index, edge_weight)
    assert q.shape == (eval_cells,)
    action = agent.act(state, edge_index, edge_weight, valid_mask=env.valid_actions(0))
    assert 0 <= action < eval_cells


def test_gnn_batched_training_shape_is_explicit() -> None:
    env = CellularNetworkEnv(LTEConfig(num_cells=20, num_ues=20))
    agent = GnnDQNAgent(num_cells=20, feature_dim=env.feature_dim, cfg=DQNConfig(hidden_dim=32))
    state = torch.from_numpy(env.build_state(0)).float()
    states = torch.cat([state, state], dim=0)
    edge_index, edge_weight = env.edge_data
    edge_index_batch = torch.cat([edge_index, edge_index + 20], dim=1)
    edge_weight_batch = edge_weight.repeat(2)

    q = agent(
        states,
        edge_index_batch,
        edge_weight_batch,
        batch_size=2,
        nodes_per_graph=20,
    )
    assert q.shape == (2, 20)
