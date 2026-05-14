"""Tests for drive-test shadow inference pipeline."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.models.gnn_dqn import DQNConfig, GnnDQNAgent


def _make_synthetic_drive_csv(path: Path, num_rows: int = 5, num_cells: int = 3) -> None:
    """Create a synthetic drive-test CSV with the expected columns."""
    pcis = list(range(100, 100 + num_cells))
    rows = []
    for i in range(num_rows):
        serving = pcis[i % num_cells]
        neighbors = [p for p in pcis if p != serving]
        rows.append({
            "timestamp": f"2026-05-12T10:00:{i:02d}",
            "ue_id": 1,
            "lat": 27.7 + i * 0.001,
            "lon": 85.3 + i * 0.001,
            "speed_mps": 5.0 + i,
            "serving_pci": serving,
            "neighbor_pcis": ",".join(str(p) for p in neighbors),
            "rsrp_dbm": -90.0 - i * 5,
            "rsrq_db": -8.0 - i,
            "sinr_db": 15.0 - i * 2,
            "handover_event": 1 if i == 2 else 0,
            "throughput_mbps": 20.0 - i * 3,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _make_untrained_agent(num_cells: int = 3) -> GnnDQNAgent:
    """Create a fresh (untrained) GnnDQNAgent."""
    cfg = DQNConfig(hidden_dim=64, dropout=0.0)
    agent = GnnDQNAgent(num_cells=num_cells, feature_dim=11, cfg=cfg, seed=42)
    agent.eval()
    return agent


class TestDriveShadowPipeline:
    """End-to-end test of the shadow inference pipeline."""

    def test_output_columns_and_row_count(self, tmp_path: Path) -> None:
        """Pipeline produces correct columns and row count."""
        # Import here to test the script module
        sys.path.insert(0, str(ROOT / "scripts"))
        # We import the functions directly from the script
        script_path = ROOT / "scripts" / "prepare_drive_data.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("prepare_drive_data", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Create synthetic CSV
        csv_path = tmp_path / "drive_test.csv"
        _make_synthetic_drive_csv(csv_path, num_rows=5, num_cells=3)

        # Create untrained agent
        agent = _make_untrained_agent(num_cells=3)

        # Load CSV
        df = pd.read_csv(csv_path)

        # Run pipeline
        out_df = mod.run_shadow_inference(df, agent, num_cells=3)

        # Check columns
        expected_cols = [
            "timestamp", "ue_id", "recommended_cell", "actual_cell",
            "agreement", "top3_agreement", "predicted_handover_useful",
            "outage_avoided", "pingpong_risk",
        ]
        assert list(out_df.columns) == expected_cols

        # Check row count matches input
        assert len(out_df) == 5

    def test_output_csv_write(self, tmp_path: Path) -> None:
        """Pipeline writes a valid output CSV."""
        script_path = ROOT / "scripts" / "prepare_drive_data.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("prepare_drive_data", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        csv_path = tmp_path / "drive_test.csv"
        out_path = tmp_path / "output.csv"
        _make_synthetic_drive_csv(csv_path, num_rows=5, num_cells=3)

        agent = _make_untrained_agent(num_cells=3)
        df = pd.read_csv(csv_path)
        out_df = mod.run_shadow_inference(df, agent, num_cells=3)

        out_df.to_csv(out_path, index=False)
        assert out_path.exists()

        # Re-read and verify
        reloaded = pd.read_csv(out_path)
        assert len(reloaded) == 5
        assert "recommended_cell" in reloaded.columns
        assert "agreement" in reloaded.columns

    def test_valid_mask_construction(self, tmp_path: Path) -> None:
        """build_state_vector produces correct valid mask."""
        script_path = ROOT / "scripts" / "prepare_drive_data.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("prepare_drive_data", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        row = pd.Series({
            "timestamp": "2026-05-12T10:00:00",
            "ue_id": 1,
            "lat": 27.7,
            "lon": 85.3,
            "speed_mps": 10.0,
            "serving_pci": 100,
            "neighbor_pcis": "101,102",
            "rsrp_dbm": -95.0,
            "rsrq_db": -9.0,
        })
        pci_to_idx = {100: 0, 101: 1, 102: 2}
        x, valid_mask = mod.build_state_vector(row, pci_to_idx, num_cells=3)

        # All 3 cells should be valid
        assert valid_mask.sum() == 3
        assert x.shape == (3, 11)
        # Serving cell should have serving_indicator = 1.0
        assert x[0, 5].item() == 1.0
        # Neighbors should have serving_indicator = 0.0
        assert x[1, 5].item() == 0.0
        assert x[2, 5].item() == 0.0

    def test_full_edge_index(self) -> None:
        """make_full_edge_index produces correct graph structure."""
        script_path = ROOT / "scripts" / "prepare_drive_data.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("prepare_drive_data", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        edge_index, edge_weight = mod.make_full_edge_index(3)
        # 3 cells fully connected (no self-loops) = 3*2 = 6 edges
        assert edge_index.shape == (2, 6)
        assert edge_weight.shape == (6,)
        # No self-loops
        for i in range(edge_index.shape[1]):
            assert edge_index[0, i].item() != edge_index[1, i].item()
