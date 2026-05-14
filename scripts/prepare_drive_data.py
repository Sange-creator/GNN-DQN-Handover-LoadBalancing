#!/usr/bin/env python3
"""Drive-test shadow inference.

Loads a trained GNN-DQN checkpoint and runs inference on drive-test CSV data
to compare model recommendations against actual handover behavior.

Usage:
    python scripts/prepare_drive_data.py \
        --drive-test data/drive_test.csv \
        --checkpoint results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt \
        --out results/drive_shadow/shadow_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.models.gnn_dqn import DQNConfig, GnnDQNAgent
from handover_gnn_dqn.rl.training import load_gnn_checkpoint

# ue_only feature order (11 features per cell):
# RSRP, RSRQ, load_proxy, rsrp_trend, rsrq_trend, serving_indicator,
# signal_usability, ue_speed, time_since_last_ho, prev_serving_indicator,
# rsrq_load_proxy
UE_ONLY_FEATURE_DIM = 11


def make_full_edge_index(num_cells: int):
    """Build fully-connected edge graph for num_cells cells (no self-loops)."""
    adj = np.ones((num_cells, num_cells), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight


def parse_neighbor_pcis(val) -> list[int]:
    """Parse neighbor_pcis column which may be comma-separated string or NaN."""
    if pd.isna(val):
        return []
    if isinstance(val, (int, float)):
        return [int(val)]
    return [int(x.strip()) for x in str(val).split(",") if x.strip()]


def build_state_vector(
    row: pd.Series,
    pci_to_idx: dict[int, int],
    num_cells: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Build a (num_cells, 11) feature tensor from one drive-test row.

    Returns the feature tensor and a boolean valid_mask of shape (num_cells,).
    """
    features = np.zeros((num_cells, UE_ONLY_FEATURE_DIM), dtype=np.float32)
    valid_mask = np.zeros(num_cells, dtype=bool)

    serving_pci = int(row["serving_pci"])
    neighbor_pcis = parse_neighbor_pcis(row["neighbor_pcis"])
    rsrp = float(row["rsrp_dbm"])
    rsrq = float(row["rsrq_db"])
    speed = float(row.get("speed_mps", 0.0))

    # Normalize RSRP to [0,1] range: typical range -140 to -44 dBm
    rsrp_norm = np.clip((rsrp + 140) / 96.0, 0.0, 1.0)
    # Normalize RSRQ to [0,1] range: typical range -20 to -3 dB
    rsrq_norm = np.clip((rsrq + 20) / 17.0, 0.0, 1.0)
    # Load proxy from RSRQ (lower RSRQ -> higher load)
    load_proxy = 1.0 - rsrq_norm
    # Signal usability: 1 if RSRP > -110 and RSRQ > -12
    signal_usability = 1.0 if rsrp > -110 and rsrq > -12 else 0.0
    # Speed normalized (assume max 40 m/s ~ 144 km/h)
    speed_norm = np.clip(speed / 40.0, 0.0, 1.0)

    # Set serving cell features
    if serving_pci in pci_to_idx:
        idx = pci_to_idx[serving_pci]
        features[idx] = [
            rsrp_norm,          # RSRP
            rsrq_norm,          # RSRQ
            load_proxy,         # load_proxy
            0.0,                # rsrp_trend (unknown from single sample)
            0.0,                # rsrq_trend (unknown from single sample)
            1.0,                # serving_indicator
            signal_usability,   # signal_usability
            speed_norm,         # ue_speed
            0.5,                # time_since_last_ho (default mid-range)
            0.0,                # prev_serving_indicator
            load_proxy,         # rsrq_load_proxy
        ]
        valid_mask[idx] = True

    # Set neighbor cell features (approximate: weaker signal assumed)
    for n_pci in neighbor_pcis:
        if n_pci in pci_to_idx:
            idx = pci_to_idx[n_pci]
            # Neighbors assumed ~6 dB weaker RSRP and ~3 dB weaker RSRQ
            n_rsrp_norm = np.clip(rsrp_norm - 6 / 96.0, 0.0, 1.0)
            n_rsrq_norm = np.clip(rsrq_norm - 3 / 17.0, 0.0, 1.0)
            n_load_proxy = 1.0 - n_rsrq_norm
            n_signal = 1.0 if (rsrp - 6) > -110 and (rsrq - 3) > -12 else 0.0
            features[idx] = [
                n_rsrp_norm,
                n_rsrq_norm,
                n_load_proxy,
                0.0,
                0.0,
                0.0,            # not serving
                n_signal,
                speed_norm,
                0.5,
                0.0,
                n_load_proxy,
            ]
            valid_mask[idx] = True

    x = torch.tensor(features, dtype=torch.float32)
    return x, valid_mask


def get_q_values(
    agent: GnnDQNAgent,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Get Q-values for all cells."""
    with torch.no_grad():
        agent.eval()
        q = agent.forward(x, edge_index, edge_weight).cpu().numpy()
    # Mask invalid cells
    q_masked = q.copy()
    q_masked[~valid_mask] = float("-inf")
    return q_masked


def run_shadow_inference(
    df: pd.DataFrame,
    agent: GnnDQNAgent,
    num_cells: int,
) -> pd.DataFrame:
    """Run shadow inference on all rows and produce output dataframe."""
    # Collect all unique PCIs
    all_pcis = set()
    for _, row in df.iterrows():
        all_pcis.add(int(row["serving_pci"]))
        all_pcis.update(parse_neighbor_pcis(row["neighbor_pcis"]))

    pci_list = sorted(all_pcis)
    if len(pci_list) > num_cells:
        # Truncate to model capacity
        pci_list = pci_list[:num_cells]

    pci_to_idx = {pci: i for i, pci in enumerate(pci_list)}
    idx_to_pci = {i: pci for pci, i in pci_to_idx.items()}

    edge_index, edge_weight = make_full_edge_index(num_cells)

    results = []
    # Look ahead for next handover target
    has_ho_event = "handover_event" in df.columns

    for i, (_, row) in enumerate(df.iterrows()):
        x, valid_mask = build_state_vector(row, pci_to_idx, num_cells)

        # Get Q-values and recommendation
        q_values = get_q_values(agent, x, edge_index, edge_weight, valid_mask)
        recommended_idx = int(np.argmax(q_values))
        recommended_pci = idx_to_pci.get(recommended_idx, -1)

        # Top-3 by Q-value
        valid_q = q_values.copy()
        top3_indices = np.argsort(valid_q)[-3:][::-1]
        top3_pcis = [idx_to_pci.get(int(idx), -1) for idx in top3_indices]

        serving_pci = int(row["serving_pci"])

        # Find the actual next handover target (next row with different serving)
        actual_target = -1
        if has_ho_event:
            # Look for explicit handover event
            for j in range(i + 1, min(i + 20, len(df))):
                future_row = df.iloc[j]
                if int(future_row["serving_pci"]) != serving_pci:
                    actual_target = int(future_row["serving_pci"])
                    break
        else:
            for j in range(i + 1, min(i + 20, len(df))):
                future_row = df.iloc[j]
                if int(future_row["serving_pci"]) != serving_pci:
                    actual_target = int(future_row["serving_pci"])
                    break

        # Agreement: does the model recommend what actually happened?
        agreement = 1 if (actual_target != -1 and recommended_pci == actual_target) else 0
        top3_agreement = 1 if (actual_target != -1 and actual_target in top3_pcis) else 0

        # Predicted handover useful: model recommends a cell different from serving
        predicted_ho_useful = 1 if recommended_pci != serving_pci else 0

        # Outage avoided: if signal is poor and model recommends a different cell
        rsrp = float(row["rsrp_dbm"])
        outage_avoided = 1 if (rsrp < -110 and predicted_ho_useful) else 0

        # Ping-pong risk: model recommends going back to prev serving cell
        # (approximated: if recommended == serving, which means stay -> no ping-pong)
        pingpong_risk = 0
        if i > 0:
            prev_serving = int(df.iloc[i - 1]["serving_pci"])
            if recommended_pci == prev_serving and prev_serving != serving_pci:
                pingpong_risk = 1

        results.append({
            "timestamp": row.get("timestamp", i),
            "ue_id": row.get("ue_id", 0),
            "recommended_cell": recommended_pci,
            "actual_cell": actual_target,
            "agreement": agreement,
            "top3_agreement": top3_agreement,
            "predicted_handover_useful": predicted_ho_useful,
            "outage_avoided": outage_avoided,
            "pingpong_risk": pingpong_risk,
        })

    return pd.DataFrame(results)


def print_summary(out_df: pd.DataFrame) -> None:
    """Print summary statistics."""
    total = len(out_df)
    has_actual = out_df[out_df["actual_cell"] != -1]
    n_with_actual = len(has_actual)

    print(f"\n{'='*60}")
    print("Drive-Test Shadow Inference Summary")
    print(f"{'='*60}")
    print(f"Total samples:              {total}")
    print(f"Samples with actual HO:     {n_with_actual}")

    if n_with_actual > 0:
        top1_rate = has_actual["agreement"].mean() * 100
        top3_rate = has_actual["top3_agreement"].mean() * 100
        print(f"Top-1 agreement rate:       {top1_rate:.1f}%")
        print(f"Top-3 agreement rate:       {top3_rate:.1f}%")

    outage_candidates = out_df[out_df["outage_avoided"] == 1]
    if len(outage_candidates) > 0:
        outage_rate = len(outage_candidates) / total * 100
        print(f"Outage-avoidance rate:      {outage_rate:.1f}% ({len(outage_candidates)}/{total})")
    else:
        print("Outage-avoidance rate:      N/A (no low-signal samples)")

    pingpong_rate = out_df["pingpong_risk"].mean() * 100
    print(f"Ping-pong risk rate:        {pingpong_rate:.1f}%")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive-test shadow inference: compare GNN-DQN recommendations to actual handovers."
    )
    parser.add_argument("--drive-test", type=Path, required=True, help="Path to drive-test CSV")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained GNN-DQN checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    if not args.drive_test.exists():
        print(f"ERROR: Drive-test CSV not found: {args.drive_test}")
        sys.exit(1)
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    agent, metadata, _ = load_gnn_checkpoint(
        args.checkpoint,
        strict_metadata=False,
    )
    num_cells = agent.num_cells
    feature_dim = agent.feature_dim
    print(f"Model: {num_cells} cells, {feature_dim} features/cell")

    # Load drive-test data
    print(f"Loading drive-test data: {args.drive_test}")
    df = pd.read_csv(args.drive_test)
    print(f"Loaded {len(df)} samples")

    # Run shadow inference
    out_df = run_shadow_inference(df, agent, num_cells)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote shadow inference results: {args.out}")

    # Print summary
    print_summary(out_df)


if __name__ == "__main__":
    main()
