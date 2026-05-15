#!/usr/bin/env python3
"""Replay drive-test CSV through all handover policies and compare KPIs.

Reads CSVs produced by the Pokhara drive-test app (columns: timestamp, rsrp,
rsrq, cell_id, pci, latitude, longitude, speed_mps, ..., n1_rsrp, n1_rsrq,
..., handover, time_since_handover, prev_cell_id, ...).

For each timestep the script asks every policy which cell it would select,
then computes comparative KPIs: handover count, ping-pong rate, signal quality
maintained, and time-in-outage.

Usage:
    python scripts/replay_drivetest.py \
        --csv path/to/drivetest.csv \
        --checkpoint results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt \
        --out results/drive_replay/lakeside.csv
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

from handover_gnn_dqn.models.gnn_dqn import GnnDQNAgent
from handover_gnn_dqn.rl.training import load_gnn_checkpoint

RSRP_MIN = -140.0
RSRP_MAX = -44.0
RSRQ_MIN = -20.0
RSRQ_MAX = -3.0
MIN_RSRP_DBM = -112.0
PINGPONG_WINDOW = 6


def normalize_rsrp(v: float) -> float:
    return np.clip((v - RSRP_MIN) / (RSRP_MAX - RSRP_MIN), 0.0, 1.0)


def normalize_rsrq(v: float) -> float:
    return np.clip((v - RSRQ_MIN) / (RSRQ_MAX - RSRQ_MIN), 0.0, 1.0)


def load_drive_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"rsrp", "rsrq", "cell_id", "pci"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def extract_cells(df: pd.DataFrame) -> list[int]:
    """Get sorted list of all unique PCIs (serving + neighbors)."""
    pcis = set(df["pci"].dropna().astype(int).unique())
    for col in df.columns:
        if col.startswith("n") and col.endswith("_rsrp"):
            pass
    # Also check neighbor measurements — neighbors are visible via n1..n5 rsrp
    # but we don't have their PCI in this format. Use cell_id as PCI proxy.
    pcis.update(df["cell_id"].dropna().astype(int).unique())
    if "prev_cell_id" in df.columns:
        pcis.update(df["prev_cell_id"].dropna().astype(int).unique())
    pcis.discard(0)
    return sorted(pcis)


def build_neighbor_rsrp(row: pd.Series) -> list[tuple[float, float]]:
    """Extract (rsrp, rsrq) for neighbors n1..n5 from row."""
    neighbors = []
    for i in range(1, 6):
        col_rsrp = f"n{i}_rsrp"
        col_rsrq = f"n{i}_rsrq"
        if col_rsrp in row.index and pd.notna(row[col_rsrp]):
            rsrp = float(row[col_rsrp])
            rsrq = float(row[col_rsrq]) if col_rsrq in row.index and pd.notna(row[col_rsrq]) else -12.0
            if rsrp < 0:  # valid RSRP
                neighbors.append((rsrp, rsrq))
    return neighbors


def build_state_from_row(
    row: pd.Series,
    prev_row: pd.Series | None,
    pci_to_idx: dict[int, int],
    num_cells: int,
) -> tuple[torch.Tensor, np.ndarray, int]:
    """Build (num_cells, 11) UE_ONLY state tensor from a drive-test row.

    Returns: (state_tensor, valid_mask, serving_idx)
    """
    features = np.zeros((num_cells, 11), dtype=np.float32)
    valid_mask = np.zeros(num_cells, dtype=bool)

    serving_pci = int(row["pci"])
    serving_idx = pci_to_idx.get(serving_pci, 0)
    rsrp = float(row["rsrp"])
    rsrq = float(row["rsrq"])
    speed = float(row.get("speed_mps", 0.0)) if pd.notna(row.get("speed_mps")) else 0.0

    # Trends
    rsrp_trend = float(row.get("rsrp_trend", 0.0)) if pd.notna(row.get("rsrp_trend")) else 0.0
    rsrq_trend = float(row.get("rsrq_trend", 0.0)) if pd.notna(row.get("rsrq_trend")) else 0.0
    time_since_ho = float(row.get("time_since_handover", 30.0)) if pd.notna(row.get("time_since_handover")) else 30.0
    prev_cell = int(row.get("prev_cell_id", serving_pci)) if pd.notna(row.get("prev_cell_id")) else serving_pci

    rsrp_n = normalize_rsrp(rsrp)
    rsrq_n = normalize_rsrq(rsrq)
    load_proxy = float(row.get("load_proxy_rsrq", 1.0 - rsrq_n)) if pd.notna(row.get("load_proxy_rsrq")) else (1.0 - rsrq_n)
    signal_usable = 1.0 if rsrp > MIN_RSRP_DBM and rsrq > -15.0 else 0.0
    speed_n = np.clip(speed / 40.0, 0.0, 1.0)
    time_ho_n = np.clip(time_since_ho / 60.0, 0.0, 1.0)

    # Serving cell
    features[serving_idx] = [
        rsrp_n, rsrq_n, load_proxy,
        np.clip(rsrp_trend / 10.0, -1.0, 1.0),
        np.clip(rsrq_trend / 5.0, -1.0, 1.0),
        1.0,  # is_serving
        signal_usable, speed_n, time_ho_n,
        0.0,  # prev_serving (set below if applicable)
        load_proxy,
    ]
    valid_mask[serving_idx] = True

    # Mark previous serving cell
    if prev_cell in pci_to_idx and prev_cell != serving_pci:
        prev_idx = pci_to_idx[prev_cell]
        features[prev_idx, 9] = 1.0  # was_previous_serving

    # Neighbors
    neighbors = build_neighbor_rsrp(row)
    # Assign neighbors to non-serving slots in order
    neighbor_slots = [i for i in range(num_cells) if i != serving_idx]
    for ni, (n_rsrp, n_rsrq) in enumerate(neighbors):
        if ni >= len(neighbor_slots):
            break
        idx = neighbor_slots[ni]
        n_rsrp_n = normalize_rsrp(n_rsrp)
        n_rsrq_n = normalize_rsrq(n_rsrq)
        n_load = 1.0 - n_rsrq_n
        n_usable = 1.0 if n_rsrp > MIN_RSRP_DBM and n_rsrq > -15.0 else 0.0
        features[idx] = [
            n_rsrp_n, n_rsrq_n, n_load,
            0.0, 0.0,
            0.0,  # not serving
            n_usable, speed_n, time_ho_n,
            1.0 if idx == pci_to_idx.get(prev_cell, -1) else 0.0,
            n_load,
        ]
        valid_mask[idx] = n_rsrp >= MIN_RSRP_DBM

    return torch.tensor(features, dtype=torch.float32), valid_mask, serving_idx


def make_full_edge_index(num_cells: int):
    adj = np.ones((num_cells, num_cells), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight


# --------------- Policy replay functions ---------------

def policy_no_handover(serving_idx: int, **kwargs) -> int:
    return serving_idx


def policy_strongest_rsrp(serving_idx: int, rsrp_all: np.ndarray, valid_mask: np.ndarray, hysteresis: float = 2.0, **kwargs) -> int:
    masked = rsrp_all.copy()
    masked[~valid_mask] = -999.0
    best = int(np.argmax(masked))
    if masked[best] > rsrp_all[serving_idx] + hysteresis:
        return best
    return serving_idx


class A3TTTState:
    def __init__(self, offset_db: float = 3.0, ttt: int = 3):
        self.offset_db = offset_db
        self.ttt = ttt
        self.candidate = -1
        self.counter = 0

    def select(self, serving_idx: int, rsrp_all: np.ndarray, valid_mask: np.ndarray) -> int:
        masked = rsrp_all.copy()
        masked[~valid_mask] = -999.0
        best = int(np.argmax(masked))
        if best != serving_idx and masked[best] > rsrp_all[serving_idx] + self.offset_db:
            if self.candidate == best:
                self.counter += 1
            else:
                self.candidate = best
                self.counter = 1
            if self.counter >= self.ttt:
                self.counter = 0
                self.candidate = -1
                return best
        else:
            self.candidate = -1
            self.counter = 0
        return serving_idx


def policy_gnn_dqn(agent: GnnDQNAgent, state: torch.Tensor, edge_index, edge_weight, valid_mask: np.ndarray, **kwargs) -> int:
    with torch.no_grad():
        return agent.act(state, edge_index, edge_weight, epsilon=0.0, valid_mask=valid_mask)


# --------------- KPI computation ---------------

def compute_kpis(decisions: list[int], rsrp_serving_history: list[float]) -> dict:
    """Compute KPIs from a sequence of cell selections."""
    handovers = 0
    pingpongs = 0
    history = []
    outage_steps = 0

    for i, cell in enumerate(decisions):
        if i > 0 and cell != decisions[i - 1]:
            handovers += 1
            # Check ping-pong: returning to a cell served within last PINGPONG_WINDOW steps
            recent = history[max(0, i - PINGPONG_WINDOW):i]
            if cell in recent:
                pingpongs += 1
        history.append(cell)
        if rsrp_serving_history[i] < MIN_RSRP_DBM:
            outage_steps += 1

    n = len(decisions)
    return {
        "handovers": handovers,
        "handover_rate_per_min": handovers / max(n / 60.0, 1.0),
        "pingpongs": pingpongs,
        "pingpong_ratio": pingpongs / max(handovers, 1),
        "outage_steps": outage_steps,
        "outage_pct": outage_steps / max(n, 1) * 100,
        "avg_rsrp": np.mean(rsrp_serving_history),
        "samples": n,
    }


def run_replay(df: pd.DataFrame, agent: GnnDQNAgent, num_cells: int) -> pd.DataFrame:
    """Run all policies on the drive-test data and return comparative KPIs."""
    pcis = extract_cells(df)
    # Pad to model capacity
    while len(pcis) < num_cells:
        pcis.append(pcis[-1] + 1000 + len(pcis))
    pci_to_idx = {pci: i for i, pci in enumerate(pcis[:num_cells])}

    edge_index, edge_weight = make_full_edge_index(num_cells)

    # Track decisions per policy
    decisions = {
        "no_handover": [],
        "strongest_rsrp": [],
        "a3_ttt": [],
        "gnn_dqn": [],
    }
    # Track what RSRP the UE would see under each policy's chosen cell
    rsrp_history = {k: [] for k in decisions}

    a3_state = A3TTTState()

    # We need per-cell RSRP at each step. We only have serving + neighbors.
    # Build a per-step RSRP vector from available data.
    current_serving = {k: None for k in decisions}

    for i in range(len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1] if i > 0 else None

        state, valid_mask, serving_idx = build_state_from_row(row, prev_row, pci_to_idx, num_cells)

        # Build RSRP vector for all cells (from what we can observe)
        rsrp_all = np.full(num_cells, -140.0)
        rsrp_all[serving_idx] = float(row["rsrp"])
        neighbors = build_neighbor_rsrp(row)
        neighbor_slots = [j for j in range(num_cells) if j != serving_idx]
        for ni, (n_rsrp, _) in enumerate(neighbors):
            if ni < len(neighbor_slots):
                rsrp_all[neighbor_slots[ni]] = n_rsrp

        # --- no_handover ---
        if current_serving["no_handover"] is None:
            current_serving["no_handover"] = serving_idx
        decisions["no_handover"].append(current_serving["no_handover"])
        rsrp_history["no_handover"].append(rsrp_all[current_serving["no_handover"]])

        # --- strongest_rsrp ---
        if current_serving["strongest_rsrp"] is None:
            current_serving["strongest_rsrp"] = serving_idx
        choice = policy_strongest_rsrp(current_serving["strongest_rsrp"], rsrp_all, valid_mask)
        current_serving["strongest_rsrp"] = choice
        decisions["strongest_rsrp"].append(choice)
        rsrp_history["strongest_rsrp"].append(rsrp_all[choice])

        # --- a3_ttt ---
        if current_serving["a3_ttt"] is None:
            current_serving["a3_ttt"] = serving_idx
        choice = a3_state.select(current_serving["a3_ttt"], rsrp_all, valid_mask)
        current_serving["a3_ttt"] = choice
        decisions["a3_ttt"].append(choice)
        rsrp_history["a3_ttt"].append(rsrp_all[choice])

        # --- gnn_dqn ---
        if current_serving["gnn_dqn"] is None:
            current_serving["gnn_dqn"] = serving_idx
        choice = policy_gnn_dqn(agent, state, edge_index, edge_weight, valid_mask)
        current_serving["gnn_dqn"] = choice
        decisions["gnn_dqn"].append(choice)
        rsrp_history["gnn_dqn"].append(rsrp_all[choice])

    # Compute KPIs
    results = []
    for name in decisions:
        kpis = compute_kpis(decisions[name], rsrp_history[name])
        kpis["policy"] = name
        results.append(kpis)

    return pd.DataFrame(results).set_index("policy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay drive-test CSV through all handover policies")
    parser.add_argument("--csv", type=Path, required=True, help="Drive-test CSV path")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="GNN-DQN checkpoint (default: multiscenario_ue)")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV for KPIs")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    # Default checkpoint
    if args.checkpoint is None:
        args.checkpoint = ROOT / "results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt"
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading model: {args.checkpoint.name}")
    agent, metadata, _ = load_gnn_checkpoint(args.checkpoint, strict_metadata=False)
    num_cells = int(metadata["max_cells"])
    print(f"Model: {num_cells} cells, {metadata['feature_dim']} features")

    print(f"Loading drive-test: {args.csv.name}")
    df = load_drive_csv(args.csv)
    print(f"  {len(df)} samples, {df['pci'].nunique()} unique PCIs")

    print("\nRunning policy replay...")
    kpi_df = run_replay(df, agent, num_cells)

    print("\n" + "=" * 70)
    print("  DRIVE-TEST POLICY COMPARISON")
    print("=" * 70)
    print(kpi_df.to_string())
    print("=" * 70)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        kpi_df.to_csv(args.out)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
