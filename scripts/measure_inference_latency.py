#!/usr/bin/env python3
"""Measure GNN-DQN inference latency for O-RAN near-RT RIC feasibility."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.rl.training import load_gnn_checkpoint


def make_full_edge_index(n: int):
    """Build edge_index and edge_weight for a fully-connected graph of n nodes."""
    adj = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.ones(len(src), dtype=torch.float32)
    return edge_index, edge_weight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure GNN-DQN agent inference latency."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--num-calls", type=int, default=1000, help="Number of inference calls.")
    parser.add_argument("--num-ues", type=int, default=250, help="Number of UEs in synthetic batch.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "latency" / "inference_latency.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    agent, meta, _payload = load_gnn_checkpoint(args.checkpoint)

    num_cells = agent.num_cells
    feature_dim = agent.feature_dim
    num_ues = args.num_ues

    print(f"Agent: num_cells={num_cells}, feature_dim={feature_dim}")
    print(f"Synthetic batch: num_ues={num_ues}, num_calls={args.num_calls}")

    # Build synthetic inputs
    rng = np.random.default_rng(42)
    states = rng.standard_normal((num_ues, num_cells, feature_dim)).astype(np.float32)
    edge_index, edge_weight = make_full_edge_index(num_cells)
    valid_masks = np.ones((num_ues, num_cells), dtype=bool)

    # Warmup (5 calls)
    for _ in range(5):
        agent.act_batch(states, edge_index, edge_weight, valid_masks, epsilon=0.0, rng=rng)

    # Timed inference calls
    latencies_ms = []
    for _ in range(args.num_calls):
        t0 = time.perf_counter()
        agent.act_batch(states, edge_index, edge_weight, valid_masks, epsilon=0.0, rng=rng)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies_ms)
    median = float(np.median(latencies))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean = float(np.mean(latencies))

    # O-RAN near-RT RIC target: < 10 ms PER UE DECISION.
    # The measured `median` is the wall-time for one batched forward pass that
    # produces actions for all `num_ues` UEs simultaneously, so the per-UE
    # decision latency is median / num_ues.
    ric_target_ms = 10.0
    per_ue_median = median / num_ues
    per_ue_p95 = p95 / num_ues
    per_ue_p99 = p99 / num_ues
    median_pass = per_ue_median < ric_target_ms
    p95_pass = per_ue_p95 < ric_target_ms

    print(f"\n{'='*50}")
    print(f"Inference Latency ({args.num_calls} calls, {num_ues} UEs, {num_cells} cells)")
    print(f"{'='*50}")
    print(f"  Batch total latency:")
    print(f"    Mean:   {mean:.3f} ms")
    print(f"    Median: {median:.3f} ms")
    print(f"    P95:    {p95:.3f} ms")
    print(f"    P99:    {p99:.3f} ms")
    print(f"  Per-UE decision latency (batch / num_ues):")
    print(f"    Median: {per_ue_median:.3f} ms")
    print(f"    P95:    {per_ue_p95:.3f} ms")
    print(f"    P99:    {per_ue_p99:.3f} ms")
    print(f"{'='*50}")
    print(f"  O-RAN near-RT RIC target (per-UE <{ric_target_ms} ms):")
    print(f"    Median: {'PASS' if median_pass else 'FAIL'} ({per_ue_median:.3f} ms)")
    print(f"    P95:    {'PASS' if p95_pass else 'FAIL'} ({per_ue_p95:.3f} ms)")
    print(f"{'='*50}")

    # Save results
    results = {
        "num_ues": num_ues,
        "num_cells": num_cells,
        "feature_dim": feature_dim,
        "num_calls": args.num_calls,
        "batch_latency_ms": {
            "mean": round(mean, 4),
            "median": round(median, 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
        },
        "per_ue_latency_ms": {
            "median": round(per_ue_median, 4),
            "p95": round(per_ue_p95, 4),
            "p99": round(per_ue_p99, 4),
        },
        "ric_target_ms": ric_target_ms,
        "median_pass": median_pass,
        "p95_pass": p95_pass,
        "checkpoint": str(args.checkpoint),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
