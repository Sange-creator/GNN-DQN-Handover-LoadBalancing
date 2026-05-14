#!/usr/bin/env python3
"""Compare simulator distributions against ns-3 reference data via KS tests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.env import CellularNetworkEnv, LTEConfig


def load_ns3_data(ns3_dir: Path) -> dict[str, np.ndarray]:
    """Load and concatenate all ns-3 sample CSVs into metric arrays."""
    import csv

    all_throughputs: list[float] = []
    all_rsrp: list[float] = []
    all_loads: list[float] = []
    all_handover_counts: list[float] = []

    for f in sorted(ns3_dir.glob("samples_run_*.csv")):
        with f.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                all_throughputs.append(float(row["dl_throughput_mbps"]))
                all_handover_counts.append(float(row["handover_count"]))
                for i in range(1, 8):
                    rsrp_key = f"rsrp_dbm_cell_{i}"
                    load_key = f"load_cell_{i}"
                    if rsrp_key in row:
                        all_rsrp.append(float(row[rsrp_key]))
                    if load_key in row:
                        all_loads.append(float(row[load_key]))

    return {
        "throughput_mbps": np.array(all_throughputs),
        "rsrp_dbm": np.array(all_rsrp),
        "cell_load": np.array(all_loads),
        "handover_count": np.array(all_handover_counts),
    }


def run_simulator(num_cells: int, num_ues: int, area_m: float, steps: int, seeds: list[int]) -> dict[str, np.ndarray]:
    """Run simulator to collect comparable distributions."""
    all_throughputs: list[float] = []
    all_rsrp: list[float] = []
    all_loads: list[float] = []

    for seed in seeds:
        cfg = LTEConfig(
            num_cells=num_cells,
            num_ues=num_ues,
            area_m=area_m,
            feature_mode="ue_only",
            prb_available=False,
        )
        env = CellularNetworkEnv(cfg)
        env.reset(seed)

        for _ in range(steps):
            env.advance_mobility()
            all_rsrp.extend(env._rsrp.flatten().tolist())
            all_loads.extend(env.cell_loads().tolist())
            all_throughputs.extend(env.user_throughputs().tolist())

    return {
        "throughput_mbps": np.array(all_throughputs),
        "rsrp_dbm": np.array(all_rsrp),
        "cell_load": np.array(all_loads),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare simulator vs ns-3 distributions (KS test).")
    parser.add_argument("--ns3-dir", type=Path, default=ROOT / "data" / "raw" / "ns3")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "calibration" / "ns3_ks_report.json")
    parser.add_argument("--sim-seeds", type=int, default=3)
    parser.add_argument("--sim-steps", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading ns-3 data from: {args.ns3_dir}")
    ns3_data = load_ns3_data(args.ns3_dir)
    print(f"  ns-3 samples: throughput={len(ns3_data['throughput_mbps'])}, "
          f"rsrp={len(ns3_data['rsrp_dbm'])}, load={len(ns3_data['cell_load'])}")

    print(f"Running simulator: 7 cells, 25 UEs, {args.sim_steps} steps x {args.sim_seeds} seeds")
    seeds = [42 + i * 1000 for i in range(args.sim_seeds)]
    sim_data = run_simulator(num_cells=7, num_ues=25, area_m=2000.0, steps=args.sim_steps, seeds=seeds)
    print(f"  sim samples: throughput={len(sim_data['throughput_mbps'])}, "
          f"rsrp={len(sim_data['rsrp_dbm'])}, load={len(sim_data['cell_load'])}")

    metrics_to_compare = ["throughput_mbps", "rsrp_dbm", "cell_load"]
    results: list[dict] = []

    print(f"\n{'Metric':<20} {'KS stat':<10} {'p-value':<12} {'n_sim':<8} {'n_ns3':<8} {'Flag'}")
    print("-" * 70)

    for metric in metrics_to_compare:
        sim_arr = sim_data[metric]
        ns3_arr = ns3_data[metric]
        if len(sim_arr) == 0 or len(ns3_arr) == 0:
            continue
        ks_stat, p_value = stats.ks_2samp(sim_arr, ns3_arr)
        flag = "CONCERN" if p_value < 0.01 else ""
        print(f"{metric:<20} {ks_stat:<10.4f} {p_value:<12.6f} {len(sim_arr):<8} {len(ns3_arr):<8} {flag}")
        results.append({
            "metric": metric,
            "ks_stat": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "n_sim": int(len(sim_arr)),
            "n_ns3": int(len(ns3_arr)),
            "significant_divergence": bool(p_value < 0.01),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.out}")

    concerns = [r for r in results if r["significant_divergence"]]
    if concerns:
        print(f"\nWARNING: {len(concerns)} metric(s) show significant divergence (p < 0.01):")
        for c in concerns:
            print(f"  - {c['metric']}: KS={c['ks_stat']:.4f}, p={c['p_value']:.6f}")
    else:
        print("\nAll metrics pass KS test (no significant divergence from ns-3).")


if __name__ == "__main__":
    main()
