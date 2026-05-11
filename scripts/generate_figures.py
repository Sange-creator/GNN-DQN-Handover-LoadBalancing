#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DISPLAY_ORDER = [
    "no_handover",
    "random_valid",
    "strongest_rsrp",
    "a3_ttt",
    "load_aware",
    "gnn_dqn",
    "son_gnn_dqn",
]

METHOD_LABELS = {
    "no_handover": "No HO",
    "random_valid": "Random",
    "strongest_rsrp": "Strongest RSRP",
    "a3_ttt": "A3-TTT",
    "load_aware": "Load-Aware",
    "gnn_dqn": "GNN-DQN (raw)",
    "son_gnn_dqn": "SON-GNN-DQN",
}

SCENARIO_LABELS = {
    "dense_urban": "Dense Urban",
    "highway": "Highway",
    "suburban": "Suburban",
    "sparse_rural": "Sparse Rural",
    "overloaded_event": "Overloaded",
    "real_pokhara": "Pokhara",
    "pokhara_dense_peakhour": "Pokhara Peak",
    "kathmandu_real": "Kathmandu (25-cell)",
    "unknown_hex_grid": "Hex Grid",
    "coverage_hole": "Coverage Hole",
    "dharan_synthetic": "Dharan",
}

HIGHLIGHT_SCENARIOS = [
    "dense_urban",
    "highway",
    "overloaded_event",
    "kathmandu_real",
]


def load_eval_dir(eval_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    data = {}
    for csv_path in sorted(eval_dir.glob("*.csv")):
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        data[csv_path.stem] = rows
    return data


def get_val(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except (ValueError, TypeError):
        return 0.0


def method_row(rows: List[Dict[str, str]], method: str) -> Dict[str, str] | None:
    for r in rows:
        if r.get("method") == method:
            return r
    return None


def fig_throughput_comparison(data: Dict, out_dir: Path) -> None:
    scenarios = [s for s in HIGHLIGHT_SCENARIOS if s in data]
    methods = [m for m in DISPLAY_ORDER if m != "random_valid"]

    x = np.arange(len(scenarios))
    width = 0.12
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, method in enumerate(methods):
        vals, errs = [], []
        for scen in scenarios:
            row = method_row(data[scen], method)
            vals.append(get_val(row, "avg_ue_throughput_mbps") if row else 0)
            errs.append(get_val(row, "avg_ue_throughput_mbps_ci95") if row else 0)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=METHOD_LABELS.get(method, method),
               capsize=2, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Avg UE Throughput (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=9)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_comparison.png", dpi=200)
    plt.close(fig)


def fig_pingpong_comparison(data: Dict, out_dir: Path) -> None:
    scenarios = [s for s in HIGHLIGHT_SCENARIOS if s in data]
    methods = [m for m in DISPLAY_ORDER if m not in ("no_handover", "random_valid")]

    x = np.arange(len(scenarios))
    width = 0.15
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, method in enumerate(methods):
        vals = []
        for scen in scenarios:
            row = method_row(data[scen], method)
            vals.append(get_val(row, "pingpong_rate") * 100 if row else 0)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(method, method),
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Ping-Pong Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=9)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "pingpong_comparison.png", dpi=200)
    plt.close(fig)


def fig_ablation_son(data: Dict, out_dir: Path) -> None:
    scenarios = sorted(data.keys())
    gnn_vals, son_vals, a3_vals = [], [], []
    labels = []

    for scen in scenarios:
        gnn_row = method_row(data[scen], "gnn_dqn")
        son_row = method_row(data[scen], "son_gnn_dqn")
        a3_row = method_row(data[scen], "a3_ttt")
        if not (gnn_row and son_row and a3_row):
            continue
        gnn_vals.append(get_val(gnn_row, "avg_ue_throughput_mbps"))
        son_vals.append(get_val(son_row, "avg_ue_throughput_mbps"))
        a3_vals.append(get_val(a3_row, "avg_ue_throughput_mbps"))
        labels.append(SCENARIO_LABELS.get(scen, scen))

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, gnn_vals, width, label="GNN-DQN (raw)", color=sns.color_palette()[3])
    ax.bar(x, son_vals, width, label="SON-GNN-DQN (ours)", color=sns.color_palette()[2])
    ax.bar(x + width, a3_vals, width, label="A3-TTT (baseline)", color=sns.color_palette()[0])

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Avg UE Throughput (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_son_layer.png", dpi=200)
    plt.close(fig)


def fig_radar_multimetric(data: Dict, out_dir: Path) -> None:
    metrics = [
        ("avg_ue_throughput_mbps", "Throughput", True),
        ("jain_load_fairness", "Fairness", True),
        ("pingpong_rate", "Stability\n(1-pingpong)", False),
        ("handovers_per_1000_decisions", "Low HO Rate", False),
        ("outage_rate", "Coverage\n(1-outage)", False),
    ]
    methods_show = ["a3_ttt", "load_aware", "son_gnn_dqn"]

    all_scenarios = sorted(data.keys())
    method_avgs: Dict[str, List[float]] = {m: [] for m in methods_show}

    for metric_key, _, higher_better in metrics:
        for method in methods_show:
            vals = []
            for scen in all_scenarios:
                row = method_row(data[scen], method)
                if row:
                    vals.append(get_val(row, metric_key))
            avg = np.mean(vals) if vals else 0
            if not higher_better:
                avg = 1.0 - avg if metric_key == "pingpong_rate" or metric_key == "outage_rate" else max(0, 1000 - avg)
            method_avgs[method].append(avg)

    for method in methods_show:
        arr = np.array(method_avgs[method])
        max_val = arr.max()
        if max_val > 0:
            method_avgs[method] = (arr / max_val).tolist()

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    for method in methods_show:
        values = method_avgs[method] + method_avgs[method][:1]
        ax.plot(angles, values, linewidth=2, label=METHOD_LABELS.get(method, method))
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "radar_multimetric.png", dpi=200)
    plt.close(fig)


def fig_topology_generalization(data: Dict, out_dir: Path) -> None:
    train_scenarios = ["dense_urban", "highway", "suburban", "sparse_rural",
                       "overloaded_event", "real_pokhara", "pokhara_dense_peakhour"]
    test_scenarios = ["kathmandu_real", "unknown_hex_grid", "coverage_hole", "dharan_synthetic"]

    train_present = [s for s in train_scenarios if s in data]
    test_present = [s for s in test_scenarios if s in data]

    if not train_present or not test_present:
        return

    method = "son_gnn_dqn"
    a3 = "a3_ttt"

    def relative_perf(scenarios):
        ratios = []
        for scen in scenarios:
            son_row = method_row(data[scen], method)
            a3_row = method_row(data[scen], a3)
            if son_row and a3_row:
                a3_val = get_val(a3_row, "avg_ue_throughput_mbps")
                son_val = get_val(son_row, "avg_ue_throughput_mbps")
                if a3_val > 0:
                    ratios.append(son_val / a3_val)
        return ratios

    train_ratios = relative_perf(train_present)
    test_ratios = relative_perf(test_present)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    positions = [1, 2]
    bp = ax.boxplot([train_ratios, test_ratios], positions=positions, widths=0.5,
                    patch_artist=True, medianprops=dict(color="black"))
    colors = [sns.color_palette()[0], sns.color_palette()[2]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="A3-TTT baseline")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Training Scenarios\n(in-distribution)", "Test Scenarios\n(unseen topologies)"])
    ax.set_ylabel("SON-GNN-DQN / A3-TTT Throughput Ratio")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "topology_generalization.png", dpi=200)
    plt.close(fig)


def fig_son_cio_utilization(data: Dict, out_dir: Path) -> None:
    scenarios = sorted(data.keys())
    avg_cio, max_cio = [], []
    labels = []

    for scen in scenarios:
        row = method_row(data[scen], "son_gnn_dqn")
        if not row:
            continue
        avg_val = get_val(row, "son_avg_abs_cio_db")
        max_val = get_val(row, "son_max_abs_cio_db")
        if avg_val > 0 or max_val > 0:
            avg_cio.append(avg_val)
            max_cio.append(max_val)
            labels.append(SCENARIO_LABELS.get(scen, scen))

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width / 2, avg_cio, width, label="Avg |CIO| (dB)")
    ax.bar(x + width / 2, max_cio, width, label="Max |CIO| (dB)")
    ax.axhline(6.0, color="red", linestyle="--", linewidth=1, label="Safety bound (6 dB)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("CIO Magnitude (dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 7)
    fig.tight_layout()
    fig.savefig(out_dir / "son_cio_utilization.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures from evaluation CSVs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    eval_dirs = [p for p in args.run_dir.glob("eval_*") if p.is_dir()]
    if not eval_dirs:
        eval_dirs = [args.run_dir / "evaluation"]
    eval_dir = max(eval_dirs, key=lambda p: p.stat().st_mtime if p.exists() else 0)

    if not eval_dir.exists() or not list(eval_dir.glob("*.csv")):
        raise SystemExit(f"No evaluation CSVs found in {eval_dir}")

    out_dir = args.run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")

    data = load_eval_dir(eval_dir)
    print(f"Loaded {len(data)} scenarios from {eval_dir}")

    fig_throughput_comparison(data, out_dir)
    print("  [1/6] throughput_comparison.png")

    fig_pingpong_comparison(data, out_dir)
    print("  [2/6] pingpong_comparison.png")

    fig_ablation_son(data, out_dir)
    print("  [3/6] ablation_son_layer.png")

    fig_radar_multimetric(data, out_dir)
    print("  [4/6] radar_multimetric.png")

    fig_topology_generalization(data, out_dir)
    print("  [5/6] topology_generalization.png")

    fig_son_cio_utilization(data, out_dir)
    print("  [6/6] son_cio_utilization.png")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
