#!/usr/bin/env python3
"""Compute deployment-readiness composite score from per-scenario eval CSVs.

The thesis argues that on a single metric (avg throughput), the demand-capped
simulator produces near-identical results across all reasonable handover
policies. To rank policies by what actually matters for deployment, we need a
multi-objective score that combines:

    - throughput (avg + tail)
    - load balance
    - signaling cost (handovers)
    - stability (pingpongs)

This script:
1. Reads every `<scenario>.csv` in an eval directory.
2. Normalizes each metric within each scenario so all methods are scored 0..1
   relative to the best and worst on that scenario.
3. Computes a weighted composite per (method, scenario).
4. Outputs:
   - `composite_per_scenario.csv` — composite score table
   - `composite_ranking.csv` — methods ranked by mean composite across scenarios
   - prints a Markdown summary suitable to paste into the thesis

Usage:
    PYTHONPATH=src python3 scripts/compute_composite_score.py \\
        --eval-dir results/runs/multiscenario_ue_defense/eval_normal \\
        --out-dir results/runs/multiscenario_ue_defense/eval_normal
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# Composite weights.  Higher weight = matters more in the ranking.
# Reasoning per weight:
#   - avg throughput 0.40:  primary user-visible KPI, but largely demand-capped
#     in low-congestion scenarios, so we don't let it dominate.
#   - p5 throughput  0.20:  tail fairness — protects worst-served UEs.
#   - load std       0.15:  load balancing objective (signed: lower=better).
#   - pingpong rate  0.15:  stability (lower=better).
#   - HO per 1000    0.10:  signaling cost (lower=better).
WEIGHTS = {
    "avg_ue_throughput_mbps": 0.40,
    "p5_ue_throughput_mbps": 0.20,
    "load_std": 0.15,
    "pingpong_rate": 0.15,
    "handovers_per_1000_decisions": 0.10,
}

# Whether higher is better for each metric (True) or lower is better (False).
HIGHER_IS_BETTER = {
    "avg_ue_throughput_mbps": True,
    "p5_ue_throughput_mbps": True,
    "load_std": False,
    "pingpong_rate": False,
    "handovers_per_1000_decisions": False,
}


def _read_scenario_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse one eval CSV into {method_name: {metric_name: value}}."""
    rows: Dict[str, Dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            rows[method] = {
                k: float(v) for k, v in row.items()
                if k != "method" and v not in ("", None)
            }
    return rows


def _normalize_within_scenario(
    values_by_method: Dict[str, float], higher_is_better: bool
) -> Dict[str, float]:
    """Map values to [0, 1] where 1 is the best method on this metric."""
    vals = list(values_by_method.values())
    if not vals:
        return {m: 0.0 for m in values_by_method}
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    if spread == 0:
        # All methods tied on this metric — give everyone a flat score.
        return {m: 0.5 for m in values_by_method}
    normalized: Dict[str, float] = {}
    for method, v in values_by_method.items():
        n = (v - lo) / spread  # 0..1, higher v = higher n
        if not higher_is_better:
            n = 1.0 - n
        normalized[method] = n
    return normalized


def compute_composite_for_scenario(
    scenario_rows: Dict[str, Dict[str, float]]
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Return (composite_per_method, normalized_metric_per_method).

    composite_per_method: {method: composite_score in [0, 1]}.
    normalized_metric_per_method: {method: {metric: normalized_value}}.
    """
    methods = list(scenario_rows.keys())
    normalized_by_metric: Dict[str, Dict[str, float]] = {}
    for metric, weight in WEIGHTS.items():
        values_by_method = {
            m: scenario_rows[m][metric]
            for m in methods
            if metric in scenario_rows[m]
        }
        if not values_by_method:
            continue
        normalized_by_metric[metric] = _normalize_within_scenario(
            values_by_method, higher_is_better=HIGHER_IS_BETTER[metric]
        )

    composite: Dict[str, float] = {m: 0.0 for m in methods}
    for metric, weight in WEIGHTS.items():
        for method, n in normalized_by_metric.get(metric, {}).items():
            composite[method] += weight * n

    # Per-method normalized values to write out for diagnostics.
    normalized_per_method: Dict[str, Dict[str, float]] = {m: {} for m in methods}
    for metric, mvals in normalized_by_metric.items():
        for method, n in mvals.items():
            normalized_per_method[method][metric] = n

    return composite, normalized_per_method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Directory of per-scenario eval CSVs.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: --eval-dir).")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Method names to exclude (e.g. random_valid).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or args.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_csvs = sorted(p for p in args.eval_dir.glob("*.csv") if p.is_file())
    # Skip any aggregate / composite files written by earlier runs.
    scenario_csvs = [p for p in scenario_csvs if not p.stem.startswith("composite")
                     and not p.stem.startswith("summary")]
    if not scenario_csvs:
        raise SystemExit(f"No scenario CSVs found in {args.eval_dir}")

    # Composite per (method, scenario).
    per_scenario_composite: Dict[str, Dict[str, float]] = defaultdict(dict)
    # Aggregate normalized values per (method, metric, scenario) for diagnostics.
    normalized_dump: List[dict] = []

    for path in scenario_csvs:
        scenario_name = path.stem
        rows = _read_scenario_csv(path)
        rows = {m: r for m, r in rows.items() if m not in args.exclude}
        if not rows:
            continue
        composite, normalized = compute_composite_for_scenario(rows)
        for method, score in composite.items():
            per_scenario_composite[method][scenario_name] = score
        for method, norm_metrics in normalized.items():
            normalized_dump.append({
                "scenario": scenario_name,
                "method": method,
                **{f"norm_{k}": round(v, 4) for k, v in norm_metrics.items()},
            })

    methods = sorted(per_scenario_composite.keys())
    scenarios = sorted({s for d in per_scenario_composite.values() for s in d})

    # 1. Per-scenario composite CSV.
    composite_csv_path = out_dir / "composite_per_scenario.csv"
    with composite_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + scenarios + ["mean", "wins"])
        ranking: List[Tuple[str, float, int]] = []
        for method in methods:
            row_scores = [per_scenario_composite[method].get(s) for s in scenarios]
            valid_scores = [s for s in row_scores if s is not None]
            mean = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            wins = 0
            for scenario in scenarios:
                vals = {
                    m: per_scenario_composite[m].get(scenario)
                    for m in methods
                    if per_scenario_composite[m].get(scenario) is not None
                }
                if not vals:
                    continue
                best = max(vals.values())
                # Allow ties within 0.5% of the leader.
                if per_scenario_composite[method].get(scenario) is not None:
                    score = per_scenario_composite[method][scenario]
                    if best > 0 and score >= best - 0.005:
                        wins += 1
            ranking.append((method, mean, wins))
            row = [method] + [
                f"{s:.4f}" if s is not None else "" for s in row_scores
            ] + [f"{mean:.4f}", str(wins)]
            writer.writerow(row)

    # 2. Ranking CSV.
    ranking_csv_path = out_dir / "composite_ranking.csv"
    ranking.sort(key=lambda r: (-r[1], -r[2]))
    with ranking_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "method", "mean_composite", "scenarios_won"])
        for i, (method, mean, wins) in enumerate(ranking, 1):
            writer.writerow([i, method, f"{mean:.4f}", wins])

    # 3. Normalized metric values JSON (diagnostics).
    norm_json_path = out_dir / "composite_normalized_diagnostics.json"
    with norm_json_path.open("w") as f:
        json.dump({
            "weights": WEIGHTS,
            "higher_is_better": HIGHER_IS_BETTER,
            "rows": normalized_dump,
        }, f, indent=2)

    # 4. Print Markdown ranking summary (paste into thesis).
    if not args.quiet:
        print()
        print("# Deployment-Readiness Composite Score")
        print()
        print("**Weights:** "
              + ", ".join(f"`{k}`={v:.2f}" for k, v in WEIGHTS.items()))
        print()
        print(f"**Scenarios evaluated:** {len(scenarios)}")
        print(f"**Methods compared:** {len(methods)}")
        if args.exclude:
            print(f"**Methods excluded:** {', '.join(args.exclude)}")
        print()
        print("## Ranking (higher = better)")
        print()
        print("| Rank | Method | Mean composite | Scenarios won (incl. ties) |")
        print("|---:|---|---:|---:|")
        for i, (method, mean, wins) in enumerate(ranking, 1):
            print(f"| {i} | `{method}` | {mean:.4f} | {wins}/{len(scenarios)} |")
        print()
        print(f"Output written:")
        print(f"  - {composite_csv_path}")
        print(f"  - {ranking_csv_path}")
        print(f"  - {norm_json_path}")


if __name__ == "__main__":
    main()
