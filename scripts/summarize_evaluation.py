#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REQUIRED_METHODS = {
    "no_handover",
    "random_valid",
    "strongest_rsrp",
    "a3_ttt",
    "load_aware",
    "gnn_dqn",
    "son_gnn_dqn",
}


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["method"]: row for row in rows}


def scenario_status(path: Path, *, a3_tolerance: float, outage_slack: float) -> tuple[list[str], str]:
    rows = read_rows(path)
    failures: list[str] = []
    missing = sorted(REQUIRED_METHODS - set(rows))
    if missing:
        failures.append(f"missing methods: {', '.join(missing)}")
        return failures, f"| {path.stem} | missing | - | - | - | - | - |"

    son = rows["son_gnn_dqn"]
    random_valid = rows["random_valid"]
    a3 = rows["a3_ttt"]
    no_handover = rows["no_handover"]

    son_avg = _float(son, "avg_ue_throughput_mbps")
    random_avg = _float(random_valid, "avg_ue_throughput_mbps")
    son_p5 = _float(son, "p5_ue_throughput_mbps")
    random_p5 = _float(random_valid, "p5_ue_throughput_mbps")
    a3_avg = _float(a3, "avg_ue_throughput_mbps")
    son_outage = _float(son, "outage_rate")
    base_outage = min(_float(a3, "outage_rate"), _float(no_handover, "outage_rate"))
    son_pingpong = _float(son, "pingpong_rate")
    a3_pingpong = _float(a3, "pingpong_rate")

    # In overloaded scenarios, random spreading can achieve near-optimal load
    # distribution by accident (uniform spread = perfect balance). Only flag
    # if son is catastrophically worse — the real test is vs a3_ttt.
    if son_avg <= random_avg * 0.80:
        failures.append("son avg far below random_valid")
    if son_p5 <= random_p5 * 0.60:
        failures.append("son p5 far below random_valid")
    if son_avg < a3_avg * a3_tolerance:
        failures.append(f"son avg below {a3_tolerance:.0%} of a3")
    if son_outage > base_outage + outage_slack:
        failures.append("son outage exceeds conservative baseline")
    if a3_pingpong > 1e-9 and son_pingpong > a3_pingpong * 1.5:
        failures.append("son ping-pong > 150% of a3")

    status = "PASS" if not failures else "FAIL"
    line = (
        f"| {path.stem} | {status} | {son_avg:.3f} | {random_avg:.3f} | "
        f"{a3_avg:.3f} | {son_p5:.3f} | {son_pingpong:.3f} |"
    )
    return failures, line


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evaluation CSVs against acceptance gates.")
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--a3-tolerance", type=float, default=0.98)
    parser.add_argument("--outage-slack", type=float, default=0.002)
    parser.add_argument("--no-fail", action="store_true", help="Print failures but exit 0.")
    args = parser.parse_args()

    csvs = sorted(args.eval_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV files found in {args.eval_dir}")

    print("| Scenario | Gate | SON avg | Random avg | A3 avg | SON P5 | SON ping-pong |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    all_failures: dict[str, list[str]] = {}
    for path in csvs:
        failures, line = scenario_status(
            path,
            a3_tolerance=args.a3_tolerance,
            outage_slack=args.outage_slack,
        )
        print(line)
        if failures:
            all_failures[path.stem] = failures

    if all_failures:
        print("\nFailures:")
        for scenario, failures in all_failures.items():
            print(f"- {scenario}: {'; '.join(failures)}")
        if not args.no_fail:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
