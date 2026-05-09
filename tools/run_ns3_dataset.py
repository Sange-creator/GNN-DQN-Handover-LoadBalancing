from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NS3_ROOT = Path("/Users/saangetamang/ns-allinone-3.40/ns-3.40")
SCRATCH_TARGET = "scratch_gnn-dqn-handover-data"
SCRATCH_BINARY = "build/scratch/ns3.40-gnn-dqn-handover-data-default"


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def summarize(samples_path: Path, handovers_path: Path, summary_path: Path) -> None:
    rows = []
    with samples_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No sample rows found in {samples_path}")

    throughputs = [float(r["dl_throughput_mbps"]) for r in rows]
    load_columns = [c for c in rows[0] if c.startswith("load_cell_")]
    max_loads = [max(float(r[c]) for c in load_columns) for r in rows]

    handover_events = 0
    if handovers_path.exists():
        with handovers_path.open(newline="") as f:
            reader = csv.DictReader(f)
            handover_events = sum(1 for r in reader if r.get("event") == "end_ok")

    ue_ids = {r["ue_id"] for r in rows}
    times = {r["time_s"] for r in rows}
    decisions = max(len(ue_ids) * len(times), 1)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rows",
                "num_ues",
                "num_samples",
                "avg_dl_throughput_mbps",
                "p5_dl_throughput_mbps",
                "avg_max_cell_load",
                "handover_events",
                "handovers_per_1000_ue_samples",
            ],
        )
        writer.writeheader()
        sorted_t = sorted(throughputs)
        p5_index = int(0.05 * (len(sorted_t) - 1))
        writer.writerow(
            {
                "rows": len(rows),
                "num_ues": len(ue_ids),
                "num_samples": len(times),
                "avg_dl_throughput_mbps": sum(throughputs) / len(throughputs),
                "p5_dl_throughput_mbps": sorted_t[p5_index],
                "avg_max_cell_load": sum(max_loads) / len(max_loads),
                "handover_events": handover_events,
                "handovers_per_1000_ue_samples": 1000.0 * handover_events / decisions,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run the ns-3 LTE handover CSV exporter.")
    parser.add_argument("--ns3-root", type=Path, default=DEFAULT_NS3_ROOT)
    parser.add_argument("--sim-time", type=float, default=30.0)
    parser.add_argument("--sample-period", type=float, default=1.0)
    parser.add_argument("--num-enbs", type=int, default=7)
    parser.add_argument("--num-ues", type=int, default=25)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "ns3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns3_root = args.ns3_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = out_dir / f"samples_run_{args.run}.csv"
    handovers_path = out_dir / f"handovers_run_{args.run}.csv"
    summary_path = out_dir / f"summary_run_{args.run}.csv"

    if not args.skip_build:
        run(["ninja", SCRATCH_TARGET], cwd=ns3_root)

    binary = ns3_root / SCRATCH_BINARY
    if not binary.exists():
        raise FileNotFoundError(f"Expected ns-3 scratch binary at {binary}")

    run(
        [
            str(binary),
            f"--numEnbs={args.num_enbs}",
            f"--numUes={args.num_ues}",
            f"--simTime={args.sim_time}",
            f"--samplePeriod={args.sample_period}",
            f"--run={args.run}",
            f"--samplePath={samples_path}",
            f"--handoverPath={handovers_path}",
        ],
        cwd=ns3_root,
    )
    summarize(samples_path, handovers_path, summary_path)
    print(f"Wrote {samples_path}")
    print(f"Wrote {handovers_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

