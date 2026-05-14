"""Tests for ns-3 comparison script output schema."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _make_fake_ns3_csv(ns3_dir: Path, num_rows: int = 10) -> None:
    """Create a minimal fake ns-3 samples CSV."""
    ns3_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "time_s,ue_id,imsi,serving_cell,x_m,y_m,z_m,dl_throughput_mbps,handover_count,"
        "rsrp_dbm_cell_1,rsrp_dbm_cell_2,rsrp_dbm_cell_3,rsrp_dbm_cell_4,"
        "rsrp_dbm_cell_5,rsrp_dbm_cell_6,rsrp_dbm_cell_7,"
        "load_cell_1,load_cell_2,load_cell_3,load_cell_4,load_cell_5,load_cell_6,load_cell_7,"
        "ue_count_cell_1,ue_count_cell_2,ue_count_cell_3,ue_count_cell_4,"
        "ue_count_cell_5,ue_count_cell_6,ue_count_cell_7"
    )
    lines = [header]
    import random
    rng = random.Random(42)
    for i in range(num_rows):
        row = (
            f"{float(i)},{i % 5},{i + 1},{rng.randint(1, 7)},"
            f"{rng.uniform(-500, 500):.1f},{rng.uniform(-500, 500):.1f},0.0,"
            f"{rng.uniform(0.1, 2.0):.3f},{rng.randint(0, 3)},"
            + ",".join(f"{rng.uniform(-100, -50):.1f}" for _ in range(7)) + ","
            + ",".join(f"{rng.uniform(0.0, 0.5):.3f}" for _ in range(7)) + ","
            + ",".join(str(rng.randint(1, 10)) for _ in range(7))
        )
        lines.append(row)
    (ns3_dir / "samples_run_1.csv").write_text("\n".join(lines))


def test_ns3_report_schema():
    """Run compare_ns3 on fake data and validate output JSON schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ns3_dir = Path(tmpdir) / "ns3"
        _make_fake_ns3_csv(ns3_dir, num_rows=20)
        out_path = Path(tmpdir) / "report.json"

        result = subprocess.run(
            [
                sys.executable, str(ROOT / "scripts" / "compare_ns3.py"),
                "--ns3-dir", str(ns3_dir),
                "--out", str(out_path),
                "--sim-seeds", "1",
                "--sim-steps", "5",
            ],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
            timeout=120,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}\n{result.stdout}"
        assert out_path.exists()

        report = json.loads(out_path.read_text())
        assert isinstance(report, list)
        assert len(report) >= 1

        required_keys = {"metric", "ks_stat", "p_value", "n_sim", "n_ns3", "significant_divergence"}
        for entry in report:
            assert required_keys.issubset(entry.keys()), f"Missing keys in {entry}"
            assert 0.0 <= entry["ks_stat"] <= 1.0
            assert 0.0 <= entry["p_value"] <= 1.0
            assert entry["n_sim"] > 0
            assert entry["n_ns3"] > 0
            assert isinstance(entry["significant_divergence"], bool)
