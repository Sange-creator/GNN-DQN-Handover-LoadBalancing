#!/usr/bin/env python3
"""
Google Colab Fine-Tuning — GNN-DQN Handover Optimization
=========================================================

3 Colab cells. Run this file locally to print copy-paste code:
  python3 scripts/colab_finetune.py

Prerequisites
-------------
1. Zip your repo: zip -r gnn-dqn-handover-loadbalancing.zip gnn-dqn-handover-loadbalancing/
   Must include: src/ configs/ scripts/ data/raw/
2. Include a resume checkpoint under:
     results/runs/ue_final_30h/checkpoints/resume/resume_ep0350.pt
   (or any resume_epXXXX.pt — the script will auto-detect the latest)
3. Set Colab runtime to GPU (A100 preferred, V100/T4 also work)

Key Changes vs Original Training
---------------------------------
- SON config tightened: CIO ±3dB (was ±8), 0.5dB steps (was 1.0), 8 updates/cycle (was 24)
- Focused on 5 core scenarios (was 11): dense_urban, highway_fast, suburban, overloaded_event, real_pokhara
- Lower LR (0.00015 vs 0.00035) and epsilon (0.15→0.008 vs 0.45→0.01)
- Removed sparse_rural, highway(slow), pokhara_peakhour, stress variants from training
"""

from __future__ import annotations

import textwrap


# =============================================================================
# COLAB CELL 1 — Install deps + upload and unzip project
# =============================================================================
CELL1 = textwrap.dedent(r'''
# ============================================================
# CELL 1: Install dependencies + upload project zip
# ============================================================
# Runtime → Change runtime type → GPU (A100 preferred)
%%capture
!pip install -q torch torch-geometric numpy pandas tqdm

import os, shutil, subprocess, sys
from pathlib import Path

from google.colab import files

print("Upload gnn-dqn-handover-loadbalancing.zip")
print("Must include: src/ configs/ scripts/ data/raw/ results/runs/ue_final_30h/checkpoints/resume/")
uploaded = files.upload()
zip_name = next(iter(uploaded.keys()))
ROOT = Path("/content/gnn-dqn-handover-loadbalancing")
if ROOT.exists():
    shutil.rmtree(ROOT)
subprocess.run(["unzip", "-q", "-o", f"/content/{zip_name}", "-d", "/content"], check=True)
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT / "src")

# Verify structure
for p in ["src/handover_gnn_dqn", "configs/experiments", "scripts/train.py", "scripts/evaluate.py"]:
    assert (ROOT / p).exists(), f"Missing: {p}"
print("✓ Project extracted. PYTHONPATH =", os.environ["PYTHONPATH"])

# Find latest resume checkpoint
resume_dir = ROOT / "results/runs/ue_final_30h/checkpoints/resume"
if resume_dir.exists():
    resumes = sorted(resume_dir.glob("resume_ep*.pt"))
    if resumes:
        print(f"✓ Found {len(resumes)} resume checkpoint(s). Latest: {resumes[-1].name}")
    else:
        print("⚠ No resume_ep*.pt files found in", resume_dir)
else:
    print("⚠ Resume dir not found:", resume_dir)
print("\nReady. Run Cell 2 next.")
''').strip()


# =============================================================================
# COLAB CELL 2 — Finetune via official train.py + resume
# =============================================================================
CELL2 = textwrap.dedent(r'''
# ============================================================
# CELL 2: Fine-tune model (~6-10h on A100, ~12-16h on T4)
# ============================================================
import os, subprocess, sys, glob
from pathlib import Path

ROOT = Path("/content/gnn-dqn-handover-loadbalancing")
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT / "src")

# Auto-detect latest resume checkpoint
resume_dir = ROOT / "results/runs/ue_final_30h/checkpoints/resume"
resumes = sorted(resume_dir.glob("resume_ep*.pt")) if resume_dir.exists() else []

if resumes:
    RESUME = resumes[-1]
    print(f"Using resume checkpoint: {RESUME.name}")
else:
    # Fallback: upload manually
    from google.colab import files
    print("No resume checkpoint found. Upload resume_epXXXX.pt:")
    up = files.upload()
    name = next(iter(up.keys()))
    RESUME = ROOT / "colab_uploaded_resume.pt"
    RESUME.write_bytes(up[name])

assert RESUME.is_file(), f"Missing resume checkpoint: {RESUME}"

# Verify config exists
CONFIG = ROOT / "configs/experiments/colab_finetune_ue.json"
assert CONFIG.is_file(), f"Missing config: {CONFIG}"

# Print key config info
import json
with open(CONFIG) as f:
    cfg = json.load(f)
print(f"Config: {CONFIG.name}")
print(f"  Episodes: {cfg['episodes']}")
print(f"  Train scenarios: {cfg['train_scenarios']}")
print(f"  SON CIO range: [{cfg['son_config']['cio_min_db']}, {cfg['son_config']['cio_max_db']}] dB")
print(f"  SON max step: {cfg['son_config']['max_cio_step_db']} dB")
print(f"  SON updates/cycle: {cfg['son_config']['max_updates_per_cycle']}")
print(f"  DQN LR: {cfg['dqn']['learning_rate']}, eps: {cfg['dqn']['epsilon_start']}→{cfg['dqn']['epsilon_end']}")

cmd = [
    sys.executable,
    str(ROOT / "scripts/train.py"),
    "--config", str(CONFIG),
    "--resume", str(RESUME),
    "--allow-existing-out-dir",
]
print(f"\nRunning: {' '.join(cmd)}\n")
subprocess.check_call(cmd)
print("\n✓ Training complete!")
print("  Best weights:", ROOT / "results/runs/colab_finetune_ue/checkpoints/gnn_dqn.pt")
''').strip()


# =============================================================================
# COLAB CELL 3 — Full evaluation + download
# =============================================================================
CELL3 = textwrap.dedent(r'''
# ============================================================
# CELL 3: Evaluate all scenarios (20 seeds) + download results
# ============================================================
import os, subprocess, sys, shutil
from pathlib import Path

ROOT = Path("/content/gnn-dqn-handover-loadbalancing")
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT / "src")

CKPT = ROOT / "results/runs/colab_finetune_ue/checkpoints/gnn_dqn.pt"
OUT = ROOT / "results/runs/colab_finetune_ue/eval_post_finetune"
assert CKPT.is_file(), f"Missing {CKPT} — run Cell 2 first"

cmd = [
    sys.executable,
    str(ROOT / "scripts/evaluate.py"),
    "--checkpoint", str(CKPT),
    "--out-dir", str(OUT),
    "--seeds", "20",
    "--steps", "80",
    "--split", "all_plus_stress",
]
print(f"Running: {' '.join(cmd)}\n")
subprocess.check_call(cmd)
print("\n✓ CSVs written:", OUT)

# Quick summary table
import csv
print("\n" + "=" * 80)
print("PUBLICATION SUMMARY")
print("=" * 80)
print(f"{'Scenario':<25} {'son_gnn_dqn':>12} {'a3_ttt':>12} {'load_aware':>12} {'Δ vs A3':>8}")
print("-" * 80)

for csvf in sorted(OUT.glob("*.csv")):
    scenario = csvf.stem
    rows = {}
    with open(csvf) as f:
        for row in csv.DictReader(f):
            rows[row["method"]] = float(row["avg_ue_throughput_mbps"])
    if "son_gnn_dqn" in rows and "a3_ttt" in rows:
        son = rows["son_gnn_dqn"]
        a3 = rows["a3_ttt"]
        la = rows.get("load_aware", 0)
        delta = (son - a3) / a3 * 100
        mark = "✓" if delta > 0 else "✗"
        print(f"{scenario:<25} {son:>12.3f} {a3:>12.3f} {la:>12.3f} {delta:>+7.2f}% {mark}")

# Zip everything for download
zip_dir = ROOT / "results/runs/colab_finetune_ue"
shutil.make_archive("/content/finetune_results", "zip", zip_dir.parent, zip_dir.name)
try:
    from google.colab import files as colab_files
    colab_files.download("/content/finetune_results.zip")
    print("\n✓ Download started!")
except Exception:
    print("\nDownload: /content/finetune_results.zip")
    print("Or: !cp /content/finetune_results.zip /content/drive/MyDrive/")
''').strip()


def main() -> None:
    print("=" * 72)
    print("COLAB CELL 1 — Install + Upload")
    print("=" * 72)
    print(CELL1)
    print("\n" + "=" * 72)
    print("COLAB CELL 2 — Fine-tune")
    print("=" * 72)
    print(CELL2)
    print("\n" + "=" * 72)
    print("COLAB CELL 3 — Evaluate + Download")
    print("=" * 72)
    print(CELL3)


if __name__ == "__main__":
    main()
