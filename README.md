# GNN-DQN Handover Optimization

Research code for topology-invariant, per-UE handover optimization in LTE/5G-style
networks. The main publishable path is **UE-only**: RSRP, RSRQ-derived load proxy,
mobility, serving-cell state, and handover history. A separate **O-RAN/E2** profile
adds true PRB/counter features when network-side telemetry is available.

The deployable framing is **SON-GNN-DQN**: the learned GNN-DQN policy produces
per-UE target preferences, then a safety-bounded SON controller aggregates those
preferences into standard handover parameter updates such as CIO/TTT. Direct
GNN-DQN remains an evaluation baseline.

## Current Status

This repo has been refactored into a training-ready structure. Results archived under
`results/archive_prefix/pre_refactor_2026-05-09/` are diagnostic only and should not be
cited as final results.

## Project Structure

```text
configs/experiments/      JSON configs for smoke, UE-only, Pokhara, and O-RAN runs
data/raw/                 OpenCellID, ns-3, synthetic, and drive-test inputs
data/processed/           Cleaned/generated datasets (ignored by git)
docs/                     Project notes, guides, and paper material
scripts/                  Config-driven train/evaluate/data/figure entrypoints
src/handover_gnn_dqn/     Reusable package code
tests/                    Unit/integration/regression acceptance tests
results/archive_prefix/   Archived pre-fix diagnostic outputs
results/runs/             New generated runs (ignored by git)
```

## Feature Profiles

- `ue_only`: drive-test compatible. Uses UE-observable radio measurements and
  `load_proxy_rsrq`. It does not require PRB utilization.
- `oran_e2`: future O-RAN/xApp-compatible profile. Adds true `prb_utilization`,
  `prb_available`, connected UE count, and cell throughput when network-side telemetry
  is available.

Drive-test data without PRB is valid. Missing PRB is represented explicitly with
`prb_available = 0`; estimated RSRQ load is never treated as true PRB utilization.

## SON Deployment Mode

Evaluation now includes `son_gnn_dqn`, a SON-compatible tuned-A3 policy. It:

- samples GNN-DQN per-UE target preferences periodically,
- aggregates preferences over serving-target cell pairs,
- applies bounded CIO changes within operator-style limits,
- increases TTT conservatively when ping-pong becomes high,
- reports SON KPIs such as update count, CIO magnitude, and rollback count.

This is the recommended defense framing for current non-O-RAN deployment.

## Commands

```bash
# Acceptance smoke: UE-only
python3 scripts/train.py --config configs/experiments/smoke_ue.json

# Acceptance smoke: O-RAN/E2 feature profile
python3 scripts/train.py --config configs/experiments/smoke_oran.json

# Main publishable training path
python3 scripts/train.py --config configs/experiments/multiscenario_ue.json

# Evaluate a saved checkpoint
python3 scripts/evaluate.py \
  --checkpoint results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt \
  --out-dir results/runs/multiscenario_ue/eval_20seed \
  --seeds 20
```

Legacy wrappers remain:

```bash
python3 run_experiment.py   # smoke UE-only
python3 run_overnight.py    # Pokhara UE-only config
python3 run_full_training.py # multi-scenario UE-only config
```

## Pre-Training Gate

Before a long run, verify:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Then run both feature-profile smoke tests:

```bash
python3 scripts/train.py --config configs/experiments/smoke_ue.json
python3 scripts/train.py --config configs/experiments/smoke_oran.json
```

Long runs can be resumed from generated checkpoints:

```bash
python3 scripts/train.py \
  --config configs/experiments/multiscenario_ue.json \
  --resume results/runs/multiscenario_ue/checkpoints/resume/resume_ep0025.pt
```
