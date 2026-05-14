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

This repo has completed a clean UE-only production training and 20-seed evaluation
run for `multiscenario_ue`.

Final production checkpoint metadata:

```text
model_version:    gnn_dqn_v3_graph_value_head
reward_version:   throughput_fairness_pingpong_v3
feature_profile:  ue_only
prb_available:    false
feature_dim:      11
checkpoint_kind:  best_model
git_commit:       9d66dd47d8a0d18989bc566eb624db42e92208b3
episodes done:    300
best_episode:     248
best_score:       15.863471935147086
```

The final 20-seed evaluation passed all 11 scenario gates:

| Scenario | Gate | SON avg | Random avg | A3 avg | SON P5 | SON ping-pong |
|---|---:|---:|---:|---:|---:|---:|
| coverage_hole | PASS | 4.971 | 4.197 | 4.971 | 2.396 | 0.000 |
| dense_urban | PASS | 4.953 | 4.133 | 4.953 | 2.343 | 0.000 |
| dharan_synthetic | PASS | 5.072 | 4.166 | 5.072 | 2.325 | 0.000 |
| highway | PASS | 4.877 | 4.050 | 4.877 | 2.335 | 0.000 |
| kathmandu_real | PASS | 4.704 | 3.720 | 4.707 | 2.172 | 0.000 |
| overloaded_event | PASS | 5.131 | 5.808 | 5.136 | 2.607 | 0.000 |
| pokhara_dense_peakhour | PASS | 4.135 | 1.526 | 4.131 | 1.693 | 0.000 |
| real_pokhara | PASS | 4.978 | 3.924 | 4.977 | 2.354 | 0.001 |
| sparse_rural | PASS | 2.933 | 2.453 | 2.934 | 1.198 | 0.000 |
| suburban | PASS | 4.997 | 4.150 | 4.998 | 2.350 | 0.000 |
| unknown_hex_grid | PASS | 4.933 | 4.109 | 4.935 | 2.308 | 0.000 |

The headline result is that `son_gnn_dqn` is A3-competitive across training and
held-out scenarios while maintaining near-zero ping-pong. It should not be claimed
as universally throughput-dominant: in `overloaded_event`, `random_valid` achieves
higher average throughput by spreading users randomly, while `son_gnn_dqn` remains
within the acceptance gate and preserves handover stability.

Generated run artifacts live under `results/runs/multiscenario_ue/`, which is
ignored by git. The checkpoint and CSVs are intentionally local artifacts, not
committed repository files.

Results archived under `results/archive_prefix/pre_refactor_2026-05-09/` are
diagnostic only and should not be cited as final results.

## Project Structure

```text
configs/experiments/      JSON configs for smoke, UE-only, Pokhara, and O-RAN runs
data/raw/                 OpenCellID, ns-3, synthetic, and drive-test inputs
data/processed/           Cleaned/generated datasets (ignored by git)
docs/thesis/              Thesis chapters and abstract
docs/reports/             Evaluation reports and narrative summaries
docs/guides/              Human-readable guides and DOCX exports
docs/references/          External reference papers and PDFs
docs/sites/               Presentation/demo websites
scripts/                  Config-driven train/evaluate/data/figure entrypoints
src/handover_gnn_dqn/     Reusable package code
tests/                    Unit/integration/regression acceptance tests
tools/                    Utility scripts for data collection and external tooling
web_dashboard/            Optional local dashboard app
results/archive_prefix/   Archived pre-fix diagnostic outputs
results/runs/             New generated runs (ignored by git)
```

See `docs/REPO_LAYOUT.md` for the full code-vs-documentation organization rule.

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
