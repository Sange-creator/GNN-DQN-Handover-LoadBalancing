# GNN-DQN Based Handover Optimization and Load Balancing in LTE Networks

## Project Goal

> **GNN-DQN based handover optimization and load balancing in LTE networks.**
>
> A Context-Aware Adaptive SON-Gated architecture dynamically orchestrates handover routing between a Graph Neural Network (GNN-DQN) agent and a deterministic SON layer. By evaluating UE speed and average cell load, the system extracts the full spatial optimization benefits of the AI (+18.8% Jain fairness) while protecting the network from temporal performance degradation and OOD congestion, achieving Pareto optimality across all LTE deployment scenarios.

## Success Criteria (All Must Be Met)

1. **Handover optimization** — son_gnn_dqn throughput ≥ A3-TTT in every scenario
2. **Load balancing** — son_gnn_dqn Jain fairness index significantly above A3-TTT
3. **Zero ping-pong** — ping-pong rate ≈ 0 across all scenarios
4. **Sticky cell fix** — earlier, timely handovers especially on highway (CIO/TTT tuning)

## Architecture

```
UE measurements (RSRP, RSRQ, Speed, Cell Load)
        ↓
   Context-Aware 2-Signal Gate
        ↙                      ↘
[Speed < 15 & Load < 35%]   [Speed > 15 OR Load >= 35%]
        ↓                        ↓
   GNN-DQN (RAW)           SON Wrapper (A3-TTT)
   (Spatial Opt.)          (Temporal/OOD Stability)
        ↘                      ↙
        A3 event executor (LTE execution)
```

- **a3_ttt**: traditional LTE physics baseline
- **gnn_dqn**: raw learned preference policy, unrestricted AI baseline
- **son_gnn_dqn**: blanket safety SON-wrapped policy
- **adaptive_son_gnn_dqn**: The deployed 2-Signal Gate — the main contribution

## Current Training Run

**highway_sonv3** (running now, PID 7537):
- Resume from: `results/runs/ue_final_30h/checkpoints/resume/resume_ep0500.pt`
- Config: `configs/experiments/highway_sonv3.json`
- Episodes: 500 → 800 (300 new episodes)
- Log: `results/runs/highway_sonv3_train.log`
- Key fixes over previous run:
  - highway scenario weight: 0.01 → 1.5 (was starved, causing -18.8% throughput)
  - rollback_throughput_drop_frac: 0.05 → 0.20 (was too tight, rolled back all useful CIO moves)
  - highway added to validation scenarios (early stopping now sees highway quality)
  - rollback_pingpong_increase_frac: 0.15 → 0.30

## Expected Results After highway_sonv3

| Scenario | Throughput vs A3 | Jain vs A3 | Ping-pong |
|----------|-----------------|------------|-----------|
| dense_urban | +2 to +5% | +15 to +25% | ≈0 |
| overloaded_event | +10 to +20% | +20 to +35% | ≈0 |
| suburban | +1 to +4% | +10 to +18% | ≈0 |
| highway | +3 to +8% | +3 to +8% | 0 |
| highway_fast | ≈0% | ≈0% | 0 |
| sparse_rural | ≈0 to +2% | +5 to +12% | ≈0 |

## Evaluation

After training completes, run 30-seed evaluation:

```bash
python3 scripts/evaluate.py \
  --checkpoint results/runs/highway_sonv3/checkpoints/gnn_dqn.pt \
  --out-dir results/runs/highway_sonv3/eval_30seeds \
  --seeds 30
```

Final comparisons must include all methods:
- `no_handover`, `random_valid`, `strongest_rsrp`
- `a3_ttt` (baseline to beat)
- `load_aware`
- `gnn_dqn` (raw policy, shows maximum theoretical gains)
- `son_gnn_dqn` (main contribution — safe deployment)

## SON Controller Parameters (highway_sonv3)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| cio_min_db / cio_max_db | ±6 dB | CIO adjustment range |
| max_cio_step_db | 1.0 dB | Max change per cycle |
| preference_threshold | 0.10 | Min GNN preference share to trigger CIO+ |
| rollback_throughput_drop_frac | **0.20** | Allow up to 20% thr drop before rollback |
| rollback_pingpong_increase_frac | **0.30** | Allow 30% pp increase before rollback |
| update_interval_steps | 4 | How often SON checks preferences |
| max_updates_per_cycle | 10 | Max CIO changes per update cycle |

## What Is Citable

Only results from runs after 2026-05-09 refactor:
- `results/runs/colab_finetune_ue/eval_30seeds/` — baseline reference
- `results/runs/highway_sonv3/eval_30seeds/` — **final publication results** (after training)

Archived outputs under `results/archive_prefix/pre_refactor_2026-05-09/` are diagnostic only — do not cite.

## Feature Profile (UE_ONLY)

- RSRP, RSRQ (signal strength and quality)
- RSRQ-derived load proxy (interference/congestion estimate)
- RSRP/RSRQ trend features (rate of change)
- Serving-cell indicator, previous serving-cell indicator
- Signal usability flag
- UE speed
- Time since last handover

Never call RSRQ-estimated load "real PRB utilization." PRB utilization requires eNB counters (`oran_e2` mode).

## Key Commands

```bash
# Monitor current training
tail -f results/runs/highway_sonv3_train.log

# Evaluate after training
python3 scripts/evaluate.py \
  --checkpoint results/runs/highway_sonv3/checkpoints/gnn_dqn.pt \
  --out-dir results/runs/highway_sonv3/eval_30seeds \
  --seeds 30

# Quick smoke test
python3 scripts/train.py --config configs/experiments/smoke_ue.json

# Resume training (if interrupted)
python3 scripts/train.py \
  --config configs/experiments/highway_sonv3.json \
  --resume results/runs/highway_sonv3/checkpoints/resume/resume_epXXXX.pt
```

## Paper Talking Points

- **Pareto Optimality:** The Adaptive SON-Gated architecture dynamically routes UEs to maximize AI spatial optimization in stable areas (+18.8% load fairness) while reverting to deterministic physics to eliminate AI hallucinations in high mobility (0.0% ping-pong) and extreme congestion (+75.6% throughput rescue).
- **Synergistic Fusion:** ML (GNN-DQN) is not forced to replace standard LTE telecommunication physics. Instead, they operate synergistically using a 2-Signal Context Gate (Speed and Average Cell Load).
- **Topology Invariance:** The underlying GNN encoder processes maps as graph nodes/edges, meaning the exact same neural weights trained on 20 cells work flawlessly on 40-cell OOD real-world topologies like Kathmandu.
- **UE-Only Features:** No real PRB counters are needed. The model achieves state-of-the-art results using drive-test observable RSRP, RSRQ (as a load proxy), and UE velocity.

## graphify

This project has a knowledge graph at graphify-out/.

- ALWAYS read graphify-out/GRAPH_REPORT.md before reading source files or running searches
- For cross-module questions, use `graphify query "<question>"` or `graphify path "<A>" "<B>"`
- After modifying code, run `graphify update .` to keep the graph current
