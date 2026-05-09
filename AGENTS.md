# GNN-DQN Handover Optimization Project

## Project Overview

**Title:** GNN-DQN: Graph Neural Network-based Deep Q-Network for Intelligent Handover Optimization in LTE/5G Networks with Cross-Region Generalization

**Goal:** Build a production-feasible GNN-DQN handover controller that:
1. Makes per-UE handover decisions using graph-aware deep RL
2. Uses only UE-observable measurements (RSRP, RSRQ as load proxy)
3. Generalizes across network topologies without retraining (Pokhara → Dharan → Nepal)
4. Outperforms rule-based (A3, ReBuHa) and learning-based (CDQL, flat-DQN) baselines
5. Is designed for O-RAN near-RT RIC xApp deployment (future work)

**Deadline:** May 14, 2026 (7 days from May 7)
**Target:** Journal publication (IEEE TVT, Computer Networks, or IEEE Access)

---

## Current Status (2026-05-08)

### Completed
- **Code:** All 8 modules in `handover_gnn_dqn/` fully implemented; 6 baselines wired up; ns-3 exporter verified (3 runs in `data/ns3/`).
- **Data:** 75 real Pokhara cells + 150 Kathmandu cells from OpenCellID in `data/opencellid/`.
- **Overnight Pokhara training:** Finished 2026-05-08 08:01 (7.2h on M1 Pro, single-topology, 15 cells × 90 UEs, 100 episodes × 80 steps). Checkpoints + history at `results/overnight/`.
- **10-seed evaluation on Pokhara:** All 6 methods compared, summary.csv written.

### Honest results from `results/overnight/summary.csv`
| Method | Avg Mbps (±CI95) | P5 Mbps | Load Std | HO/1000 |
|---|---:|---:|---:|---:|
| load_aware | 1.179 ± 0.111 | 0.067 | 4.65 | 91.2 |
| strongest_rsrp | 0.999 ± 0.096 | 0.059 | 5.74 | 38.5 |
| a3_ttt | 0.965 ± 0.092 | 0.061 | 5.48 | 5.6 |
| no_handover | 0.921 ± 0.110 | 0.062 | 5.08 | 0.0 |
| **gnn_dqn (ours)** | **0.898 ± 0.048** | 0.053 | 5.99 | 45.2 |
| flat_dqn | 0.379 ± 0.037 | 0.027 | 10.06 | 11.2 |

- **GNN-DQN vs strongest_rsrp:** −10.1% avg throughput, −9.1% P5, +4% worse load std.
- **GNN-DQN vs flat_dqn (ablation):** +137% throughput, +40.5% better load std — the GNN component clearly helps over an MLP, but the policy is not yet beating signal-strength rules.
- Training history (`gnn_history.json`) shows the agent does not converge in 100 episodes: episode-100 throughput (0.71) is below episode-5 (0.91); loss bottoms ~episode 85 then climbs. Likely undertrained or reward-misweighted.

### REPORT.md is superseded
The numbers in `REPORT.md` (GNN-DQN +6.4% over strongest_rsrp) are from a 12-episode run on a 9-cell synthetic topology (`run_experiment.py --train-episodes 12 --steps 60 --test-episodes 5`) and **do not match** the production overnight results above. Treat `REPORT.md` as a dev-stage artifact, not a citable result.

### In progress
- **Multi-scenario training** (`run_full_training.py`, PID 45328, started 11:40, episode ~30/120, ETA ~22:00 today). Trains a single GNN-DQN across 6 scenarios (dense_urban, highway, suburban, sparse_rural, overloaded_event, real_pokhara) then evaluates on 3 unseen scenarios (kathmandu_real, dharan_synthetic, unknown_hex_grid). This is the run that proves the topology-invariance contribution. Output will land in `results/full_training/`.

### Not started
- Diagnosis of why GNN-DQN regresses on the single-topology Pokhara run (reward weights, epsilon schedule, action masking, training length).
- Zero-shot eval of the existing `gnn_dqn_pokhara.pt` checkpoint on Kathmandu / Dharan (independent of the multi-scenario run).
- Figures (training curves, comparison bars with CI, CDF, topology map, load heatmap).
- LaTeX paper / thesis report draft.
- Git init.

### Calibration vs original AGENTS.md targets
| Target (original) | Actual | Gap |
|---|---|---|
| 500 episodes per run | 100 | 5× short |
| 20 evaluation seeds | 10 | 2× short |
| Replay 100K | 100K | ✓ |
| epsilon decay 300 ep | 80 ep (overnight) | decays too fast for short runs |
| Inference <10ms | not measured | TBD |
| 95% CIs from 20+ seeds | 95% CIs from 10 seeds | acceptable, not target |

---

## Key Design Decisions

### Why GNN-DQN (not PPO-CIO)?
- **Per-UE decisions** (fine-grained) vs PPO-CIO's global bias adjustment (coarse)
- **GNN enables topology-invariant generalization** — same model works on any cell layout
- **Value-based (DQN)** — simpler, more sample-efficient for discrete actions
- **Novel combination** — no published paper combines GNN + DQN for per-UE handover

### Feature Design (UE-Observable Only)
The model uses ONLY measurements a real UE can obtain:
- RSRP per neighbor cell (measurable)
- RSRQ per neighbor cell (measurable → load proxy via Raida framework)
- RSRP/RSRQ trends (derived from history)
- Serving cell indicator, time since last HO, UE speed class
- NO direct PRB utilization (eNB-side only, not available to UE)

Exception: In O-RAN xApp mode, E2 interface provides direct PRB data.

### Model Architecture (Target: ~50K parameters)
```
Input (12 features per cell node)
  → GCNConv Layer 1 (128 hidden) + ReLU + Dropout(0.1)
  → GCNConv Layer 2 (128 hidden) + ReLU + Dropout(0.1)
  → GCNConv Layer 3 (64 hidden) + ReLU
  → Dueling Q-Head:
      → Value stream: Linear(64 → 32 → 1)
      → Advantage stream: Linear(64 → 32 → 1)
      → Q = V + (A - mean(A))
```

### Training Configuration
- Episodes: 500+
- Steps per episode: 200
- Transitions target: 2-5 million
- Seeds: 20 (for statistical confidence)
- Optimizer: Adam (lr=3e-4)
- Replay buffer: 100K transitions
- Target network update: every 500 steps
- Epsilon: 1.0 → 0.05 (linear decay over 300 episodes)

---

## Data Sources

1. **OpenCellID** — Real eNodeB positions for Nepal (MCC=429, filter LTE)
   - Pokhara bbox: 28.17-28.27N, 83.93-84.03E
   - Kathmandu bbox: 27.65-27.75N, 85.28-85.38E
   - Dharan bbox: 26.78-26.83N, 87.27-87.32E

2. **ns-3.40 Simulator** — Located at `/Users/saangetamang/ns-allinone-3.40/ns-3.40`
   - Scratch program: `scratch/gnn-dqn-handover-data.cc`
   - Generates per-UE RSRP, throughput, handover events

3. **G-NetTrack Pro** — Android drive test app
   - Logs RSRP, RSRQ, SINR, cell ID, GPS, handover events
   - Used for model validation (not training)

4. **Synthetic** — Python simulator with configurable topology
   - Accepts real cell positions from OpenCellID
   - Generates unlimited training transitions

---

## Comparison Methods (6 total)

| # | Method | Type | Description |
|---|--------|------|-------------|
| 1 | No Handover | Lower bound | Stay on initial cell |
| 2 | Strongest RSRP | 3GPP A3-like | Signal-strength handover with hysteresis |
| 3 | A3 TTT | 3GPP standard | A3 event with time-to-trigger |
| 4 | Load-Aware Heuristic | ReBuHa-like | Signal + load utility rule |
| 5 | Flat DQN (no GNN) | ML baseline | Standard DQN without graph structure |
| 6 | **GNN-DQN (ours)** | Proposed | Graph-aware deep Q-network |

---

## Key Metrics (KPIs)

### Primary (headline results):
- Average UE throughput (Mbps)
- 5th percentile UE throughput (Mbps) — fairness indicator
- Load standard deviation — balance indicator
- Jain's fairness index (load)
- Handover rate (per 1000 decisions)
- Ping-pong handover rate

### Secondary (paper completeness):
- Outage rate (RSRP < threshold)
- Overload rate (cells > 100% capacity)
- Training convergence time (episodes to 95% performance)
- Inference time (ms) — must be <10ms for O-RAN
- Generalization gap (% performance drop on unseen topology)

---

## Project Structure

```
gnn-dqn-handover/
├── data/
│   ├── opencellid/           — Real tower positions
│   ├── drive_test/           — G-NetTrack measurements
│   ├── ns3/                  — ns-3 simulator output
│   └── synthetic/            — Generated training data
├── handover_gnn_dqn/         — Core model code
│   ├── simulator.py          — LTE network environment
│   ├── topology.py           — Load real cell positions → graph
│   ├── gnn_dqn.py           — GNN-DQN agent (3-layer GCN + Dueling DQN)
│   ├── flat_dqn.py          — Flat DQN baseline (no GNN)
│   ├── policies.py          — Rule-based handover policies
│   ├── experiment.py        — Evaluation framework
│   ├── data_loader.py       — Parse drive test & ns-3 data
│   └── visualize.py         — Plotting & figures
├── results/
│   ├── figures/             — Publication-quality plots (PDF)
│   ├── tables/              — LaTeX tables
│   └── models/              — Saved model checkpoints
├── paper/                    — Research paper (LaTeX)
├── website/                  — Project demo website
├── tools/                    — Utility scripts
├── run_experiment.py         — Main entry point
└── AGENTS.md                — This file
```

---

## Related Work (Key References)

### Direct Competitors:
1. Salvatori et al. (2026) "Dual-Graph MARL for HO" — TD3 + GNN on dual graph, CIO control
2. Eskandarpour & Soleimani (2025) "PPO QoS-Aware Load Balancing" — PPO + CIO, 6 KPIs
3. Gonzalez Bermudez et al. (2025) "GNN for O-RAN Mobility" — GNN link prediction

### Baselines Referenced:
4. CDQL — Contextual Deep Q-Learning (value-based baseline)
5. ReBuHa — Rule-based Resource Block Utilization Handover Algorithm

### Our Advantages Over These:
- Per-UE decisions (finer granularity than CIO-based)
- Value-based DQN (simpler, fewer hyperparameters than PPO/TD3)
- Real topology (Pokhara Valley eNodeBs)
- Cross-region generalization proof (Pokhara → Kathmandu → Dharan)
- UE-observable features only (deployable without eNB cooperation)

---

## Development Phases

### Phase 1: Foundation — DONE
- [x] Basic simulator working
- [x] Initial GNN-DQN with PyTorch Geometric
- [x] ns-3 data export pipeline (3 runs in data/ns3/)
- [x] 3-layer GCN architecture (30,658 params)
- [x] 15 UE-observable features (RSRQ-based load proxy) in simulator.py
- [x] Topology loader for real positions (topology.py)
- [x] Flat-DQN baseline (flat_dqn.py)
- [x] OpenCellID download — 75 Pokhara, 150 Kathmandu
- [x] Dharan: synthetic topology (no OpenCellID coverage)

### Phase 2: Training — IN PROGRESS
- [x] 100-episode single-topology Pokhara run (run_overnight.py, completed 2026-05-08 08:01)
- [x] Flat-DQN trained on same topology
- [x] 10-seed evaluation of all 6 methods (results/overnight/summary.csv)
- [ ] **Diagnose why GNN-DQN regresses on Pokhara** (reward weights, epsilon decay, training length, action masking)
- [ ] Retrain GNN-DQN with fixes (target: beat strongest_rsrp on at least P5 throughput or load_std)
- [ ] 500-episode run if M1 budget allows

### Phase 3: Generalization — IN PROGRESS
- [x] Multi-scenario training started (run_full_training.py, PID 45328, ETA 2026-05-08 ~22:00)
- [ ] Zero-shot eval on Kathmandu_real (25 cells, real positions)
- [ ] Zero-shot eval on Dharan_synthetic (20 cells)
- [ ] Zero-shot eval on unknown_hex_grid (19 cells, 3GPP standard layout)
- [ ] Generalization gap report (% throughput drop on unseen vs trained)
- [ ] Optional: 20-episode few-shot fine-tune per region

### Phase 4: Visualization — NOT STARTED
- [ ] Training convergence curves (GNN vs flat-DQN, loss + throughput per episode)
- [ ] Comparison bar charts with 95% CI error bars (6 methods × 8 KPIs)
- [ ] CDF of per-UE throughput
- [ ] Pokhara network topology map (cell positions + adjacency)
- [ ] Load heatmap (before vs after GNN-DQN)
- [ ] Generalization results plot (per-scenario performance)

### Phase 5: Paper / Thesis — NOT STARTED
- [ ] System model + problem formulation
- [ ] Methodology (GNN-DQN architecture, state/action/reward, training)
- [ ] Results section using overnight + multi-scenario data
- [ ] Honest discussion of trade-offs (GNN-DQN gives up some throughput vs heuristics for X)
- [ ] Related work
- [ ] Abstract + intro + conclusion

### Phase 6: Repo / Polish — NOT STARTED
- [ ] `git init` + commit history
- [ ] README cleanup (remove or annotate the toy 12-episode numbers)
- [ ] Project website with results dashboard
- [ ] Final proofreading
- [ ] Defense slides

---

## Commands

### Run training:
```bash
python3 run_experiment.py --train-episodes 500 --steps 200 --test-episodes 20
```

### Generate ns-3 data:
```bash
python3 tools/run_ns3_dataset.py --sim-time 60 --sample-period 1 --run 1
```

### Quick smoke test:
```bash
python3 run_experiment.py --train-episodes 4 --steps 35 --test-episodes 3
```

---

## Key Design Properties (For Defense/Paper)

### Why GNN Works on Unknown Sites:
- GNN learns RELATIVE features (RSRP difference, load difference between neighbors)
- Not absolute identities (tower PCI, location coordinates)
- Same physics everywhere: signals decay with distance, congestion lowers RSRQ
- Only requirement: provide graph structure (cell neighbor map)

### Handling Network Changes:
- New tower added → add node + edges → GNN works immediately (no retraining)
- Tower removed → remove node → GNN works immediately
- Flat DQN BREAKS on topology changes (fixed input size)
- This is GNN's key advantage and paper's main contribution

### Capacity Limitation (What to Say in Paper):
- Model optimizes WITHIN physical capacity constraints
- Cannot create bandwidth from nothing (Shannon bound)
- Value: maximizes fairness and minimizes degradation under congestion
- Complementary to infrastructure investment, not a replacement

### RSRQ as Load Proxy:
- PRB utilization is eNB-side only (UE can't measure it directly)
- RSRQ = N * RSRP / RSSI → encodes interference from loaded neighbors
- Low RSRQ ≈ high cell load (Raida et al. framework)
- In O-RAN xApp mode, direct PRB data available via E2 (future work)

---

## Critical Constraints

- **Budget:** $0 (M1 Pro handles all training locally)
- **Training time budget:** 3-6 hours max per full run
- **Model must be:** topology-invariant (works on any cell count)
- **Features must be:** UE-observable only (for deployment realism)
- **Results must have:** 95% confidence intervals from 20+ seeds
- **Inference must be:** <10ms (for near-RT RIC compatibility)

---

## Notes

- The model learns RELATIVE feature relationships between cells, not absolute tower identities
- RSRQ is used as load proxy (Raida et al. framework): low RSRQ ≈ high cell load
- GNN operates on graph structure → same weights work for different topology sizes
- The combination (GNN + DQN + per-UE handover) is novel; whether it actually outperforms simple heuristics on a real topology is currently the open empirical question for this project — the 100-episode Pokhara run did not show that, and the multi-scenario run is the next data point.
- Primary competitor is Salvatori 2026 (dual-graph MARL): TD3 + CIO, coarser granularity than per-UE.
