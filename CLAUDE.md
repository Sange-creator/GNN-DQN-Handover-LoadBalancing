# GNN-DQN Handover Optimization Project

## Project Overview

**Title:** GNN-DQN: Graph Neural Network-based Deep Q-Network for Intelligent Handover Optimization in LTE/5G Networks with Cross-Region Generalization

**Goal:** Build a production-feasible GNN-DQN handover controller that:
1. Makes per-UE handover decisions using graph-aware deep RL
2. Uses only UE-observable measurements (RSRP, RSRQ as load proxy)
3. Generalizes across network topologies without retraining (Pokhara → Dharan → Nepal)
4. Outperforms rule-based (A3, ReBuHa) and learning-based (CDQL, flat-DQN) baselines
5. Is designed for O-RAN near-RT RIC xApp deployment (future work)

**Deadline:** May 14, 2026 (5 days remaining from May 9)
**Target:** Journal publication (IEEE TVT, Computer Networks, or IEEE Access)

---

## Current Status (2026-05-09)

### Completed
- **Code:** All 8 modules in `handover_gnn_dqn/` fully implemented; 6 baselines wired up; ns-3 exporter verified (3 runs in `data/ns3/`).
- **Data:** 75 real Pokhara cells + 150 Kathmandu cells from OpenCellID in `data/opencellid/`.
- **Overnight Pokhara training:** Finished 2026-05-08 08:01 (7.2h on M1 Pro, single-topology, 15 cells × 90 UEs, 100 episodes × 80 steps). Checkpoints + history at `results/overnight/`.
- **10-seed evaluation on Pokhara:** All 6 methods compared, summary.csv written.
- **Git initialized:** Repo on main branch with initial commit.

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
- **Multi-scenario training** (`run_full_training.py`, started 2026-05-08 11:40). Trains a single GNN-DQN across 6 scenarios (300 episodes × 80 steps) then evaluates on 3 unseen scenarios. Output: `results/full_training/`.

### Not started
- Diagnosis of why GNN-DQN regresses on the single-topology Pokhara run (reward weights, epsilon schedule, action masking, training length).
- Zero-shot eval of the existing `gnn_dqn_pokhara.pt` checkpoint on Kathmandu / Dharan (independent of the multi-scenario run).
- Figures (training curves, comparison bars with CI, CDF, topology map, load heatmap).
- LaTeX paper / thesis report draft.

---

## Known Issues (2026-05-09)

### Critical: GNN-DQN underperforms heuristics
The model is 10% worse than "strongest RSRP" — a trivial signal-following rule. Root causes:
1. **Undertrained** — 100 episodes is insufficient; loss still climbing at termination. Need 500+.
2. **Reward function misweighted** — agent optimizes a blend of throughput + load + handover penalty, but weights may not push throughput high enough.
3. **Epsilon decays too fast** — overnight run decayed over 80 episodes (should be 300). Agent exploits a bad policy before exploring enough.
4. **Only 80 steps/episode** — target is 200 steps. Too few transitions per episode.

### Multi-scenario training concerns
5. **Replay buffer split thin** — 100K / 6 scenarios = ~16K per scenario. May be insufficient.
6. **Only 5 eval seeds** — need 20 for publishable confidence intervals.
7. **300 episodes across 6 scenarios** — each scenario gets ~50 episodes on average. May need 500+ total.

### Paper/deadline risks
8. **No paper draft** — deadline is May 14 (5 days).
9. **Generalization unproven** — multi-scenario run results not yet available.
10. **REPORT.md has wrong numbers** — based on toy 12-episode run, not real results.

### Calibration vs targets
| Target (original) | Actual | Gap |
|---|---|---|
| 500 episodes per run | 100 (overnight), 300 (multi) | 2-5× short |
| 20 evaluation seeds | 5-10 | 2-4× short |
| 200 steps/episode | 80 | 2.5× short |
| Replay 100K | 100K (but split across 6) | ~16K effective |
| epsilon decay 300 ep | 80 (overnight), 210 (multi) | overnight too fast |
| Inference <10ms | not measured | TBD |

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
- Episodes: 500+ (target); currently 100 (overnight), 300 (multi-scenario)
- Steps per episode: 200 (target); currently 80
- Transitions target: 2-5 million; currently ~720K (300 ep × 80 steps × ~30 UEs avg)
- Seeds: 20 (target); currently 5-10
- Optimizer: Adam (lr=3e-4)
- Replay buffer: 100K transitions (split ~16K per scenario in multi-scenario mode)
- Target network update: every 500 steps
- Epsilon: 1.0 → 0.05 (linear decay over 210 episodes in multi-scenario run)
- Train every: 4 decisions
- Batch size: 64

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

## Training & Test Scenarios

### Training Scenarios (6 — model cycles through all)
| # | Scenario | Cells | UEs | What it simulates |
|---|----------|-------|-----|-------------------|
| 1 | Dense Urban | 20 | 250 | Lakeside/Kathmandu crowded area |
| 2 | Highway | 10 | 50 | Fast-moving vehicles (80-120 km/h) |
| 3 | Suburban | 15 | 105 | Medium density residential |
| 4 | Sparse Rural | 7 | 25 | Hills, few towers, low traffic |
| 5 | Overloaded Event | 12 | 240 | Festival/concert extreme congestion |
| 6 | Real Pokhara | ~20 | variable | Actual cell positions from OpenCellID |

### Test Scenarios (3 — NEVER seen during training, proves generalization)
| # | Scenario | Cells | UEs | Purpose |
|---|----------|-------|-----|---------|
| 1 | Kathmandu Real | 25 | ~175 | Real city, real positions → works on new place |
| 2 | Dharan Synthetic | 20 | 100 | Synthetic terrain (no OpenCellID coverage) |
| 3 | Unknown Hex Grid | 19 | 133 | Standard 3GPP layout, completely different structure |

The GNN handles any new place (Biratnagar, Birgunj, etc.) by just providing the cell neighbor graph — no retraining needed. Flat DQN breaks on topology changes (fixed input size).

---

## Comparison Methods (6 total)

### Essential comparisons for paper (minimum 3):
| # | Method | Type | What it proves |
|---|--------|------|----------------|
| 1 | Strongest RSRP | 3GPP default | "Better than dumb signal-following" |
| 2 | Load-Aware Heuristic | ReBuHa-like | "Justifies ML over smart rules" |
| 3 | Flat DQN (no GNN) | ML ablation | "GNN architecture matters" |
| 4 | **GNN-DQN (ours)** | Proposed | Graph-aware deep Q-network |

### Optional (nice-to-have but not critical):
| # | Method | Type | Description |
|---|--------|------|-------------|
| 5 | No Handover | Lower bound | Stay on initial cell (trivially bad) |
| 6 | A3 TTT | 3GPP standard | A3 event with time-to-trigger |

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
└── CLAUDE.md                — This file
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
- [x] Multi-scenario training started (`run_full_training.py`, 300 episodes × 80 steps across 6 training scenarios)
- [ ] Check multi-scenario run results in `results/full_training/`
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

### Phase 6: Repo / Polish — PARTIALLY DONE
- [x] `git init` + initial commit on main
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
