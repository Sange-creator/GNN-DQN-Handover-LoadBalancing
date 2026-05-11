# Work Division & Tracking

> **Detailed prompts:** See `docs/prompts/PROMPT_CODEX.md`, `docs/prompts/PROMPT_GEMINI.md`, `docs/prompts/PROMPT_CLAUDE.md`
> **Master tracker:** See `docs/MASTER_TRACKER.md`

## Overview

Three AI assistants are working on this project in parallel:

| Agent | Role | Strengths |
|-------|------|-----------|
| **Claude Code (Opus)** | Core engineering — code, architecture, debugging, training infra | Deep code understanding, file editing, shell access, iterative debugging |
| **Codex** | Batch computation — long training runs, evaluation sweeps, figure generation | Runs long jobs, good at executing defined scripts, parallel batch work |
| **Gemini** | Research & writing — thesis prose, literature, LaTeX, defense prep | Large context for papers, citation search, document synthesis |

---

## Agent 1: Claude Code (You / Me)

### Responsibilities
1. Fix environment issues (pytest, dependencies)
2. Implement missing evaluation baselines (`random_valid`, `strongest_rsrp`, `load_aware`, standalone `gnn_dqn`)
3. Implement `scripts/generate_figures.py` (publication-quality plots)
4. Code review and bug fixes during training convergence
5. Any architectural changes to the GNN-DQN, SON controller, or simulator
6. Git hygiene — commits, branches, PR management

### Current Status
- [x] Soft target updates, cosine LR, weight decay added
- [x] Dueling DQN value stream fix
- [x] SON controller improvements
- [x] Checkpoint compatibility metadata
- [x] Run validation and result hygiene tools
- [ ] Complete baseline set in evaluator
- [ ] `generate_figures.py` implementation
- [ ] pytest environment fix
- [ ] Final code review before production run

---

## Agent 2: Codex

### Responsibilities
1. Run the full `multiscenario_ue` training (300 episodes, ~hours)
2. Run the full evaluation sweep (20 seeds, all baselines, all scenarios)
3. Generate summary CSVs and figures from evaluation results
4. Run any ablation studies if needed

### Prompt for Codex

```
You are working on a GNN-DQN handover optimization project.
Your job is to execute the production training and evaluation pipeline.

## Environment Setup
```bash
cd /path/to/gnn-dqn-handover-loadbalancing
pip install -r requirements.txt  # or: pip install torch torch-geometric numpy pandas pytest
export PYTHONPATH=src
```

## Step 1: Run Tests (sanity check)
```bash
python3 -m pytest -q
```
If tests fail, STOP and report the failures. Do not proceed.

## Step 2: Production Training
```bash
python3 scripts/train.py --config configs/experiments/multiscenario_ue.json
```
This trains for 300 episodes across 7 scenarios. It will produce:
- `results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt` (final model)
- `results/runs/multiscenario_ue/checkpoints/resume/resume_ep*.pt` (every 25 eps)
- Training logs

Monitor for:
- Loss convergence (should decrease over first 100 episodes)
- Epsilon decay (reaches 0.03 by episode 250)
- Any NaN or crash

If it crashes, save the error and the last resume checkpoint path.

## Step 3: Full Evaluation
```bash
python3 scripts/evaluate.py \
  --checkpoint results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt \
  --out-dir results/runs/multiscenario_ue/eval_20seed \
  --seeds 20
```
This evaluates all methods (no_handover, random_valid, strongest_rsrp, a3_ttt,
load_aware, gnn_dqn, son_gnn_dqn) across all test scenarios with 20 random
seeds for statistical confidence.

## Step 4: Generate Figures
```bash
python3 scripts/generate_figures.py --run-dir results/runs/multiscenario_ue
```
(Only if this script is implemented. If it errors with "not implemented", skip.)

## Step 5: Summarize
```bash
python3 scripts/summarize_evaluation.py --eval-dir results/runs/multiscenario_ue/eval_20seed
```

## Deliverables
Report back:
1. Training convergence: did loss stabilize? Final epsilon?
2. Evaluation results: paste the summary table or CSV
3. Any errors or warnings encountered
4. File paths of all generated artifacts
```

---

## Agent 3: Gemini

### Responsibilities
1. Write the thesis chapters (Introduction, Related Work, Methodology, Results, Conclusion)
2. Format comparison tables in LaTeX
3. Write the abstract
4. Prepare defense slides narrative
5. Literature search for additional citations
6. Proofread and ensure academic tone

### Prompt for Gemini

```
You are helping write a Master's thesis on intelligent handover optimization
in 5G networks. The system is called **SON-GNN-DQN**.

## Project Summary
SON-GNN-DQN is a topology-generalized GNN-DQN preference model with a
safety-bounded SON translation layer for A3/CIO-style handover optimization.
It uses UE-only measurements (RSRP, RSRQ as load proxy) and does not require
network-side PRB counters.

## Key Contributions (frame these in the thesis)
1. GNN-based DQN that generalizes across arbitrary cell topologies (train on
   3-cell, deploy on 25-cell zero-shot)
2. SON safety layer that translates RL preferences into bounded CIO/TTT
   updates (±6 dB cap, rate limiting, rollback)
3. UE-ONLY observability — works with standard 3GPP measurement reports,
   no O-RAN E2 dependency for basic deployment
4. Multi-objective reward: throughput + fairness + stability (anti-ping-pong)

## System Architecture
- State Builder: constructs per-UE graph from RSRP/RSRQ/speed/history
- GNN-DQN Agent: 3-layer GCN → Dueling Q-head → per-cell Q-values
- SON Controller: maps Q-value preferences to CIO updates with safety bounds
- Decision Engine: standard A3 logic with SON-adjusted parameters

## What I Need From You

### Chapter 1: Introduction (~2000 words)
- Motivate the problem: 5G densification → more handovers → need intelligence
- State the gap: existing DRL is topology-fixed, unsafe for deployment
- State the contribution: topology-generalized + safety-bounded + UE-observable
- Outline thesis structure

### Chapter 2: Related Work (~3000 words)
Use the following structure:
- 2.1 DRL for Handover (DQN, DDQN, PPO approaches)
- 2.2 GNN for RAN Management (topology generalization, transference)
- 2.3 Self-Organizing Networks (CIO/TTT tuning, MLB/MRO)
- 2.4 Gap Analysis (what's missing → our contribution)

Key references to include:
- Shen et al. (2023) — GNN generalization O(n) vs O(n²)
- Eisen & Ribeiro (2020) — transference in wireless GNNs
- Eskandarpour & Soleimani (2025) — DRL for QoS-aware load balancing
- 3GPP TS 38.331 — RRC specification (A3 event, CIO, TTT)
- Moysen & Garcia-Lozano (2018) — SON survey

### Chapter 3: Methodology (~4000 words)
- 3.1 System Model (network model, UE mobility, channel model)
- 3.2 Problem Formulation (MDP: state, action, reward)
- 3.3 GNN-DQN Architecture (GCN layers, dueling head, training)
- 3.4 SON Translation Layer (CIO mapping, safety bounds, rollback)
- 3.5 Training Strategy (multi-scenario, epsilon decay, soft updates)

### Chapter 4: Results (~3000 words)
NOTE: Results data will be provided separately once training completes.
Prepare the structure with placeholder tables:
- 4.1 Experimental Setup (scenarios, baselines, metrics)
- 4.2 Training Convergence Analysis
- 4.3 Scenario-Specific Performance
- 4.4 Topology Generalization (train small → test large)
- 4.5 Ablation: SON layer impact (gnn_dqn vs son_gnn_dqn)
- 4.6 Safety Analysis (rollback events, CIO bounds utilization)

### Chapter 5: Conclusion (~1000 words)
- Summarize contributions
- Limitations (simulation gap, no real deployment yet)
- Future work (O-RAN E2 integration, online fine-tuning, multi-agent)

## Style Requirements
- Academic tone, third person
- IEEE citation style [1], [2], ...
- All equations numbered
- Tables in LaTeX format
- Keep claims precise — don't overclaim without data

## Context Files (I will provide)
- Defense Q&A: docs/thesis_support.md (already written)
- Evaluation report: docs/SON_GNN_DQN_Evaluation_Report.md
- Results CSVs: will be provided after training completes
```

---

## Coordination Timeline

```
Phase 1 (NOW):
  Claude Code → fix baselines, implement figure generation
  Gemini     → write Chapters 1-3 (no results needed)

Phase 2 (After Claude Code fixes):
  Codex      → run multiscenario_ue training + evaluation
  Gemini     → continue Chapter 3, prepare Chapter 4 structure

Phase 3 (After Codex delivers results):
  Claude Code → verify results, debug any issues
  Gemini      → fill Chapter 4 tables, write Chapter 5
  Codex       → run ablation studies if needed

Phase 4 (Final):
  Claude Code → final code cleanup, generate figures
  Gemini      → proofread, format bibliography, abstract
```

---

## Progress Tracker

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Soft target updates + cosine LR | Claude | DONE | commit 9d66dd4 |
| Dueling DQN fix | Claude | DONE | commit befc795 |
| SON controller improvements | Claude | DONE | commit befc795 |
| Run validation tools | Claude | DONE | commit 2dba3ed |
| Fix pytest environment | Claude | TODO | Python 3.14 missing pytest |
| Implement missing baselines in evaluator | Claude | TODO | random_valid, strongest_rsrp, load_aware, gnn_dqn |
| Implement generate_figures.py | Claude | TODO | Currently a placeholder |
| Multiscenario training (300 eps) | Codex | TODO | Blocked on baseline fix |
| Full evaluation sweep (20 seeds) | Codex | TODO | After training |
| Chapter 1: Introduction | Gemini | TODO | Can start now |
| Chapter 2: Related Work | Gemini | TODO | Can start now |
| Chapter 3: Methodology | Gemini | TODO | Can start now |
| Chapter 4: Results | Gemini | TODO | Blocked on eval data |
| Chapter 5: Conclusion | Gemini | TODO | Blocked on eval data |
| LaTeX tables from CSVs | Gemini | TODO | Blocked on eval data |
| Defense slides narrative | Gemini | TODO | After chapters done |
| Final figures for paper | Claude/Codex | TODO | After eval complete |
| Ablation: with/without SON layer | Codex | TODO | After main eval |
| Code cleanup + final commit | Claude | TODO | Last step |

---

## File Locations

| Artifact | Path |
|----------|------|
| Training config | `configs/experiments/multiscenario_ue.json` |
| Final checkpoint | `results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt` |
| Evaluation CSVs | `results/runs/multiscenario_ue/eval_20seed/*.csv` |
| Figures | `results/runs/multiscenario_ue/figures/` |
| Thesis chapters | `docs/thesis/` (Gemini output) |
| Defense Q&A | `docs/thesis_support.md` |
| Eval report | `docs/SON_GNN_DQN_Evaluation_Report.md` |
