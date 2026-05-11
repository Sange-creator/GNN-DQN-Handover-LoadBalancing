# MASTER TRACKER — SON-GNN-DQN Project

> Last updated: 2026-05-10
> Goal: Complete thesis submission with trained model, evaluation results, and written chapters.

---

## What Is The Project?

A 5G handover optimization system with three layers:
1. **GNN-DQN** — learns which cell a user should connect to (the "preference brain")
2. **SON Safety Layer** — translates preferences into safe, bounded 3GPP parameters
3. **Standard A3 Logic** — actual handover uses normal telecom rules with SON-adjusted params

The contribution is `son_gnn_dqn` = GNN-DQN + SON layer working together.

---

## Current State (Honest Assessment)

### What Works
- All code is written and functional (26 Python files in `src/`)
- All 7 baseline policies are implemented and tested
- Evaluation pipeline produces proper CSVs with CI95
- SON controller with safety bounds, rate limiting, rollback — all functional
- Diagnostic run (60 episodes) completed successfully

### What The Diagnostic Shows
The model has NOT converged yet (only 60 eps, tiny network):
- `son_gnn_dqn` matches `a3_ttt` (SON safety defaults to A3 when RL is weak)
- Raw `gnn_dqn` collapses on large topologies (proves SON layer is necessary)
- This is EXPECTED — 60 eps is a sanity check, not a production run

### What's Missing
1. **Production training** — 300 episodes (the `multiscenario_ue` run)
2. **Figure generation** — `scripts/generate_figures.py` is a stub
3. **Python packages** — pytest, pandas, matplotlib, seaborn not installed
4. **Thesis document** — structure exists in `docs/thesis_support.md` but no formal chapters

---

## Three-Agent Strategy

```
┌─────────────────────────────────────────────────────────┐
│  PHASE 1: PREPARATION (Claude Code — NOW)               │
│  Fix environment, implement figures, verify pipeline     │
├─────────────────────────────────────────────────────────┤
│  PHASE 2: EXECUTION (Codex — overnight/long run)        │
│  Run training 300 eps + full evaluation + figures        │
├─────────────────────────────────────────────────────────┤
│  PHASE 3: WRITING (Gemini — parallel with Phase 2)      │
│  Write thesis chapters 1-3 (no results needed)          │
├─────────────────────────────────────────────────────────┤
│  PHASE 4: ASSEMBLY (All three)                          │
│  Claude: verify results, debug                          │
│  Gemini: fill Chapter 4 tables, write Chapter 5         │
│  Codex: ablation runs if needed                         │
└─────────────────────────────────────────────────────────┘
```

---

## Progress Table

### Infrastructure (Claude Code)
| # | Task | Status | Blocker |
|---|------|--------|---------|
| 1 | Core model architecture (GNN-DQN + Dueling head) | DONE | — |
| 2 | SON controller (CIO bounds, TTT, rollback) | DONE | — |
| 3 | All 7 baseline policies | DONE | — |
| 4 | Evaluation pipeline (multi-seed, CSV output) | DONE | — |
| 5 | Summarize script (acceptance gates) | DONE | — |
| 6 | Training convergence fixes (soft targets, cosine LR) | DONE | — |
| 7 | Install missing packages (pytest, pandas, matplotlib, seaborn) | DONE | — |
| 8 | Implement `generate_figures.py` | DONE | 6 figures, tested |
| 9 | Final pre-flight test pass | DONE | 18/18 tests pass |
| 10 | Post-training result verification | TODO | needs Codex results |

### Training & Evaluation (Codex)
| # | Task | Status | Blocker |
|---|------|--------|---------|
| 11 | Run pytest (pre-flight) | READY | env fixed, tests pass |
| 12 | Run multiscenario_ue training (300 eps) | TODO | needs #9 |
| 13 | Run full evaluation (20 seeds, all scenarios) | TODO | needs #12 |
| 14 | Run summarize_evaluation | TODO | needs #13 |
| 15 | Run generate_figures.py | TODO | needs #8, #13 |
| 16 | Report results back | TODO | needs #14 |
| 17 | Ablation: SON layer impact study | STRETCH | needs #13 |

### Thesis Writing (Gemini)
| # | Task | Status | Blocker |
|---|------|--------|---------|
| 18 | Chapter 1: Introduction | TODO | none |
| 19 | Chapter 2: Literature Review | TODO | none |
| 20 | Chapter 3: Methodology | TODO | none |
| 21 | Chapter 4: Results & Discussion | TODO | needs Codex results |
| 22 | Chapter 5: Conclusion & Future Work | TODO | needs #21 |
| 23 | Abstract | TODO | needs #22 |
| 24 | LaTeX formatting + bibliography | TODO | needs #21 |
| 25 | Defense slide narrative | STRETCH | needs #22 |

---

## Key Files & Where Things Live

| What | Path |
|------|------|
| Source code | `src/handover_gnn_dqn/` |
| GNN-DQN model | `src/handover_gnn_dqn/models/gnn_dqn.py` |
| SON controller | `src/handover_gnn_dqn/son/controller.py` |
| Baseline policies | `src/handover_gnn_dqn/policies/policies.py` |
| Simulator/env | `src/handover_gnn_dqn/env/simulator.py` |
| Training loop | `src/handover_gnn_dqn/rl/training.py` |
| Scenarios | `src/handover_gnn_dqn/topology/scenarios.py` |
| Train script | `scripts/train.py` |
| Eval script | `scripts/evaluate.py` |
| Figure script | `scripts/generate_figures.py` (STUB) |
| Summarize script | `scripts/summarize_evaluation.py` |
| Production config | `configs/experiments/multiscenario_ue.json` |
| Diagnostic results | `results/runs/diagnostic_ue/evaluation/*.csv` |
| Production results | `results/runs/multiscenario_ue/` (NOT YET) |
| Thesis support | `docs/thesis_support.md` |
| Eval report | `docs/SON_GNN_DQN_Evaluation_Report.md` |

---

## Expected Results (Realistic)

Based on diagnostic patterns + architecture:

| Method | Expected Throughput vs A3-TTT | Ping-pong | Story |
|--------|-------------------------------|-----------|-------|
| son_gnn_dqn | +1% to +5% | 0% | Safe improvement |
| gnn_dqn (raw) | -5% to -30% on large topo | varies | Proves SON needed |
| load_aware | ~same throughput | 15-35% | Unstable |
| strongest_rsrp | slightly worse | 10-25% | Signal-greedy, unsafe |

**The defensible thesis narrative:**
> son_gnn_dqn provides comparable or better throughput than A3-TTT while
> guaranteeing zero ping-pongs, topology generalization, and bounded safety.
> The improvement is modest in throughput but significant in operational
> reliability and deployability.

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model doesn't converge in 300 eps | No thesis results | Can increase to 500 eps, reduce lr |
| Throughput gain < 1% over A3 | Weak "outperforms" claim | Pivot narrative to safety+generalization |
| Codex environment issues | Delayed training | Provide exact pip install + env setup |
| Large topology collapse persists | Limits generalization claim | Focus on medium topologies, note limitation |

---

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-05-09 | Switched to soft target updates (τ=0.005) | Hard target copies caused instability |
| 2026-05-09 | Added cosine LR schedule | Prevent late-training oscillation |
| 2026-05-09 | Fixed dueling head to use graph-pooled value | Per-node value was wrong architecture |
| 2026-05-10 | Diagnostic confirms pipeline OK | 60 eps too few for convergence, expected |
| 2026-05-10 | Production run = 300 eps mandatory | Need convergence for thesis claims |
