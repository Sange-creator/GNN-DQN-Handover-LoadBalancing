# Codex Master Plan — Final-Year SON-GNN-DQN Project

> **Audience:** Codex (or any agent) that will re-plan from this document and then execute.
> **Goal:** Ship one focused training run that produces **citable results** plus the supporting bug fixes, evaluation hardening, and defense materials needed to defend a final-year thesis.
> **Why this document exists:** The previous Codex plan (the 8-phase "Step-by-Step Plan To Improve Model Effectiveness" sent on 2026-05-12) contained ~30% hallucinated content. This document corrects those errors, reflects the actual codebase as of 2026-05-12 14:00 NPT, and prioritizes work in a way that fits a tight remaining timeline.
> **2026-05-12 update:** §9 (Performance ambition) and §10 (Episode budget) were added in response to the user's request to maximize the model's win-rate across all scenarios. The earlier sections (§1–§8) are unchanged so previous reviews of the plan remain valid.

## Honest framing of "win in every condition"

The user asked for a model that **significantly outperforms every baseline in every condition** (highway, dense urban, fast highway, etc.). Codex must internalize the following before proceeding:

- Classical 3GPP handover rules (`strongest_rsrp`, `a3_ttt`, `load_aware`) are **strong** in the conditions they were each designed for. `strongest_rsrp` is near-optimal on uncongested layouts; `load_aware` is a greedy optimum for the load-balancing objective; `a3_ttt` is engineered specifically against ping-pong.
- A learned policy beating *all three at once* on every scenario is **possible but not guaranteed**. Recent published work (Salvatori 2026, Eskandarpour 2025) typically reports wins on *stress* scenarios (congestion, mobility, hotspots) and **competitive parity** on normal scenarios — not universal dominance.
- The plan below maximizes the probability of universal dominance. If results show universal dominance, great — make that the headline. If results show "wins on stress + parity on normal," **that is still a strong publishable thesis**, and the canonical run remains citable. Both framings are pre-defined in §2 of this document so the agent does not need to triage in the moment.

---

## 0 · Critique of the previous Codex plan

Codex's earlier plan got the *strategy* mostly right but invented file names and proposed redundant work. Treat the items below as **not actionable** and do not re-introduce them:

| Codex's claim | Reality | Action |
|---|---|---|
| `son_gnn_dqn_true_prb` and `son_gnn_dqn_network_assisted` are new methods to add | `son_gnn_dqn_true_prb` already exists in `default_policy_factories` (metrics/experiment.py:120). The `network_assisted` name is invented; the codebase uses `oran_e2` for the same idea. | **Skip.** No new method aliases. |
| Rename `oran_e2` → `network_assisted` | Cosmetic. `AGENTS.md` is already explicit that O-RAN is the future-work track. | **Skip.** |
| Add `n_step: 3`, `per_alpha: 0.6`, `learning_rate: 1e-4` as new DQN parameters | All three already in `configs/experiments/multiscenario_ue_v2.json` (lines 48-50). | **Skip.** No change. |
| "Make scenarios hard enough" | Already done. `multiscenario_ue_v2.json` includes `coverage_hole`, `highway_fast`, `overloaded_event`, `pokhara_dense_peakhour`. | **Mostly skip;** verify validation_scenarios cover at least one stress scenario. |
| Stop the current bad training | Already stopped on 2026-05-12 12:19; no Python process running. | **Already done.** |

Codex's plan was **correct** about:
- Resume without replay is a real bug (now fixed by Claude on 2026-05-12).
- Validation gating during training is missing (now fixed).
- Drive-test shadow inference is missing.
- ns-3 calibration script is missing.
- Don't claim universal dominance; require significant wins on stress scenarios.
- Lightweight resume checkpoints biased Q-target distribution.

This master plan keeps those correct calls and discards the hallucinated ones.

---

## 1 · Current state (verified 2026-05-12)

**Project root:** `/Users/saangetamang/gnn-dqn-handover-loadbalancing/` (refactored 2026-05-09 from `gnn-dqn-handover/`).

**Code health:**
- Layout: `src/handover_gnn_dqn/{env,models,policies,metrics,rl,son,oran,topology,data,visualization}/`
- 28/28 tests pass (`PYTHONPATH=src python3 -m pytest -q`, runtime ~52s).
- All 7 baseline policies implemented: `no_handover`, `random_valid`, `strongest_rsrp`, `a3_ttt`, `load_aware`, `gnn_dqn`, `son_gnn_dqn`, `son_gnn_dqn_true_prb`.
- SON controller in `src/handover_gnn_dqn/son/controller.py` (245 lines) with bounded CIO/TTT adjustments, rate limiting, rollback.
- Reward function v4 (speed-aware, proactive, P5-protective) in `simulator.py`.

**Recent landing (2026-05-12 by Claude):**
- Bug 1 (resume w/o replay): `checkpoint_include_replay: true` set in config; hard warning emitted in `scripts/train.py`; stale `multiscenario_ue_v2` run annotated `exploratory: true` in `METADATA.json`.
- Bug 2 (validation gating): new `_run_validation_pass` (eval mode, ε=0, separate seeds) in `rl/training.py`; episode loop runs validation at the configured cadence; best-checkpoint now selects on held-out score.
- 4 new tests in `tests/unit/test_validation_gate.py`. All green.

**Known issues remaining** (concrete file:line):
| # | Issue | File | Severity |
|---|---|---|---|
| I-1 | `num_gcn_layers` config field is **dead code** — `gnn_dqn.py:26` declares it; layer construction at `gnn_dqn.py:212-214` hardcodes 3 GCN layers regardless | `src/handover_gnn_dqn/models/gnn_dqn.py` | P2 |
| I-2 | `flat_dqn` baseline is **never trained** by `scripts/train.py` — it appears in `default_policy_factories` only when a `flat_agent` is provided, and no training script trains one | `scripts/train.py`, `src/handover_gnn_dqn/rl/` | P0 (ablation evidence missing) |
| I-3 | Top-level shim files (`src/handover_gnn_dqn/{gnn_dqn,simulator,experiment,scenarios,flat_dqn}.py`) are backward-compat re-exports; harmless clutter | `src/handover_gnn_dqn/*.py` | P3 |
| I-4 | `prepare_drive_data.py` is still a column-checker, not a shadow-inference pipeline | `scripts/prepare_drive_data.py` | P1 |
| I-5 | No automated ns-3 calibration script | `scripts/` | P2 |
| I-6 | Multi-process / multi-seed parallel training not used — single seed wallclock dominates | `scripts/train.py` | P2 (deferred to speedup plan) |
| I-7 | `pre_refactor_2026-05-09` archive in `results/` is correctly flagged but `REPORT.md` (legacy doc) still cites pre-refactor numbers | `REPORT.md` if present | P3 |
| I-8 | Wallclock per 480-episode run is ~40 hours on M1 — too long for the remaining timeline without parallelism | scenario configs, training loop | P1 |
| I-9 | The smoke run revealed that with two equal-tied validation scores, train.py reports the **first** episode as "best" but the loaded model is the **last** one — cosmetic but misleading log line | `scripts/train.py` (best_row selection) | P3 |

**Strategic gap:** the project still lacks a *single, clean, reproducible* overnight run whose results are unambiguously citable. The "one training that makes my model better" framing is exactly the right one.

---

## 2 · Strategy — "One Training To Rule Them All"

We will produce **one** training run whose results are the canonical evidence for the thesis. It will be:

1. **Clean from episode 0** — no resume from any pre-fix checkpoint.
2. **Held-out validation gated** — best-checkpoint chosen on ε=0 validation, never on training-noise.
3. **Replay-preserved** — `checkpoint_include_replay: true` so any later resume is honest.
4. **Multi-scenario** — trained on 8 scenarios; evaluated on training + 4 unseen scenarios.
5. **Replay-paired with a Flat-DQN ablation** trained on the same scenarios (proves the GNN component is what helps).
6. **Run on M1 overnight + optionally mirrored on Kaggle / Oracle Cloud** for redundancy.

The headline claim of the thesis is:
> A SON-GNN-DQN per-UE handover preference model, gated by a safety-bounded A3/CIO SON layer, **matches or beats** classical handover rules on normal scenarios and **significantly outperforms** them on congestion and mobility-stress scenarios, while remaining stable across unseen topologies.

If the trained model cannot defend this claim, the thesis pivots to:
> An honest evaluation framework for learned handover controllers, demonstrating that **classical A3 rules are stronger baselines than commonly assumed**, that **flat-DQN fails on this task**, and that **graph-aware DQN matched against a safety SON layer is the most credible deployment path**.

Either framing is publishable. Both depend on producing the same clean training + evaluation. Pick the framing after results are in.

---

## 3 · Priority-ordered fix list

### P0 — Must land before the canonical training (ship-blocker)

These are the items without which the canonical training cannot produce citable results.

#### P0-a · Train a Flat-DQN baseline in the same pipeline (Issue I-2)
**Why:** The thesis needs the GNN-vs-MLP ablation. Without a flat_dqn baseline trained on the same scenarios, you can't claim "GNN is what helped" — only "DQN with these features helped."

**Implementation:**
1. Add `flat_dqn_episodes` (default 0) to `scripts/train.py` config schema. When > 0, after the GNN-DQN training and before evaluation, train a `FlatDQNAgent` with the same scenario rotation, the same `dqn_cfg`, and the same `validation_*` config.
2. Save the flat agent to `results/runs/<run>/checkpoints/flat_dqn.pt` with proper metadata.
3. Pass `flat_agent=` into `default_policy_factories` during evaluation so the `flat_dqn` row appears in summary CSVs.
4. The flat agent does not need its own validation pass (use the same one as the GNN, but with the flat agent in eval mode).

**Acceptance:**
- New test `tests/unit/test_flat_dqn_baseline.py`: trains a tiny flat agent (4 episodes, 1 scenario) and confirms a checkpoint is saved with `model_class == "FlatDQNAgent"`.
- A run completed with `flat_dqn_episodes: <N>` produces evaluation CSVs containing the `flat_dqn` row.
- `pytest -q` green.

**Estimated effort:** 2-3 hours.

#### P0-b · Verify and fix `num_gcn_layers` dead-code (Issue I-1)
**Why:** A reviewer reading the config + model code will immediately notice the inconsistency. Fix or remove before the canonical run, not after.

**Decision:** *Remove* the config field. Keeping 3 layers hard-coded is the right architectural choice (proven by the existing model's parameter count). Trying to wire `num_gcn_layers` into a `nn.ModuleList` introduces variation that hasn't been tested.

**Implementation:**
1. Remove `num_gcn_layers: int = 3` from `DQNConfig` in `src/handover_gnn_dqn/models/gnn_dqn.py:26`.
2. Remove `"num_gcn_layers": 3` from all configs in `configs/experiments/*.json`.
3. Document in `docs/REPO_LAYOUT.md` that the GCN depth is fixed at 3 layers.

**Acceptance:**
- `pytest -q` green.
- `grep -rn num_gcn_layers .` returns zero hits inside `src/`, `scripts/`, `configs/`.

**Estimated effort:** 30 min.

#### P0-c · Pre-flight integrity check
Run before the canonical training kicks off:

```bash
PYTHONPATH=src python3 -m pytest -q   # 28+ tests pass
PYTHONPATH=src python3 scripts/train.py \
    --config configs/experiments/smoke_validation_gate.json   # smoke completes
PYTHONPATH=src python3 scripts/train.py \
    --config configs/experiments/diagnostic_ue.json \
    --allow-existing-out-dir   # ~1 hour diagnostic, validation runs at least 3 times
```

Diagnostic acceptance: GNN-DQN's `val_holdout_validation_score` at episode 60 should be **above** the same scenario's `strongest_rsrp` score. If not, the canonical training will not converge either — stop and triage reward weights or training cadence before kicking the long run.

---

### P1 — High-value adds that can run in parallel with the canonical training

#### P1-a · Drive-test shadow inference (Issue I-4)
**Why:** Strongest external-validity argument for the defense. Real measurements from real phones.

**Implementation:** rewrite `scripts/prepare_drive_data.py` so its single command does:
1. Parse drive-test CSV (timestamp, UE id, lat/lon, speed, serving cell, neighbor cells, RSRP, RSRQ, optional SINR, optional handover events).
2. For each row, project to the simulator's coordinate system, build a UE-only feature state (set `prb_utilization=0`, `prb_available=0`).
3. Load the trained model from `--checkpoint`.
4. Run the model in eval mode → recommended target cell.
5. Compare against actual serving cell and actual next-handover target.
6. Report: top-1 / top-3 agreement, predicted-handover usefulness (would the recommendation have avoided a known too-late HO?), ping-pong avoidance rate, outage avoidance.
7. Output: `results/runs/<run>/drive_test_validation.csv`.

**Acceptance:**
- A test on synthetic drive-test data produces a non-empty CSV.
- Drive-test top-1 agreement on a *random* policy is ≤ 1/num_cells, on the trained model is > that — sanity check that the script is doing real inference.

**Estimated effort:** 3-4 hours.

#### P1-b · ns-3 calibration KS-test (Issue I-5)
**Why:** Calibration evidence. The simulator must produce distributions close to ns-3 packet-level traces for the results to be credible.

**Implementation:** add `scripts/compare_ns3.py` that:
1. Reads `data/raw/ns3/samples_run_*.csv` (or wherever they live post-refactor).
2. Runs the simulator with matched cell layout and UE count.
3. For each KPI (`avg_ue_throughput_mbps`, `handovers_per_1000_decisions`, RSRP histogram), runs Kolmogorov-Smirnov two-sample test simulator-vs-ns3.
4. Outputs `results/calibration/ns3_ks_report.json` with `{"metric": ..., "ks_stat": ..., "p_value": ..., "interpretation": ...}` rows.
5. Reports any metric where p < 0.01 (distributions significantly differ) as a calibration warning, not as a project failure.

**Acceptance:**
- Script runs end-to-end on the existing 3 ns-3 runs.
- A JSON report is produced.
- Calibration warnings are documented in the thesis as "external validation outcome" (positive or negative is acceptable; the existence of the comparison is what matters).

**Estimated effort:** 1-2 hours.

#### P1-c · Wallclock speedups (Issue I-6, I-8)
**Reference:** all items in `docs/guides/CODEX_TRAINING_SPEEDUP_PLAN.md` (already written). Codex should execute that plan in the order specified there.

**Key constraints repeated here so this doc is self-sufficient:**
- Items A (vectorize state), B (UE-count reduction), D (`set_num_threads`), and H (cheaper validation) together should give ≥2× speedup at low risk.
- Items C (curriculum) and E (early stopping) add another 1.3× combined.
- Item G (`torch.compile`) is opt-in only. Do not enable by default.

**Acceptance:** wallclock for `smoke_validation_gate.json` decreases monotonically as items are landed. Record every measurement in `docs/guides/SPEEDUP_MEASUREMENTS.md`.

**Estimated effort:** 8-10 hours total across all items.

---

### P2 — Strengthen the evidence base (can be done after canonical training starts)

#### P2-a · Inference latency measurement
**Why:** `AGENTS.md` mentions O-RAN near-RT RIC requires inference < 10 ms. The thesis should report this number.

**Implementation:** add a `scripts/measure_inference_latency.py` that loads a checkpoint, builds states for the largest training scenario (20 cells, 250 UEs), runs `agent.act_batch` 1000 times, reports median + P95 + P99 latency.

**Acceptance:** numbers go into the thesis methodology section.

**Estimated effort:** 1 hour.

#### P2-b · 25-cell and 40-cell unseen-topology stress test
**Why:** Confirms topology generalization (the central GNN claim) at scales beyond the training distribution.

**Implementation:** add two synthetic scenarios with hex_grid layouts at 25 and 40 cells to the test_scenarios list of the canonical config. Verify the agent loads and runs without crash.

**Acceptance:** evaluation CSVs include rows for these two scenarios; outage rate stays below 35% (otherwise the model is failing).

**Estimated effort:** 1 hour.

#### P2-c · Generate publication-quality figures
**Reference:** `scripts/generate_figures.py` is already implemented (per MASTER_TRACKER.md, with 6 figures).

**Action:** Run it. Verify it produces:
- Training convergence curve (training score + validation score per episode).
- Comparison bar chart with 95% CI for all 8 methods on the headline scenario.
- CDF of per-UE throughput across methods.
- Network topology map (Pokhara cells).
- Load heatmap before/after GNN-DQN.
- Generalization plot (per-scenario performance, train vs unseen).

**Acceptance:** PNGs and PDFs land in `results/runs/<run>/figures/`. Each is referenced in the thesis chapter outline.

**Estimated effort:** 30 min after training completes.

---

### P3 — Cosmetic / polish (do last, time permitting)

- **P3-a** Remove backward-compat shim files in `src/handover_gnn_dqn/{gnn_dqn,simulator,experiment,scenarios,flat_dqn}.py`. Skip if any test or non-package code still imports them.
- **P3-b** Fix the best-episode tie-breaker in `scripts/train.py` so the reported "Best episode" matches the loaded model.
- **P3-c** Move or delete the legacy `REPORT.md` (numbers from a 12-episode toy run) so it isn't accidentally cited.
- **P3-d** Add a top-level `Makefile` or `Taskfile` with one-liner targets for `make smoke`, `make canonical`, `make figures`, `make eval` — easier to live-demo to the examiner.

---

## 4 · Canonical training playbook

This is the **one command** that produces the citable results. Run it once all P0 items have landed and pre-flight passes.

> **Note (2026-05-12 update):** §10.5 contains the authoritative config snapshot for the canonical run, including the techniques from §9 (behavioral cloning, scenario weights, fine-tuning) and the chosen speedup items. The command in this section remains valid; only the config has been refined.

### Command
```bash
cd /Users/saangetamang/gnn-dqn-handover-loadbalancing
PYTHONPATH=src caffeinate -i python3 scripts/train.py \
    --config configs/experiments/multiscenario_ue_v2.json \
    2>&1 | tee results/runs/multiscenario_ue_v2/canonical.log
```

### Required config additions (already in place after P0)
- `checkpoint_include_replay: true` ✓ (landed by Claude)
- `validation_scenarios: ["suburban", "real_pokhara", "overloaded_event"]` ✓
- `validation_seeds: 5` ✓
- `validate_every_episodes: 20` ✓
- `validation_steps: 60` ✓
- `flat_dqn_episodes: 360` (NEW — add as part of P0-a)
- `early_stopping_patience: 5` (NEW — add as part of P1-c Item E)
- `early_stopping_min_episodes: 240` (NEW)

### Expected duration
- **Before speedups:** ~40 hours wallclock for 480 episodes.
- **After speedups A, B, D, H:** ~12-15 hours.
- **After all speedups including curriculum + early stopping:** ~8-10 hours.

### Acceptance gates (must pass before declaring the run "canonical")

1. **Tests:** `pytest -q` green at start of run.
2. **Validation curve:** Held-out `val_holdout_validation_score` increases monotonically (with smoothing) and plateaus after at least one of (a) ε ≤ 0.05, (b) 60% of total episodes elapsed.
3. **Replay continuity:** Resume checkpoint at episode 60 contains `replay_state` (file size ≥ 50 MB for `multiscenario_ue_v2`).
4. **Headline result:** On the validation scenarios at the best episode, **at least one** of these is true:
   - `son_gnn_dqn` has higher P5 throughput than `strongest_rsrp` AND `a3_ttt`.
   - `son_gnn_dqn` has lower load_std than `strongest_rsrp` AND `a3_ttt`, while throughput is within 5%.
   - `son_gnn_dqn` has at least 50% lower pingpong_rate than `load_aware`, while throughput is within 5%.
5. **Ablation:** `gnn_dqn` outperforms `flat_dqn` on at least 6 of 8 KPIs across training scenarios (proves the GNN component matters).
6. **Generalization:** On unseen `kathmandu_real` and `dharan_synthetic`, `son_gnn_dqn`'s throughput is ≥ 90% of its training-scenario throughput (generalization gap < 10%).
7. **Stability:** `flat_dqn` is allowed to underperform classical baselines (that's actually the desired ablation outcome). It must NOT have outage_rate > 60% (would indicate a training crash).

If gates 1-3 fail: stop and triage. The run is broken, not just underperforming.
If gates 4-6 fail: the headline claim must be reframed (use the backup framing in §2). The run is still citable.

---

## 5 · Defense materials checklist

In parallel with the canonical training, prepare:

- [ ] **Slide 1:** Problem — handover in 5G under congestion, mobility, coverage holes.
- [ ] **Slide 2:** Why GNN-DQN — topology invariance + per-UE granularity + UE-observable features.
- [ ] **Slide 3:** Why SON layer — safety bounds on a learned controller, 3GPP-compatible A3/CIO output.
- [ ] **Slide 4:** Architecture diagram (state → 3-layer GCN → dueling Q-head → SON translator → A3 parameters).
- [ ] **Slide 5:** Training setup table (scenarios, episodes, replay, validation gating).
- [ ] **Slide 6:** Headline results table (the 8 methods × 6 KPIs from the canonical run).
- [ ] **Slide 7:** Convergence curve.
- [ ] **Slide 8:** Per-scenario breakdown — where SON-GNN-DQN wins, where it doesn't, honest framing.
- [ ] **Slide 9:** Topology generalization — Kathmandu and Dharan unseen.
- [ ] **Slide 10:** GNN-vs-Flat ablation.
- [ ] **Slide 11:** Drive-test shadow inference (if P1-a landed).
- [ ] **Slide 12:** ns-3 calibration (if P1-b landed).
- [ ] **Slide 13:** Inference latency vs O-RAN target.
- [ ] **Slide 14:** Limitations + future work (real O-RAN xApp deployment, real PRB via E2 KPM, larger drive-test corpus).

Thesis chapter mapping:
- Chapter 1 (intro): slide 1-3.
- Chapter 2 (related work): existing notes in `docs/references/`.
- Chapter 3 (methodology): slide 4-5; cross-reference `src/handover_gnn_dqn/`.
- Chapter 4 (results): slides 6-13; tables and figures from `results/runs/<run>/figures/`.
- Chapter 5 (conclusion): slide 14.

---

## 6 · Order of operations (Codex roadmap)

Execute in this order. Each step has explicit acceptance; do not proceed if the previous step fails.

| Step | Item | Effort | Blocks |
|---|---|---:|---|
| 1 | Read this document end to end | 15 min | nothing |
| 2 | Read `docs/guides/CODEX_TRAINING_SPEEDUP_PLAN.md` | 10 min | nothing |
| 3 | Read `src/handover_gnn_dqn/rl/training.py`, `scripts/train.py`, `configs/experiments/multiscenario_ue_v2.json` | 20 min | nothing |
| 4 | Land **P0-b** (remove `num_gcn_layers` dead code) — smallest change, builds confidence | 30 min | step 5 |
| 5 | Land **P0-a** (Flat-DQN baseline training in pipeline) | 2-3 hr | step 8 |
| 6 | Land **P1-c speedup items A, B, D, H** from speedup plan (lowest risk, highest impact) | 5-6 hr | step 8 |
| 7 | Run **P0-c pre-flight** (pytest + smoke + diagnostic) | 1 hr | step 8 |
| 8 | Kick off **canonical training** in a `screen` session with `caffeinate -i` | 8-15 hr (background) | step 11 |
| 9 | While training runs in background: land **P1-a** (drive-test shadow inference) | 3-4 hr | none |
| 10 | While training runs in background: land **P1-b** (ns-3 KS-test) | 1-2 hr | none |
| 11 | Verify **acceptance gates** on canonical run; produce figures via `scripts/generate_figures.py` | 1 hr | step 12 |
| 12 | Run **drive-test shadow inference** on the canonical checkpoint | 30 min | step 13 |
| 13 | Run **inference latency measurement** (P2-a) | 30 min | step 14 |
| 14 | Update `docs/SON_GNN_DQN_Evaluation_Report.md` with the canonical numbers | 1 hr | step 15 |
| 15 | Hand off to the thesis-writing agent (Gemini per `MASTER_TRACKER.md`) with figures + tables | 30 min | end |

Total elapsed: ~25-30 hours of active work + ~10-15 hours of background training. Fits comfortably in 3 days if started immediately.

---

## 7 · Forbidden moves

Repeating the most important constraints so they are impossible to miss:

- ❌ Do **not** undo Claude's Bug 1 / Bug 2 fixes from 2026-05-12.
- ❌ Do **not** resume from any checkpoint saved before 2026-05-12 14:00 — those are pre-fix and biased.
- ❌ Do **not** add new method aliases beyond what's already in `default_policy_factories`.
- ❌ Do **not** rename `oran_e2`.
- ❌ Do **not** reduce `hidden_dim`, `replay_capacity`, `n_step`, `per_alpha`, or `epsilon_decay_episodes`.
- ❌ Do **not** edit `REPORT.md` to "update" it with new numbers — it is a legacy artifact, leave it or delete it (P3-c).
- ❌ Do **not** start a long training run without first passing `pytest -q` and the smoke run.
- ❌ Do **not** commit checkpoint files (`.pt`) to git — they belong in `results/runs/<run>/`, which is in `.gitignore`.
- ❌ Do **not** report success without recording wallclock + acceptance-gate outcomes in `docs/SON_GNN_DQN_Evaluation_Report.md`.

---

## 8 · How to report back

After step 15, post a single status block with:

1. **Wallclock:** start, end, total hours.
2. **Acceptance gates:** which of the 7 gates passed; for any that failed, the actual number vs the threshold and a 1-line interpretation.
3. **Top-line numbers:** the headline table (8 methods × 6 KPIs on the headline scenario, plus generalization gap).
4. **Items skipped or reverted:** with reasoning.
5. **Files added or modified:** so the human reviewer can `git diff` quickly.
6. **Outstanding risks:** anything that might fail at the defense.

That's the complete handoff. The human takes it from there.

---

## 9 · Performance ambition — techniques to maximize win-rate across all scenarios

This section addresses the user's explicit request: make the model **significantly outperform every baseline in every condition** (highway, highway_fast, dense_urban, suburban, overloaded_event, coverage_hole, real_pokhara, pokhara_dense_peakhour, sparse_rural, plus unseen kathmandu_real / dharan_synthetic / unknown_hex_grid).

The techniques below are ordered by **expected impact per hour of implementation effort**. Codex should implement them in the order listed. Each item lists the scenarios it most helps with and the rough effort.

### 9.1 · Behavioral-cloning warm-start (HIGHEST IMPACT, P0 of §9)
**What it does:** Before DQN training begins, train the GNN agent to **imitate a teacher policy** for a small number of episodes. Pick the teacher per-scenario:
- For `highway`, `highway_fast`, `sparse_rural` → teacher is `strongest_rsrp` (signal-strength wins in low-congestion mobility).
- For `dense_urban`, `overloaded_event`, `pokhara_dense_peakhour`, `real_pokhara` → teacher is `load_aware` (load balancing wins under congestion).
- For `suburban`, `coverage_hole` → mixed; use `a3_ttt` (stability matters).

This gives the agent a starting policy that **already matches the best baseline per scenario**. DQN then refines on top of that, instead of starting from scratch.

**Implementation:**
1. Add `behavioral_clone_episodes: 40` to `multiscenario_ue_v2.json` (zero by default → backward compatible).
2. Add a `_run_behavioral_cloning(agent, scenarios, teachers_per_scenario, episodes, steps)` function in `src/handover_gnn_dqn/rl/training.py`. For each step, the teacher policy picks the action; the agent's loss is supervised cross-entropy on the teacher's choice.
3. Call this BEFORE the DQN loop in `train_multi_scenario`. After warm-start finishes, the agent's Q-values approximate the teacher's preferences; DQN then improves on them.
4. Save a `bc_warmstart.pt` checkpoint alongside the resume checkpoints so the BC-only policy can be evaluated as an ablation.

**Why this is high impact:** Most DQN failures in this codebase are because the agent spends the first 100+ episodes recovering basic competence (don't pick weak cells, don't ping-pong). Behavioral cloning skips that phase entirely. The agent enters DQN training as a competent baseline and only has to *improve* on it.

**Helps with:** every scenario, but especially `highway` and `sparse_rural` where the agent currently underperforms `strongest_rsrp`.

**Effort:** 3-4 hours.

**Acceptance:** After BC-only warm-start (DQN disabled), evaluation should show `gnn_dqn` within 10% of the teacher policy on its respective scenarios. Add `tests/unit/test_behavioral_cloning.py` to verify.

### 9.2 · Scenario importance sampling (P0 of §9)
**What it does:** Currently the training loop cycles through scenarios uniformly. Replace this with a **weighted sampler** that picks harder scenarios more often.

**Implementation:**
1. Add `scenario_weights` to `multiscenario_ue_v2.json` (default: uniform). Recommended starting weights:
   ```json
   "scenario_weights": {
     "dense_urban": 1.0,
     "highway": 1.2,
     "highway_fast": 1.5,
     "suburban": 0.8,
     "sparse_rural": 0.6,
     "overloaded_event": 1.5,
     "real_pokhara": 1.0,
     "pokhara_dense_peakhour": 1.4
   }
   ```
   Higher weight = more training time = the agent gets more practice on that scenario. Stress scenarios get 1.5×; easy scenarios get 0.6-0.8×.
2. In the episode loop (`training.py:202-207`), replace the round-robin scenario selection with weighted sampling.
3. After every 60 episodes, run a *per-scenario* validation pass and **dynamically reweight**: scenarios where `son_gnn_dqn` is losing badly to the best classical baseline get their weight bumped 1.2×.

**Why this works:** Gives the agent more samples from exactly the scenarios where it currently struggles, without inflating total episode count.

**Helps with:** `highway_fast`, `overloaded_event`, `coverage_hole`.

**Effort:** 2 hours.

**Acceptance:** Log shows scenarios sampled in proportion to their weights (within ±15% across the run).

### 9.3 · Reward shaping audit + per-scenario reward weights (P1 of §9)
**What it does:** Audit the current `user_reward()` in `src/handover_gnn_dqn/env/simulator.py` and tune weights. Specifically:
- The current reward is a single scalar that mixes throughput, load_std, overload, handover_cost, pingpong, outage.
- For `highway_fast`, the handover cost is currently too punitive — fast UEs need to hand over frequently. **Reduce handover penalty when `ue_speed > median_speed`.** (The function already has speed-conditioning; verify weights.)
- For `overloaded_event` and `dense_urban`, increase the `load_std` and `overload` weights.
- For `coverage_hole`, increase the `outage` penalty 2×.

**Implementation:**
1. Add `reward_weights_per_scenario: dict | None` to `LTEConfig`.
2. When constructing an env via `make_env_from_scenario`, pass the per-scenario weights if provided.
3. Default weights stay as-is for backward compatibility.

**Why this works:** A single reward function cannot be optimal across radically different scenarios. Per-scenario weights let the agent learn scenario-appropriate behavior without changing the architecture.

**Helps with:** every scenario, but biggest gains on `highway_fast` (currently over-penalized HOs) and `coverage_hole` (currently under-penalized outages).

**Effort:** 2-3 hours.

**Acceptance:** New test confirms different scenarios produce different reward magnitudes for the same UE state. Episode-end metrics align with the intended objective per scenario.

### 9.4 · Per-scenario fine-tuning after multi-scenario training (P1 of §9)
**What it does:** After the main 480-episode multi-scenario training finishes, save the master checkpoint, then run **20 episodes of fine-tuning per test scenario** with a low learning rate. This produces a *family* of specialized checkpoints, plus the original generalist.

**Implementation:**
1. Add config block:
   ```json
   "fine_tune_per_scenario": {
     "enabled": true,
     "episodes": 20,
     "steps_per_episode": 80,
     "learning_rate": 5e-5,
     "epsilon_start": 0.1,
     "epsilon_end": 0.02
   }
   ```
2. After the main training loop, for each scenario in `train_scenarios + test_scenarios`, copy the master agent, fine-tune for `episodes` episodes, save as `checkpoints/finetuned_<scenario>.pt`.
3. Evaluation reports both:
   - `son_gnn_dqn` (generalist) — primary headline.
   - `son_gnn_dqn_finetuned_<scenario>` — per-scenario specialist (ablation).

**Why this works:** The generalist is the deployable model. The specialists show how much performance is left on the table by not fine-tuning. If the specialists *significantly* outperform the generalist, the thesis discussion can frame this as "production deployment would benefit from periodic on-device fine-tuning" — a credible O-RAN angle.

**Helps with:** the unseen test scenarios where generalization gap is largest.

**Effort:** 3 hours (the loop is mechanical; most effort is in evaluation plumbing).

**Acceptance:** Each finetuned checkpoint exists; evaluation CSVs include rows for both generalist and specialist; specialist's KPIs ≥ generalist's on its own scenario.

### 9.5 · Multi-seed training for variance reduction (P1 of §9)
**What it does:** Train **3 seeds in parallel** using Python's `multiprocessing`. Each seed produces its own checkpoint. At evaluation, ensemble the three policies by majority vote on action.

**Why this works:** RL is high-variance. A single-seed model that "wins" on one seed might lose on another. An ensemble averages out seed-luck.

**Caveat:** Tripples disk usage; needs ~3× RAM if all subprocesses run simultaneously. On M1 Pro with 32GB, 3 parallel seeds is feasible. Reduce to 2 if memory pressure shows up.

**Implementation:**
1. Add `parallel_seeds: 1` config field.
2. New script `scripts/train_parallel.py` that spawns N subprocesses, each running `scripts/train.py` with a different `seed` value.
3. New script `scripts/ensemble_eval.py` that loads N checkpoints and reports both the per-seed and the ensemble metrics.

**Helps with:** statistical strength of the headline claim — the thesis can report "across 3 independent seeds, son_gnn_dqn beat strongest_rsrp on scenario X with p < 0.05 (paired t-test)."

**Effort:** 4 hours.

**Acceptance:** 3 checkpoints land in `results/runs/multiscenario_ue_v2/checkpoints/seed_{42,43,44}/`; ensemble evaluation report shows reduced variance per metric.

### 9.6 · Auxiliary tasks for richer feature learning (P2 of §9, optional)
**What it does:** Add an auxiliary loss that the GCN must also predict next-step throughput. The Q-head still produces Q-values; an additional regression head predicts per-cell `expected_throughput_if_assigned`. Both losses backprop through the shared GCN backbone.

**Implementation:**
1. Add a new head to `GnnDQNAgent` that outputs per-cell throughput prediction.
2. In `_train_step`, add a regression loss against the observed next-step throughputs.
3. Tunable weight `aux_loss_weight: 0.1` in DQNConfig.

**Why this works:** Auxiliary tasks regularize feature learning; the GCN backbone is forced to encode physically meaningful representations, not just Q-values.

**Caveat:** This changes the model interface. Skip if time is tight.

**Effort:** 4-5 hours.

**Acceptance:** Test verifies the new head exists; training still converges; main Q-loss is not destabilized.

### 9.7 · NoisyNet exploration replacing epsilon-greedy (P3 of §9, experimental)
**What it does:** Replace ε-greedy with parametric noise injected into the Q-network weights. Allows state-dependent exploration that ε-greedy cannot represent.

**Implementation:** Replace the final dense layers of the agent with NoisyLinear layers from a small utility module (write inline).

**Why this is P3:** It's high-quality DQN literature (Fortunato et al. 2017), but introduces architectural risk. Skip unless other items land cleanly and time allows.

**Effort:** 3-4 hours.

### 9.8 · What NOT to do for "more performance"
- ❌ **Don't increase `hidden_dim` past 256.** Beyond 256, on this state size, training stalls — empirically observed in prior runs.
- ❌ **Don't use a deeper GCN** (4+ layers). Over-smoothing collapses node embeddings. Stick with 3.
- ❌ **Don't disable the SON layer** "to let the agent be free." The SON layer is the safety mechanism for the *deployable* method `son_gnn_dqn`. Raw `gnn_dqn` is a research baseline only.
- ❌ **Don't change the action space** to "pick a CIO bias." Per-UE handover decisions are the project's contribution; abandoning that is a different paper.
- ❌ **Don't add reward terms that look at evaluation-only KPIs** (e.g., explicit reward for ping-pong). That overfits to the evaluation. The reward already has a pingpong term that is sufficient.

### 9.9 · Honest framing of which items help which scenarios

| Scenario | Currently losing to | Items most likely to help |
|---|---|---|
| `highway` | `strongest_rsrp` | 9.1 (BC warm-start with RSRP teacher), 9.3 (reduce HO penalty for fast UEs) |
| `highway_fast` | `strongest_rsrp`, `a3_ttt` | 9.1, 9.2 (more sampling), 9.3 |
| `dense_urban` | `load_aware` | 9.1 (BC with load_aware teacher), 9.6 (aux throughput prediction) |
| `overloaded_event` | `load_aware` | 9.1, 9.2, 9.3 (stronger load_std weight) |
| `coverage_hole` | `no_handover` | 9.3 (stronger outage penalty), 9.4 (fine-tune) |
| `sparse_rural` | `strongest_rsrp`, `no_handover` | 9.1, 9.4 |
| `suburban` | competitive | minor wins from 9.4 |
| `real_pokhara` | `load_aware` | 9.1, 9.2 |
| `pokhara_dense_peakhour` | `load_aware` | 9.1, 9.2, 9.3 |
| `kathmandu_real` (unseen) | not yet measured | 9.4 (fine-tune); 9.5 (ensemble) |
| `dharan_synthetic` (unseen) | not yet measured | 9.4, 9.5 |
| `unknown_hex_grid` (unseen) | not yet measured | 9.4, 9.5 |

If only **9.1 + 9.2 + 9.3** are implemented (~7-9 hours), the model should win on stress scenarios while staying within 5% of baselines on normal scenarios. That is the realistic strong-thesis outcome. Items 9.4-9.7 are upside.

---

## 10 · Episode budget — concrete recommendation

This section answers the user's direct question: *how many episodes are we running?*

### 10.1 · Current state
- `configs/experiments/multiscenario_ue_v2.json` is configured for **480 episodes × 120 steps**.
- Prior run aborted at episode 144 because of the now-fixed replay bug.
- At ~5 min/episode on M1, 480 episodes ≈ **40 hours wallclock** without speedups.

### 10.2 · Recommended episode budget for the canonical run

**Recommendation: 600 episodes total**, structured in 4 phases. This is up from the current 480 because §9 introduces a behavioral-cloning warm-start that needs its own episode budget, and §9.4 reserves episodes for fine-tuning.

| Phase | Episodes | Purpose | Effective epsilon |
|---|---:|---|---|
| BC warm-start | 40 | Imitate per-scenario teacher policies (§9.1). DQN loss off; supervised cross-entropy. | n/a |
| Exploration | 240 | Standard DQN with high ε. Replay fills up; agent moves from teacher policy to refined policy. | 1.0 → 0.5 |
| Refinement | 240 | Lower ε. Validation gating actively selects best checkpoint. | 0.5 → 0.05 |
| Per-scenario fine-tune | 80 (10 scenarios × 8 ep) | After main loop, fine-tune for each scenario (§9.4). | 0.1 → 0.02 |

**Why 600 not 800 or 1000:** Beyond ~500 episodes of DQN training on this problem, validation curves plateau in the diagnostic runs. The marginal benefit of more episodes is small; the marginal cost in wallclock is large.

### 10.3 · Wallclock budget with speedups

Assuming the speedup items from `CODEX_TRAINING_SPEEDUP_PLAN.md` land (A: vectorized state, D: `set_num_threads(6)`, H: cheaper validation):

| Configuration | Wallclock estimate (M1 Pro) |
|---|---:|
| 600 episodes, no speedups | ~50 hours |
| 600 episodes, items A + D + H | ~18-22 hours |
| 600 episodes, items A + D + H + parallel seeds (§9.5) | ~22-26 hours for 3 seeds total |
| 480 episodes, items A + D + H (minimum acceptable fallback) | ~14-17 hours |

**If wallclock is the binding constraint:** stick with 480 episodes (the existing config) but still implement §9.1 + §9.2 + §9.3. The fine-tuning phase (§9.4) can run as a separate post-hoc job.

**If wallclock is not binding:** go to 600 episodes with the full phase structure above.

### 10.4 · Speedup strategy — what to apply and what to skip

From `docs/guides/CODEX_TRAINING_SPEEDUP_PLAN.md`:

| Speedup item | Apply for canonical run? | Reason |
|---|---|---|
| A — vectorized `build_all_states()` | **YES** | Pure engineering speedup, no semantic change |
| B — reduce UE counts per scenario | **NO** before canonical, MAYBE after | Changes scenario semantics. The autonomous plan you wrote already correctly defers this. Save it for a speed-only ablation after the canonical run lands. |
| C — `steps_per_episode` curriculum | NO for first canonical | Adds a moving target; harder to attribute results. Apply only if a second canonical pass is needed. |
| D — `torch.set_num_threads(6)` | **YES** | One-line change, ~10% gain |
| E — early stopping on val plateau | NO for first canonical | We want the full episode budget so validation curves are complete in the thesis. |
| F — skip val while ε > 0.5 | YES (minor) | Saves a few minutes |
| G — `torch.compile` | NO for first canonical | Opt-in only; PyTorch 2.11 / Python 3.14 risk |
| H — cheaper validation envs (`validation_ue_cap`, `validation_steps_override`) | **YES** | Saves significant validation overhead with no quality loss |

**Net wallclock saving from "YES" items only: ~50-55% off baseline.** That puts a 600-episode run at ~18-22 hours on M1.

### 10.5 · Config snapshot — what `multiscenario_ue_v2.json` should look like for the canonical run

```jsonc
{
  "run_name": "multiscenario_ue_v2",
  "out_dir": "results/runs/multiscenario_ue_v2",
  "feature_mode": "ue_only",
  "prb_available": false,
  "seed": 42,

  "episodes": 600,                           // 9.1+9.4 phase structure inside
  "steps_per_episode": 120,
  "behavioral_clone_episodes": 40,           // §9.1
  "flat_dqn_episodes": 360,                  // P0-a — Flat-DQN ablation
  "fine_tune_per_scenario": {                // §9.4
    "enabled": true,
    "episodes": 8,
    "steps_per_episode": 80,
    "learning_rate": 5e-5
  },

  "eval_seeds": 20,
  "checkpoint_every_episodes": 60,
  "checkpoint_include_replay": true,         // Bug 1 fix

  "validation_scenarios": [
    "highway_fast", "overloaded_event", "real_pokhara"   // stress + production
  ],
  "validation_seeds": 5,
  "validate_every_episodes": 20,
  "validation_steps": 60,
  "validation_ue_cap": 60,                   // Speedup H
  "validation_steps_override": 40,           // Speedup H

  "scenario_weights": {                      // §9.2
    "dense_urban": 1.0, "highway": 1.2, "highway_fast": 1.5,
    "suburban": 0.8, "sparse_rural": 0.6,
    "overloaded_event": 1.5,
    "real_pokhara": 1.0, "pokhara_dense_peakhour": 1.4
  },

  "train_scenarios": [
    "dense_urban", "highway", "highway_fast", "suburban",
    "sparse_rural", "overloaded_event", "real_pokhara", "pokhara_dense_peakhour"
  ],
  "test_scenarios": [
    "kathmandu_real", "dharan_synthetic",
    "unknown_hex_grid", "coverage_hole"
  ],

  "dqn": {
    "hidden_dim": 256,
    "dropout": 0.08,
    "gamma": 0.97,
    "learning_rate": 0.0003,
    "batch_size": 128,
    "replay_capacity": 400000,
    "train_every": 4,
    "target_update_every": 1000,
    "tau": 0.005,
    "grad_clip": 1.0,
    "lr_min": 5e-6,
    "weight_decay": 1e-5,
    "epsilon_start": 1.0,
    "epsilon_end": 0.02,
    "epsilon_decay_episodes": 480,
    "dueling": true,
    "double_dqn": true,
    "n_step": 3,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0
  }
}
```

Note `num_gcn_layers` is intentionally absent (P0-b removed it as dead code).

### 10.6 · Execution order summary

1. Land Bug fixes — done.
2. Land P0-a (Flat-DQN baseline), P0-b (remove `num_gcn_layers`).
3. Land §9.1 (BC warm-start), §9.2 (scenario weights), §9.3 (reward audit).
4. Land speedups A, D, F, H from speedup plan.
5. Pre-flight: `pytest -q` + smoke run + diagnostic run.
6. Kick off canonical training (600 episodes, ~18-22 h wallclock with speedups).
7. While training runs: implement §9.4 (fine-tune), §9.5 (multi-seed) if time, P1-a (drive-test), P1-b (ns-3).
8. Evaluate, summarize, generate figures.
9. Report back per §8.

**Total elapsed time to thesis-ready artifacts: ~3-4 days from now**, assuming the agent can work focused.
