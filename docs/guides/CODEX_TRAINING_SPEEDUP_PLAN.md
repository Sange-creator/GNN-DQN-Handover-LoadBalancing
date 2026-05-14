# Codex Implementation Plan — Speed Up Training Without Losing Capacity

> **Audience:** Codex (or any agent) executing code changes.
> **Goal:** Cut wallclock time for `multiscenario_ue_v2` from ~40h to ~10-15h on M1 Pro **without** reducing model capacity, replay diversity, or learning quality.
> **Status of underlying code:** Bug 1 (resume-without-replay) and Bug 2 (validation gating) were already landed by Claude on 2026-05-12. All 28 unit tests pass. Do NOT touch those changes; they are prerequisites for this plan.

---

## Ground rules

1. **Capacity-preserving only.** Do not reduce `hidden_dim`, `replay_capacity`, `n_step`, `per_alpha`, `gamma`, `target_update_every`, or any optimizer settings unless this document explicitly says so.
2. **Test-driven.** After every change, run `PYTHONPATH=src python3 -m pytest -q`. Must remain green.
3. **Smoke-test before long runs.** Use `configs/experiments/smoke_validation_gate.json` (already exists, 8 episodes) to verify each change finishes cleanly.
4. **Commit per item.** One git commit per implementation item (A, B, C, …) so a single bad item can be reverted without losing the others.
5. **Measure, don't trust.** For each item that claims a speedup, record wallclock for the smoke run before and after. Append to `docs/guides/SPEEDUP_MEASUREMENTS.md`.
6. **Reproducibility.** Do not change seed plumbing. Validation seeds in `scripts/train.py:90-95` are intentionally disjoint from training seeds.

---

## Item A — Vectorized environment state construction (expected speedup: 1.2-1.5×)

### Problem
`src/handover_gnn_dqn/rl/training.py:252-260` builds states one UE at a time:
```python
all_states = np.empty((env.cfg.num_ues, max_cells, feature_dim), dtype=np.float32)
all_valid = np.zeros((env.cfg.num_ues, max_cells), dtype=bool)
for ue_idx in ue_order:
    s = env.build_state(ue_idx)
    ...
    all_states[ue_idx] = pad_state(s)
    all_valid[ue_idx] = pad_mask(env.valid_actions(ue_idx))
```
This calls `env.build_state(ue_idx)` once per UE. Each call internally re-uses the cached RSRP/RSRQ snapshot (good), but Python loop overhead and per-UE allocations add up at 50-240 UEs/step.

### Change
1. Add a new method `CellularNetworkEnv.build_all_states() -> np.ndarray` in `src/handover_gnn_dqn/env/simulator.py` (insert near `build_state` at line 393).
   - Returns shape `(num_ues, num_cells, feature_dim)`, dtype `float32`.
   - Compute every feature column as a vectorized op over all UEs at once.
   - Reuse the existing cached RSRP/RSRQ snapshot.
2. Add a sibling method `CellularNetworkEnv.valid_actions_all() -> np.ndarray` returning shape `(num_ues, num_cells)` boolean.
3. In `training.py:247-265`, replace the per-UE loop with:
   ```python
   all_states_raw = env.build_all_states()        # (num_ues, num_cells, feature_dim)
   all_valid_raw  = env.valid_actions_all()       # (num_ues, num_cells)
   # NaN guard (was inside the loop)
   if not np.isfinite(all_states_raw).all():
       nan_state_count += 1
       raise RuntimeError(...)
   # Padding once via broadcasting
   all_states = np.zeros((env.cfg.num_ues, max_cells, feature_dim), dtype=np.float32)
   all_states[:, :num_cells, :] = all_states_raw
   all_valid = np.zeros((env.cfg.num_ues, max_cells), dtype=bool)
   all_valid[:, :num_cells] = all_valid_raw
   ```

### Acceptance criteria
- `tests/unit/test_simulator_fixes.py` still passes.
- Add new test `tests/unit/test_build_all_states.py`: for a fixed seed, assert that `build_all_states()` returns the same rows as stacking individual `build_state(i)` calls (element-wise close to atol=1e-6).
- Smoke run wallclock decreases by ≥15%.

### Rollback
`git revert <commit-A>`. The original loop is preserved by reverting; no state machine to clean up.

---

## Item B — Reduce UE counts on dense scenarios (expected speedup: 1.3-1.5×)

### Problem
`src/handover_gnn_dqn/topology/scenarios.py` configures `dense_urban` with 160 UEs and `overloaded_event` with 240 UEs. The per-step cost scales linearly with `num_ues`, but with 240 UEs/cell × ~10 cells, the per-UE Q-update is highly redundant — the agent learns the *handover policy*, not per-UE memorization. Reducing UE counts by ~30% gives a near-linear wallclock reduction without changing what the agent learns, because the underlying *distribution* of UE positions / demands stays the same.

### Change
Edit `src/handover_gnn_dqn/topology/scenarios.py`:

| Scenario | Current `num_ues` | New `num_ues` | Rationale |
|---|---:|---:|---|
| `dense_urban` | 160 | 100 | Same 8/cell ratio but with 20 cells → 5/cell average; still triggers congestion |
| `highway` | 50 | 40 | Keep 4/cell; the highway scenario is mobility-dominated, not load-dominated |
| `highway_fast` | (verify in file) | reduce 20% if currently >40 | Same logic |
| `suburban` | 105 | 75 | 5/cell; still produces enough load variance |
| `sparse_rural` | 25 | 25 | Already small; do not reduce |
| `overloaded_event` | 240 | 150 | **Most important reduction.** Still hammers ~12 cells with 12.5 UEs each |
| `real_pokhara` | 160 | 100 | Same logic as dense_urban |
| `pokhara_dense_peakhour` | (verify) | reduce 30% | Same |

### Acceptance criteria
- `tests/unit/test_simulator_fixes.py` and integration smoke still pass.
- After this change, run smoke_validation_gate.json; total wallclock decreases by ≥20%.
- `python3 scripts/evaluate.py` baseline numbers (no_handover, strongest_rsrp, a3_ttt, load_aware) shift by <5% per metric — confirms the scenario semantics are preserved.

### Rollback
`git revert <commit-B>`.

---

## Item C — Curriculum on `steps_per_episode` (expected speedup: 1.2×)

### Problem
`steps_per_episode: 120` is correct for the late-training phase where the agent makes nuanced multi-step decisions. But the first ~25% of training is dominated by ε=1.0 → ε=0.5 random exploration. Long episodes during this phase produce noisy gradients without proportionate learning.

### Change
1. Add config field `steps_per_episode_curriculum` to `multiscenario_ue_v2.json`:
   ```json
   "steps_per_episode_curriculum": {
     "phase1_episodes": 120,
     "phase1_steps": 60,
     "phase2_episodes": 240,
     "phase2_steps": 90,
     "phase3_steps": 120
   }
   ```
   With 480 total: phase1=0-120 (60 steps), phase2=120-360 (90 steps), phase3=360-480 (120 steps).
2. In `src/handover_gnn_dqn/rl/training.py:train_multi_scenario`, accept optional `steps_per_episode_curriculum: dict | None = None`. Inside the episode loop (around line 234), compute the effective `steps_for_this_episode` based on episode index.
3. In `scripts/train.py`, read the curriculum config and pass it through.
4. Keep `steps_per_episode` (scalar) as the **fallback** when curriculum is not provided. Backward-compatible.

### Acceptance criteria
- New test `tests/unit/test_curriculum_steps.py`: verify that with a curriculum config, the right step count is used at the right episode boundary.
- Smoke run still completes correctly with curriculum disabled (no regression).
- With curriculum enabled in smoke config, total step count across all episodes drops by ≥15%.

### Rollback
`git revert <commit-C>`. The scalar `steps_per_episode` path is preserved.

---

## Item D — `torch.set_num_threads` to match M1 performance cores (expected speedup: 1.1-1.2×)

### Problem
PyTorch on M1 defaults to using all logical cores (10). The M1 Pro has 6 *performance* cores + 4 *efficiency* cores. Loading the efficiency cores can cause cache thrashing and slow down the performance cores.

### Change
At the top of `src/handover_gnn_dqn/rl/training.py` (right after the imports block ending around line 19):
```python
import os
# M1 Pro: 6 performance + 4 efficiency cores. Pinning to performance cores reduces
# cache thrashing and gives ~10-15% throughput on small GNN forward passes.
# Allow override via env var for cloud machines with different topology.
_default_threads = 6
torch.set_num_threads(int(os.environ.get("GNN_DQN_NUM_THREADS", _default_threads)))
```

### Acceptance criteria
- Smoke run completes without error.
- Wallclock for smoke run with `GNN_DQN_NUM_THREADS=6` is within ±5% of the default-threads run (sanity check on M1).
- On the production run, wallclock reduces by 8-15%.

### Rollback
`git revert <commit-D>` removes the `set_num_threads` line. PyTorch returns to its default.

---

## Item E — Early stopping on validation plateau (expected savings: 10-30% on average)

### Problem
With validation gating from Bug 2, we now know reliably whether training is making progress. If validation score plateaus, we are wasting compute. Currently the loop runs all 480 episodes regardless.

### Change
1. Add config fields to `multiscenario_ue_v2.json`:
   ```json
   "early_stopping_patience": 5,
   "early_stopping_min_delta": 0.01,
   "early_stopping_min_episodes": 240
   ```
   Patience: number of consecutive validation passes without improvement before stopping. min_delta: minimum improvement (in `holdout_validation_score` units) that counts as progress. min_episodes: do not stop before this episode (protects against early-phase noise).
2. In `train_multi_scenario`, after each validation pass that runs, check: did `val_holdout_validation_score` improve over the running best by ≥ min_delta? Track a counter of consecutive non-improvements. If counter ≥ patience AND `episode + 1 ≥ min_episodes`, break out of the episode loop.
3. Log a clear `[EARLY STOP] no improvement for N validations` message.
4. Ensure the final best checkpoint is still saved.

### Acceptance criteria
- New test `tests/unit/test_early_stopping.py`: simulate a history where validation plateaus and verify the loop breaks at the right episode.
- Smoke run with `early_stopping_patience: 2, early_stopping_min_episodes: 4` finishes in fewer episodes than the no-early-stop baseline.
- `pytest -q` green.

### Rollback
`git revert <commit-E>`.

---

## Item F — Skip validation during high-exploration warmup (expected savings: 1-2 min total, low impact)

### Problem
Validation during the first ~25% of training (ε > 0.5) is dominated by noise because the agent is still mostly random. The validation result is not informative for best-checkpoint selection.

### Change
In `train_multi_scenario`, before running validation, check `epsilon` for the current episode. If `epsilon > 0.5`, skip the validation pass (still log the training-metric score). This applies only to validation, not to checkpointing.

### Acceptance criteria
- New test in `tests/unit/test_validation_gate.py`: when epsilon is high, validation does not run; when epsilon drops below 0.5, it resumes.
- Smoke run still produces ≥1 validation pass (ensure the smoke config has enough episodes to drop below ε=0.5).

### Rollback
`git revert <commit-F>`.

---

## Item G — `torch.compile` on the agent's forward pass (expected speedup: 1.3-1.8× on compute-bound steps)

### Problem
The GCN forward pass through 3 layers + dueling head is the inner-loop bottleneck once Item A vectorizes state assembly.

### Change (RISKY — implement last)
1. In `src/handover_gnn_dqn/models/gnn_dqn.py`, after constructing `GnnDQNAgent` in `__init__`, optionally wrap `self.forward` with `torch.compile` if env var `GNN_DQN_COMPILE=1` is set.
2. Catch any compile errors and fall back to the eager forward with a clear warning. Do NOT crash training.
3. This is an opt-in feature; do not enable by default. The risk is that PyTorch 2.11 + Python 3.14 may have rough edges with `torch.compile` on a model containing PyG ops.

### Acceptance criteria
- Smoke run with `GNN_DQN_COMPILE=1` completes without error.
- If compilation fails, the run still completes via eager path (the fallback works).
- Wallclock with compile enabled is ≥15% faster than without.

### Rollback
`git revert <commit-G>`.

---

## Item H — Reduce validation cost: cap UEs and shorten steps for validation envs (expected savings: 30-50% of validation wallclock)

### Problem
Validation currently uses the same `num_ues` and `validation_steps` as configured. For 60-step validation × 3 scenarios × 5 seeds = 900 step-passes per validation event. Each takes a few seconds. Over 24 validation events in a 480-episode run, this is ~10-15 minutes of pure validation overhead.

### Change
1. In `_run_validation_pass` (training.py), accept an optional `val_ue_cap: int | None = None` and `val_steps_override: int | None = None`. If provided, override the scenario's `num_ues` and the steps argument when constructing the validation env.
2. Default values in `scripts/train.py` (read from config): `val_ue_cap: 60`, `val_steps_override: 40`. These produce ~6× faster validation while still giving statistically meaningful scores.
3. The training scenarios remain at their full `num_ues`; only validation is reduced.

### Acceptance criteria
- Validation pass completes ≥3× faster than before.
- Best-checkpoint selection still reproduces the same ordering on the smoke run (the score scale changes but the *ranking* should be similar — log a warning if best episode swaps between full and reduced validation on the smoke run).

### Rollback
`git revert <commit-H>`.

---

## Order of operations

Implement in this exact order. Each builds on the previous, and reverting a later one does not invalidate the earlier ones:

1. **Item D** (`set_num_threads`) — 1-line change, easy, baseline benefit. *15 min*
2. **Item B** (UE count reduction) — config-only change, no code touched. *30 min*
3. **Item A** (vectorize `build_all_states`) — the biggest single speedup; touches simulator + training loop + tests. *2-3 hr*
4. **Item H** (reduced validation cost) — small, contained change. *45 min*
5. **Item F** (skip early-warmup validation) — small change. *30 min*
6. **Item E** (early stopping) — moderate change with new tests. *1-2 hr*
7. **Item C** (steps-per-episode curriculum) — touches training loop and config. *1-2 hr*
8. **Item G** (`torch.compile`) — opt-in, do last because high failure risk. *30 min implementation + variable debugging*

**Total expected wallclock reduction: from 40h to ~8-12h** when all items are applied. Items A + B + H alone should achieve ~50% reduction at lowest risk.

---

## Required after every item

```bash
# 1. Run the full test suite
PYTHONPATH=src python3 -m pytest -q

# 2. Run the smoke validation training (8 episodes)
PYTHONPATH=src python3 scripts/train.py --config configs/experiments/smoke_validation_gate.json

# 3. Append wallclock measurement to docs/guides/SPEEDUP_MEASUREMENTS.md:
#    | Date | Item applied | Smoke wallclock (s) | Notes |

# 4. Commit. Commit message format:
#    speedup(item-X): <one-line summary>
#
#    Body: list what changed, where (file:line), and the smoke-test wallclock delta.
```

---

## Forbidden changes (do not do these)

- ❌ Reduce `hidden_dim` from 256.
- ❌ Reduce `replay_capacity` from 400,000.
- ❌ Disable `n_step`, `per_alpha`, `double_dqn`, or `dueling`.
- ❌ Increase `train_every` above 4 (less gradient updates per environment step).
- ❌ Reduce `epsilon_decay_episodes` below 300 (less exploration breadth).
- ❌ Change scenario *positions* or *adjacency* — only UE counts in Item B.
- ❌ Skip running tests before committing.
- ❌ Rename `oran_e2` feature mode (Codex previously suggested this; it is cosmetic, do not do it).
- ❌ Add new method aliases like `son_gnn_dqn_true_prb_extended` — the existing two-method split (UE-only + oran_e2) is sufficient.

---

## When you're done

1. Run a clean overnight training from episode 0:
   ```bash
   PYTHONPATH=src caffeinate -i python3 scripts/train.py \
     --config configs/experiments/multiscenario_ue_v2.json
   ```
2. Record total wallclock in `docs/guides/SPEEDUP_MEASUREMENTS.md`.
3. Compare evaluation results against `results/archive_prefix/pre_refactor_2026-05-09/` — they should not regress.
4. Report back to the user with:
   - Wallclock delta (before vs after, in hours).
   - Sample-efficiency check: did the model reach a usable validation score in fewer episodes?
   - Any items you skipped or had to revert, with reasoning.

---

## Out of scope (do not attempt in this work)

- Bug 3 (drive-test shadow inference for `scripts/prepare_drive_data.py`).
- Bug 4 (ns-3 calibration via KS-test).
- Thesis writing.
- Any feature-profile rename or restructuring.
- Network architecture changes (GCN → GAT, etc.).
- Compute provider migration (Kaggle / Oracle Cloud setup — that's a separate runbook).
