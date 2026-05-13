# Implementation Runbook — Step by Step

> Paste these into a Claude Code terminal session, one phase at a time.
> Stop and inspect output after every phase. Do not skip the verification commands.
> If anything in a phase fails, fix it before moving on — do not proceed past a red signal.

---

## Where we are right now (2026-05-12)

✅ Bug 1 (resume + replay) — landed
✅ Bug 2 (validation gating) — landed
✅ Bug 3 (drive-test) — scaffolded, needs implementation
✅ Bug 4 (ns-3) — not yet started
✅ Codex landed §9 machinery in `src/handover_gnn_dqn/rl/training.py`:
   - `set_num_threads(6)` ✓
   - `_run_behavioral_cloning` ✓
   - `train_flat_multi_scenario` ✓
   - `score_son_against_baselines` ✓
   - `scenario_sampling_weights` param ✓
   - `validation_ue_cap`, `validation_steps_override` ✓
   - `early_stopping_*` ✓

🟡 BUT `scripts/train.py` does NOT yet pass these to `train_multi_scenario`
🟡 AND `configs/experiments/multiscenario_ue_v2.json` does NOT yet enable them

**This runbook closes that gap, then launches the canonical training.**

---

## Phase 0 — Pre-flight (5 min)

```bash
cd /Users/saangetamang/gnn-dqn-handover-loadbalancing

# 1. Make sure nothing is currently training
ps aux | grep -E "train\.py|run_overnight|run_full" | grep -v grep
# Expected: no output (or just the grep itself)

# 2. Make sure git is clean enough that you can revert if needed
git status --short | head

# 3. Verify the tests still pass
PYTHONPATH=src python3 -m pytest -q 2>&1 | tail -5
# Expected: "X passed in Ys"  with X ≥ 28
```

**Stop and inspect.** If pytest fails, paste the failure into a fresh Claude session with:
> "Tests are failing in `pytest -q`. Here is the output: [paste]. Fix the failing tests without changing test intent. Report back when green."

If pytest passes → continue to Phase 1.

---

## Phase 1 — Wire the new config fields into `scripts/train.py` (45 min)

Codex added 7 new parameters to `train_multi_scenario` but the script that loads the config never reads them. We need to bridge that.

**Paste this prompt into a fresh Claude Code session inside the project directory:**

> ```
> Open `scripts/train.py` and `src/handover_gnn_dqn/rl/training.py`.
> The function `train_multi_scenario` now accepts these new parameters that the
> script does NOT currently read from the config:
>
> 1. `validation_ue_cap: int | None`
> 2. `validation_steps_override: int | None`
> 3. `skip_validation_epsilon_above: float | None`
> 4. `early_stopping_min_episodes: int`
> 5. `early_stopping_patience: int`
> 6. `early_stopping_min_delta: float`
> 7. `scenario_sampling_weights: dict[str, float] | None`
> 8. `behavioral_clone_episodes: int`
>
> ALSO `train_flat_multi_scenario` exists for the Flat-DQN ablation but is
> never called.
>
> Tasks:
> A. In `scripts/train.py`, read each of the 8 fields above from `cfg` with
>    sensible defaults (None or 0). Pass them to `train_multi_scenario`.
> B. Add a new config field `flat_dqn_episodes: int = 0`. When > 0, after the
>    GNN training completes, call `train_flat_multi_scenario` with the SAME
>    scenarios, dqn_cfg, validation config, and a sibling out_dir. Save the
>    flat agent at `<out_dir>/checkpoints/flat_dqn.pt`.
> C. Pass `flat_agent=` into the existing call to `default_policy_factories`
>    in the eval phase so the `flat_dqn` row appears in summary CSVs.
> D. Run `PYTHONPATH=src python3 -m pytest -q`. All tests must still pass.
> E. Show me the diff with `git diff scripts/train.py`.
>
> Do NOT modify the model code, the simulator, or the SON controller.
> Do NOT change tests other than to add new ones.
> ```

**Verification after this phase:**
```bash
# Confirm new fields are read
grep -nE 'cfg\.get\("(validation_ue_cap|early_stopping|scenario_sampling|behavioral_clone|flat_dqn_episodes)"' scripts/train.py
# Expected: 6+ matches

# Confirm tests still green
PYTHONPATH=src python3 -m pytest -q 2>&1 | tail -3
```

If green → continue. If red → paste output back to Claude with "fix without scope creep."

---

## Phase 2 — Remove `num_gcn_layers` dead code (15 min)

**Paste into Claude:**

> ```
> The DQN config field `num_gcn_layers` is dead code. The model at
> `src/handover_gnn_dqn/models/gnn_dqn.py:212-214` hardcodes 3 GCN layers
> regardless of this value.
>
> Tasks:
> 1. Remove the `num_gcn_layers: int = 3` line from the `DQNConfig` dataclass.
> 2. Remove all `"num_gcn_layers": 3` entries from every config in
>    `configs/experiments/*.json`.
> 3. Run `grep -rn num_gcn_layers .` and report all remaining matches.
>    The only acceptable matches are inside docs/ explaining that the depth
>    is fixed.
> 4. Run `PYTHONPATH=src python3 -m pytest -q`. Must stay green.
> ```

---

## Phase 3 — Update `multiscenario_ue_v2.json` to the §10.5 canonical config (10 min)

Paste this exact JSON over `configs/experiments/multiscenario_ue_v2.json` (overwrite the file):

```json
{
  "run_name": "multiscenario_ue_v2",
  "out_dir": "results/runs/multiscenario_ue_v2",
  "feature_mode": "ue_only",
  "prb_available": false,
  "seed": 42,

  "episodes": 600,
  "steps_per_episode": 120,
  "behavioral_clone_episodes": 40,
  "flat_dqn_episodes": 360,

  "eval_seeds": 20,
  "checkpoint_every_episodes": 60,
  "checkpoint_include_replay": true,

  "validation_scenarios": ["highway_fast", "overloaded_event", "real_pokhara"],
  "validation_seeds": 5,
  "validate_every_episodes": 20,
  "validation_steps": 60,
  "validation_ue_cap": 60,
  "validation_steps_override": 40,
  "skip_validation_epsilon_above": 0.5,

  "early_stopping_min_episodes": 360,
  "early_stopping_patience": 5,
  "early_stopping_min_delta": 0.5,

  "scenario_sampling_weights": {
    "dense_urban": 1.0,
    "highway": 1.2,
    "highway_fast": 1.5,
    "suburban": 0.8,
    "sparse_rural": 0.6,
    "overloaded_event": 1.5,
    "real_pokhara": 1.0,
    "pokhara_dense_peakhour": 1.4
  },

  "train_scenarios": [
    "dense_urban", "highway", "highway_fast", "suburban",
    "sparse_rural", "overloaded_event", "real_pokhara", "pokhara_dense_peakhour"
  ],
  "test_scenarios": [
    "kathmandu_real", "dharan_synthetic", "unknown_hex_grid", "coverage_hole"
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
    "use_gat": false,
    "gat_heads": 4,
    "n_step": 3,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0
  }
}
```

**Verification:**
```bash
python3 -c "
import json
c = json.load(open('configs/experiments/multiscenario_ue_v2.json'))
assert c['episodes'] == 600
assert c['behavioral_clone_episodes'] == 40
assert c['flat_dqn_episodes'] == 360
assert c['scenario_sampling_weights']['highway_fast'] == 1.5
assert c['validation_ue_cap'] == 60
print('Config valid')
"
```

---

## Phase 4 — Run the smoke test (10 min)

```bash
PYTHONPATH=src python3 scripts/train.py \
    --config configs/experiments/smoke_validation_gate.json \
    --allow-existing-out-dir 2>&1 | tee /tmp/smoke.log

# Verify smoke produced validation scores
grep "val_holdout_validation_score\|VAL ep=" /tmp/smoke.log | head -5

# Verify a best checkpoint was selected
ls -la results/runs/smoke_validation_gate/checkpoints/ 2>/dev/null
```

**Acceptance:** at least one `VAL ep=` line shows a non-zero validation score, AND `best.pt` exists.

If broken → paste `/tmp/smoke.log` (last 100 lines) into Claude:
> "Smoke training failed. Here's the log: [paste]. Diagnose and fix. The config is `configs/experiments/smoke_validation_gate.json`. Do not change the config; fix the code path."

---

## Phase 5 — Diagnostic run (1-2 hours, optional but recommended)

```bash
# Create a diagnostic config (smaller version of canonical)
cat > configs/experiments/diagnostic_canonical.json <<'EOF'
{
  "run_name": "diagnostic_canonical",
  "out_dir": "results/runs/diagnostic_canonical",
  "feature_mode": "ue_only",
  "prb_available": false,
  "seed": 42,
  "episodes": 80,
  "steps_per_episode": 60,
  "behavioral_clone_episodes": 8,
  "flat_dqn_episodes": 40,
  "eval_seeds": 3,
  "checkpoint_every_episodes": 20,
  "checkpoint_include_replay": true,
  "validation_scenarios": ["highway_fast", "overloaded_event"],
  "validation_seeds": 3,
  "validate_every_episodes": 20,
  "validation_steps": 40,
  "validation_ue_cap": 40,
  "validation_steps_override": 25,
  "scenario_sampling_weights": {
    "dense_urban": 1.0, "highway": 1.2, "highway_fast": 1.5,
    "suburban": 0.8, "overloaded_event": 1.5
  },
  "train_scenarios": ["dense_urban", "highway", "highway_fast", "suburban", "overloaded_event"],
  "test_scenarios": ["kathmandu_real"],
  "dqn": {
    "hidden_dim": 256, "dropout": 0.08, "gamma": 0.97,
    "learning_rate": 0.0003, "batch_size": 128, "replay_capacity": 50000,
    "train_every": 4, "target_update_every": 500, "tau": 0.005, "grad_clip": 1.0,
    "lr_min": 5e-6, "weight_decay": 1e-5, "epsilon_start": 1.0, "epsilon_end": 0.05,
    "epsilon_decay_episodes": 60, "dueling": true, "double_dqn": true,
    "n_step": 3, "per_alpha": 0.6, "per_beta_start": 0.4, "per_beta_end": 1.0
  }
}
EOF

# Run diagnostic
PYTHONPATH=src caffeinate -i python3 scripts/train.py \
    --config configs/experiments/diagnostic_canonical.json \
    --allow-existing-out-dir 2>&1 | tee /tmp/diagnostic.log
```

**Acceptance:** At episode 60 of the diagnostic, the latest validation score should be **higher than at episode 20**. If it isn't, the training pipeline is broken or the reward is mis-tuned — debug before launching the 20-hour canonical run.

If validation curve is flat or declining → paste the last 200 lines of `/tmp/diagnostic.log` into Claude:
> "Diagnostic training validation curve is not improving. Log: [paste]. Identify whether this is (a) BC warm-start not enabled, (b) replay not filling, (c) reward saturated, or (d) something else. Suggest concrete fix."

If validation curve improves → continue to Phase 6.

---

## Phase 6 — Launch the canonical training (background, ~18-22 hours)

Open a fresh terminal window. Use `screen` so the run survives disconnects.

```bash
cd /Users/saangetamang/gnn-dqn-handover-loadbalancing

# Move any old multiscenario_ue_v2 outputs out of the way
if [ -d "results/runs/multiscenario_ue_v2" ]; then
  mv results/runs/multiscenario_ue_v2 results/runs/multiscenario_ue_v2.archive_$(date +%Y%m%d_%H%M)
fi

# Start a named screen session
screen -S canonical

# (inside screen) Launch the training
PYTHONPATH=src caffeinate -i python3 scripts/train.py \
    --config configs/experiments/multiscenario_ue_v2.json \
    2>&1 | tee results/runs/multiscenario_ue_v2/canonical.log

# DETACH: Ctrl+A then D
# You can now close the terminal — training keeps running
```

**Plug your laptop into power. Don't close the lid. Don't sleep the machine.**

If you have to leave: run `caffeinate -ds` in another terminal to prevent display sleep too.

**Monitoring:**
```bash
# From any terminal:
tail -f results/runs/multiscenario_ue_v2/canonical.log

# Or just check latest progress:
grep -E "^\[|VAL ep=" results/runs/multiscenario_ue_v2/canonical.log | tail -10

# Reattach to live session:
screen -r canonical
```

---

## Phase 7 — Parallel work while training runs (4-6 hours)

These don't compete with the training for compute (mostly disk/CPU work).

### 7a. Drive-test shadow inference (P1-a, 3-4 h)

**Paste into a fresh Claude session:**

> ```
> Read `scripts/prepare_drive_data.py` and `docs/guides/CODEX_MASTER_PLAN.md` §3 P1-a.
>
> Rewrite `scripts/prepare_drive_data.py` so it does:
> 1. Parse a drive-test CSV with columns: timestamp, ue_id, lat, lon, speed_mps,
>    serving_pci, neighbor_pcis (list), rsrp_dbm, rsrq_db, optional sinr_db,
>    optional handover_event, optional throughput_mbps.
> 2. Build a state vector for each row matching the `ue_only` feature profile
>    (set prb_utilization=0, prb_available=False).
> 3. Load the trained model from `--checkpoint`.
> 4. Run agent.act() in eval mode (epsilon=0) → recommended target_cell.
> 5. Compare to the actual serving_pci and the actual next-handover target.
> 6. Output `results/runs/<run>/drive_test_validation.csv` with columns:
>    timestamp, ue_id, recommended_cell, actual_cell, agreement (bool),
>    predicted_handover_useful (bool), outage_avoided (bool), pingpong_risk (bool).
> 7. Print summary: top-1 agreement rate, top-3 agreement rate, predicted-HO
>    usefulness rate, outage-avoidance rate.
>
> Add `tests/unit/test_drive_shadow.py` that runs the pipeline on synthetic
> drive-test data and confirms output schema. Tests must pass.
>
> Do NOT modify the training pipeline or model.
> ```

### 7b. ns-3 KS-test calibration (P1-b, 1-2 h)

**Paste into Claude:**

> ```
> Create `scripts/compare_ns3.py` per `docs/guides/CODEX_MASTER_PLAN.md` §3 P1-b.
>
> 1. Read `data/raw/ns3/samples_run_*.csv` (or wherever ns-3 data lives —
>    grep the repo to locate it).
> 2. Build a comparable simulator scenario (matched cell count, area, UE count).
> 3. Run the simulator for the same duration to collect distributions of:
>    avg_ue_throughput_mbps, handovers_per_1000_decisions, rsrp histogram,
>    cell load distribution.
> 4. For each metric, run `scipy.stats.ks_2samp` between simulator and ns-3.
> 5. Output `results/calibration/ns3_ks_report.json` with rows:
>    {"metric": ..., "ks_stat": ..., "p_value": ..., "n_sim": ..., "n_ns3": ...}.
> 6. Print a summary; flag any metric where p < 0.01 as "distributions
>    significantly differ — calibration concern."
>
> Add `tests/unit/test_ns3_compare.py` for the JSON schema.
> ```

### 7c. Inference latency measurement (P2-a, 30 min)

**Paste into Claude:**

> ```
> Create `scripts/measure_inference_latency.py`.
> 1. Load a checkpoint (CLI arg --checkpoint).
> 2. Build a 20-cell, 250-UE scenario from `dense_urban`.
> 3. Call `agent.act_batch()` 1000 times on synthetic states.
> 4. Report median, P95, P99 latency in milliseconds.
> 5. Print whether median + P95 are below 10 ms (the O-RAN near-RT RIC target).
> 6. Save `results/runs/<run>/inference_latency.json`.
> ```

---

## Phase 8 — When canonical training finishes (1-2 hours)

```bash
# Check it actually completed
tail -30 results/runs/multiscenario_ue_v2/canonical.log
# Expected: "TOTAL TIME: X.Y hours" or similar completion banner

# Confirm best.pt exists
ls -la results/runs/multiscenario_ue_v2/checkpoints/best.pt

# Run final 20-seed evaluation
PYTHONPATH=src python3 scripts/evaluate.py \
    --checkpoint results/runs/multiscenario_ue_v2/checkpoints/best.pt \
    --out-dir results/runs/multiscenario_ue_v2/eval_20seed \
    --seeds 20 2>&1 | tee /tmp/eval.log

# Generate figures
PYTHONPATH=src python3 scripts/generate_figures.py \
    --run-dir results/runs/multiscenario_ue_v2

# Run drive-test validation against final checkpoint
PYTHONPATH=src python3 scripts/prepare_drive_data.py \
    --drive-test data/drive_test/your_data.csv \
    --checkpoint results/runs/multiscenario_ue_v2/checkpoints/best.pt \
    --out results/runs/multiscenario_ue_v2/drive_test_validation.csv

# Run ns-3 calibration
PYTHONPATH=src python3 scripts/compare_ns3.py \
    --out results/runs/multiscenario_ue_v2/ns3_ks_report.json

# Measure inference latency
PYTHONPATH=src python3 scripts/measure_inference_latency.py \
    --checkpoint results/runs/multiscenario_ue_v2/checkpoints/best.pt
```

---

## Phase 9 — Verify acceptance gates (30 min)

Check the 7 acceptance gates from `CODEX_MASTER_PLAN.md` §4:

```bash
PYTHONPATH=src python3 -c "
import json, csv
from pathlib import Path

run_dir = Path('results/runs/multiscenario_ue_v2')

# Gate 1: pytest green
import subprocess
r = subprocess.run(['python3', '-m', 'pytest', '-q'], capture_output=True, text=True,
                   env={'PYTHONPATH': 'src', **__import__('os').environ})
print('Gate 1 (tests):', 'PASS' if 'passed' in r.stdout and 'failed' not in r.stdout else 'FAIL')

# Gate 2: validation curve increases — check history
import json
hist = json.load(open(run_dir / 'history.json'))
val_scores = [h.get('val_holdout_validation_score', 0) for h in hist if 'val_holdout_validation_score' in h]
if val_scores:
    print(f'Gate 2 (val curve): early={val_scores[0]:.2f}, late={val_scores[-1]:.2f}',
          'PASS' if val_scores[-1] > val_scores[0] else 'FAIL')

# Gate 3: replay continuity
import torch
ckpt = torch.load(run_dir / 'checkpoints' / 'resume_ep0060.pt', map_location='cpu', weights_only=False)
print('Gate 3 (replay):', 'PASS' if 'replay_state' in ckpt else 'FAIL')

# Gate 4: headline win — read eval summary
rows = list(csv.DictReader(open(run_dir / 'eval_20seed' / 'summary.csv')))
by = {r['method']: r for r in rows}
son = by.get('son_gnn_dqn', {})
rsrp = by.get('strongest_rsrp', {})
a3 = by.get('a3_ttt', {})
load = by.get('load_aware', {})
p5_son = float(son.get('p5_ue_throughput_mbps', 0))
p5_rsrp = float(rsrp.get('p5_ue_throughput_mbps', 0))
p5_a3 = float(a3.get('p5_ue_throughput_mbps', 0))
gate4 = p5_son > p5_rsrp and p5_son > p5_a3
print(f'Gate 4 (P5 win): SON={p5_son:.3f}, RSRP={p5_rsrp:.3f}, A3={p5_a3:.3f}',
      'PASS' if gate4 else 'CHECK other gate options')

# Gate 5: ablation
gnn = by.get('gnn_dqn', {})
flat = by.get('flat_dqn', {})
if gnn and flat:
    gnn_thr = float(gnn.get('avg_ue_throughput_mbps', 0))
    flat_thr = float(flat.get('avg_ue_throughput_mbps', 0))
    print(f'Gate 5 (GNN > Flat): GNN={gnn_thr:.3f}, Flat={flat_thr:.3f}',
          'PASS' if gnn_thr > flat_thr * 1.1 else 'FAIL')
"
```

---

## Phase 10 — Hand off to thesis writing

Update `docs/SON_GNN_DQN_Evaluation_Report.md` with:
- Final 8-method × 6-KPI table from `eval_20seed/summary.csv`
- Per-scenario breakdown from the per-scenario CSVs
- Generalization-gap percentages for the 4 test scenarios
- Drive-test shadow-inference summary (if Phase 7a ran)
- ns-3 KS-test results (if Phase 7b ran)
- Inference latency (Phase 7c)

Then: per `docs/MASTER_TRACKER.md`, Gemini handles thesis chapters from this evaluation report.

---

## What to do if something goes wrong mid-training

### Canonical training crashed
```bash
# Get the last few lines of the log
tail -50 results/runs/multiscenario_ue_v2/canonical.log

# Find the latest resume checkpoint
ls -lt results/runs/multiscenario_ue_v2/checkpoints/resume_ep*.pt | head -3

# Resume from it (RESUME WITH REPLAY because Bug 1 is fixed)
PYTHONPATH=src caffeinate -i python3 scripts/train.py \
    --config configs/experiments/multiscenario_ue_v2.json \
    --resume results/runs/multiscenario_ue_v2/checkpoints/resume_ep0060.pt
```

### Training is too slow (running over 24 hours)
1. Detach from screen, leave it running.
2. Open a new terminal.
3. Don't kill it — let early stopping (`patience=5`, kicks in after episode 360) end it gracefully if validation plateaus.

### Validation score is stuck
- The reward function may need re-tuning. Open a Claude session with the latest 30 lines of training log + `src/handover_gnn_dqn/env/simulator.py user_reward` function and ask: "Validation score has not improved for 60 episodes. Suggest one minimal reward-weight change to try." Do NOT change the architecture mid-run.

---

## Total estimated timeline

| Day | Phases | Time |
|---|---|---|
| Day 1 (today) | Phases 0–5 | ~3-4 hours active |
| Day 2 | Phase 6 launches, Phase 7 runs in parallel | ~18-22h training + 4-6h parallel work |
| Day 3 morning | Phase 8 + 9 (eval + gates) | ~2 hours |
| Day 3 afternoon | Phase 10 (handoff to thesis writing) | ~2 hours |

**Total active engineering time: ~10 hours over 3 days.** Most of the wall-time is the canonical training running unattended.

---

## One-line summary

> Wire config → smoke → diagnostic → 600-episode canonical with BC warm-start + scenario weights + Flat-DQN ablation → evaluate → drive-test + ns-3 → hand to thesis writer.
