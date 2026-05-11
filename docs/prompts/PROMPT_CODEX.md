# Codex Prompt — Production Training & Evaluation

## Context (Read This First)

You are working on a **5G handover optimization** research project. The system
is called **SON-GNN-DQN** — it uses a Graph Neural Network to learn handover
preferences, then a Self-Organizing Network (SON) safety layer translates
those preferences into bounded 3GPP parameters (CIO/TTT).

The codebase is complete and tested. A diagnostic run (60 episodes) confirmed
the pipeline works but the model needs more training to converge. Your job is
to run the **production training** (300 episodes) and **full evaluation**.

**You are NOT modifying code. You are running the pipeline.**

---

## Environment Setup

```bash
cd /Users/saangetamang/gnn-dqn-handover-loadbalancing

# Install missing packages
pip install pytest pandas matplotlib seaborn

# Verify all deps present
python3 -c "import torch, torch_geometric, numpy, pandas, scipy, matplotlib, seaborn; print('All OK')"

# Set Python path
export PYTHONPATH=src
```

---

## Step 1: Pre-flight Tests

```bash
python3 -m pytest -q
```

**Expected:** All tests pass. If ANY test fails, STOP immediately and report
the failure output. Do not proceed to training with failing tests.

---

## Step 2: Production Training

```bash
python3 scripts/train.py --config configs/experiments/multiscenario_ue.json 2>&1 | tee results/runs/multiscenario_ue/training.log
```

### What this does:
- Trains a GNN-DQN agent for **300 episodes**
- Each episode has **120 decision steps**
- Trains across **7 scenarios**: dense_urban, highway, suburban, sparse_rural,
  overloaded_event, real_pokhara, pokhara_dense_peakhour
- Uses epsilon-greedy exploration: starts at 1.0, decays to 0.03 by episode 250
- Saves resume checkpoints every 25 episodes
- Saves final model as `results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt`

### Expected runtime: 2-8 hours depending on hardware

### What to watch for:
- **Loss should decrease** over the first 100 episodes then stabilize around 0.01-0.5
- **If loss goes NaN:** gradient explosion — report immediately
- **If OOM error:** report the error message
- **If it crashes:** note the last completed episode and resume:

```bash
# Find latest checkpoint
ls -t results/runs/multiscenario_ue/checkpoints/resume/ | head -1

# Resume from it (replace XXX with actual episode number)
python3 scripts/train.py \
  --config configs/experiments/multiscenario_ue.json \
  --resume results/runs/multiscenario_ue/checkpoints/resume/resume_ep0XXX.pt \
  2>&1 | tee -a results/runs/multiscenario_ue/training.log
```

### If training takes more than 10 hours:
Something is wrong. Kill the process and report.

---

## Step 3: Full Evaluation (20 seeds)

After training completes successfully:

```bash
python3 scripts/evaluate.py \
  --checkpoint results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt \
  --out-dir results/runs/multiscenario_ue/eval_20seed \
  --seeds 20
```

### What this does:
- Loads the trained model
- Evaluates **7 methods** on each scenario:
  - `no_handover` — never changes cell (lower bound)
  - `random_valid` — random target (sanity check)
  - `strongest_rsrp` — always pick strongest signal
  - `a3_ttt` — standard 3GPP handover (industry baseline)
  - `load_aware` — heuristic that avoids congested cells
  - `gnn_dqn` — raw RL output without safety (ablation)
  - `son_gnn_dqn` — our full system with SON safety layer
- Uses **20 random seeds** per scenario for statistical confidence
- Produces one CSV per scenario in `results/runs/multiscenario_ue/eval_20seed/`

### Expected runtime: 30-90 minutes

---

## Step 4: Summarize Results

```bash
python3 scripts/summarize_evaluation.py results/runs/multiscenario_ue/eval_20seed --no-fail
```

This runs acceptance gates and prints a table showing whether `son_gnn_dqn`
meets minimum performance thresholds.

---

## Step 5: Quick Result Extraction

```bash
echo "=== KEY COMPARISON: son_gnn_dqn vs a3_ttt ==="
for f in results/runs/multiscenario_ue/eval_20seed/*.csv; do
  echo "--- $(basename $f .csv) ---"
  awk -F',' 'NR==1 {for(i=1;i<=NF;i++) if($i=="avg_ue_throughput_mbps") t=i; if($i=="handovers_per_1000_decisions") h=i; if($i=="pingpong_rate") p=i} NR>1 && ($1=="a3_ttt" || $1=="son_gnn_dqn") {printf "  %-15s thr=%.4f  HO/1k=%.2f  pingpong=%.4f\n", $1, $t, $h, $p}' "$f"
done
```

---

## Step 6: Generate Figures (if implemented)

```bash
python3 scripts/generate_figures.py --run-dir results/runs/multiscenario_ue
```

If this says "not implemented" — that's fine, skip it. Claude Code will
implement it separately.

---

## Deliverables — What To Report Back

### 1. Training Summary
- Did training complete all 300 episodes? (yes/no)
- Copy the **last 20 lines** of training output
- Any warnings or errors?
- How long did it take (wall clock)?

### 2. Full Result Table
Copy the complete output of the summarize_evaluation command (Step 4).

### 3. Key Metrics (copy-paste these numbers)
For each scenario, report `son_gnn_dqn` vs `a3_ttt`:
- Average throughput (Mbps)
- Handovers per 1000 decisions
- Ping-pong rate
- Jain load fairness

### 4. Convergence Check
- Does `son_gnn_dqn` beat `a3_ttt` in throughput in ANY scenario? Which ones?
- Does `son_gnn_dqn` have 0 ping-pong rate everywhere?
- Does raw `gnn_dqn` collapse on `kathmandu_real`? (throughput < 4.0?)

### 5. File Listing
```bash
ls -la results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt
ls results/runs/multiscenario_ue/checkpoints/resume/
ls results/runs/multiscenario_ue/eval_20seed/
```

---

## Success Criteria

The run is SUCCESSFUL if:
- [x] Training completes 300 episodes without crash
- [x] `son_gnn_dqn` throughput >= 98% of `a3_ttt` in ALL scenarios
- [x] `son_gnn_dqn` ping-pong rate = 0 in ALL scenarios
- [x] `son_gnn_dqn` throughput > `random_valid` in ALL scenarios
- [x] All evaluation CSVs contain all 7 methods

The run has BONUS results if:
- [ ] `son_gnn_dqn` throughput > `a3_ttt` by >1% in overloaded_event
- [ ] `gnn_dqn` raw DOESN'T collapse on large topologies (would be great!)
- [ ] `son_gnn_dqn` Jain fairness > `a3_ttt` fairness in dense scenarios

---

## CRITICAL RULES
1. Do NOT modify any source code
2. Do NOT modify any config files
3. Do NOT delete any existing results
4. If anything fails, report the FULL error — don't try to fix code
5. Save ALL output to log files (the `tee` commands above do this)
6. If you need to resume training, use the EXACT resume command shown above
