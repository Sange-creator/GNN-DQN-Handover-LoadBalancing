# Defense Training Plan

## Current Decision

Use `ue_only` as the primary thesis path and headline `son_gnn_dqn` as the
deployable method. The raw `gnn_dqn` remains the learned preference baseline.
O-RAN/E2 and true PRB optimization stay as a future-work or demo extension.

## Why the Previous Runs Are Not Final

- `diagnostic_canonical` converged and is useful evidence that the model can
  learn, but `son_gnn_dqn` was still mostly tied with classical A3/RSRP on
  highway and slightly behind on some throughput margins.
- `multiscenario_ue_v2` is diagnostic only because it was resumed without replay
  and was aborted around episode 144.
- The active clean run is `multiscenario_ue_defense`, started from episode 0
  with replay checkpoints enabled and held-out SON validation.

## Code Fixes Before Full Training

- JSON `son_config` is now passed into training validation and final evaluation.
- `scripts/evaluate.py` reloads the saved `son_config` from checkpoint metadata.
- `strongest_rsrp` and `a3_ttt` now mask invalid target cells before selecting a
  handover target.
- Behavioral cloning repairs any impossible teacher label to the UE's current
  serving cell and records `bc_repaired_teacher_targets`.

## Main Training Run

```bash
PYTHONPATH=src GNN_DQN_NUM_THREADS=6 /opt/homebrew/bin/python3 -u \
  scripts/train.py \
  --config configs/experiments/multiscenario_ue_defense.json
```

The run is launched inside a detached `screen` session named
`gnn_dqn_defense`.

Monitor:

```bash
tail -f results/runs/multiscenario_ue_defense/train.log
ps -p "$(cat results/runs/multiscenario_ue_defense/train.pid)" -o pid,etime,%cpu,%mem,command
screen -ls
```

Resume only from replay-including checkpoints:

```bash
PYTHONPATH=src GNN_DQN_NUM_THREADS=6 /opt/homebrew/bin/python3 -u \
  scripts/train.py \
  --config configs/experiments/multiscenario_ue_defense.json \
  --resume results/runs/multiscenario_ue_defense/checkpoints/resume/resume_epXXXX.pt
```

## Training Strategy

- 600 RL episodes with a step curriculum:
  - episodes 1-180: 40 steps
  - episodes 181-380: 80 steps
  - episodes 381-600: 120 steps
- 40 behavioral-cloning warm-start episodes from conservative teachers.
- Post-BC exploration starts at epsilon 0.55, not 1.0, because a fully random
  policy after BC creates unrealistic handover churn and slows the simulator.
- Per-scenario replay buffers with PER and 3-step returns.
- Oversample weak/important regimes:
  - `highway_fast`: 1.7
  - `overloaded_event`: 1.6
  - `pokhara_dense_peakhour`: 1.5
  - `highway`: 1.25
- Keep `sparse_rural` present but lower weight so the model learns coverage
  safety without spending the run on low-density cases.
- Use training-only UE caps for practical wall-clock time while keeping
  validation and final evaluation scenarios uncapped.
- Select best checkpoints using held-out `son_gnn_dqn` validation, not noisy
  training reward.

## Validation Strategy

Held-out validation scenarios:

- `highway_fast`: fast mobility and proactive handover pressure.
- `overloaded_event`: dense area and congestion stress.
- `real_pokhara`: real-cell layout generalization.
- `kathmandu_real`: unseen dense topology.

Validation uses disjoint seeds, capped UEs, and shorter validation episodes so
the run remains practical without losing the important scenario signal.

## Post-Training Evaluation

Run 20-seed evaluation after the checkpoint is saved:

```bash
PYTHONPATH=src /opt/homebrew/bin/python3 scripts/evaluate.py \
  --checkpoint results/runs/multiscenario_ue_defense/checkpoints/gnn_dqn.pt \
  --out-dir results/runs/multiscenario_ue_defense/eval_20seed \
  --seeds 20 \
  --steps 120
```

Summarize gates:

```bash
PYTHONPATH=src /opt/homebrew/bin/python3 scripts/summarize_evaluation.py \
  results/runs/multiscenario_ue_defense/eval_20seed
```

Final tables must include:

- `no_handover`
- `random_valid`
- `strongest_rsrp`
- `a3_ttt`
- `load_aware`
- `gnn_dqn`
- `son_gnn_dqn`

## Acceptance Gates

- Full tests pass.
- Smoke UE defense run completes with finite losses and saved checkpoint.
- Behavioral cloning loss stays finite across sparse rural and highway fast.
- Resume checkpoints include replay.
- `son_gnn_dqn` is not below 98% of A3 average throughput on normal scenarios.
- `son_gnn_dqn` does not increase outage beyond conservative baselines.
- `son_gnn_dqn` improves or matches ping-pong stability on highway scenarios.
- Final claim is based on `son_gnn_dqn`, not raw `gnn_dqn`.

## Defense Framing

State that the framework is SON-compatible, not a live O-RAN xApp. UE-only
features are compatible with drive-test and UE report data. RSRQ is a
UE-observable load/interference proxy, not real PRB utilization. The optional
network-side path can later replace the proxy with true PRB/KPM counters through
O-RAN/E2 without changing the main UE-only training claim.
