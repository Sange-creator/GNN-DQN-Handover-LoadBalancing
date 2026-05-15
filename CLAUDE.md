# GNN-DQN Handover Optimization Notes

## Current Project Frame

The project is now framed as:

> SON-GNN-DQN: a topology-generalized GNN-DQN preference model with a
> safety-bounded SON translation layer for A3/CIO-style handover optimization.

The main path is **UE_ONLY**. O-RAN/E2 is a separate future/demo path.

## What Is Citable

Only use results generated after the current training-readiness fixes:

- explicit GNN batching
- fixed topology normalization
- fixed reward ping-pong timing
- checkpoint compatibility metadata
- `random_valid` baseline
- `son_gnn_dqn` evaluation
- UE_ONLY feature profile

Archived outputs under `results/archive_prefix/pre_refactor_2026-05-09/` are
diagnostic only. They showed the old model was weak and should not be used as
final evidence.

## Feature Profiles

`ue_only`:

- RSRP
- RSRQ
- RSRQ-derived load proxy
- RSRP/RSRQ trend features
- serving-cell indicator
- signal usability
- UE speed
- time since last handover
- previous serving-cell indicator
- no real PRB dependency

`oran_e2`:

- all UE_ONLY features
- real/simulated PRB utilization
- `prb_available`
- connected UE count
- cell throughput counter

For drive-test data without PRB, use:

```text
prb_utilization = 0
prb_available = 0
```

Never call RSRQ-estimated load real PRB utilization.

## Training Sequence

```bash
PYTHONPATH=src python3 -m pytest -q
python3 scripts/train.py --config configs/experiments/smoke_ue.json
python3 scripts/train.py --config configs/experiments/smoke_oran.json
python3 scripts/train.py --config configs/experiments/diagnostic_ue.json
python3 scripts/train.py --config configs/experiments/multiscenario_ue.json
```

The long run can be resumed only with the exact same config. Resume checkpoints
are lightweight by default; full replay snapshots are opt-in because they are
large.

```bash
python3 scripts/train.py \
  --config configs/experiments/multiscenario_ue.json \
  --resume results/runs/multiscenario_ue/checkpoints/resume/resume_ep0025.pt
```

## Evaluation Methods

Final comparisons should include:

- `no_handover`
- `random_valid`
- `strongest_rsrp`
- `a3_ttt`
- `load_aware`
- `gnn_dqn`
- `son_gnn_dqn`

The main deployable method is `son_gnn_dqn`. Direct `gnn_dqn` is the learned
preference policy and research baseline.

## Main Commands

```bash
python3 scripts/train.py --config configs/experiments/smoke_ue.json
python3 scripts/train.py --config configs/experiments/diagnostic_ue.json
python3 scripts/train.py --config configs/experiments/multiscenario_ue.json
python3 scripts/evaluate.py \
  --checkpoint results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt \
  --out-dir results/runs/multiscenario_ue/eval_20seed \
  --seeds 20
```

Compatibility wrappers:

```bash
python3 run_experiment.py
python3 run_overnight.py
python3 run_full_training.py
```

## Defense Talking Points

- This is a SON-compatible system today, not a live O-RAN deployment.
- The model uses UE-observable measurements, so it is compatible with drive-test
  validation.
- RSRQ acts as a practical load/interference proxy when PRB is unavailable.
- GNN inference is topology-size tolerant and works across cell counts.
- The SON layer translates per-UE preferences into bounded CIO/TTT updates.
- Future O-RAN work can add real PRB and E2/KPM counters through `oran_e2`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
