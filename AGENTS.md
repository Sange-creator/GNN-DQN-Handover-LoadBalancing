# GNN-DQN Handover Optimization Project

## Current Direction

Build a defense-ready, SON-compatible GNN-DQN handover optimization framework.

Main deployable claim:

> A topology-generalized GNN-DQN learns per-UE handover preferences from
> UE-observable measurements, then a safety-bounded SON layer translates those
> preferences into standard A3/CIO-style handover parameter updates.

Use **UE_ONLY** as the primary training and paper path. Use **ORAN_E2** only as a
future-work/demo profile.

## Non-Negotiable Choices

- Main feature profile: `ue_only`
- Main deployable method: `son_gnn_dqn`
- Direct `gnn_dqn`: learned preference model and research baseline
- O-RAN/PRB: separate demo/future extension, not the main headline result
- Old checkpoints/results: diagnostic only, not citable
- Missing PRB in drive-test data: `prb_utilization = 0`, `prb_available = 0`
- RSRQ load proxy must never be described as real PRB utilization

## Active Project Structure

```text
configs/experiments/      JSON experiment configs
data/raw/                 OpenCellID, ns-3, drive-test inputs
data/processed/           Generated/cleaned data, ignored by git
docs/                     Guides and paper/defense notes
scripts/                  Config-driven train/evaluate/demo entrypoints
src/handover_gnn_dqn/     Reusable package code
tests/                    Unit and integration acceptance tests
results/archive_prefix/   Pre-refactor diagnostic outputs
results/runs/             New generated runs, ignored by git
```

## Codebase Status

Implemented or hardened:

- `src/` package layout
- Config-driven `scripts/train.py` and `scripts/evaluate.py`
- UE-only and O-RAN/E2 feature profiles
- Explicit PRB availability mask
- Coordinate normalization for real/synthetic topologies
- Highway road-corridor mobility
- Event clustered mobility
- Reward ping-pong timing based on pre-action history
- Explicit GNN batching so unseen topology sizes do not depend on `max_cells`
- DQN target inference with dropout disabled
- Checkpoint metadata/compatibility validation
- Lightweight resume checkpoints for long training; full replay snapshots are
  opt-in because they are large
- `random_valid` baseline
- SON controller and `son_gnn_dqn` tuned-A3 evaluation path

## Training Order

Run in this order:

```bash
PYTHONPATH=src python3 -m pytest -q
python3 scripts/train.py --config configs/experiments/smoke_ue.json
python3 scripts/train.py --config configs/experiments/smoke_oran.json
python3 scripts/train.py --config configs/experiments/diagnostic_ue.json
python3 scripts/train.py --config configs/experiments/multiscenario_ue.json
```

Resume only with the same config:

```bash
python3 scripts/train.py \
  --config configs/experiments/multiscenario_ue.json \
  --resume results/runs/multiscenario_ue/checkpoints/resume/resume_ep0025.pt
```

Evaluate:

```bash
python3 scripts/evaluate.py \
  --checkpoint results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt \
  --out-dir results/runs/multiscenario_ue/eval_20seed \
  --seeds 20
```

## Acceptance Gate Before Full Training

- Unit/integration tests pass.
- UE_ONLY and ORAN_E2 smoke runs complete without NaNs or invalid actions.
- Kathmandu-style 25-cell and larger 40-cell inference work.
- Checkpoint reload rejects incompatible feature profiles.
- `random_valid` baseline is present in evaluation.
- Short diagnostic GNN-DQN beats random before any long run is trusted.
- Normal urban/suburban/highway scenarios do not show artificial outage.

## Results Policy

Pre-refactor outputs under `results/archive_prefix/pre_refactor_2026-05-09/`
showed that old GNN-DQN checkpoints were not reliably stronger than RSRP or
load-aware heuristics. Treat them as diagnostic evidence for why the refactor was
necessary.

Final tables must include:

- `no_handover`
- `random_valid`
- `strongest_rsrp`
- `a3_ttt`
- `load_aware`
- `gnn_dqn`
- `son_gnn_dqn`

Headline the deployable method as `son_gnn_dqn`, not raw direct GNN-DQN.

## Defense Narrative

- The current system is SON-compatible, not a live O-RAN xApp.
- UE_ONLY features match drive-test and UE-report data.
- RSRQ is used as a UE-observable load/interference proxy.
- The GNN supports different cell counts and topologies.
- The SON layer converts learned preferences into bounded CIO/TTT updates.
- O-RAN/E2 can later add real PRB utilization and network KPIs.
