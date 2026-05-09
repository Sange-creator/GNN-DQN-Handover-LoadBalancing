# Training Readiness Master Plan

## Priority

Train `ue_only` first. It matches manual drive-test data and protects the main claim:
topology-invariant SON-compatible handover optimization using UE-observable
measurements. Use `oran_e2` only as a separate demo/future-work path with true or
simulated network-side PRB.

The main deployable method is `son_gnn_dqn`: direct GNN-DQN learns per-UE target
preferences, while the SON layer translates them into bounded CIO/TTT updates for
standard A3 execution.

## Fixed Before Training

- Generated outputs archived under `results/archive_prefix/pre_refactor_2026-05-09/`.
- Source package moved under `src/handover_gnn_dqn/`.
- Config-driven training/evaluation added under `scripts/`.
- Feature profiles added:
  - `ue_only`: no true PRB required.
  - `oran_e2`: true PRB/counters plus explicit availability mask.
- Cell coordinates normalized into the same frame as UE positions.
- Highway UEs spawn and move along a road corridor.
- Ping-pong reward uses pre-action handover history.
- GNN inference supports unseen topology sizes such as 25-cell Kathmandu.
- GNN batching is explicit, so unseen 40-cell graphs are not confused with
  batched 20-cell training data.
- DQN target calculation runs with dropout disabled.
- Checkpoints include feature/profile/model/reward metadata and reject
  incompatible reloads.
- Long runs can emit lightweight resume checkpoints with optimizer, target
  network, and RNG state. Full replay snapshots are opt-in because they become
  very large.
- Evaluation includes a `random_valid` acceptance baseline.
- SON controller added for bounded CIO/TTT parameter updates.
- Evaluation includes `son_gnn_dqn` alongside direct `gnn_dqn`.

## Required Before Full Run

- Manual gate tests pass.
- `smoke_ue` and `smoke_oran` train without NaNs or invalid actions.
- Kathmandu 25-cell evaluation does not crash.
- Urban/suburban/highway scenarios do not show artificial outage from geometry bugs.

## Training Order

1. `python3 scripts/train.py --config configs/experiments/smoke_ue.json`
2. `python3 scripts/train.py --config configs/experiments/smoke_oran.json`
3. `python3 scripts/train.py --config configs/experiments/diagnostic_ue.json`
4. `python3 scripts/train.py --config configs/experiments/multiscenario_ue.json`
5. Evaluate with 20 seeds and treat `son_gnn_dqn` as the deployable method.
6. Run `oran_demo` only after the UE-only result is stable.

Resume a long run only with the same config:

```bash
python3 scripts/train.py \
  --config configs/experiments/multiscenario_ue.json \
  --resume results/runs/multiscenario_ue/checkpoints/resume/resume_ep0025.pt
```

## Defense Narrative

The main model is deployable as a SON optimizer with drive-test/UE-report-visible
measurements via RSRQ-derived load proxy. In an O-RAN/NTC deployment, the same
pipeline can consume real PRB counters through E2/KPM or operator telemetry using
`prb_available=1`.
