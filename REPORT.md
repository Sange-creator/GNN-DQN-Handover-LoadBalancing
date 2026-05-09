# Superseded Diagnostic Report

This file intentionally does **not** contain final result claims.

Older toy-run and pre-refactor tables were removed because they mixed incompatible
simulator, reward, feature, and training settings. The archived outputs under
`results/archive_prefix/pre_refactor_2026-05-09/` are useful for diagnosis only:
they showed that the old GNN-DQN checkpoint did not reliably beat strong RSRP or
load-aware baselines.

Use fresh runs from `results/runs/` after the current UE-only training pipeline,
explicit GNN batching fix, checkpoint compatibility checks, and SON-GNN-DQN
evaluation path are in place.

Final defense reporting should compare:

- `no_handover`
- `random_valid`
- `strongest_rsrp`
- `a3_ttt`
- `load_aware`
- `gnn_dqn`
- `son_gnn_dqn`

The deployable method is `son_gnn_dqn`; direct `gnn_dqn` is the learned preference
model and research baseline.
