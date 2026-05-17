# Adaptive SON-Gated GNN-DQN for Cellular Handover Optimization

This repository contains the research and implementation code for a topology-invariant, per-UE handover optimization framework designed for LTE and future 5G/O-RAN networks.

The core contribution of this project is a **Context-Aware Adaptive SON-Gated Architecture**. By employing a 2-signal context gate (evaluating real-time UE Speed and Average Cell Load), the system dynamically routes handover decisions between a Deep Reinforcement Learning agent (GNN-DQN) and a deterministic safety layer (3GPP A3-TTT / SON).

## Architecture & Findings

Machine learning models inherently struggle with the physical dichotomy of cellular networks:
1. **Spatial Congestion:** Requires intelligent load balancing.
2. **Temporal Transition:** Requires stable, seamless connections at high speeds.

Unrestricted ML agents (like a pure GNN-DQN) excel at spatial optimization but experience catastrophic performance degradation during high-velocity mobility or out-of-distribution (OOD) extreme congestion. 

Our Adaptive Architecture achieves **Pareto Optimality**:
* **In Dense Urban (Light Load, Low Speed):** The system grants autonomy to the GNN-DQN, achieving an **18.8% improvement in Jain's Load Fairness** over standard heuristics.
* **In Highway & Extreme Congestion (High Speed or High Load):** The system detects the physical constraints and routes users to the deterministic SON bounds. This completely eliminates ML-induced ping-pong failures (0.0%) and restores throughput to the theoretical maximum.

## Project Structure

This codebase has been rigorously refactored into a config-driven, production-ready state:

```text
configs/experiments/      JSON configs for training, fine-tuning, and scenarios
data/raw/                 OpenCellID, synthetic, and drive-test inputs
docs/reports/             LaTeX generation files, empirical results, and master references
scripts/                  Unified entry points (train.py, evaluate.py)
src/handover_gnn_dqn/     Core package: Simulator, GNN architecture, Policies, RL Loop
tests/                    Unit/integration/regression acceptance tests
results/runs/             Generated runs, models, and CSVs (ignored by git)
archive/                  Legacy scripts and diagnostic tools (ignored by git)
```

## Policy Wrappers (`src/handover_gnn_dqn/policies/policies.py`)
1. **`A3-TTT`**: The deterministic LTE physics standard baseline.
2. **`RAW GNN-DQN`**: The unrestricted AI.
3. **`SON-Tuned A3`**: The blanket safety wrapper.
4. **`Adaptive SON-GNN`**: The deployment champion featuring the 2-Signal routing gate.

## Execution Commands

**1. Run Acceptance Tests**
```bash
PYTHONPATH=src python3 -m pytest -q
python3 scripts/train.py --config configs/experiments/smoke_ue.json
```

**2. Evaluate the Champion Checkpoint**
```bash
python3 scripts/evaluate_targeted.py \
  --checkpoint results/runs/colab_finetune_ue/checkpoints/gnn_dqn.pt \
  --out-dir final_defense_results \
  --seeds 20
```

**3. Base Training**
```bash
python3 scripts/train.py --config configs/experiments/multiscenario_ue.json
```

## Note on Feature Profiles
- `ue_only`: Drive-test compatible. Uses UE-observable radio measurements and RSRQ load proxy.
- `oran_e2`: Future O-RAN/xApp-compatible profile. Adds true PRB utilization when network-side telemetry is available.
