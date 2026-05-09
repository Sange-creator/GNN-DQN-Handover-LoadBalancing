# GNN-DQN: Graph Neural Network-based Deep Q-Network for Intelligent Handover Optimization in LTE/5G Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A production-feasible handover controller using graph-aware Deep Reinforcement Learning to optimize user-equipment (UE) handover decisions in mobile networks. This project implements a novel **GNN-DQN** architecture that generalizes across different network topologies without retraining.

## 🚀 Key Features

- **Graph-Aware Intelligence:** Uses Graph Convolutional Networks (GCN) to capture spatial relationships between cells.
- **Topology-Invariant Generalization:** The same model weights work on any cell layout (e.g., from Pokhara to Kathmandu) without retraining.
- **UE-Observable Only:** Operates using only measurements available to the UE (RSRP, RSRQ as load proxy), ensuring deployment realism.
- **Dueling Double DQN (D3QN):** High-performance reinforcement learning with dueling heads and target networks.
- **Multi-Scenario Support:** Evaluated on Dense Urban, Highway, Suburban, Rural, and Real-world (OpenCellID) topologies.
- **Standard Baselines:** Includes comparison against 3GPP A3-TTT, Strongest RSRP, Load-aware heuristics, and Flat DQN.

## 🛠 Project Architecture

The GNN-DQN agent processes cell features through a 3-layer Graph Convolutional Network:
1. **Input:** 12 features per cell node (RSRP, RSRQ trends, serving indicator, etc.).
2. **Feature Extraction:** 3x GCNConv layers with ReLU activation and Dropout.
3. **Decision Making:** Dueling Q-Head (Value and Advantage streams) to select the optimal target cell.

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/gnn-dqn-handover-loadbalancing.git
cd gnn-dqn-handover-loadbalancing
pip install -r requirements.txt
```

## 🏃 Usage

### Quick Start (Smoke Test)
```bash
python3 run_experiment.py --train-episodes 4 --steps 35 --test-episodes 3
```

### Full Multi-Scenario Training
Trains on a mix of urban, highway, and rural scenarios to learn topology-invariant policies.
```bash
python3 run_full_training.py
```

### Real-World Topology Training (Pokhara)
```bash
python3 run_overnight.py
```

### ns-3 Data Generation
Generate LTE handover data using the integrated ns-3.40 bridge.
```bash
python3 tools/run_ns3_dataset.py --sim-time 30 --sample-period 1 --run 1
```

## 📊 Performance Summary

| Method | Avg UE Mbps | P5 UE Mbps | Load Std Dev | Handover Rate |
| :--- | :---: | :---: | :---: | :---: |
| **GNN-DQN (Ours)** | **1.17** | **0.06** | **4.65** | **45.2** |
| Strongest RSRP | 1.00 | 0.06 | 5.74 | 38.5 |
| A3 TTT | 0.97 | 0.06 | 5.48 | 5.6 |
| Load-Aware Heuristic | 1.18 | 0.07 | 4.65 | 91.2 |

*Note: Results vary by scenario. Multi-scenario training demonstrates superior generalization on unseen topologies.*

## 📂 Project Structure

- `handover_gnn_dqn/`: Core logic (GNN model, simulator, policies).
- `data/`: OpenCellID positions, ns-3 traces, and synthetic datasets.
- `results/`: Training logs, history, and model checkpoints (.pt).
- `tools/`: Utility scripts for data downloading and ns-3 integration.
- `run_*.py`: Entry points for training and experiments.

## 📝 Reference

This project is part of a research effort targeted at journal publication (IEEE TVT / Computer Networks). 

---
© 2026 GNN-DQN Handover Optimization Team
