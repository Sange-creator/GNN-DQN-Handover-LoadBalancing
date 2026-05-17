# Thesis Master Reference: GNN-DQN Handover Optimization

This document serves as the comprehensive "cheat sheet" and structural master guide for generating the final LaTeX thesis report via Claude Code. It contains the exact narrative, training analysis, mathematical logic, and finalized metrics that have been proven to survive academic scrutiny.

## 1. Codebase Architecture & Redundancy Removal
The project has been aggressively refactored to an industry-standard, config-driven architecture. Over 30 legacy scripts and redundant configurations have been safely archived (and `.gitignore`d). 
* **`src/handover_gnn_dqn/`**: The core package housing the simulator, GNN architecture, and the handover policies.
* **`scripts/`**: The execution layer. All evaluation is routed through unified scripts (`train.py`, `evaluate.py`, `colab_finetune.py`), eliminating script redundancy.
* **`configs/experiments/`**: The JSON config layer that isolates parameters from the codebase, ensuring exact reproducibility without modifying Python files.

## 2. Model Definitions & Policy Wrappers
There is only **ONE trained Neural Network** (`gnn_dqn.pt`). The different "models" compared in the thesis are actually different deployment policies (found in `policies.py`) that dictate how the network utilizes the AI's output:
1. **`A3-TTT` (The Baseline):** The deterministic LTE physics standard.
2. **`RAW GNN-DQN`:** The unrestricted AI. Excellent at spatial load balancing, but dangerous in fast temporal shifts.
3. **`SON-Tuned A3`:** The blanket safety wrapper. Safe, but chokes the AI's load-balancing capabilities in the city.
4. **`Adaptive SON-GNN` (The Champion):** The novel 2-Signal Gate that routes UEs between the RAW AI and the SON wrapper based on contextual physics (Speed and Load).

## 3. The Training Pipeline (From Scratch to Finetune)
1. **Graph Representation:** The network state is mapped into a Graph Neural Network (nodes=cells, edges=adjacencies). This allows the model to be topology-invariant, generalizing from a 20-cell training map to a 40-cell real-world map (e.g., Kathmandu).
2. **Behavioral Cloning (Warm Start):** The agent begins by mimicking the deterministic A3-TTT teacher to learn fundamental LTE physics (e.g., signal boundaries) and avoid 100% Radio Link Failure rates during early exploration.
3. **Multi-Scenario Exploration (Base Training):** The Deep Q-Network (DQN) explores diverse simulated topologies (urban, highway, rural) using an Epsilon-Greedy strategy, learning the generalized mapping between RF signatures and handover rewards.
4. **Targeted Fine-Tuning (Exploitation):** The generalized checkpoint is fine-tuned specifically on high-density/overloaded topologies to prioritize Jain Load Fairness, producing the final deployable weights.

## 4. The 2-Signal Gate Logic (The Architectural Contribution)
The core thesis contribution is the mathematical routing gate that solves the conflict between Spatial Congestion (where AI excels) and Temporal Transition (where Physics excel).
* **Rule 1 (The AI Zone):** If `Speed < 15 m/s` AND `Average Cell Load < 35%` -> **Route to RAW GNN-DQN**. (Safe to perform active load balancing).
* **Rule 2 (The Safety Zone):** If `Speed > 15 m/s` OR `Average Cell Load >= 35%` -> **Route to SON/A3-TTT**. (Signal decay is too fast, or OOD congestion is causing AI hallucinations. Enforce deterministic stability).

## 5. The Definitive Metrics (For LaTeX Tables)
* **Dense Urban (Light Load, Low Speed -> Uses RAW AI):** Adaptive Policy matches RAW GNN, achieving a **+18.8% improvement in Jain Load Fairness** over A3-TTT, effectively clearing spatial congestion.
* **Highway (High Speed -> Uses SON):** RAW GNN fails (-25.8% throughput drop, 3.2% ping-pong). Adaptive Policy correctly detects the >15m/s speed and routes to SON, restoring throughput to optimal 4.78 Mbps and eliminating ping-pongs (**0.0%**).
* **Pokhara/Kathmandu (Extreme Congestion -> Uses SON):** RAW GNN panics in untrained, saturated topologies (up to 75% handover failure). Adaptive Policy detects the >=35% Load Gate and routes to SON, preventing AI collapse and rescuing up to **+75.6% throughput**.

## 6. The IEEE-Standard Conclusion (For the Paper)
> "To address the dichotomy between spatial load balancing and temporal handover stability in LTE networks, this paper proposed a Context-Aware Adaptive SON-Gated architecture. Experimental results demonstrate that while unrestricted Deep Reinforcement Learning (GNN-DQN) achieves state-of-the-art spatial optimization, it suffers from severe performance degradation in high-velocity transitions and out-of-distribution network saturation. By introducing a 2-signal routing gate utilizing User Equipment (UE) velocity and average cell load, the proposed architecture orchestrates a synergistic fusion between the ML agent and deterministic 3GPP bounds. 
>
> Quantitative evaluations confirm the architecture achieves Pareto optimality across all deployment environments. In stable urban scenarios, the adaptive framework extracted the full capabilities of the fine-tuned GNN-DQN, yielding an 18.8% improvement in Jain’s Fairness Index over the traditional A3-TTT baseline. During high-velocity highway transitions ($v > 15$ m/s), the SON gate completely mitigated the AI's 25.8% throughput degradation, restoring connection stability with a 0.0% ping-pong rate. Furthermore, under extreme out-of-distribution congestion (load $\ge$ 35%), the architecture successfully prevented ML convergence failures, yielding up to a 75.6% throughput rescue compared to unconstrained agents. 
> 
> These findings prove that for next-generation automated networks, Machine Learning must dynamically operate within established deterministic physics rather than replace them. While this study successfully utilized UE-observable proxies (RSRQ) for load estimation, future work will integrate the model with O-RAN E2 interfaces to ingest true Physical Resource Block (PRB) utilization metrics for expanded multi-agent orchestration."

## 7. Master KPI Cheat Sheet (For Defense Preparation)
*   **Base Training KPIs (Exploration Stage)**
    *   **Goal:** Teach the agent fundamental LTE physics (RSRP signal boundaries).
    *   **Method:** Multi-scenario Deep Q-Network exploration via Epsilon-Greedy strategy combined with Behavioral Cloning.
    *   **Result:** The GNN successfully mapped RF features and topologies to Q-values without causing 100% Radio Link Failures.
*   **Fine-Tuning KPIs (Exploitation Stage)**
    *   **Goal:** Maximize spatial resource distribution.
    *   **Method:** Targeted exposure to high-density environments with an annealed learning rate.
    *   **Result:** Produced the champion `gnn_dqn.pt` weights that generated the **+18.8% Jain Fairness** spike.
*   **Deployment/Inference KPIs (The Adaptive Gate)**
    *   **Goal:** Protect the network from the fine-tuned model's aggressiveness in untrained states.
    *   **Method:** The 2-Signal Gate (`Speed > 15m/s` OR `Load >= 35%`).
    *   **Result:** Beats A3-TTT by 18.8% in fairness; Beats Load-Aware heuristics by eliminating their 19% ping-pong flaw; Beats pure ML by saving 25.8% throughput on highways and 75.6% throughput in extreme congestion.
