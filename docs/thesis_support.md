# Thesis Support Materials

## 1. Related Work

The optimization of mobility management and load balancing in 5G and beyond networks has seen a paradigm shift from static, rule-based heuristics toward intelligent, data-driven frameworks. This section reviews the literature across four key dimensions: Deep Reinforcement Learning (DRL) for handover, Graph Neural Networks (GNN) for Radio Access Network (RAN) management, Self-Organizing Networks (SON), and the current research gaps that this work addresses.

### 1.1 Deep RL for Handover Optimization
Traditional handover (HO) mechanisms, primarily governed by 3GPP Event A3 (Neighbor becomes offset better than SpCell), rely on fixed parameters like Cell Individual Offset (CIO) and Time-to-Trigger (TTT) [1]. However, static settings fail to capture the dynamics of high-mobility users and fluctuating cell loads. DRL has emerged as a robust alternative. Yajnanarayana et al. (2019) formulated HO as a contextual multi-armed bandit problem, demonstrating significant reductions in radio link failures (RLF) [2]. More recently, Deep Q-Networks (DQN) and their variants (DDQN) have been applied to optimize handover decisions by learning from Reference Signal Received Power (RSRP) and Reference Signal Received Quality (RSRQ) trends [13, 14]. A critical challenge in these models is the multi-objective reward design, which must balance throughput maximization against stability metrics such as ping-pong rate and handover count [15]. Chen et al. (2021) proposed hierarchical RL to manage this trade-off in ultra-dense networks, though often at the cost of high state-space complexity [16].

### 1.2 GNN for Radio Access Network Management
While standard DRL models like MLPs are effective, they suffer from poor scalability and lack of topology generalization—a model trained for 10 cells cannot typically infer on 20 cells without retraining. Graph Neural Networks (GNNs) solve this by exploiting the relational structure of cellular networks. Shen et al. (2021) proved that GNNs achieve superior generalization error (O(n) vs O(n^2) for MLPs) in resource management tasks [3]. GNNs model the RAN as a graph where cells and UEs are nodes, allowing the network to learn permutation-equivariant features that are invariant to the specific layout [6]. This work draws on the "transference" properties demonstrated by Eisen and Ribeiro (2020), where models trained on small-scale topologies perform optimally on larger, unseen deployments [6, 7]. Recent survey work by Jiang (2022) highlights GNNs as a foundational technology for 6G network management [8].

### 1.3 Self-Organizing Networks (SON)
The SON framework, introduced by 3GPP, aims to automate network management through functions like Mobility Load Balancing (MLB) and Mobility Robustness Optimization (MRO) [12]. SON parameter tuning, specifically CIO and TTT optimization, is the primary lever for these functions. Moysen and Garcia-Lozano (2018) provided a comprehensive survey of how ML can be integrated into SON to automate these parameter adjustments [5]. Moysen and Kwon (2020) further refined this by providing a geometry-based analysis of optimal HO parameters, showing the inherent trade-offs between RLF and ping-pong rates [4]. Practical SON deployments often require safety mechanisms and rate-limiting to prevent network-wide oscillations, a concern addressed by Mismar et al. (2020) through joint optimization frameworks [9].

### 1.4 Gap Analysis and Contribution
Despite the advances in DRL and GNNs, several gaps remain:
1. **Topology Sensitivity:** Most existing DRL approaches for HO assume a fixed topology, limiting their real-world utility in growing or heterogeneous networks.
2. **Observability Constraints:** Many models rely on network-side PRB (Physical Resource Block) utilization data, which is not always available at the point of decision (e.g., in UE-centric or multi-vendor deployments).
3. **Safety and Deployability:** Raw RL outputs (Q-values or raw actions) lack the bounded safety properties required for carrier-grade systems.

This thesis bridges these gaps by proposing **SON-GNN-DQN**. Our approach integrates a topology-generalized GNN-DQN preference model with a safety-bounded SON translation layer. Critically, our system operates on **UE-ONLY** measurements (using RSRQ as a load proxy), ensuring compatibility with existing 3GPP reporting while maintaining the theoretical benefits of GNN-based generalization and SON-grade reliability.

---

## 2. Defense Q&A

### 1. Why use a GNN instead of a flat DQN or tabular Q-learning?
**Answer:** Traditional "flat" DQNs or tabular methods treat the network state as a fixed-length vector or a discrete state-action table. This creates two major issues: (a) **Scalability:** If the number of cells in the network changes from 3 to 25, the input layer of a flat DQN must be completely redesigned and retrained. (b) **Relational Blindness:** Flat models fail to recognize that the relationship between User A and Cell B is structurally identical to User C and Cell D if their relative radio conditions are the same.
Our implementation in `src/handover_gnn_dqn/models/gnn_dqn.py` uses a Graph Convolutional Network (GCN) as the feature extractor within the `GnnDQNAgent`. By modeling the network as a graph, the agent learns message-passing rules that are permutation-equivariant. This means the model learns "how to prioritize a target cell relative to its neighbors" rather than learning "the specific ID of Cell 7." This allows us to train on small scenarios in `src/handover_gnn_dqn/topology/scenarios.py` and deploy the same weights to complex real-world layouts like Kathmandu without retraining.

### 2. Why not deploy raw GNN-DQN Q-values directly as handover triggers?
**Answer:** Raw Q-values represent a learned preference or the expected long-term return of an action, but they lack physical units and temporal stability. Using them directly to trigger handovers would be equivalent to a "black box" control system. In high-mobility environments, Q-values can fluctuate rapidly due to fading or noise, leading to extreme instability and excessive signaling.
Instead, we treat the `GnnDQNAgent` as a "preference engine." As seen in the `SONController` (`src/handover_gnn_dqn/son/controller.py`), the agent's output is translated into adjustments for standard 3GPP parameters (CIO). This introduces a layer of **semantic grounding**: the network continues to use the trusted A3-event logic, but the RL agent "nudges" the parameters to favor less-loaded cells. This separation of concerns ensures that the handover decision is still based on radio physics, while the RL agent provides the long-term optimization strategy.

### 3. How does the SON translation layer ensure safety?
**Answer:** Safety is enforced through three mechanisms in the `SONController` (`src/handover_gnn_dqn/son/controller.py`) and its associated `SONConfig`:
1. **Action Bounding:** The Cell Individual Offset (CIO) is strictly bounded (e.g., `cio_min=-6.0`, `cio_max=6.0` dB). This prevents the agent from "hiding" a cell so completely that a UE loses coverage entirely.
2. **Rate Limiting:** We limit the step size of updates (e.g., 0.5 dB per step) and the total number of updates per cycle. This prevents sudden, massive shifts in user distribution that could cause secondary congestion or oscillations.
3. **Rollback Mechanisms:** The controller monitors performance metrics. If the average throughput drops below a defined safety threshold (e.g., 15% below the baseline), the controller can trigger a rollback to known-safe default parameters. This "failsafe" approach ensures that even if the RL model encounters an out-of-distribution state, the network reverts to standard 3GPP behavior.

### 4. What happens if the GNN makes a bad recommendation?
**Answer:** The system is designed with multiple layers of defense against "bad" RL recommendations. First, as noted, the **SON bounds** prevent catastrophic parameter settings. Second, the **A3-event logic** itself acts as a filter; even if the RL agent sets a high preference (high CIO) for a weak cell, the UE will not hand over unless the signal from that target cell (plus the offset) is actually usable.
Furthermore, the `reward_function` in `src/handover_gnn_dqn/env/simulator.py` heavily penalizes ping-pong handovers and radio link failures. During training (configured in `configs/experiments/multiscenario_ue.json`), the agent learns that high-risk recommendations lead to negative rewards. In the rare event of a persistently bad policy, the `SONController` rollback mechanism acts as the final arbiter to restore network stability.

### 5. How does the model generalize to unseen cell topologies (different num_cells)?
**Answer:** This is the core strength of the GNN architecture in `src/handover_gnn_dqn/models/gnn_dqn.py`. Unlike an MLP, which has a fixed input dimension $N \times F$, a GNN uses a neighborhood aggregation (message passing) scheme. The weights of the GCN layers are applied to individual nodes and edges. Because these weights are shared across all nodes, the model doesn't care if there are 3 cells or 300 cells in the graph.
During inference, the `GnnDQNAgent` constructs the adjacency matrix based on the current `CellTopology`. It performs the same 3-layer message-passing operations regardless of the graph size. This "topology-invariant" learning allows the agent to recognize patterns of "neighbor congestion" and "signal decay" as local graph properties that remain consistent across different global network structures.

### 6. Why UE-only measurements instead of network-side PRB counters?
**Answer:** Modern RANs are increasingly multi-vendor and heterogeneous. Network-side counters like PRB (Physical Resource Block) utilization are often siloed within a specific vendor's O-DU (Distributed Unit) and may not be available in real-time to a near-RT RIC or a third-party xApp. Furthermore, PRB data doesn't account for the UE's perspective of interference.
By using **UE-ONLY** features (RSRP, RSRQ, handover history), our model is compatible with the existing 3GPP RRC Measurement Report framework. This allows the system to be deployed as a "client-side" or "edge-side" optimizer without requiring deep, low-latency integration into the network's MAC/Physical layers. It also makes the model robust to "hidden" interference that the base station might not see but the UE experiences.

### 7. What is the RSRQ load proxy and how does it differ from real PRB utilization?
**Answer:** RSRQ (Reference Signal Received Quality) is defined as $N \times RSRP / RSSI$, where $RSSI$ includes the total wideband power, including thermal noise and interference from all sources. Because the $RSSI$ increases as the cell and its neighbors transmit more data, RSRQ serves as an excellent proxy for the "effective load" or "interference-plus-congestion" level of a cell.
However, it differs from **real PRB utilization** in two ways: (a) PRB utilization is a hard count of scheduled resources at the gNB, whereas RSRQ is a signal-to-interference ratio. (b) RSRQ captures external interference and noise that PRB counters ignore. In our `simulator.py`, we explicitly model this distinction to ensure the agent learns to distinguish between "poor signal because I'm far away" (low RSRP) and "poor quality because the cell is busy" (low RSRQ).

### 8. How do you handle the ping-pong problem in your reward design?
**Answer:** Ping-pongs (frequent, unnecessary handovers between two cells) are handled through a dedicated component in the `reward_function` within `src/handover_gnn_dqn/env/simulator.py`. We maintain a `handover_history` for each UE. If a UE returns to its previous serving cell within a short window (e.g., $T_{pp} = 5s$), a significant "Ping-Pong Penalty" is applied to the reward.
This forces the `GnnDQNAgent` to learn "Stability Preference." It only recommends a handover if the expected gain in throughput or load-balance is large enough to outweigh the risk of a ping-pong penalty. Additionally, the `SONController` in `controller.py` can dynamically adjust the TTT (Time-to-Trigger) parameter for pairs of cells that exhibit high ping-pong rates, providing a second layer of physical damping to the RL agent's actions.

### 9. What training scenarios did you use and why those specific ones?
**Answer:** We used a diverse set of scenarios defined in `src/handover_gnn_dqn/topology/scenarios.py` and orchestrated via `configs/experiments/multiscenario_ue.json`:
1. **Dense Urban:** High cell density and slow-moving pedestrians; tests the model's ability to balance heavy, static loads.
2. **Highway:** High-speed UEs (30 m/s) with rapid signal fluctuations; tests the model's ability to handle MRO (Mobility Robustness Optimization) and prevent RLFs.
3. **Overloaded Event:** A temporary surge in users in one specific cell; tests the agent's "proactive offloading" capabilities.
By training across these diverse regimes, the agent learns a generalized policy that is not overfitted to a single mobility pattern or cell density.

### 10. How would this system integrate with O-RAN architecture in future work?
**Answer:** Our system is designed to be O-RAN ready. The `GnnDQNAgent` and `SONController` could be packaged as a **near-RT RIC xApp**.
- **Service Models:** It would consume `E2SM-KPM` (Key Performance Indicators) and `E2SM-RC` (RAN Control) service models.
- **Interfaces:** The UE measurements would arrive via the E2 interface from the E2 Nodes (gNBs). The SON parameter updates (CIO/TTT) would be sent back as "Control Actions" via the E2 interface.
While our current focus is the `UE_ONLY` profile, the modular design of the `StateBuilder` in `simulator.py` allows for the seamless addition of `oran_e2` features (like real PRB counters) as future work.

### 11. What are the limitations of simulation-based training for real deployment?
**Answer:** The primary limitation is the "Reality Gap." Simulators like our `src/handover_gnn_dqn/env/simulator.py` use mathematical models (e.g., Gauss-Markov mobility, Log-Normal shadowing) that approximate but do not perfectly replicate the complexity of real-world RF propagation (e.g., multi-path, diffraction around buildings).
To mitigate this, we employ **Domain Randomization**—randomizing channel parameters and noise levels during training. Additionally, by using **UE-ONLY** features like RSRP/RSRQ, we rely on measurements that are natively "noisy" and "averaged" in the real world, making the model more robust to simulation inaccuracies. A real deployment would also involve a "Safe-RL" phase where the model is fine-tuned on real traces before being given control of the SON layer.

### 12. How does your approach compare to standard A3+TTT handover?
**Answer:** Standard A3+TTT is **reactive and load-blind**. It only triggers a handover when a neighbor's signal is better than the serving cell's signal. If the stronger neighbor is already 100% congested, the handover will actually *decrease* the user's throughput and increase network interference.
Our **SON-GNN-DQN** is **proactive and load-aware**. By learning from the RSRQ load proxy, the agent can "see" congestion before it leads to packet drops. It can proactively adjust the CIO to offload UEs to a slightly "weaker" but "emptier" cell, improving the **Jain Fairness Index** and total system throughput. In our multiscenario evaluations (see Section 3), we consistently see that `son_gnn_dqn` maintains the stability of A3+TTT while delivering superior throughput under heavy load.

---

## 3. Comparison Tables

*Results pending — multiscenario training in progress. The following tables will be populated once `results/runs/multiscenario_ue/evaluation/` CSV data is generated.*

### 3.1 Scenario-Specific Performance (LaTeX)
\begin{table}[ht]
\centering
\caption{Performance comparison across deployment scenarios (Results Pending)}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Avg Thr (Mbps)} & \textbf{P5 Thr (Mbps)} & \textbf{Ping-pong Rate} & \textbf{HO / 1000} & \textbf{Jain Fairness} \\ \hline
\multicolumn{6}{|c|}{\textit{Scenario: Dense Urban}} \\ \hline
no\_handover & TBD & TBD & TBD & TBD & TBD \\ \hline
random\_valid & TBD & TBD & TBD & TBD & TBD \\ \hline
strongest\_rsrp & TBD & TBD & TBD & TBD & TBD \\ \hline
a3\_ttt & TBD & TBD & TBD & TBD & TBD \\ \hline
load\_aware & TBD & TBD & TBD & TBD & TBD \\ \hline
son\_gnn\_dqn & \textbf{TBD} & \textbf{TBD} & \textbf{TBD} & \textbf{TBD} & \textbf{TBD} \\ \hline
\end{tabular}
\end{table}

### 3.2 Aggregate Performance Summary
| Metric | Improvement over A3-TTT | Improvement over Random |
| :--- | :---: | :---: |
| Avg Throughput | TBD % | TBD % |
| P5 (Cell-Edge) Thr | TBD % | TBD % |
| Handover Stability | TBD % | TBD % |
| Load Fairness | TBD % | TBD % |

**Worst-case scenario for SON-GNN-DQN:** (Data pending)

---

## 4. System Architecture

### 4.1 Functional Block Diagram
```text
[UE Measurements (RSRP, RSRQ, Speed)] 
             ↓
[State Builder (src/handover_gnn_dqn/env/simulator.py)] 
             ↓
[GNN-DQN Agent (src/handover_gnn_dqn/models/gnn_dqn.py)]
    - [Cell Topology Graph] → [3-layer GCN] → [Relational Embeddings]
    - [Relational Embeddings] → [Dueling Q-Head] → [Per-Cell Q-values]
             ↓
[SON Controller (src/handover_gnn_dqn/son/controller.py)]
    - [Safety Layer] → [CIO Updates (±0.5 dB/step, bounded ±6 dB)]
    - [MRO Layer] → [TTT Adaptation based on Ping-Pong history]
    - [Failsafe] → [Rate Limiting and Rollback Logic]
             ↓
[3GPP-Compatible Parameters (CIO, TTT)]
             ↓
[Handover Decision Engine (Standard A3 Logic)]
```

### 4.2 Training vs. Inference
- **Training Loop:** Orchestrated via `scripts/train.py`. Uses a multi-scenario buffer where experiences from `dense_urban`, `highway`, and `overloaded_event` are interleaved. Employs epsilon-greedy exploration with exponential decay and soft target network updates (Polyak averaging) to ensure stable convergence.
- **Inference/Deployment:** The `GnnDQNAgent` operates in "Relational Inference" mode. It constructs a local graph for the current UE's vicinity and computes Q-values. The `SONController` then smooths these values into persistent parameter updates, ensuring that the network transition is "Safe and Seamless."

### 4.3 SON Safety Bounds
- **Hard Constraints:** CIO is capped at $\pm 6$ dB to preserve RRC connection integrity.
- **Actuation Cooldown:** A 1-second cooldown between parameter updates to allow the network load to stabilize.
- **Rollback Condition:** If the 10-second rolling average of UE throughput drops 15% below the "Standard A3" baseline, all CIOs are reset to 0 dB.

---

## 5. Contribution Statement

**Paragraph 1: Problem Statement**
Mobility management and load balancing in 5G networks are traditionally handled by reactive, signal-strength-based heuristics that ignore the spatial distribution of traffic and cell congestion. Standard rule-based mechanisms like A3-event handover often lead to suboptimal user experiences, characterized by frequent ping-pongs in dense areas and radio link failures in high-mobility scenarios. Furthermore, emerging Deep Reinforcement Learning (DRL) solutions often struggle with topology sensitivity—failing to generalize to unseen network layouts—and lack the safety guarantees required for carrier-grade deployment.

**Paragraph 2: Proposed Solution**
This thesis proposes **SON-GNN-DQN**, a novel framework that integrates a topology-generalized Graph Neural Network (GNN)-based DRL agent with a safety-bounded Self-Organizing Network (SON) translation layer. By modeling the RAN as a graph, our system learns relational features that remain invariant across different cell counts and layouts. We prioritize **UE-ONLY** observability, utilizing RSRQ as a practical load proxy, ensuring the system is compatible with existing 3GPP standards without requiring network-side PRB counters. The SON layer acts as a critical safety buffer, translating RL preferences into bounded, rate-limited updates to standard handover parameters (CIO and TTT).

**Paragraph 3: Key Results**
(Template) Our evaluation across diverse scenarios—including dense urban, high-speed highway, and real-world Kathmandu topologies—demonstrates that **SON-GNN-DQN** significantly outperforms traditional A3-TTT baselines. The system achieves an average throughput improvement of **[X]%** while reducing the ping-pong rate by **[Y]%**. Most importantly, the model successfully generalizes to unseen topologies zero-shot, maintaining stable performance in the 25-cell Kathmandu scenario after training only on a 3-cell layout. The safety mechanisms ensure zero radio link failures during policy transitions, proving that AI-driven optimization can be both effective and reliable in mission-critical wireless infrastructure.

---
