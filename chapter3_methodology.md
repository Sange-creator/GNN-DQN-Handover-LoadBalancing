# Chapter 3: Methodology

## 3.1 System Architecture

The proposed SON-GNN-DQN framework is designed as a three-layer intelligent control system for 5G handover optimization. The architecture is structured to decouple high-level preference learning from low-level network safety and 3GPP standards compliance.

1.  **Layer 1: GNN-DQN (The Preference Brain):** This layer consumes processed UE measurements and the network topology graph. It utilizes a Graph Convolutional Network (GCN) to extract relational features across the cell layout. The output is a set of Q-values, where each value represents the "preference score" for a candidate target cell relative to the current serving cell.
2.  **Layer 2: SON Translation Layer (The Safety Filter):** This layer acts as a semantic bridge. It translates the raw Q-value preferences into discrete adjustments of 3GPP handover parameters (CIO and TTT). It enforces strict operational constraints, such as parameter bounds, rate-limiting, and performance-based rollback logic, ensuring that the RL agent cannot destabilize the network.
3.  **Layer 3: Standard 3GPP A3 Logic (The Execution Engine):** The final handover decision is made by the standard 3GPP A3-event logic using the parameters "nudged" by the SON layer. This ensures that the system is carrier-grade and backwards-compatible, as the RL agent only influences the *parameters* of the decision, never overrides the underlying physics of the radio link.

The high-level data flow is as follows:
$$UE \text{ Measurements} \xrightarrow{State Builder} GNN\text{-}DQN \xrightarrow{Preferences} SON \text{ Controller} \xrightarrow{CIO/TTT} A3 \text{ Engine} \xrightarrow{Handover} Network$$

## 3.2 Network Model

The network environment is modeled as a discrete-time simulator reproducing the behavior of a 5G NR Radio Access Network.

### Cell Topology
The cell layout is represented as an undirected graph $G = (V, E)$, where the set of vertices $V$ corresponds to base stations (gNodeBs) and the set of edges $E$ represents the interference or adjacency relationships. Each cell $i \in V$ is characterized by its spatial coordinates $(x_i, y_i)$, transmit power $P_{tx}$, and carrier frequency.

### Propagation and Channel Model
The signal strength observed by a UE at position $(x, y)$ from cell $i$ is modeled using a distance-dependent path loss model combined with log-normal shadowing:
$$RSRP_i(dBm) = P_{tx} - PL(d_i) + X_{\sigma}$$
where $PL(d_i)$ is the path loss at distance $d_i$, and $X_{\sigma} \sim \mathcal{N}(0, \sigma^2)$ represents shadowing with $\sigma = 8$ dB to reflect a typical urban environment. Rayleigh fading is applied to the small-scale variations of the signal.

### Mobility Model
UE movement follows the **Gauss-Markov mobility model**, which provides temporally correlated trajectories closer to real-world vehicular or pedestrian motion. The velocity $v_t$ and heading $\theta_t$ at time $t$ are updated as:
$$v_t = \alpha v_{t-1} + (1-\alpha)\bar{v} + \sqrt{1-\alpha^2}\delta_v$$
$$\theta_t = \alpha \theta_{t-1} + (1-\alpha)\bar{\theta} + \sqrt{1-\alpha^2}\delta_{\theta}$$
where $\alpha$ is the memory parameter, and $\delta$ represents Gaussian noise.

### Capacity and Load Model
The capacity of cell $i$ is calculated based on the Shannon-Hartley theorem, considering the SINR of all connected users. A **Proportional Fair (PF)** scheduler is assumed, which divides the total capacity $C_i$ among the $N_i$ connected UEs. The instantaneous throughput for UE $u$ connected to cell $i$ is:
$$Thr_{u,i} = \frac{C_i}{N_i} \cdot \omega_{u,i}$$
where $\omega_{u,i}$ is the scheduling weight based on channel quality and fairness.

## 3.3 MDP Formulation

We cast the handover optimization problem as an episodic MDP defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

### State Space ($\mathcal{S}$)
The state at time $t$ is defined by the node feature matrix $X \in \mathbb{R}^{|V| \times F}$ and the adjacency matrix $A \in \{0, 1\}^{|V| \times |V|}$. For the **UE-ONLY** profile, each node (cell) $i$ has $F=11$ features:
1.  RSRP from cell $i$.
2.  RSRQ from cell $i$.
3.  Estimated load proxy (derived from RSRQ).
4.  RSRP trend (first derivative).
5.  RSRQ trend.
6.  Serving cell indicator (1 if serving, 0 otherwise).
7.  Signal usability (binary).
8.  UE speed.
9.  Time since last handover.
10. Previous serving cell indicator.
11. Distance to cell $i$.

### Action Space ($\mathcal{A}$)
The action $a_t$ corresponds to selecting a target cell from the candidate set: $a_t \in \{1, 2, ..., |V|\}$. Note that selecting the current serving cell corresponds to a "stay" decision.

### Reward Function ($\mathcal{R}$)
The reward function is multi-objective, designed to balance throughput improvement against stability:
$$r_t = w_{thr} \Delta Thr + w_{fair} \Delta Jain - w_{ho} I(HO) - w_{pp} I(PP) - w_{rlf} I(RLF)$$
where $\Delta Thr$ is throughput gain, $\Delta Jain$ is the improvement in Jain's Fairness Index, $I(HO)$ is a handover indicator, $I(PP)$ is a ping-pong penalty, and $I(RLF)$ is a radio link failure penalty.

### Discount Factor ($\gamma$)
We set $\gamma = 0.97$ to emphasize long-term stability over instantaneous throughput gains.

## 3.4 GNN-DQN Architecture

The agent architecture uses a **Dueling Double DQN** with a GNN-based feature extractor.

### Graph Convolutional Layers
The feature extractor consists of 3 GCN layers. The update rule for layer $l$ is:
$$h^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} h^{(l)} W^{(l)})$$
where $\tilde{A} = A + I$ is the adjacency matrix with self-loops, $\tilde{D}$ is the degree matrix, and $W^{(l)}$ are the learned weights. The dimensions are $11 \to 128 \to 128 \to 128$.

### Dueling Head
The final node embeddings $h^{(3)}$ are passed to a dueling head:
1.  **Value Stream:** Computes the state value $V(s)$ by mean-pooling all node embeddings and passing them through an MLP. This represents the overall quality of the current network state.
2.  **Advantage Stream:** Computes the advantage $A(s, a)$ for each cell $i$ by passing $h^{(3)}_i$ through an MLP. This represents the relative benefit of handing over to cell $i$.

The Q-value is then combined as:
$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|V|} \sum_{a'} A(s, a') \right)$$

## 3.5 Training Methodology

The model is trained using **Double DQN** to decouple action selection from value estimation, reducing overoptimism.

*   **Experience Replay:** A buffer of 300,000 transitions is used, with separate sub-buffers for each training scenario to prevent catastrophic forgetting.
*   **Soft Target Updates:** The target network weights $\theta_{target}$ are updated using Polyak averaging: $\theta_{target} \leftarrow \tau\theta + (1-\tau)\theta_{target}$ with $\tau = 0.005$.
*   **Learning Rate:** A cosine annealing schedule is used, decaying from $1 \times 10^{-4}$ to $1 \times 10^{-5}$.
*   **Exploration:** $\epsilon$-greedy exploration decays linearly from $1.0$ to $0.03$ over the first 250 episodes.
*   **Multi-Scenario Training:** In each training epoch, the agent is exposed to all 7 training scenarios (dense urban, highway, etc.) to ensure domain robustness.

## 3.6 SON Translation Layer

The SON controller translates the GNN-DQN Q-values into persistent 3GPP parameter updates.

### CIO Update Logic
Every update interval (1 second), the controller calculates the difference between the Q-value of a neighbor $j$ and the serving cell $i$: $\Delta Q_{ij} = Q_j - Q_i$.
*   If $\Delta Q_{ij} > \text{threshold}$, $CIO_{ij} \leftarrow \min(CIO_{ij} + 0.5, 6.0)$ dB.
*   If $\Delta Q_{ij} < -\text{threshold}$, $CIO_{ij} \leftarrow \max(CIO_{ij} - 0.5, -6.0)$ dB.

### Safety and Rollback
The controller monitors the 10-second rolling average of UE throughput. If the throughput drops more than 15% below the standard A3 baseline (indicating a bad policy update), a **Rollback** is triggered, resetting all $CIO$ values to 0 dB.

## 3.7 Evaluation Methodology

The system is evaluated using 20 independent random seeds per scenario to generate 95% confidence intervals (CI95). Testing is conducted on both in-distribution scenarios used during training and out-of-distribution "Zero-Shot" scenarios (e.g., the 25-cell Kathmandu layout and unseen coverage holes). Performance is compared against six baselines, ranging from `no_handover` to `load_aware` heuristics.
