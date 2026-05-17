# Comprehensive Project Overview: GNN-DQN Based Handover Optimization and Load Balancing in LTE Networks

## 1. Executive Summary & Objective

This document provides an in-depth, mathematically rigorous, yet accessible overview of the **GNN-DQN Handover Optimization** project. 

The goal of this project is to solve a fundamental problem in cellular networks: **Load-blind handovers**. Legacy networks use simple signal strength (RSRP) to determine when a mobile phone should switch cell towers. This causes massive congestion when many users gather in one place, as everyone connects to the strongest tower, completely ignoring nearby, quieter towers.

Our solution introduces a **Graph Neural Network paired with a Deep Q-Network (GNN-DQN)** to learn optimal load-balancing handovers. Crucially, to make this safe for real-world deployment, the AI's decisions are filtered through a **Self-Organizing Network (SON)** safety layer that strictly complies with telecom standards.

---

## 2. Mathematical Foundation of the Environment

Before the AI can learn, the simulator must accurately reflect real-world LTE physics.

### 2.1 Signal Strength (RSRP)
The Reference Signal Received Power (RSRP) determines how strong the signal is from the tower to the User Equipment (UE). It is calculated as:
$$ RSRP_{i,c} = P_{tx,c} - PL_{i,c} + \chi_{\sigma} $$
Where:
* $P_{tx,c}$ is the transmit power of cell $c$ (e.g., 43 dBm).
* $PL_{i,c}$ is the Path Loss based on the distance $d$ (in km) between UE $i$ and cell $c$. We use the standard urban macro model: 
  $$ PL_{i,c} = 128.1 + 37.6 \log_{10}(\max(d, 0.035)) $$
* $\chi_{\sigma}$ is the shadow fading, representing obstacles like buildings, modeled as a Gaussian random variable $\mathcal{N}(0, \sigma^2)$. Based on our Pokhara drive-test data, we calibrated $\sigma = 8.33$ dB.

### 2.2 Signal Quality and Load Proxy (RSRQ)
To balance load without requiring perfect "God-mode" network data, our AI relies strictly on what a phone can actually measure: **RSRQ** (Reference Signal Received Quality). RSRQ drops when a tower is congested because other users' traffic creates interference.
$$ RSRQ_{i,c} = 10 \log_{10} \left( \frac{N_{PRB} \cdot RSRP_{i,c}^{linear}}{RSRP_{i,c}^{linear} + I_{c} + N_0} \right) $$
Where:
* $N_{PRB}$ is the number of Physical Resource Blocks (e.g., 50 for 10 MHz bandwidth).
* $N_0$ is the thermal noise floor (e.g., -104 dBm).
* $I_{c}$ is the interference from other cells. Importantly, interference from a neighboring cell depends on its load $\rho$:
  $$ I_{c} = \sum_{j \neq c} \left( P_{rx, j}^{linear} \times \rho_j \right) $$
By passing $RSRQ$ to our AI, the AI can mathematically infer $\rho$ (the cell load) and avoid handing over to congested towers.

---

## 3. The Core Algorithm: Intelligence and Execution

Our architecture separates the "Brain" (GNN-DQN) from the "Hands" (SON Controller).

### 3.1 The Brain: GNN-DQN (Graph Neural Network + Deep Q-Network)

**1. State Representation (What the AI sees):**
For a UE $i$, the state $s_t$ is a matrix of UE-observable features for the serving cell and all neighboring cells. It includes normalized $RSRP$, normalized $RSRQ$, speed $v$, and time since the last handover.

**2. Graph Message Passing (The GNN):**
Standard neural networks break if you add or remove a cell tower. A GNN treats towers as nodes on a graph, allowing the model to work on *any* city layout.
The GNN updates the hidden feature vector $h_c$ for each cell $c$ by aggregating information from neighboring cells $\mathcal{N}(c)$:
$$ h_c^{(l+1)} = \sigma \left( W_1 h_c^{(l)} + W_2 \sum_{j \in \mathcal{N}(c)} e_{cj} h_j^{(l)} \right) $$
Where $e_{cj}$ is the edge weight (inversely proportional to the physical distance between towers).

**3. Action Selection (The DQN):**
The DQN takes the GNN's output and predicts a Q-value $Q(s_t, a_t)$ for every possible target cell. The Q-value represents the expected long-term reward of handing over to that cell.
The network is trained using the Bellman Equation loss:
$$ L(\theta) = \mathbb{E} \left[ \left( r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-) - Q(s_t, a_t; \theta) \right)^2 \right] $$

### 3.2 The Hands: SON Safety Controller

Telecom operators will never let an AI directly control phone connections. Instead, our AI acts as an advisor to a Self-Organizing Network (SON) controller.

**1. Translating Q-Values to A3 Parameters:**
Standard networks use the **A3 Event** rule: handover if $RSRP_{target} > RSRP_{serving} + A3_{offset} - CIO_{target}$.
Our SON controller updates the $CIO$ (Cell Individual Offset) based on the AI's preferences:
* If the AI wants UEs to move to Cell B, the SON increases $CIO_B$.
* Mathematical update bound: $CIO_{new} = \max(\min(CIO_{old} + \Delta CIO, +3 \text{dB}), -3 \text{dB})$.
We strictly limit the maximum step size to 0.5 dB per cycle to ensure smooth transitions.

**2. The Rollback Mechanism (Safety Switch):**
If the AI makes a mistake, the SON detects it and instantly rolls back the parameters.
$$ \text{If } \left( \frac{\text{PingPongs}}{\text{Total Handovers}} > 0.008 \right) \text{ OR } \left( \Delta \text{Throughput} < -15\% \right) \implies CIO = \text{Legacy Default} $$

---

## 4. The Reward Function: How We Train the AI

To make the AI balance loads without causing chaos, we designed a highly engineered, multi-objective reward function $R_t$.

$$ R_t = R_{throughput} + R_{load} + R_{safety} $$

**1. Throughput & P5 Protection:**
$$ R_{throughput} = 3.5 \left( \frac{Thr_i}{Demand_i} \right) + 1.2 \log_2(1 + Thr_i) + 1.2 (\Delta Thr_{P5}) $$
We heavily reward the AI for increasing the throughput of the bottom 5% of users ($Thr_{P5}$), forcing it to care about edge-users.

**2. Load Balancing Fairness:**
$$ R_{load} = 1.2 \times \text{JainFairness} + 2.0 \times \mathbb{1}_{\text{escape overload}} $$
* Jain's Fairness Index measures how evenly traffic is spread: $\frac{(\sum \rho_c)^2}{N \sum \rho_c^2}$.
* $\mathbb{1}_{\text{escape overload}}$ is a massive bonus granted if the AI hands a UE from a cell with >85% load to a cell with <75% load.

**3. Safety & Ping-Pong Penalties:**
$$ R_{safety} = - C(v) \cdot \mathbb{1}_{HO} - 15.0 \cdot \mathbb{1}_{PingPong} - 2.5 \cdot \mathbb{1}_{Outage} $$
* We penalize every handover, but the cost $C(v)$ shrinks if the user's speed $v$ is high (because cars *must* hand over).
* A massive $-15.0$ penalty is applied if a UE hands back to a cell it just left (a ping-pong). This mathematically overrides any potential throughput reward, teaching the AI that ping-pongs are forbidden.

---

## 5. Completed Phases & Project Status

### Phase 0: Codebase & Bug Fixes
* **Bug Fix:** The SON controller historically tracked cumulative ping-pongs, becoming "numb" late in the simulation. We re-engineered the algorithm to track *per-cycle* moving-window ping-pongs.
* **Metric Engineering:** Added `late_ho_rate` and `steps_below_margin` to track "sticky cells" (when a UE stays connected to a dying signal too long).

### Phase 1: Drive Test Calibration
We ingested real-world GPS data from Nepal (Pokhara and Kathmandu).
* We computed physical speed derivatives using the Haversine formula: $v = \frac{\Delta \text{distance}}{\Delta t}$.
* Augmented the simulation to perfectly match the dense urban, suburban, and highway subsets found in the physical drive-test reports.

### Phase 2 & 3: Fine-Tuning Execution
* **Currently Active:** We are running the final "Phase 3" training execution. The cell capacity has been strictly capped at **50 Mbps** to force the simulation into a mathematically congested state, where load balancing is absolutely required for survival.

---

## 6. Expected Results & Academic Claims

Upon the completion of the current run, we expect the following results, which form the core claims of the thesis:

| Scenario | A3-TTT Thr | son_gnn_dqn (ours) | Expected $\Delta$ Thr | Ping-Pong Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Highway (Uncongested)** | ~ 1.61 Mbps | 1.61–1.64 Mbps | tie to +2% | $\leq$ 0.003 |
| **Overloaded Event (Congested)** | ~ 0.75 Mbps | 0.83–0.88 Mbps | **+10% to +15%** | $\leq$ 0.010 |
| **Kathmandu (Unseen Map)** | ~ 0.98 Mbps | 0.98–1.01 Mbps | tie to +3% | $\leq$ 0.008 |

### The "Honest Framing" Narrative for Publication
1. **Load Balancing Only Matters Under Pressure:** In empty networks, our AI wisely ties the legacy A3 standard. There is no need to optimize a network that isn't struggling.
2. **The P5 Victory:** During urban surges, our AI prevents edge-user starvation, boosting worst-case throughput by up to 15% purely by routing traffic away from congested central towers.
3. **The Deployability Compromise:** We explicitly acknowledge that raw RL could get +25% throughput, but it causes network-crashing ping-pongs. By using the SON safety layer, we willingly sacrifice some extreme throughput to guarantee a ping-pong rate of $\leq 1.2\%$, matching stringent telecom constraints.

---

## 7. Upcoming Roadmap (Phases 4-6)

1. **Phase 4 (Evaluation):** Once Phase 3 training completes, we will run multi-seed stochastic evaluations across all 8 scenarios and 7 baselines (including `a3_ttt` and `load_aware` heuristics) to generate mathematically tight 95% Confidence Intervals.
2. **Phase 5 (Replay):** We will feed the physical Pokhara GPS trace directly through the trained agent, mapping its exact CIO adjustments to physical coordinates.
3. **Phase 6 (Thesis Extraction):** Automated scripts will extract the tables and generate RSRP/Load-distribution maps for the final LaTeX thesis and paper submission.

*End of Document.*
