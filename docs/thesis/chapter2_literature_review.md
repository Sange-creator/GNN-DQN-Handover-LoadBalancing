# Chapter 2: Literature Review

## 2.1 Handover in 5G Networks

Mobility management is a core function of the Radio Access Network (RAN) that ensures continuous connectivity as a User Equipment (UE) moves across the coverage area of different cells. The 3GPP standards define a structured handover framework consisting of four phases: measurement, reporting, decision, and execution [1].

In 5G NR (New Radio), the process begins with the network configuring the UE to perform measurements of the Reference Signal Received Power (RSRP) and Reference Signal Received Quality (RSRQ) from the serving cell and neighboring cells. These measurements are filtered and evaluated against specific criteria, known as measurement events. The most critical event for intra-frequency mobility is **Event A3**, which is triggered when a neighbor cell’s signal strength becomes better than the serving cell’s signal by a predefined offset. The condition for Event A3 is expressed as:
$$M_n + O_{cn} - Hys > M_s + O_{cs} + Off$$
where $M_n$ and $M_s$ are the measurements of the neighbor and serving cells respectively, $O_{cn}$ and $O_{cs}$ are cell-specific offsets (Cell Individual Offset, CIO), $Hys$ is the hysteresis parameter to prevent rapid toggling, and $Off$ is the A3 offset threshold [10].

Once the condition is met continuously for a duration known as the **Time-to-Trigger (TTT)**, the UE sends a Measurement Report to the gNodeB. The network then makes a handover decision, typically resulting in an RRC Connection Reconfiguration message that directs the UE to the target cell.

The primary limitation of this framework is the static nature of the handover parameters ($CIO$, $TTT$, $Hys$). In dynamic environments with fluctuating traffic loads and user densities, static parameters fail to adapt. For instance, a cell with a high RSRP may be fully congested; a handover to such a cell would lead to a "stampede" effect, where many users connect to the same overloaded base station, drastically reducing individual throughput and increasing network interference.

## 2.2 Deep Reinforcement Learning for Mobility Management

To overcome the rigidity of rule-based systems, researchers have increasingly applied Deep Reinforcement Learning (DRL) to optimize handover decisions. DRL models the handover problem as a Markov Decision Process (MDP), where an agent interacts with the network environment to learn an optimal policy that maximizes long-term rewards [8].

Standard DRL architectures, such as the **Deep Q-Network (DQN)** and its improvements (Double DQN, Dueling DQN), have been widely explored. **Yajnanarayana et al. (2019)** proposed a contextual multi-armed bandit approach for 5G handover, demonstrating significant reductions in radio link failures by adapting to user speed and channel conditions [2]. **Chen et al. (2021)** introduced a hierarchical RL framework for ultra-dense heterogeneous networks (HetNets), separating the macrocell load-balancing decisions from small-cell user associations [9].

A major challenge in DRL for mobility is the design of a multi-objective reward function. Handover optimization involves a fundamental trade-off between throughput maximization and network stability. Frequent handovers may improve instantaneous signal quality but increase signaling overhead and the risk of "ping-pong" events. **Eskandarpour and Soleimani (2025)** utilized Proximal Policy Optimization (PPO) with a multi-KPI reward covering throughput, delay, and handover counts, showing that RL can achieve a superior Quality of Service (QoS) balance under observation uncertainty [6]. **Rivera and Erol-Kantarci (2021)** highlighted the importance of addressing Q-value overestimation in wireless environments through the use of Clipped Double Q-Learning [7].

## 2.3 Graph Neural Networks for Wireless Systems

Despite the success of standard DRL, traditional neural networks like Multi-Layer Perceptrons (MLPs) suffer from a lack of topology invariance. An MLP-based DQN is typically tied to a specific input dimension, meaning the model is restricted to a fixed number of cells. Furthermore, MLPs do not natively exploit the relational structure of the RAN, where cells interfere with their immediate neighbors.

**Graph Neural Networks (GNNs)** address these limitations by representing the network as a graph $G = (V, E)$, where $V$ denotes base stations and $E$ represents the interference or adjacency relationships between them. GNNs employ a "message-passing" mechanism where each node updates its state based on the features of its neighbors. This architecture ensures **permutation equivariance**: the model's output changes consistently if the labels of the nodes are swapped.

Theoretical analysis by **Shen et al. (2021)** demonstrated that GNNs for radio resource management achieve a generalization error that scales linearly with the number of nodes $O(n)$, whereas unstructured MLPs scale as $O(n^2)$ [3]. This makes GNNs essential for scaling to large-scale 5G deployments. Furthermore, **Eisen and Ribeiro (2020)** proved the "transference" property of GNNs, showing that a model trained on a small-scale random graph can be deployed on a much larger, unseen topology with minimal performance loss [4]. Recent applications of GNNs include power control, dynamic scheduling, and intelligent connection management in O-RAN RIC xApps.

## 2.4 Self-Organizing Networks (SON)

The concept of Self-Organizing Networks (SON), introduced by the 3GPP, aims to automate network planning, configuration, and optimization to reduce operational expenditure (OPEX). Key SON functions include **Mobility Load Balancing (MLB)** and **Mobility Robustness Optimization (MRO)** [5].

MLB aims to redistribute traffic from congested cells to underutilized neighbors by adjusting the Cell Individual Offset (CIO). By increasing the $CIO$ of a target cell, the network "artificially" improves its perceived signal strength, encouraging earlier handovers. MRO focuses on reducing handover failures and ping-pongs by tuning the TTT and hysteresis parameters.

A comprehensive survey by **Moysen and Garcia-Lozano (2018)** categorized the integration of machine learning into SON, emphasizing the need for automated parameter tuning that respects carrier-grade safety requirements [5]. Practical SON deployments often require **Safety Bounds**—ensuring that parameters like CIO do not exceed ranges that might create coverage holes—and **Rollback Mechanisms** to restore defaults if performance degrades. **Mismar et al. (2020)** proposed a joint optimization framework for beamforming and handover that explicitly avoids conflicts between different SON functions [8].

## 2.5 Research Gap Analysis

While the literature provides a strong foundation, several critical gaps remain, as summarized in Table 2.1.

| Work | Topology General. | Safety Bounded | UE-Only | Multi-Objective |
|------|:-:|:-:|:-:|:-:|
| Chen et al. (2021) | ✗ | ✗ | ✗ | ✓ |
| Eskandarpour (2025) | ✗ | partial | ✗ | ✓ |
| Shen et al. (2023) | ✓ | ✗ | — | — |
| Eisen & Ribeiro (2020) | ✓ | ✗ | — | — |
| **This work (SON-GNN-DQN)** | **✓** | **✓** | **✓** | **✓** |

**Table 2.1: Comparison of the proposed work with existing literature.**

Most existing DRL approaches for handover optimization are topology-dependent and rely on network-side PRB counters that are not always available. Furthermore, the lack of a safety translation layer makes raw RL policies risky for live deployment. Our work, **SON-GNN-DQN**, bridges these gaps by combining topology-invariant GNN learning with a safety-bounded SON controller, operating exclusively on UE-observable measurements.
