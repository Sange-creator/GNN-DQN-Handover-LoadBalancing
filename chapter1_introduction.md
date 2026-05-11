# Chapter 1: Introduction

## 1.1 Background and Motivation

The rapid evolution of mobile communications, from the initial digital voice services of the second generation (2G) to the multi-gigabit throughputs of the fifth generation (5G) and beyond, has fundamentally transformed global connectivity. As specified by the ITU-R IMT-2020 requirements, 5G networks are designed to support a diverse set of use cases, including Enhanced Mobile Broadband (eMBB), Ultra-Reliable Low-Latency Communications (URLLC), and Massive Machine-Type Communications (mMTC). To meet these demanding requirements, network operators are increasingly adopting densification strategies, deploying a heterogeneous mix of macrocells, small cells, and femtocells to enhance capacity and coverage.

However, network densification introduces significant challenges in mobility management. As cells become smaller and more numerous, the frequency of handover (HO) events—the process of transferring a mobile user’s (User Equipment, UE) connection from one base station (gNodeB) to another—increases dramatically. In dense urban environments, a high-speed vehicular user may experience several handovers per minute. Each handover event is a potential failure point; inefficient mobility management can lead to radio link failures (RLF), excessive signaling overhead, and a degraded user experience (QoE). Statistics from commercial network deployments indicate that up to 20–50% of user complaints regarding service quality are directly or indirectly related to handover failures and network instability.

Traditional handover mechanisms, established by the 3rd Generation Partnership Project (3GPP), primarily rely on signal-strength-based triggers such as the Event A3. While these rule-based heuristics are computationally efficient and standardized, they are essentially reactive and "load-blind." They prioritize the strongest signal without considering the congestion level of the target cell. In modern 5G networks, where traffic distributions are highly uneven, a UE may be handed over to a stronger but heavily congested cell, resulting in immediate throughput collapse. Conversely, a cell-edge user may suffer from frequent "ping-pong" handovers—rapidly switching back and forth between two cells—due to small signal fluctuations, wasting network resources and increasing latency.

## 1.2 Problem Statement

The integration of artificial intelligence, particularly Deep Reinforcement Learning (DRL), into mobility management has shown promise in addressing these limitations. However, existing research and proposed DRL-based solutions face three critical sub-problems that hinder their practical adoption in carrier-grade networks:

1.  **Topology Sensitivity:** Most existing DRL models for handover optimization are designed for a fixed network topology. A model trained to manage a specific layout of $N$ cells typically fails when deployed in a different environment with $M \neq N$ cells or a different spatial arrangement. This lack of generalization necessitates expensive retraining for every unique deployment site.
2.  **The Safety Gap:** Raw RL agents often exhibit unpredictable or extreme behavior during the "trial-and-error" learning phase or when encountering out-of-distribution states. In a mission-critical 5G network, a "black box" RL agent that takes arbitrary handover actions represents a significant operational risk, potentially causing wide-scale service disruptions if coverage holes are accidentally created.
3.  **Observability Constraints:** Many state-of-the-art RL models for load balancing assume the availability of network-side telemetry, such as Physical Resource Block (PRB) utilization. However, in multi-vendor or disaggregated RAN environments (like O-RAN), such granular data may be siloed or unavailable in real-time. There is a need for intelligent models that can operate using only the measurements already observed and reported by the UE.

## 1.3 Research Questions

This thesis seeks to address these gaps by investigating the following research questions:

*   **RQ1:** Can a Graph Neural Network (GNN)-based DRL agent learn handover preferences that are invariant to the network topology and can generalize to unseen cell counts and layouts without retraining?
*   **RQ2:** Can a Self-Organizing Network (SON) translation layer bridge the gap between intelligent RL recommendations and standardized 3GPP parameters, ensuring operational safety through hard bounds and rollback mechanisms?
*   **RQ3:** Can competitive load-balancing and mobility performance be achieved using a "UE-ONLY" feature profile, utilizing Reference Signal Received Quality (RSRQ) as a practical load proxy?

## 1.4 Contributions

The primary contributions of this work are summarized as follows:

1.  **Topology-Invariant GNN-DQN Architecture:** We propose a novel DRL agent that utilizes Graph Convolutional Networks (GCN) as a feature extractor. This architecture allows the model to learn relational features of the cellular graph, enabling zero-shot transfer from small-scale training scenarios to complex, real-world topologies (e.g., a 25-cell deployment in Kathmandu).
2.  **Safety-Bounded SON Translation Layer:** We design a SON controller that translates RL-derived "preference scores" into safe adjustments of 3GPP parameters (Cell Individual Offset and Time-to-Trigger). This layer enforces strict operational bounds and rollback logic, ensuring the network remains stable even if the RL agent provides suboptimal recommendations.
3.  **UE-ONLY Observability Profile:** We define and validate a feature set based exclusively on UE-observable measurements. By using RSRQ as a load proxy, we demonstrate that effective load-aware handover can be achieved without requiring proprietary network-side PRB counters.
4.  **Multi-Scenario Training Methodology:** We develop a robust training pipeline that cycles the agent through diverse mobility and traffic regimes (e.g., dense urban, high-speed highway, overloaded events), ensuring the learned policy is resilient to varying network conditions.
5.  **Comprehensive Benchmarking:** We evaluate the proposed system against six standard and state-of-the-art baselines across ten distinct deployment scenarios, demonstrating superior stability and load fairness.

## 1.5 Thesis Organization

The remainder of this thesis is organized as follows. **Chapter 2** provides a detailed review of the literature on 5G handover mechanisms, Deep Reinforcement Learning in wireless systems, and the foundations of Graph Neural Networks. **Chapter 3** describes the proposed SON-GNN-DQN methodology, including the graph-based MDP formulation and the safety controller design. **Chapter 4** presents the experimental setup, training convergence analysis, and a detailed discussion of the results across various scenarios. Finally, **Chapter 5** concludes the thesis and outlines directions for future work.
