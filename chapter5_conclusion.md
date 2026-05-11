# Chapter 5: Conclusion and Future Work

## 5.1 Summary

Mobility management in 5G and beyond networks faces significant challenges due to increased cell densification and dynamic traffic patterns. Traditional handover mechanisms, while stable and standardized, are inherently reactive and load-blind, often leading to suboptimal resource utilization and degraded user experience. This thesis has proposed **SON-GNN-DQN**, a novel framework that integrates topology-invariant Deep Reinforcement Learning (DRL) with a safety-bounded Self-Organizing Network (SON) translation layer.

By modeling the Radio Access Network (RAN) as a graph and employing Graph Convolutional Networks (GCNs), the proposed system learns relational preference features that are invariant to the network topology. The SON translation layer acts as a critical safety buffer, mapping RL recommendations to standardized 3GPP parameters (CIO and TTT) within strict operational bounds. Critically, the system operates using a "UE-ONLY" feature profile, utilizing RSRQ as a load proxy to ensure practical deployability without requiring network-side PRB counters.

## 5.2 Key Findings

The comprehensive evaluation of the SON-GNN-DQN framework has yielded several key findings:

1.  **Topology Generalization via GNN:** The GNN architecture successfully enabled zero-shot transfer from small-scale training topologies (3–7 cells) to complex real-world layouts, such as the 25-cell Kathmandu scenario. This proves that GNNs can learn universal "principles" of handover that are decoupled from specific cell counts.
2.  **Structural Necessity of the SON Layer:** Ablation studies demonstrated that the SON safety layer is not merely an enhancement but a structural necessity. Raw RL policies frequently collapse or take aggressive actions in unseen environments, leading to outages and coverage holes. The SON layer ensures the system remains "safe-by-design" by bounding the parameter space.
3.  **Competitive UE-Only Performance:** The results show that competitive load-balancing gains can be achieved using only UE-observable measurements. By using RSRQ as an effective load proxy, the system matched or slightly exceeded the throughput of proprietary network-side heuristics while maintaining significantly higher stability.
4.  **Stability and Safety Coexistence:** The multi-objective reward design, which penalizes handovers and ping-pongs, successfully taught the agent to be conservative. The resulting policy achieved stable network operation with zero ping-pongs across all test scenarios, proving that intelligent optimization does not have to come at the cost of network reliability.

## 5.3 Limitations

Despite the promising results, several limitations of the current work should be acknowledged:

*   **Simulation-to-Reality Gap:** While the simulator incorporates high-fidelity propagation and mobility models, it cannot perfectly replicate the stochastic complexity of a real-world urban environment. Validation on real network traces or a pilot deployment is required to confirm the findings.
*   **Noisy Load Approximation:** The RSRQ-derived load proxy, while practical, is a noisy approximation of physical PRB utilization. In scenarios with extreme external interference, the proxy may trigger conservative offloading decisions that do not perfectly align with the physical capacity constraints.
*   **Single-Agent Perspective:** The current framework optimizes handover from the perspective of an individual UE. While this is compatible with existing RRC standards, it does not explicitly coordinate decisions across multiple UEs to proactively manage network-wide interference or "herd" effects.

## 5.4 Future Work

Building on the foundations established in this thesis, several promising directions for future research are identified:

1.  **O-RAN E2 Integration:** Future iterations of the system could be packaged as a **near-RT RIC xApp**. Integrating with the O-RAN E2 interface would allow the model to consume real-time PRB counters via the `E2SM-KPM` service model, potentially improving the precision of load-aware offloading.
2.  **Multi-Agent GNN Formulations:** Transitioning from a single-agent to a multi-agent Reinforcement Learning (MARL) framework would allow for coordinated handover decisions. This could prevent situations where multiple UEs simultaneously offload to the same neighbor, potentially creating secondary congestion.
3.  **Online Fine-Tuning and Domain Adaptation:** Developing mechanisms for online fine-tuning would allow the model to adapt its pre-trained GNN weights to the specific characteristics of a local deployment site using real-time feedback from network traces.
4.  **Hierarchical Control:** Investigating a hierarchical architecture where a macro-level agent manages cell-wide load distribution (by tuning SON parameters) while a micro-level agent assists individual UEs with fine-grained association decisions could further enhance network performance.
5.  **Real-World Pilot Validation:** Partnering with a network operator (e.g., Nepal Telecom) to validate the framework using anonymized drive-test data and real RSRP/RSRQ logs from a commercial 5G cluster would be a crucial step toward commercialization.

In conclusion, the SON-GNN-DQN framework represents a significant step toward the realization of truly autonomous and safe mobility management in next-generation wireless networks. By combining the relational power of GNNs with the safety of SON controllers, it provides a robust and deployable path for AI-driven network optimization.
