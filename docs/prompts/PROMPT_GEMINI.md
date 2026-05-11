# Gemini Prompt — Thesis Writing

## Your Role

You are writing the **full Master's thesis** for a Computer Engineering /
Telecommunications degree. The thesis presents a novel system called
**SON-GNN-DQN** for intelligent handover optimization in 5G cellular networks.

Your job is to produce **publication-quality academic writing** — clear,
precise, properly cited, with proper LaTeX formatting for equations and tables.

---

## Project Title

**"SON-GNN-DQN: Topology-Generalized Deep Reinforcement Learning with
Safety-Bounded Self-Organizing Network Translation for 5G Handover
Optimization"**

---

## System Understanding (Internalize Before Writing)

### The Problem
In 5G networks, as cell density increases, users (UEs) must frequently switch
between cells (handover). The standard method (3GPP A3 event) is reactive
and load-blind — it only considers signal strength, ignoring cell congestion.
This leads to:
- Users connecting to strong but overloaded cells → poor throughput
- Excessive ping-pong handovers in dense areas
- No adaptation to traffic patterns

### The Solution: Three-Layer Architecture

```
Layer 1: GNN-DQN (Preference Brain)
├── Models the network as a graph (cells = nodes, adjacency = interference)
├── Learns relational features invariant to topology size
├── Output: Q-value per candidate cell = "preference score"
└── Key property: same model works on 3 cells OR 25 cells (zero-shot)

Layer 2: SON Translation (Safety Filter)
├── Maps Q-value preferences → standard 3GPP parameters (CIO, TTT)
├── CIO bounded: ±6 dB maximum (prevents coverage holes)
├── Step size limited: ±0.5 dB per update (prevents oscillation)
├── Rollback: if throughput drops 15%, reset to defaults
└── Key property: system is safe even if RL agent is wrong

Layer 3: Standard A3 Logic (Handover Decision)
├── Normal 3GPP: "hand over if neighbor RSRP + CIO > serving for TTT"
├── RL only nudges parameters, never overrides physics
└── Key property: always 3GPP-compatible, carrier-grade
```

### Why This Matters (The Contribution)
| Dimension | Prior Work | This Work |
|-----------|-----------|-----------|
| Topology | Fixed to N cells | GNN generalizes to any N |
| Safety | Raw RL output (risky) | SON-bounded (safe-by-design) |
| Observability | Needs PRB counters | UE-only (RSRQ as proxy) |
| Deployability | Requires O-RAN/E2 | Works with standard 3GPP |

---

## Feature Set (UE-ONLY Profile)

The model uses ONLY user-equipment-observable measurements:
1. **RSRP** — Reference Signal Received Power (signal strength, dBm)
2. **RSRQ** — Reference Signal Received Quality (signal quality, accounts for interference)
3. **RSRQ-derived load proxy** — maps RSRQ to estimated cell load [0,1]
4. **RSRP trend** — first derivative (signal getting better/worse?)
5. **RSRQ trend** — first derivative
6. **Serving cell indicator** — binary (is this node the current cell?)
7. **Signal usability** — is RSRP above usable threshold?
8. **UE speed** — velocity of the user (m/s)
9. **Time since last handover** — prevents rapid switching
10. **Previous serving cell** — for ping-pong awareness

**NO** Physical Resource Block (PRB) data required. This means:
- Compatible with drive-test validation
- Works in multi-vendor deployments where PRB is siloed
- Deployable as edge/client-side optimizer

---

## Baselines Explained (For Results Chapter)

| Method | What It Does | Weakness |
|--------|-------------|----------|
| `no_handover` | Never changes cell | Terrible if user moves away |
| `random_valid` | Random target selection | No intelligence, sanity check |
| `strongest_rsrp` | Always pick best signal | Ignores load, causes stampede |
| `a3_ttt` | Standard 3GPP (CIO=0, TTT=3 steps) | Reactive, load-blind |
| `load_aware` | Avoids high-load cells (heuristic) | High ping-pong rate (~30%) |
| `gnn_dqn` | Raw RL without safety layer | Collapses on unseen topologies |
| `son_gnn_dqn` | **Full system** (RL + SON safety) | Our contribution |

---

## Training Details

- **Algorithm:** Double DQN with Dueling architecture
- **Feature extractor:** 3-layer Graph Convolutional Network (GCN)
- **Replay buffer:** 300,000 transitions (per-scenario buffers)
- **Target network:** soft updates (Polyak averaging, τ = 0.005)
- **Learning rate:** cosine schedule, 1×10⁻⁴ → 1×10⁻⁵
- **Exploration:** ε-greedy, 1.0 → 0.03 over 250 episodes
- **Training scenarios:** 7 (dense_urban, highway, suburban, sparse_rural,
  overloaded_event, real_pokhara, pokhara_dense_peakhour)
- **Test scenarios (unseen):** kathmandu_real (25 cells), unknown_hex_grid,
  coverage_hole, dharan_synthetic
- **Episodes:** 300, steps per episode: 120

---

## Reward Function (Multi-Objective)

```
R = w_throughput × throughput_gain
  + w_fairness × fairness_improvement
  + w_handover × handover_penalty
  + w_pingpong × pingpong_penalty
  + w_rlf × radio_link_failure_penalty
```

Key design: The agent is penalized for instability (handovers, ping-pongs)
even if they improve instantaneous throughput. This teaches conservative,
stable behavior that the SON layer can then safely actuate.

---

## Evaluation Metrics

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| Avg throughput (Mbps) | Mean user data rate | Primary QoE metric |
| P5 throughput | 5th percentile rate | Cell-edge user experience |
| Handovers/1000 decisions | Switching frequency | Signaling overhead |
| Ping-pong rate | % handovers reversed within 5s | Network instability |
| Jain fairness index | Load balance across cells | Resource efficiency |
| Outage rate | % time below minimum signal | Coverage reliability |
| SON update count | How often CIO changes | Actuation frequency |
| SON CIO utilization | How much of ±6dB used | Safety margin usage |

---

## CHAPTER SPECIFICATIONS

### Chapter 1: Introduction (2000-2500 words)

**1.1 Background and Motivation**
- 5G and beyond: network densification trend (small cells, HetNets)
- More cells = more handover events = more failure points
- ITU-R IMT-2020 requirements: reliability, low latency, seamless mobility
- Current networks: 20-50% of user complaints relate to handover failures

**1.2 Problem Statement**
Frame as three sub-problems:
1. **Topology sensitivity:** DRL models trained for N cells fail on M≠N
2. **Safety gap:** Raw RL decisions lack carrier-grade reliability
3. **Observability constraint:** PRB counters not universally available

**1.3 Research Questions**
- RQ1: Can a GNN-based DQN agent learn handover preferences that generalize
  across arbitrary cell topologies without retraining?
- RQ2: Can a SON translation layer ensure operational safety while preserving
  the learned optimization benefits?
- RQ3: Can competitive performance be achieved using only UE-observable
  measurements (without network-side PRB)?

**1.4 Contributions**
List clearly:
1. A GNN-DQN architecture for topology-invariant handover preference learning
2. A safety-bounded SON translation layer (CIO/TTT) with rollback mechanisms
3. A UE-ONLY feature profile using RSRQ as load proxy
4. Multi-scenario training methodology for domain robustness
5. Comprehensive evaluation against 6 baselines across 10+ scenarios

**1.5 Thesis Organization**
One paragraph summarizing chapters.

---

### Chapter 2: Literature Review (3000-3500 words)

**2.1 Handover in 5G Networks**
- 3GPP handover framework: measurement → reporting → decision → execution
- A3 event: neighbor signal > serving + offset for duration TTT
- Parameters: CIO (Cell Individual Offset), TTT (Time-to-Trigger), hysteresis
- Problem: static parameters can't adapt to dynamic load

**2.2 Deep Reinforcement Learning for Mobility Management**
- MDP formulation for handover decisions
- DQN → DDQN → Dueling DQN → prioritized experience replay
- Multi-objective reward challenges in wireless (throughput vs stability)
- Key works:
  - Yajnanarayana et al. (2019): contextual bandit for HO, reduced RLF
  - Chen et al. (2021): hierarchical RL for ultra-dense networks
  - Eskandarpour & Soleimani (2025): PPO for QoS-aware load balancing
  - Rivera & Erol-Kantarci (2021): clipped double Q-learning for wireless

**2.3 Graph Neural Networks for Wireless Systems**
- Why graphs: RAN is naturally relational (cells interfere, UEs connect)
- Message passing neural networks (MPNN) framework
- Key insight: permutation equivariance = topology invariance
- Scaling: GNN generalizes O(n) vs MLP O(n²) (Shen et al. 2023)
- Transference property (Eisen & Ribeiro 2020): train small → works large
- Recent applications: power control, scheduling, resource management

**2.4 Self-Organizing Networks (SON)**
- 3GPP SON framework: Self-configuration, self-optimization, self-healing
- MLB (Mobility Load Balancing): adjusts CIO to redistribute users
- MRO (Mobility Robustness Optimization): tunes TTT to reduce failures
- Safety requirements: bounded parameters, rate limiting, conflict resolution
- Key works:
  - Moysen & Garcia-Lozano (2018): comprehensive SON ML survey
  - Mismar et al. (2020): joint optimization avoiding SON conflicts

**2.5 Research Gap Analysis**

Present as a comparison table:

| Work | Topology General. | Safety Bounded | UE-Only | Multi-Objective |
|------|:-:|:-:|:-:|:-:|
| Chen et al. (2021) | ✗ | ✗ | ✗ | ✓ |
| Eskandarpour (2025) | ✗ | partial | ✗ | ✓ |
| Shen et al. (2023) | ✓ | ✗ | — | — |
| Eisen & Ribeiro (2020) | ✓ | ✗ | — | — |
| **This work** | **✓** | **✓** | **✓** | **✓** |

---

### Chapter 3: Methodology (4000-4500 words)

**3.1 System Architecture**
- High-level block diagram description (refer to the three-layer diagram above)
- Data flow: UE measurements → state builder → GNN-DQN → SON → A3 decision

**3.2 Network Model**
- Cell topology: undirected graph G = (V, E), V = cells, E = interference links
- Propagation: path loss (distance-dependent) + log-normal shadowing (σ = 8 dB)
- Mobility: Gauss-Markov model with scenario-specific parameters
- Capacity: Shannon-based with proportional fair scheduling
- Load model: shared capacity divided among connected UEs

**3.3 MDP Formulation**

State space:
```
s_t = (X ∈ R^{|V| × F}, A ∈ {0,1}^{|V| × |V|})
```
where X is node feature matrix (F=11 features) and A is adjacency.

Action space:
```
a_t ∈ {1, 2, ..., |V|}  (select target cell)
```

Reward:
```
r_t = α·Δthroughput + β·Δfairness - γ·I(handover) - δ·I(pingpong) - ζ·I(RLF)
```

Discount factor: γ = 0.97

**3.4 GNN-DQN Architecture**

Present with equations:
- GCN layer: h^{(l+1)} = σ(D^{-1/2} A D^{-1/2} h^{(l)} W^{(l)})
- 3 GCN layers: F → 128 → 128 → 128
- Dueling head:
  - Value stream: V(s) = MLP(mean_pool(h^{(3)}))  [graph-level]
  - Advantage stream: A(s,a) = MLP(h^{(3)}_a)  [per-node]
  - Q(s,a) = V(s) + A(s,a) - mean(A)

**3.5 Training Methodology**
- Double DQN: decouple action selection from value estimation
- Experience replay: 300K capacity, per-scenario buffers
- Soft target updates: θ_target ← τ·θ + (1-τ)·θ_target, τ=0.005
- Learning rate: cosine annealing from 10⁻⁴ to 10⁻⁵
- Weight decay: 10⁻⁵ (L2 regularization)
- Multi-scenario training: cycle through 7 scenarios per epoch
- ε-greedy: linear decay 1.0 → 0.03 over 250 episodes

**3.6 SON Translation Layer**

Algorithm pseudocode:
```
every update_interval:
  for each cell pair (i, j):
    q_diff = Q(j) - Q(serving)
    if q_diff > threshold:
      CIO[i→j] += min(step_size, cio_max - CIO[i→j])
    elif q_diff < -threshold:
      CIO[i→j] -= min(step_size, CIO[i→j] - cio_min)
  
  # Safety check
  if rolling_throughput < baseline * 0.85:
    ROLLBACK: CIO = 0 for all pairs
```

Parameters:
- `cio_min, cio_max`: ±6 dB
- `step_size`: 0.5 dB
- `update_interval`: 1 second cooldown
- `rollback_threshold`: 15% throughput drop

**3.7 Evaluation Methodology**
- 20 random seeds per scenario for CI95
- Both in-distribution (training scenarios) and out-of-distribution testing
- Metrics: throughput, fairness, stability, safety (defined above)

---

### Chapter 4: Results and Discussion (3000-3500 words)

**NOTE:** Actual numbers will be provided once training completes.
Write the complete structure with [PLACEHOLDER] values. Use realistic
placeholder patterns based on these diagnostic observations:
- son_gnn_dqn typically matches or slightly beats a3_ttt (+0% to +3%)
- son_gnn_dqn always has 0 ping-pong (key advantage)
- load_aware has 15-35% ping-pong (key weakness)
- gnn_dqn collapses on unseen large topologies (proves SON necessary)

**4.1 Experimental Setup**
- Hardware: [to be filled]
- Software: Python 3.x, PyTorch, torch_geometric
- Training: 300 episodes × 120 steps × 7 scenarios
- Evaluation: 20 seeds × 10+ scenarios

**4.2 Training Convergence Analysis**
- Loss curve analysis (reference figure)
- Epsilon schedule verification
- Identify convergence episode

**4.3 Overall Performance Comparison**
Main result table (ALL methods × ALL scenarios × key metrics)
- Use LaTeX tabular format
- Bold the best value per column
- Include CI95 values

**4.4 Scenario-Specific Deep Dives**
Pick 3-4 most interesting scenarios:
- **Overloaded Event:** where SON-GNN-DQN should show biggest gain (proactive offloading)
- **Kathmandu Real (25 cells):** topology generalization proof
- **Highway:** high-mobility stress test
- **Dense Urban:** load balancing capability

For each: explain WHY our method performs as it does (connect to architecture).

**4.5 Topology Generalization Analysis**
- Performance on unseen topologies vs training topologies
- Compare: if we had used MLP instead of GNN, what would happen?
- Zero-shot transfer: trained on 3-7 cells, tested on 25

**4.6 Ablation: SON Safety Layer**
- Compare `gnn_dqn` (raw) vs `son_gnn_dqn` (with SON)
- Key finding: raw policy collapses on large topologies
- SON layer prevents catastrophic failure
- CIO utilization stats: how much of the ±6dB range is actually used?

**4.7 Discussion and Limitations**
- Simulation-to-reality gap
- RSRQ as imperfect load proxy
- Modest throughput gains (frame as safety tradeoff)
- Single-agent (no multi-UE coordination)

---

### Chapter 5: Conclusion and Future Work (1000-1200 words)

**5.1 Summary**
Restate contributions and key findings.

**5.2 Key Findings**
1. GNN enables topology generalization (train small → deploy large)
2. SON layer is structurally necessary (not just helpful)
3. UE-only features achieve competitive performance
4. Safety and stability can coexist with intelligent optimization

**5.3 Limitations**
- Simulation-based (no real deployment validation yet)
- RSRQ proxy is noisy approximation of true load
- Throughput gains modest vs A3-TTT (1-5%), though stability gains significant
- No multi-agent coordination

**5.4 Future Work**
1. O-RAN E2 integration: add real PRB counters via E2SM-KPM
2. Online fine-tuning: adapt model to real network traces
3. Multi-agent: coordinate handover decisions across UEs
4. Hierarchical: separate macro (load balance) and micro (per-UE) decisions
5. Real-world pilot: partner with Nepal Telecom for drive-test validation

---

### Abstract (250 words, write LAST)

Structure:
- Problem (2 sentences)
- Gap (1 sentence)
- Proposed solution (2-3 sentences)
- Key results (2-3 sentences with specific numbers)
- Significance (1 sentence)

---

## Style Requirements

1. **Tone:** Academic third person ("This work proposes..." not "We propose...")
2. **Citations:** IEEE numeric [1], [2], [3]
3. **Equations:** All numbered, referenced in text as "Equation (1)"
4. **Tables:** LaTeX format, caption ABOVE table
5. **Figures:** Referenced as "Fig. X", caption BELOW figure
6. **Precision:** Never say "significantly outperforms" without p < 0.05
   Use "marginally improves" or "achieves comparable performance with
   enhanced stability" for modest gains
7. **Honesty:** Don't overclaim. If throughput gain is 1-2%, say
   "comparable throughput with zero ping-pong overhead" not "superior
   throughput performance"

## Critical Framing Guidance

**The thesis narrative should emphasize (in order of strength):**
1. **Safety** — zero ping-pongs, bounded parameters, rollback protection
2. **Generalization** — one model works across any topology size
3. **Practical deployability** — UE-only, 3GPP compatible, no O-RAN needed
4. **Modest throughput improvement** — bonus, not the main claim

**Do NOT claim:**
- "Dramatically outperforms" A3-TTT (it matches or slightly beats)
- "Real-time O-RAN deployment" (it's simulation, O-RAN is future work)
- "Optimal policy" (it's learned approximation with bounded safety)

**DO claim:**
- Novel combination (GNN + SON safety layer = new contribution)
- Topology generalization (proven by testing on unseen layouts)
- Safe-by-design (proven by SON bounds preventing collapses)
- Practical (no dependency on proprietary network data)

---

## References to Include (Expand With Your Search)

[1] 3GPP TS 38.331 v17.x — NR RRC Protocol Specification
[2] V. Yajnanarayana et al., "5G Handover using Reinforcement Learning," 2019
[3] Y. Shen et al., "Graph Neural Networks for Scalable Radio Resource Mgmt," IEEE TSP 2021
[4] M. Eisen & A. Ribeiro, "Optimal Wireless Resource Allocation with GNNs," IEEE TSP 2020
[5] J. Moysen & L. Giupponi, "From 4G to 5G: Self-Organized Network Management," Comp. Comm. 2018
[6] R. Eskandarpour & V. Soleimani, "DRL for QoS-Aware Load Balancing Under Mobility," 2025
[7] J. Rivera & M. Erol-Kantarci, "QoS-Aware Load Balancing with Clipped Double Q," 2021
[8] F.B. Mismar et al., "Deep Reinforcement Learning for 5G Networks," IEEE JSAC 2020
[9] S. Chen et al., "Hierarchical RL for Ultra-Dense HetNets," IEEE TWC 2021
[10] 3GPP TS 38.214 — NR Physical Layer Procedures for Data
[11] R. Jain et al., "A Quantitative Measure of Fairness," DEC-TR 1984

---

## Output Format

Deliver each chapter as a separate markdown file with LaTeX equations
(using $...$ for inline and $$...$$ for display). Tables in LaTeX tabular
format wrapped in markdown code blocks.

File naming:
- `chapter1_introduction.md`
- `chapter2_literature_review.md`
- `chapter3_methodology.md`
- `chapter4_results.md` (with placeholders)
- `chapter5_conclusion.md`
- `abstract.md`
