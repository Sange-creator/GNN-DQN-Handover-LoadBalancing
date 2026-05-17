# Key Terms Glossary

Quick reference for all technical terms used in this project.

## Cellular Network Terms

| Term | Full Name | Simple Explanation |
|---|---|---|
| **UE** | User Equipment | Your phone or device |
| **eNodeB** | Evolved Node B | LTE cell tower / base station |
| **Cell** | — | Coverage area of one antenna sector |
| **Serving cell** | — | The cell your UE is currently connected to |
| **Target cell** | — | A cell your UE might handover to |
| **LTE** | Long-Term Evolution | 4G mobile network standard |
| **PRB** | Physical Resource Block | Smallest schedulable radio resource unit |

## Signal Measurements

| Term | Full Name | Simple Explanation |
|---|---|---|
| **RSRP** | Reference Signal Received Power | Signal strength in dBm (more negative = weaker) |
| **RSRQ** | Reference Signal Received Quality | Signal quality accounting for interference and load |
| **RSSI** | Received Signal Strength Indicator | Total power received (signal + noise + interference) |
| **dBm** | Decibels relative to milliwatt | Logarithmic power unit. -70 = strong, -112 = weak |
| **Path loss** | — | Signal weakening over distance (128.1 + 37.6×log10(d_km)) |
| **Shadowing** | — | Random signal variation from buildings/terrain |
| **Fading** | — | Rapid signal fluctuation from multipath |

## Handover Terms

| Term | Full Name | Simple Explanation |
|---|---|---|
| **HO** | Handover | Switching UE from one cell to another |
| **A3 Event** | 3GPP Measurement Event A3 | Standard HO trigger: target > serving + offset |
| **TTT** | Time-to-Trigger | How long A3 must hold before HO executes (ms or steps) |
| **CIO** | Cell Individual Offset | Per-cell-pair bias added to RSRP in A3 decision |
| **Ping-pong** | — | UE bouncing back to previous cell within short window |
| **HO interruption** | — | Brief throughput drop during handover (~50-100ms) |
| **Too-late HO** | — | Handover triggered after signal already failed |
| **Too-early HO** | — | Handover to a cell you quickly leave |
| **Outage** | — | Signal below minimum threshold → service failure |
| **Hysteresis** | — | Margin preventing HO on small signal differences |

## SON Terms

| Term | Full Name | Simple Explanation |
|---|---|---|
| **SON** | Self-Organizing Network | System that auto-tunes network parameters |
| **MLB** | Mobility Load Balancing | SON function that distributes users across cells |
| **MRO** | Mobility Robustness Optimization | SON function that reduces HO failures and PP |
| **Rollback** | — | Undoing a parameter change that made things worse |

## Machine Learning Terms

| Term | Full Name | Simple Explanation |
|---|---|---|
| **RL** | Reinforcement Learning | Learning by trial-and-error with rewards |
| **DQN** | Deep Q-Network | Neural network that predicts action values |
| **GNN** | Graph Neural Network | Neural network that operates on graph-structured data |
| **GCN** | Graph Convolutional Network | Type of GNN using neighbor averaging |
| **GAT** | Graph Attention Network | Type of GNN using learned attention weights |
| **Q-value** | Quality value | Expected total future reward for taking an action |
| **State** | — | The current situation (features the agent observes) |
| **Action** | — | What the agent does (which cell to connect to) |
| **Reward** | — | Feedback signal after taking an action |
| **Episode** | — | One complete training run (120 steps on one scenario) |
| **ε-greedy** | Epsilon-greedy | Explore ε% randomly, exploit (1-ε)% using Q-values |
| **Replay buffer** | Experience replay | Memory bank of past transitions for training |
| **PER** | Prioritized Experience Replay | Sampling important transitions more often |
| **Target network** | — | Slowly-updated copy of model for stable training |
| **Double DQN** | — | Using two networks to prevent Q-value overestimation |
| **Dueling DQN** | — | Decomposing Q into value V(s) + advantage A(s,a) |
| **N-step return** | — | Using N steps of actual reward before bootstrap |
| **Bellman equation** | — | Q(s,a) = reward + γ × max Q(s', a') |
| **γ (gamma)** | Discount factor | How much to value future vs immediate rewards (0.975) |
| **τ (tau)** | Soft update rate | How fast target tracks online network (0.003) |
| **LR** | Learning rate | Step size for gradient descent (0.00015) |
| **Batch size** | — | Number of transitions per training step (128) |
| **Grad clip** | Gradient clipping | Prevents exploding gradients (0.8) |

## Evaluation Metrics

| Term | Simple Explanation | Good Value |
|---|---|---|
| **avg_throughput** | Mean Mbps across all UEs | Higher is better |
| **p5_throughput** | 5th percentile throughput (worst 5% of UEs) | Higher = fewer starved users |
| **HO/1k decisions** | Handovers per 1000 decision points | Lower is better (15-30 typical) |
| **PP rate** | Fraction of handovers that are ping-pong | < 3% for publication |
| **load_std** | Standard deviation of cell loads | Lower = more balanced |
| **Jain fairness** | Fairness index (0-1) | Closer to 1.0 is better |
| **outage_rate** | Fraction of UEs below RSRP threshold | Should be 0% or near-0% |

## Our Project-Specific Terms

| Term | Meaning |
|---|---|
| **ue_only** | Feature profile using only UE-observable measurements (11 features) |
| **oran_e2** | Feature profile with network-side data (15 features, future work) |
| **son_gnn_dqn** | Our deployable method: GNN-DQN + SON controller + A3 execution |
| **gnn_dqn** | Direct GNN-DQN policy (research baseline, not deployable) |
| **resume checkpoint** | Training snapshot with model + optimizer + replay buffer |
| **best_state_dict** | Model weights from the best validation episode |
| **behavioral cloning** | Pre-training by imitating a heuristic (load-aware) policy |
| **feature_dim = 11** | Number of per-cell features in ue_only mode |
| **max_cells = 20** | Largest cell count in training scenarios |
| **validation score** | Composite metric: SON throughput margin - PP penalty |
