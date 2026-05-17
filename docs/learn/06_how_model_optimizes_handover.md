# How Our Model Optimizes Handover (The Full Pipeline)

## The Big Picture

Our system has **two layers** working together:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: GNN-DQN (Brain)                               │
│  "For each UE, which cell would be best right now?"      │
│  Output: per-UE cell preferences                         │
└────────────────────┬────────────────────────────────────┘
                     ↓ preferences aggregated
┌─────────────────────────────────────────────────────────┐
│  Layer 2: SON Controller (Safety Layer)                   │
│  "Translate preferences into safe, bounded A3 parameter  │
│   updates (CIO + TTT)"                                   │
│  Output: adjusted A3 thresholds per cell pair             │
└────────────────────┬────────────────────────────────────┘
                     ↓ modified A3 parameters
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Standard A3 Handover (Execution)               │
│  "Execute handovers using standard LTE procedure with    │
│   SON-adjusted CIO and TTT values"                       │
└─────────────────────────────────────────────────────────┘
```

This is called **son_gnn_dqn** — our headline deployable method.

## Why Two Layers? (The SON Approach)

### Why Not Let GNN-DQN Directly Handover?

We could let the GNN-DQN directly control handovers (that's what `gnn_dqn`
policy does). But this has problems for real deployment:

1. **Non-standard**: Real LTE networks use A3 events. Replacing the entire HO
   mechanism requires modifying the RAN software.
2. **Safety risk**: If the ML model has a bad prediction, it could cause mass
   handovers or outages.
3. **Interpretability**: Network operators can't understand or audit ML decisions.

### The SON Solution

Instead, we use the GNN-DQN as an **advisor** that tunes the parameters of the
standard A3 handover mechanism:

- **CIO (Cell Individual Offset)**: Bias added to RSRP for specific cell pairs.
  Positive CIO → makes target cell look stronger → encourages handover to it.
  Negative CIO → makes target cell look weaker → discourages handover to it.

- **TTT (Time-to-Trigger)**: How long the A3 condition must hold before handover.
  Higher TTT → more conservative → fewer handovers, less ping-pong.
  Lower TTT → more aggressive → faster response, more ping-pong risk.

```
Standard A3:  target_rsrp > serving_rsrp + offset
SON-tuned A3: target_rsrp + CIO > serving_rsrp + offset   (CIO biases the decision)
```

## Step-by-Step: What Happens Each Time Step

### 1. Mobility Update
```
All UEs move based on their velocity (speed + direction).
Highway UEs: 80-120 km/h along the road
Urban UEs: 0.5-15 km/h random walk
Event UEs: 0-5 km/h clustered near center
```

### 2. Channel Snapshot
```
For every (UE, Cell) pair:
  - Calculate path loss: 128.1 + 37.6 × log10(distance_km)
  - Add shadowing (slow fading)
  - Add measurement noise (fast fading)
  - Result: RSRP(UE, Cell) in dBm
  
Also compute: RSRQ, loads, throughputs (all cached for consistency)
```

### 3. SON Controller Check (every 10 steps)
```
If enough steps have passed since last update:
  a. Collect GNN-DQN preferences for ALL UEs
  b. Aggregate: "How many UEs on Cell A prefer Cell B?"
  c. For each (source, target) cell pair:
     - If many UEs prefer target → increase CIO by 0.5 dB
     - If target is overloaded → decrease CIO by 0.5 dB
  d. Apply up to 8 updates this cycle
  e. Safety check: rollback if throughput dropped or PP increased
```

### 4. Per-UE Handover Decision (A3 with SON CIO)
```
For each UE:
  - adjusted_rsrp[cell] = rsrp[cell] + CIO[serving, cell]
  - Find best cell among valid cells
  - A3 condition: adjusted_rsrp[best] > rsrp[serving] + A3_offset?
    - Yes: increment TTT counter for this (UE, target) pair
    - No: reset counter
  - If counter >= TTT threshold → EXECUTE HANDOVER
```

## The SON Safety Bounds

Our SON controller is intentionally conservative:

| Parameter | Value | Purpose |
|---|---|---|
| CIO range | ±3 dB | Can't make a cell look 3+ dB better/worse than reality |
| CIO step | 0.5 dB | Small nudges, not dramatic swings |
| Updates per cycle | 8 | Limited number of changes at once |
| Update interval | 10 steps | Let changes settle before changing again |
| Preference threshold | 15% | Only act if 15%+ of UEs on a cell prefer target |
| Rollback on PP | > 3% | Undo changes if ping-pong exceeds 3% |
| Rollback on throughput | > 5% drop | Undo changes if throughput drops 5%+ |

This means even if the GNN-DQN gives bad advice, the damage is bounded.

## Comparison: How Different Policies Decide

### No Handover
```
Always stay on current cell.
result: best signal initially, but can't adapt to mobility or load changes.
```

### Strongest RSRP (+ hysteresis)
```
If best_cell_rsrp > serving_rsrp + 2 dB → handover.
result: follows signal, ignores load.
```

### A3/TTT (Standard LTE)
```
If target_rsrp > serving_rsrp + 3 dB for 3 consecutive steps → handover.
result: stable signal-following with ping-pong protection. The baseline.
```

### Load-Aware Heuristic
```
Score = signal_quality - 0.48 × estimated_load - HO_cost.
Pick highest score.
result: considers load but uses a fixed formula.
```

### GNN-DQN (Direct)
```
Q-values = GNN(state, graph).
Pick cell with highest Q among valid cells.
result: learned policy, but bypasses standard A3 (non-standard).
```

### SON-GNN-DQN (Our Method) ⭐
```
GNN-DQN preferences → SON aggregation → CIO/TTT adjustment → Standard A3.
result: learned intelligence + standard execution + safety bounds.
```

## What Makes Our Method Better Than A3

### 1. Load Awareness
A3 only sees signal strength. Our GNN sees RSRQ-derived load estimates and
learns that moving from a loaded cell to a lighter one improves throughput,
even if the lighter cell has slightly weaker signal.

### 2. Topology Awareness
The GNN propagates information across the cell graph. It knows that if Cell A is
overloaded and Cell B is its neighbor, biasing handovers from A→B will help.
But it also knows not to overload Cell B in the process (because Cell B's state
propagates back through the graph).

### 3. Speed-Adapted Behavior
The model sees `ue_speed_class` as a feature. For fast UEs (highway), it
learns to be more proactive about handovers. For slow UEs (pedestrian), it
learns to be more conservative.

### 4. Predictive (Not Just Reactive)
The model receives `rsrp_trend` (is signal getting better or worse?) and
`time_since_ho` (how recently did we handover?). It can learn patterns like
"signal is degrading fast → start handover NOW before outage."

### 5. Multi-Objective Optimization
The reward function balances throughput, ping-pong, load balance, fairness, and
outage avoidance simultaneously. No fixed formula can capture this — the GNN-DQN
learns the optimal trade-off from experience.

## Training vs Evaluation Flow

### Training (learns the policy)
```
Environment → random scenario (dense_urban, highway_fast, etc.)
Agent takes actions directly (ε-greedy)
Reward computed → stored in replay buffer
Agent updates weights via DQN loss
→ Agent learns: "what do good handover decisions look like?"
```

### Evaluation (tests the deployable policy)
```
Environment → specific scenario with fixed seed
SON controller uses agent's preferences to adjust CIO/TTT
Standard A3 makes actual handover decisions
Metrics computed: throughput, PP rate, HO rate, load balance
→ Compare against baselines (no_handover, a3_ttt, load_aware, etc.)
```

## The Defense Narrative

For your thesis/paper, the story is:

> "We train a GNN-DQN to learn per-UE handover preferences from UE-observable
> measurements. A safety-bounded SON controller translates these preferences
> into standard A3/CIO-style handover parameter updates. The system achieves
> higher throughput and lower ping-pong than traditional A3/TTT, while
> remaining fully compatible with existing LTE infrastructure."

Key claims:
1. **SON-compatible**: Uses standard A3, not custom HO logic
2. **UE-only features**: No network-side data needed (realistic deployment)
3. **Topology-generalized**: Works on unseen cell counts and layouts
4. **Safety-bounded**: CIO/TTT changes are clamped and rollback-protected
5. **Multi-scenario**: Dense urban, highway, events, suburban, real positions
