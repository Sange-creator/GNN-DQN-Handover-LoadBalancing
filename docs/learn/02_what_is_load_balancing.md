# What is Load Balancing?

## The Simple Analogy

Imagine a food court with 12 counters. Counter #3 has the best burger, so
everyone lines up there — 50 people waiting. Meanwhile, Counter #7 (also great
burgers) has only 5 people. **Load balancing** means gently directing some people
from Counter #3 to Counter #7 so everyone gets served faster.

In cellular networks, the "counters" are cell towers and the "people" are phone
users. Each cell tower has a **finite capacity** (like a kitchen that can only
make so many burgers per minute). When too many users pile onto one tower, everyone
on that tower gets slower service.

## Technical Definition

**Load balancing** in LTE is the process of distributing user traffic across cells
so that no single cell is overloaded while others are underutilized. This
maximizes the **aggregate network throughput** and ensures **fairness** among users.

## How Cell Load Works

Each cell has a capacity (in our simulator: 50 Mbps). The **load** is:

```
Cell Load = (Total demand of users on this cell) / Cell Capacity

Example:
  Cell A: 10 users × 6 Mbps average demand = 60 Mbps demand
  Cell A capacity = 50 Mbps
  Cell A load = 60 / 50 = 1.20 (120% — OVERLOADED!)

  Cell B: 3 users × 5 Mbps average demand = 15 Mbps demand
  Cell B load = 15 / 50 = 0.30 (30% — lots of headroom)
```

When load > 1.0 (100%), the cell is **overloaded**:
- Users share the limited bandwidth
- Everyone's throughput drops proportionally
- It's like too many people on one Wi-Fi: everything slows down

## Why Load Balancing Matters

Without load balancing:
```
Cell A: load = 1.5   → each user gets 67% of their demand
Cell B: load = 0.3   → each user gets 100% of their demand
Cell C: load = 1.2   → each user gets 83% of their demand
Cell D: load = 0.4   → each user gets 100% of their demand
```

With good load balancing:
```
Cell A: load = 0.8   → everyone gets 100%
Cell B: load = 0.7   → everyone gets 100%
Cell C: load = 0.9   → everyone gets 100%
Cell D: load = 0.6   → everyone gets 100%
```

**Moving just a few users from overloaded cells to underloaded cells can
dramatically improve the experience for ALL users.**

## Measuring Load Balance Quality

### Load Standard Deviation (load_std)
How uneven is the load distribution?

```
Perfect balance:  loads = [0.7, 0.7, 0.7, 0.7]  → std = 0.0 (ideal)
Poor balance:     loads = [1.5, 0.2, 1.3, 0.1]  → std = 0.67 (bad)
```

Lower std = more balanced = better.

### Jain's Fairness Index
A number between 0 and 1 measuring how fairly resources are distributed:

```
Jain = (Σ loads)² / (N × Σ loads²)

Perfect fairness: Jain = 1.0
Worst case:       Jain = 1/N (one cell has everything)
```

### Overload Rate
Fraction of cells where load > 1.0 (capacity exceeded):
```
loads = [0.8, 1.3, 0.6, 1.1]
overloaded = [1.3, 1.1]  →  overload_rate = 2/4 = 0.50
```

## The Challenge: Signal vs Load Trade-off

Here's the fundamental tension:

**Signal strength says**: "Connect to the closest/strongest tower"
**Load balancing says**: "Connect to the least loaded tower"

These often conflict! The closest tower is also the one serving everyone else nearby.

```
Scenario: Stadium with 3 cells
  Cell A (closest): RSRP = -82 dBm, load = 1.4 (overloaded!)
  Cell B (medium):  RSRP = -91 dBm, load = 0.6
  Cell C (far):     RSRP = -105 dBm, load = 0.3

  Standard A3: stays on Cell A (strongest signal) → slow for everyone
  Smart LB:    moves to Cell B (good signal + low load) → faster!
```

## How Traditional Methods Handle It

### 1. Strongest RSRP (No load balancing)
Always pick the strongest signal. Simple, but ignores load completely.

### 2. A3/TTT (Standard LTE)
Only hands over when a neighbor becomes significantly stronger. Purely
signal-based — no load awareness at all.

### 3. Load-Aware Heuristic
Combines signal quality with estimated load:
```
Score(cell) = signal_quality - 0.48 × estimated_load - handover_cost
Pick the cell with highest score.
```
Better than pure signal, but uses a fixed formula that can't adapt.

## How Our System Does It

Our GNN-DQN learns the optimal trade-off between signal and load **from
experience**. It doesn't use a fixed formula — it discovers the right balance
through millions of decisions.

The key innovation: the model can see the **topology** (which cells are
neighbors) through the GNN, so it understands that moving users from Cell A to
Cell B will reduce Cell A's load but increase Cell B's load. It thinks about the
**network-wide effect**, not just one user's perspective.

## RSRQ: The UE-Observable Load Proxy

In real deployment, your phone can't directly see a cell's load. But it CAN
measure **RSRQ** (Reference Signal Received Quality):

```
RSRQ = N × RSRP / RSSI

Where:
  N = number of resource blocks
  RSRP = reference signal power (what you want)
  RSSI = total received power (signal + interference + noise)
```

When a cell is heavily loaded, it transmits more data → more interference →
higher RSSI → **lower RSRQ**. So RSRQ drops when a cell is busy.

Our model uses RSRQ as a **load proxy**:
```python
estimated_load = (-3.0 - rsrq_db) / 17.0   # maps RSRQ [-20, -3] to load [0, 1]
```

This is a UE-observable measurement — no network cooperation needed. This is
important for our defense: we only use what a real phone can measure.

## In Our Code

The load balancing components appear in:

| Component | Where | What it does |
|---|---|---|
| `_loads` | simulator.py | True cell loads (demand/capacity) |
| `load_from_rsrq()` | simulator.py | UE-observable load estimate |
| `load_proxy_rsrq` | build_state() | Feature #11 fed to GNN |
| `load_std_improvement` | reward function | Rewards reducing load imbalance |
| `overload_escape` | reward function | Big reward for leaving overloaded cell |
| `jain_fairness` | reward function | Rewards even load distribution |
| `sticky_penalty` | reward function | Penalizes staying on overloaded cell |
| `LoadAwarePolicy` | policies.py | Baseline heuristic for comparison |
