# How GNN Works (Graph Neural Network)

## The Simple Analogy

Imagine you're deciding which restaurant to eat at. You don't just look at each
restaurant independently — you also consider:
- "Restaurant A is next to Restaurant B, which is always crowded, so A probably
  gets overflow customers"
- "Restaurant C is isolated, so its load is independent of others"

You reason about **relationships between places**, not just each place alone.

A **Graph Neural Network** does exactly this for cell towers. Instead of
evaluating each cell in isolation, it passes information along the connections
(edges) between neighboring cells, so each cell's representation includes
knowledge about its neighbors.

## What is a Graph?

A graph has **nodes** (things) and **edges** (connections between things).

```
In our case:
  Nodes = Cell towers (e.g., 20 cells in dense_urban)
  Edges = Which cells are neighbors (based on physical proximity)

     [Cell 1] ---0.8--- [Cell 2] ---0.6--- [Cell 3]
        |                   |
       0.7                0.9
        |                   |
     [Cell 4] ---0.5--- [Cell 5]

  Edge weights = how close the cells are (higher = closer neighbors)
```

In our code, each cell is connected to its 4 nearest neighbors
(`graph_neighbors = 4`), with weights based on distance:
```python
weight = exp(-distance / scale)  # closer = higher weight
```

## Why Not Just a Regular Neural Network?

A regular neural network treats the 20 cells as a flat list: [Cell 1, Cell 2, ..., Cell 20].
It doesn't know that Cell 1 is next to Cell 2 but far from Cell 15.

Problem: If you train on 20 cells but need to evaluate on 25 cells, a regular
network breaks — wrong input size!

A GNN works on the **graph structure**, not a fixed list. It processes each node
using its neighbors. So:
- 7 cells? Works. (sparse_rural)
- 20 cells? Works. (dense_urban)
- 25 cells? Works. (kathmandu)
- 40 cells? Works. (never seen during training!)

**This is our publication strength: topology generalization.**

## How GNN Message Passing Works

The core idea is **message passing**: each node collects information from its
neighbors, combines it with its own features, and produces an updated
representation.

### Step-by-Step (One Layer)

```
1. GATHER:  Each node collects features from its neighbors
2. AGGREGATE: Combine neighbor features (weighted average)
3. UPDATE: Mix with own features → new representation

Example for Cell 2 (neighbors: Cell 1, Cell 3, Cell 5):

  Cell 2's features: [signal=0.7, load=0.8, ...]
  Cell 1's features: [signal=0.9, load=0.3, ...]  × weight 0.8
  Cell 3's features: [signal=0.5, load=0.6, ...]  × weight 0.6
  Cell 5's features: [signal=0.6, load=0.9, ...]  × weight 0.9

  Aggregated neighbors = weighted_sum / normalization
  New Cell 2 repr = ReLU(W × [Cell 2 features + aggregated neighbors])
```

### Stacking Layers = Wider View

- **1 GNN layer**: Each node knows about its immediate neighbors (1-hop)
- **2 GNN layers**: Each node knows about neighbors-of-neighbors (2-hop)
- **3 GNN layers**: Each node knows about the extended neighborhood (3-hop)

```
3-hop example for Cell 5:

  Layer 1: Cell 5 learns about Cells 2, 4 (direct neighbors)
  Layer 2: Cell 5 now knows about Cells 1, 3 (through Cell 2)
  Layer 3: Cell 5 now has awareness of the whole cluster
```

Our model uses **3 GCN layers** — so every cell's representation incorporates
information from cells up to 3 hops away.

## Our GNN Architecture

```
Input: (num_cells, 11) — 11 features per cell for one UE's perspective
  ↓
GCN Layer 1: 11 → 256 features, LayerNorm, ReLU, Dropout(0.06)
  ↓
GCN Layer 2: 256 → 256 features, LayerNorm, ReLU, Dropout(0.06)
  ↓
GCN Layer 3: 256 → 64 features, LayerNorm, ReLU
  ↓
Dueling Head:
  ├── Value Stream:     mean(all 64-dim node embeddings) → 64 → 1 (state value)
  └── Advantage Stream: each 64-dim node embedding → 64 → 1 (per-action advantage)
  ↓
Output: Q(s, a) for each cell = V(s) + A(s,a) - mean(A)
```

Total parameters: **94,914** (very lightweight!)

## GCN vs GAT (What We Use)

### GCN (Graph Convolutional Network) — Our choice
- Aggregates neighbors using **fixed weights** (from the adjacency matrix)
- Simple, fast, stable for training
- `GCNConv` from PyTorch Geometric

### GAT (Graph Attention Network) — Available but not used
- Learns **attention weights** dynamically (which neighbors matter more)
- More expressive but harder to train
- Our config: `use_gat: false`

We chose GCN because:
1. The edge weights (distance-based) already capture the right neighborhood structure
2. 94K parameters is enough for our problem size
3. Simpler model = more stable training = more reliable results for publication

## The 11 Input Features (UE-Only Profile)

For each cell, from one UE's perspective:

| # | Feature | Range | What it captures |
|---|---|---|---|
| 1 | `rsrp_norm` | [0, 1] | Signal strength to this cell |
| 2 | `rsrq_norm` | [0, 1] | Signal quality (load proxy) |
| 3 | `rsrp_delta` | [0, 1] | Signal diff vs serving cell |
| 4 | `rsrq_delta` | [0, 1] | Quality diff vs serving cell |
| 5 | `rsrp_trend` | [0, 1] | Is signal improving or degrading? |
| 6 | `is_serving` | {0, 1} | Am I connected to this cell? |
| 7 | `signal_usable` | {0, 1} | Is RSRP above minimum threshold? |
| 8 | `ue_speed_class` | [0, 1] | How fast am I moving? |
| 9 | `time_since_ho` | [0, 1] | How long since last handover? |
| 10 | `was_previous` | {0, 1} | Was I on this cell before? (ping-pong detector) |
| 11 | `load_proxy_rsrq` | [0, 1] | Estimated load from RSRQ |

**All 11 features are UE-observable** — a real phone can measure all of them.
No network cooperation needed. This is our "ue_only" feature profile.

## Why GNN is Perfect for This Problem

1. **Variable topology**: GNN works on any number of cells (7 to 40+)
2. **Spatial awareness**: Knows which cells are neighbors
3. **Information flow**: Load on one cell affects decisions for neighboring cells
4. **Per-node output**: Produces a Q-value for each cell = one action per cell
5. **Efficient**: Same weights for all nodes (parameter sharing)

## In Our Code

| Component | File | What it does |
|---|---|---|
| `GnnDQNAgent` | models/gnn_dqn.py:183 | The full GNN model |
| `conv1, conv2, conv3` | GCNConv layers | 3-layer message passing |
| `layer_norm1/2/3` | LayerNorm | Stabilizes training |
| `value_stream` | nn.Sequential | Dueling: state value V(s) |
| `advantage_stream` | nn.Sequential | Dueling: per-action A(s,a) |
| `forward()` | Main forward pass | Supports single + batched graphs |
| `act()` | Action selection | ε-greedy with valid mask |
| `_make_graph()` | simulator.py | Builds adjacency from cell positions |
| `edge_data` | Property | Returns (edge_index, edge_weight) for PyG |
