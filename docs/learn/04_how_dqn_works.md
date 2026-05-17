# How DQN Works (Deep Q-Network)

## The Simple Analogy

Imagine you're playing a video game. At each moment, you see the screen (state)
and choose a button to press (action). After pressing, you get points or lose
points (reward). Over thousands of games, you learn:

"When I see THIS situation, pressing THIS button gives the most points."

A **DQN** does exactly this. It learns a function Q(state, action) that predicts
**how many total future points** you'll get if you take that action in that state.
Then it always picks the action with the highest predicted Q-value.

## The Q-Value Intuition

**Q(s, a) = "If I'm in state s and take action a, how much total reward will I
get from now until the end?"**

Example:
```
State: Cell A serving, Cell A load=1.3, Cell B load=0.4, Cell B signal=good
  Q(stay on A) = 2.1      ← staying on overloaded cell = moderate future reward
  Q(handover to B) = 4.8  ← moving to less loaded cell = higher future reward
  Q(handover to C) = -1.2 ← C has bad signal = negative future reward

  Decision: handover to B (highest Q)
```

The magic: Q-values account for **all future consequences**, not just the
immediate reward. Moving to Cell B might cost a small interruption NOW, but saves
you from being stuck on overloaded Cell A for the next 20 steps.

## How DQN Learns: The Bellman Equation

The fundamental equation of Q-learning:

```
Q(s, a) = reward + γ × max_a'[Q(s', a')]

Where:
  s  = current state
  a  = action taken
  reward = immediate reward received
  γ (gamma) = discount factor (0.975 in our case)
  s' = next state (after taking action a)
  a' = best action in the next state
```

In words: **The Q-value of an action equals the immediate reward PLUS the
discounted value of the best future from the next state.**

### The Discount Factor (γ = 0.975)

Why not γ = 1.0 (value all future rewards equally)?

- γ = 0.975 means "a reward 1 step from now is worth 97.5% of the same reward now"
- A reward 10 steps away is worth 0.975^10 = 0.776 (77.6%)
- A reward 40 steps away is worth 0.975^40 = 0.360 (36%)

This makes the agent prefer **sooner rewards** but still plan ahead. At 0.975,
the agent effectively plans ~40 steps into the future.

## Training Loop (Simplified)

```
Repeat for 920 episodes:
  1. Reset environment (random scenario, random seed)
  2. For 120 steps:
     a. For each UE:
        - Build state: 11 features × num_cells
        - Choose action: ε-greedy (explore or exploit)
        - Execute action: handover or stay
        - Observe: next_state, reward
        - Store (s, a, r, s') in replay buffer
     b. Every 4 steps:
        - Sample 128 random transitions from replay
        - Compute target: r + γ × Q_target(s', best_a')
        - Compute loss: (Q_predicted - target)²
        - Update network weights via gradient descent
     c. Every 800 steps:
        - Soft-update target network
```

## Key DQN Enhancements (All Used in Our Model)

### 1. Experience Replay
Instead of learning from transitions in order, we store them in a big buffer
(550,000 transitions) and sample randomly. This breaks correlation between
consecutive steps and makes training more stable.

```
Without replay: learns from [step1, step2, step3, ...] — correlated!
With replay:    learns from [step847, step12, step5003, ...] — independent!
```

### 2. Prioritized Experience Replay (PER)
Not all transitions are equally useful. A transition where the agent made a big
mistake (large TD error) is more informative than one where it predicted correctly.

PER samples important transitions more often:
```
Transition with TD error = 5.0: sampled 10× more often
Transition with TD error = 0.1: sampled rarely
```

This is like a student spending more time on problems they got wrong.

### 3. Target Network
We use TWO copies of the network:
- **Online network**: makes decisions, gets updated every training step
- **Target network**: provides stable targets, updated slowly (soft update)

Without this, training is unstable — like trying to hit a moving target.

```python
# Soft update: target slowly follows online (τ = 0.003)
target_params = 0.997 × target_params + 0.003 × online_params
```

### 4. Double DQN
Standard DQN tends to **overestimate** Q-values. Double DQN fixes this:

```
Standard DQN:  target = r + γ × max Q_target(s', a')
                                     ↑ target picks the action AND evaluates it

Double DQN:    target = r + γ × Q_target(s', argmax Q_online(s', a'))
                                               ↑ online picks    ↑ target evaluates
```

By separating action selection from evaluation, overestimation is reduced.

### 5. Dueling Architecture
Instead of learning Q(s,a) directly, we decompose it:

```
Q(s, a) = V(s) + A(s, a) - mean(A)

Where:
  V(s)    = "How good is this state overall?" (one number)
  A(s, a) = "How much better/worse is action a compared to average?" (per action)
```

This helps because in many states, the value doesn't depend much on which
specific action you take — most cells are fine to stay on.

### 6. N-Step Returns (n=3)
Instead of learning from 1-step transitions, we chain 3 steps:

```
1-step: target = r₁ + γ × Q(s₂)
3-step: target = r₁ + γ×r₂ + γ²×r₃ + γ³ × Q(s₄)
```

This gives faster credit assignment — the agent sees 3 steps of consequences
immediately rather than slowly propagating rewards backward.

## ε-Greedy Exploration

The agent needs to explore (try random actions) to discover good strategies, but
also exploit (use what it's learned) to maximize reward.

```
With probability ε: pick a random valid action (explore)
With probability 1-ε: pick the action with highest Q-value (exploit)
```

During training, ε decays:
```
Fine-tuning: ε starts at 0.15 → ends at 0.008
             (starts 15% random, ends 0.8% random)

Original training started at ε=0.45 (45% random!)
```

## The Full Picture: GNN + DQN Together

```
        UE's perspective of 20 cells
        (11 features each = 20×11 matrix)
                    ↓
        ┌───────────────────────┐
        │   GNN (3 GCN layers)  │ ← uses cell adjacency graph
        │   Learns: "what does  │
        │   each cell look like │
        │   considering its     │
        │   neighborhood?"      │
        └───────────┬───────────┘
                    ↓
            64-dim embedding per cell
                    ↓
        ┌───────────────────────┐
        │   Dueling DQN Head    │
        │   V(s) + A(s,a)       │
        │   Learns: "what's the │
        │   Q-value of each     │
        │   cell for this UE?"  │
        └───────────┬───────────┘
                    ↓
        Q-value per cell: [3.2, 4.8, 1.1, ...]
                    ↓
        Pick action with highest Q (among valid cells)
                    ↓
        Action = cell index to connect to
        (same cell = stay, different cell = handover)
```

## In Our Code

| Component | File | What it does |
|---|---|---|
| `DQNConfig` | models/gnn_dqn.py | All hyperparameters (γ, LR, ε, etc.) |
| `GnnDQNAgent.act()` | models/gnn_dqn.py | ε-greedy action selection |
| `_train_step()` | models/gnn_dqn.py | One gradient update step |
| `PrioritizedReplayBuffer` | models/gnn_dqn.py | PER storage + sampling |
| `NStepBuffer` | models/gnn_dqn.py | Chains 3-step returns |
| `train_multi_scenario()` | rl/training.py | Full training loop |
| `target_net` | Separate model copy | Stable target for Bellman equation |
