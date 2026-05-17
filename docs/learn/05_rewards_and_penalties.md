# How Rewards and Penalties Work

## The Simple Analogy

Imagine training a dog. You give treats (rewards) for good behavior and say "no!"
(penalties) for bad behavior. Over time, the dog learns what to do.

Our GNN-DQN agent is the dog. Every time it decides to handover a UE or keep it
on the current cell, we compute a reward that tells it **how good that decision
was**. The agent learns to maximize total rewards.

But unlike a simple treat/punishment, our reward is a **sophisticated score**
that balances multiple competing objectives simultaneously.

## The Multi-Objective Challenge

We want the agent to optimize **all of these at once**:

| Objective | We want... | But... |
|---|---|---|
| High throughput | Each UE gets fast speeds | Requires good cell selection |
| Low ping-pong | No bouncing between cells | Too conservative = sticky cells |
| Load balance | Even load across cells | May require "unnecessary" handovers |
| No outage | Signal always above threshold | May need risky handovers to escape |
| Few handovers | Minimal interruptions | May miss load-balancing opportunities |
| Fairness | Tail users (worst 5%) are okay | Helping tail users may hurt average |

The reward function combines ALL of these into one number. Let's break it down.

## Reward Components: The Positive Side (Carrots)

### 1. Satisfaction Bonus (weight: 3.5)
```
satisfaction = min(throughput / demand, 1.2)
reward += 3.5 × satisfaction
```
"Are you getting what you need?" If your demand is 5 Mbps and you're getting
5 Mbps, satisfaction = 1.0. Getting 6 Mbps? Satisfaction capped at 1.2.

**Analogy**: Getting your food order within the expected time.

### 2. Throughput Bonus (weight: 1.2)
```
thr_bonus = log₂(1 + throughput) / 4.0
reward += 1.2 × thr_bonus
```
Logarithmic bonus — going from 0→2 Mbps is more valuable than 8→10 Mbps.
This ensures the agent cares about getting users out of zero-throughput.

**Analogy**: First $1000 of salary matters more than going from $99K to $100K.

### 3. Throughput Delta (weight: 1.8)
```
thr_delta = (throughput_after - throughput_before) / demand
reward += 1.8 × clip(thr_delta, -0.5, 0.8)
```
"Did this action IMPROVE throughput?" Positive delta = good action.

### 4. P5 (5th Percentile) Protection (weight: 1.2)
```
p5_delta = (p5_after - p5_before) / p5_before
reward += 1.2 × clip(p5_delta, -0.5, 0.8)
```
Protects the **worst-off users** in the network. Even if average throughput
stays the same, improving the bottom 5% is rewarded.

**Analogy**: A teacher who helps struggling students, not just top performers.

### 5. Stay Bonus (variable)
```
if not handover AND signal > threshold + 6 dB AND load < 85%:
    reward += 1.0
```
"You correctly decided NOT to handover — good restraint!"

This is crucial. Without it, the agent learns to handover too often. Most of the
time, staying put is the right decision.

**Analogy**: "If it ain't broke, don't fix it."

### 6. Jain Fairness (weight: 1.2)
```
jain = (Σloads)² / (N × Σloads²)
reward += 1.2 × jain
```
Rewards even load distribution. Jain = 1.0 means perfectly balanced.

### 7. Load Gain (weight: 2.5 × load_diff)
```
if handover AND source_load > target_load + 0.03:
    load_gain = 2.5 × min(load_diff, 0.6)
    reward += load_gain
```
"You moved a UE from a loaded cell to a lighter one — smart!"

**Analogy**: Switching from a crowded checkout line to an empty one.

### 8. Proactive Handover Bonus
```
if handover AND not_in_outage AND serving_margin < 10 dB AND target_margin > serving + 3 dB:
    reward += 0.9 × margin_improvement
```
"You handed over BEFORE the signal got critical — great foresight!"

Especially important at highway speed — you can't wait for the signal to die.

### 9. Overload Escape Bonus (up to 2.8)
```
if handover AND source_load > 0.85 AND target_load < 0.75:
    reward += 2.0  (or 2.8 if source > 100%)
```
"You escaped an overloaded cell — critical for load balancing!"

### 10. Load Improvement Bonuses
```
reward += 3.0 × clip(load_std_improvement, -0.25, 0.25)
reward += 1.5 × clip(overload_improvement, -0.5, 0.5)
```
Network-wide improvements in load balance and overload.

## Reward Components: The Negative Side (Sticks)

### 1. Handover Cost (variable, 0.5 to 2.2)
```
ho_cost = clip(2.2 - 1.6 × (speed_ratio^1.2), 0.5, 2.2)
penalty -= ho_cost × handover
```
Handovers aren't free — they cause a brief interruption. But the cost is
**speed-dependent**: fast UEs (highway) get a LOWER penalty because they NEED
frequent handovers to maintain coverage.

```
Slow UE (1 m/s):   ho_cost ≈ 2.2 (expensive — don't handover unless really needed)
Fast UE (30 m/s):  ho_cost ≈ 0.6 (cheap — handover proactively)
```

**Analogy**: Changing lanes on a highway is normal. Changing lanes every 5 seconds
in city traffic is dangerous.

### 2. Ping-Pong Penalty (weight: -6.0)
```
if handover back to previous cell within 6 steps:
    penalty -= 6.0  (SEVERE!)
```
This is the **biggest single penalty**. Ping-pong is the worst behavior — it
wastes two handover interruptions for no net benefit.

### 3. Recency Penalty (up to -1.2)
```
if handover AND steps_since_last_handover < recency_window:
    penalty -= 1.2 × (1 - steps_since_last / window)
```
"You just handed over 2 steps ago — are you sure you need another one?"
Discourages rapid successive handovers even if they're not to the same cell.

### 4. Outage Penalty (weight: -2.5 + -1.8 × severity)
```
if target_rsrp < -112 dBm:
    penalty -= 2.5
    penalty -= 1.8 × min(distance_below_threshold / 10, 1.0)
```
"You connected to a cell where the signal is too weak!" Deeper outage = harsher
penalty.

### 5. Tail Penalty (weight: -1.8)
```
if UE_throughput < 20th_percentile:
    penalty -= 1.8 × (1 - throughput / p20_threshold)
```
"This UE is in the bottom 20% of throughput — fix it!"

### 6. Load-Related Penalties
```
penalty -= 1.5 × load_std       # High load variance is bad
penalty -= 2.0 × overload       # Any cells above capacity
penalty -= 0.4 × (handover AND target_load > 0.85)  # Don't handover TO loaded cell
```

## The Complete Reward Formula

```python
reward = (
    + 3.5 × satisfaction          # Are you getting your demand?
    + 1.2 × thr_bonus            # Absolute throughput value
    + 1.8 × thr_delta            # Did throughput improve?
    + 1.2 × p5_delta             # Did worst-off users improve?
    + stay_bonus                  # Reward for good restraint
    + 1.2 × jain_fairness        # Load distribution fairness
    + load_gain                   # Reward for smart load moves
    + proactive_bonus             # Reward for forward-thinking HOs
    + overload_escape             # Reward for escaping congestion
    + 3.0 × load_std_improvement # Network-wide balance improvement
    + 1.5 × overload_improvement # Network-wide overload reduction
    - 1.8 × tail_penalty         # Penalize being in throughput tail
    - 1.5 × load_std             # Penalize high load variance
    - 2.0 × overload             # Penalize any overloaded cells
    - ho_cost                    # Cost of handover interruption
    - ho_recency_penalty         # Cost of rapid consecutive HOs
    - 6.0 × pingpong             # SEVERE cost of ping-pong
    - 2.5 × outage               # Cost of losing signal
    - 1.8 × outage_severity      # Extra cost for deep outage
    - 0.4 × (HO to loaded cell)  # Don't move to congested cell
)
```

## Typical Reward Values

| Scenario | Good decision | Bad decision |
|---|---|---|
| Stay on good cell, low load | +6 to +8 | N/A |
| Smart load-balancing HO | +4 to +7 | N/A |
| Proactive HO before outage | +5 to +9 | N/A |
| Unnecessary HO | N/A | -1 to -3 |
| Ping-pong | N/A | -8 to -12 |
| HO to outage cell | N/A | -5 to -8 |
| Stay on overloaded cell | N/A | -2 to -4 |

## Why Reward Design is Critical

The reward function is the **specification** of what we want. Get it wrong, and
the agent optimizes the wrong thing:

- Too much stay bonus → agent never handovers → sticky cells
- Too little PP penalty → agent ping-pongs → connection instability
- Too much HO cost → agent won't load-balance → congestion persists
- Too little load reward → agent ignores load → no improvement over A3

Our fine-tuning adjusts these weights to align training with evaluation conditions.

## In Our Code

| Component | File | Line | What it does |
|---|---|---|---|
| `user_reward()` | simulator.py | 653 | Default reward function |
| `patch_reward()` | colab_finetune.py | Fine-tuning | Modified weights per phase |
| `satisfaction` | reward | Core | Throughput/demand ratio |
| `pingpong_penalty` | reward | Critical | -6.0 × pingpong flag |
| `ho_cost_base` | reward | Speed-aware | Lower cost for fast UEs |
| `load_gain` | reward | Differentiator | What makes us better than A3 |
| `stay_bonus` | reward | Stabilizer | Prevents over-handover |
