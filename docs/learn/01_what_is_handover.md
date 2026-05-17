# What is Handover?

## The Simple Analogy

Imagine you're on a long phone call while riding a bus. As the bus moves, you pass
through areas covered by different cell towers (like Wi-Fi routers, but for
cellular). **Handover** is when your phone silently switches from one tower to
another — without you noticing or your call dropping.

It's like passing a baton in a relay race. The "old" tower hands your connection
to the "new" tower. If the baton drop is smooth, you never notice. If it's
clumsy, your call stutters or drops.

## Technical Definition

**Handover (HO)** is the process of transferring an active UE (User Equipment =
your phone) connection from one **serving cell** (eNodeB in LTE) to a **target
cell** while maintaining service continuity.

```
Before HO:   [UE] ←——→ [Cell A (serving)]     [Cell B (idle)]
During HO:   [UE] ←--→ [Cell A] ←sync→ [Cell B]
After HO:    [UE]       [Cell A (idle)]   ←——→ [Cell B (serving)]
```

## Why Handover is Needed

1. **Mobility**: You move. Signal from Cell A weakens, Cell B gets stronger.
2. **Load balancing**: Cell A is overloaded (100 users), Cell B has capacity (20 users).
3. **Quality**: Cell A has interference, Cell B is cleaner.

## The Key Measurement: RSRP

**RSRP** (Reference Signal Received Power) = how strong the signal is from each
cell tower, measured in dBm (decibels relative to milliwatt).

| RSRP Value | What it means |
|---|---|
| -70 dBm | Excellent — you're right next to the tower |
| -90 dBm | Good — normal indoor coverage |
| -100 dBm | Fair — edge of reliable coverage |
| -112 dBm | Poor — our minimum threshold |
| -120 dBm | Outage — connection likely drops |

Your phone constantly measures RSRP from ALL nearby cells, not just the one it's
connected to. It reports these measurements to the network.

## Standard Handover: A3 Event

In LTE, the standard handover trigger is the **A3 Event**:

> "Target cell RSRP > Serving cell RSRP + Offset, sustained for TTT"

Breaking this down:

- **Offset** (typically 3 dB): How much better the target must be. Prevents
  unnecessary switching when signals are similar.
- **TTT** (Time-to-Trigger, typically 320ms = ~3 steps): How long the condition
  must hold. Prevents handover on momentary signal spikes.

```
Example:
  Serving cell RSRP = -95 dBm
  Target cell RSRP  = -89 dBm
  Offset = 3 dB

  Is -89 > -95 + 3 = -92?  Yes! (-89 > -92)
  → Start TTT timer...
  → If still true after 3 steps → Execute handover
```

## What Happens During Handover

1. **Measurement**: UE reports RSRP/RSRQ to serving cell
2. **Decision**: Network decides to trigger HO (A3 event met)
3. **Preparation**: Serving cell asks target cell "ready to accept this UE?"
4. **Execution**: UE detaches from serving, synchronizes to target
5. **Completion**: Target cell confirms, old resources released

**Interruption time**: ~50-100ms. During this window, throughput drops (our
simulator models this as 18% reduction).

## The Problems with Standard Handover

### 1. Too Late Handover
At highway speeds (120 km/h = 33 m/s), you travel 33 meters per second. If TTT
is 320ms, you move 10 meters during the trigger delay. By the time handover
executes, you may have already passed the optimal point.

### 2. Ping-Pong
You're at the boundary of two cells. Signal fluctuates:
```
Step 1: Cell A = -90, Cell B = -88  → HO to B
Step 3: Cell A = -87, Cell B = -91  → HO back to A  ← PING-PONG!
Step 5: Cell A = -91, Cell B = -89  → HO to B again  ← WORSE!
```
Each handover interrupts your connection. Ping-pong = repeated unnecessary HOs.

### 3. No Load Awareness
Standard A3 only looks at signal strength. It doesn't know that Cell B has 50
users while Cell C has 5. It might hand you to the strongest but most congested
cell.

### 4. Sticky Cell Problem
Without handover optimization, some cells accumulate too many users while
neighboring cells sit nearly empty. The "sticky" cell can't serve everyone well,
but A3 won't move users away because the signal is still adequate.

## In Our Project

Our simulator models all of this:
- `min_rsrp_dbm = -112 dBm` (outage threshold)
- `pingpong_window_steps = 6` (if you return to previous cell within 6 steps = ping-pong)
- `handover_interruption_frac = 0.18` (18% throughput loss during HO)
- `A3HandoverPolicy` implements the standard A3 event

The goal: **beat A3** by making smarter handover decisions.

## Key Terms Quick Reference

| Term | Meaning |
|---|---|
| **UE** | User Equipment (phone/device) |
| **eNodeB / Cell** | Base station / cell tower |
| **Serving cell** | The cell currently connected to |
| **Target cell** | The cell we might hand over to |
| **RSRP** | Signal strength (dBm) |
| **RSRQ** | Signal quality (accounts for interference) |
| **A3 Event** | Standard HO trigger: target > serving + offset |
| **TTT** | Time-to-Trigger: how long A3 condition must hold |
| **CIO** | Cell Individual Offset: bias added to adjust A3 |
| **Ping-pong** | UE bouncing back and forth between cells |
| **Outage** | Signal below minimum → service failure |
