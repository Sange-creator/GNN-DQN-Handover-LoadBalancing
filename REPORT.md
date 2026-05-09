# Experiment Summary

Command used:

```bash
python3 run_experiment.py --train-episodes 12 --steps 60 --test-episodes 5
```

## Result Table

| Method | Avg UE Mbps | P5 UE Mbps | Total Mbps | Load Std | Jain Load | Outage | Overload | HO/1000 | Ping-pong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GNN-DQN | 9.777 | 5.452 | 527.961 | 0.138 | 0.974 | 0.000 | 0.202 | 2.346 | 0.000 |
| Load-aware heuristic | 9.512 | 5.266 | 513.674 | 0.064 | 0.994 | 0.000 | 0.014 | 226.111 | 0.385 |
| Strongest RSRP | 9.192 | 5.061 | 496.383 | 0.296 | 0.886 | 0.000 | 0.271 | 14.568 | 0.000 |
| A3 time-to-trigger | 9.189 | 5.080 | 496.196 | 0.298 | 0.885 | 0.000 | 0.269 | 13.025 | 0.000 |
| No handover | 8.937 | 4.709 | 482.580 | 0.356 | 0.848 | 0.000 | 0.309 | 0.000 | 0.000 |

## Headline Comparison

Compared with the best regular handover baseline, `strongest_rsrp`, the GNN-DQN controller achieved:

- 6.4% higher average UE throughput
- 7.7% higher fifth-percentile UE throughput
- 53.5% lower cell-load standard deviation
- 83.9% fewer handovers than strongest RSRP

Compared with the myopic load-aware heuristic, GNN-DQN achieved higher throughput with far fewer handovers, but the heuristic had lower load variance because it aggressively reassigns users and creates many ping-pong events.

## Interpretation

Regular RSRP and A3 handover rules optimize radio signal quality, so users tend to concentrate on locally strong cells even when those cells are congested. The GNN-DQN state includes signal strength, serving-cell status, cell load, spare capacity, UE distribution, and graph-neighbor context. This lets the learned policy choose target cells that trade radio quality against congestion and handover cost.

This prototype is a system-level simulation, not a packet-level ns-3 validation. For publication-grade work, the same policy interface should be connected to ns-3 LTE/EPC traces and evaluated with confidence intervals across more seeds.

## ns-3 Trace Export

An ns-3.40 LTE/EPC trace exporter was added at:

```text
/Users/saangetamang/ns-allinone-3.40/ns-3.40/scratch/gnn-dqn-handover-data.cc
```

Verified command:

```bash
python3 tools/run_ns3_dataset.py --sim-time 30 --sample-period 1 --run 3 --skip-build
```

Generated files:

```text
data/ns3/samples_run_3.csv
data/ns3/handovers_run_3.csv
data/ns3/summary_run_3.csv
```

The verified ns-3 run produced 725 per-UE sample rows across 25 UEs, with 16 successful handover events. The exporter uses LTE/EPC traffic, A3-RSRP handover, 7 eNBs, 25 moving UEs, per-UE downlink throughput, serving cell, position, estimated RSRP to each cell, per-cell load, and per-cell UE count.
