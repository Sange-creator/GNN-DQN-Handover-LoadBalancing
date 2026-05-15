#!/bin/bash
# Auto-evaluation watcher: waits for fine-tune to complete, then runs eval.
# Usage: nohup bash scripts/auto_eval_on_completion.sh > auto_eval.log 2>&1 &

cd /Users/saangetamang/gnn-dqn-handover-loadbalancing
TRAIN_PID=25654
RUN_DIR=results/runs/colab_finetune_ue
EVAL_DIR=$RUN_DIR/eval_30seeds

echo "[$(date '+%H:%M:%S')] Watcher started. Monitoring PID $TRAIN_PID..."

# Wait for the training process to finish
while ps -p $TRAIN_PID > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date '+%H:%M:%S')] Training process exited. Starting evaluation..."

# Pick the best checkpoint
CKPT=$RUN_DIR/checkpoints/gnn_dqn.pt
if [ ! -f "$CKPT" ]; then
    # Fallback: use latest resume checkpoint
    CKPT=$(ls -t $RUN_DIR/checkpoints/resume/resume_ep*.pt 2>/dev/null | head -1)
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "[$(date '+%H:%M:%S')] ERROR: No checkpoint found. Aborting."
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Evaluating: $CKPT"
mkdir -p $EVAL_DIR

PYTHONPATH=src python3 scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --out-dir "$EVAL_DIR" \
    --seeds 30 \
    --split all_plus_stress

echo "[$(date '+%H:%M:%S')] Evaluation complete. Generating comparison table..."

# Generate readable summary
python3 << 'PYEOF'
import csv, glob, os
print(f"\n{'='*100}")
print(f"FINE-TUNE EVAL RESULTS: son_gnn_dqn vs a3_ttt (30 seeds each)")
print(f"{'='*100}")
print(f"{'Scenario':<26} {'Method':<14} {'AvgT':>7} {'P5':>7} {'LStd':>7} {'Jain':>7} {'PP':>8} {'HO/1k':>7}")
print("-"*92)

eval_dir = "results/runs/colab_finetune_ue/eval_30seeds"
for fp in sorted(glob.glob(f"{eval_dir}/*.csv")):
    s = os.path.basename(fp).replace(".csv","")
    rows = {}
    with open(fp) as f:
        for r in csv.DictReader(f): rows[r["method"]] = r
    for m in ["a3_ttt","son_gnn_dqn"]:
        if m not in rows: continue
        r = rows[m]
        tag = " ◄" if m=="son_gnn_dqn" else ""
        print(f"{s:<26} {m:<14} {float(r['avg_ue_throughput_mbps']):>7.3f} {float(r['p5_ue_throughput_mbps']):>7.3f} {float(r['load_std']):>7.3f} {float(r['jain_load_fairness']):>7.3f} {float(r['pingpong_rate']):>8.4f} {float(r['handovers_per_1000_decisions']):>7.1f}{tag}")
    print()
PYEOF

echo "[$(date '+%H:%M:%S')] DONE."
