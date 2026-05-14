#!/usr/bin/env bash
# Post-training pipeline: run everything after the canonical training finishes.
#
# Run this once the canonical training (multiscenario_ue_defense) completes.
# It runs normal eval, stress eval, latency benchmark, ns-3 calibration,
# composite scoring, and the previous-vs-new comparison.
#
# Usage:
#   bash scripts/post_training_pipeline.sh
#
# Override the run name (e.g. if you ran a different config):
#   RUN_NAME=multiscenario_ue_v2 bash scripts/post_training_pipeline.sh
set -euo pipefail

RUN_NAME=${RUN_NAME:-multiscenario_ue_defense}
PREV_RUN=${PREV_RUN:-multiscenario_ue}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

export PYTHONPATH="$ROOT/src"

RUN_DIR="results/runs/$RUN_NAME"
PREV_DIR="results/runs/$PREV_RUN"
CKPT="$RUN_DIR/checkpoints/best.pt"

echo "===================================================================="
echo " Post-training pipeline for: $RUN_NAME"
echo "===================================================================="

# ---- Step 0: sanity checks ----
if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: $RUN_DIR does not exist."
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: best.pt not found at $CKPT"
    echo "       The training may not have completed. Check the latest log:"
    echo "       tail -30 $RUN_DIR/train.log"
    exit 1
fi

# Detect still-running training
if pgrep -f "scripts/train.py.*${RUN_NAME}" >/dev/null 2>&1; then
    echo "WARNING: training process for $RUN_NAME appears to still be running."
    echo "         Press Ctrl+C in the next 10 seconds to abort, or this will"
    echo "         continue and evaluate against the current best.pt."
    sleep 10
fi

echo "Checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1))"
echo

# ---- Step 1: normal eval (20 seeds, 11 scenarios) ----
echo "[1/7] Normal eval (20 seeds × 11 scenarios) ..."
python3 scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --out-dir "$RUN_DIR/eval_normal" \
    --seeds 20

# ---- Step 2: stress eval (20 seeds × 4 high-congestion scenarios) ----
echo
echo "[2/7] Stress eval (20 seeds × 4 high-congestion scenarios) ..."
python3 scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --out-dir "$RUN_DIR/eval_stress" \
    --split stress \
    --seeds 20

# ---- Step 3: composite score on normal eval ----
echo
echo "[3/7] Composite deployment-readiness score (normal eval) ..."
python3 scripts/compute_composite_score.py \
    --eval-dir "$RUN_DIR/eval_normal" \
    --out-dir "$RUN_DIR/eval_normal" \
    --exclude random_valid

echo
echo "      Composite excluding no_handover (deployable methods only):"
python3 scripts/compute_composite_score.py \
    --eval-dir "$RUN_DIR/eval_normal" \
    --out-dir "$RUN_DIR/eval_normal/composite_deployable" \
    --exclude random_valid no_handover

# ---- Step 4: composite score on stress eval ----
echo
echo "[4/7] Composite deployment-readiness score (stress eval) ..."
python3 scripts/compute_composite_score.py \
    --eval-dir "$RUN_DIR/eval_stress" \
    --out-dir "$RUN_DIR/eval_stress" \
    --exclude random_valid

# ---- Step 5: inference latency ----
echo
echo "[5/7] Inference latency benchmark ..."
python3 scripts/measure_inference_latency.py \
    --checkpoint "$CKPT" \
    --num-ues 250 \
    --num-calls 1000 \
    --out "$RUN_DIR/inference_latency.json"

# ---- Step 6: ns-3 calibration ----
echo
echo "[6/7] ns-3 KS-test calibration ..."
python3 scripts/compare_ns3.py \
    --sim-seeds 3 \
    --sim-steps 100 \
    --out "$RUN_DIR/ns3_calibration.json"

# ---- Step 7: comparison vs previous run ----
echo
echo "[7/7] Previous-vs-new run comparison ..."
if [[ -d "$PREV_DIR/eval_20seed" ]]; then
    python3 scripts/compare_runs.py \
        --old "$PREV_DIR/eval_20seed" \
        --new "$RUN_DIR/eval_normal" \
        --out "$RUN_DIR/comparison_vs_previous.md"
else
    echo "      Skipped: $PREV_DIR/eval_20seed not found."
fi

# ---- Step 8: figures (best-effort) ----
echo
if [[ -f scripts/generate_figures.py ]]; then
    echo "[+] Generating figures ..."
    python3 scripts/generate_figures.py --run-dir "$RUN_DIR" || \
        echo "      generate_figures.py failed (non-fatal)."
fi

# ---- Final summary ----
echo
echo "===================================================================="
echo " DONE.  Headline numbers:"
echo "===================================================================="
echo
echo "Stress eval (the new headline result):"
for csv in "$RUN_DIR/eval_stress"/stress_*.csv; do
    [[ -f "$csv" ]] || continue
    name=$(basename "$csv" .csv)
    echo "  $name (avg_ue_throughput_mbps):"
    awk -F',' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
               $1=="son_gnn_dqn"||$1=="a3_ttt"||$1=="load_aware"||$1=="strongest_rsrp"||$1=="no_handover" {
                   printf "    %-18s %s\n", $1, $(h["avg_ue_throughput_mbps"])
               }' "$csv"
done

echo
echo "Composite ranking (deployable methods, normal eval):"
if [[ -f "$RUN_DIR/eval_normal/composite_deployable/composite_ranking.csv" ]]; then
    cat "$RUN_DIR/eval_normal/composite_deployable/composite_ranking.csv"
fi

echo
echo "Output artifacts:"
echo "  $RUN_DIR/eval_normal/"
echo "  $RUN_DIR/eval_stress/"
echo "  $RUN_DIR/inference_latency.json"
echo "  $RUN_DIR/ns3_calibration.json"
echo "  $RUN_DIR/comparison_vs_previous.md"
echo
echo "Next step: hand $RUN_DIR/comparison_vs_previous.md and the CSVs"
echo "to the thesis writer."
