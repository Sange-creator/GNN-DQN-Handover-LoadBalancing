# Claude Code Plan — Immediate Work

## My Role

I (Claude Code) handle code-level work: implementing missing features, fixing
bugs, verifying results, and maintaining code quality. I do NOT run long
training jobs (that's Codex) or write thesis prose (that's Gemini).

---

## Immediate Tasks (Phase 1 — Before Codex Can Run)

### Task 1: Install Missing Packages
```bash
pip install pytest pandas matplotlib seaborn
```
Verify with:
```bash
PYTHONPATH=src python3 -m pytest -q
```

### Task 2: Implement `scripts/generate_figures.py`

Currently a stub. Needs to produce publication-quality figures from eval CSVs:

**Required figures:**
1. **Bar chart: Throughput comparison** — all methods × key scenarios
   (grouped bars, error bars from CI95)
2. **Bar chart: Ping-pong rate** — all methods × key scenarios
3. **Radar/spider plot:** Multi-metric comparison (throughput, stability,
   fairness, safety) for top methods
4. **Topology generalization:** Performance on training scenarios vs unseen
   scenarios (shows transfer capability)
5. **Ablation chart:** `gnn_dqn` vs `son_gnn_dqn` across scenarios
   (shows SON layer necessity)
6. **SON CIO utilization:** Histogram of CIO values used (shows safety margin)

**Input:** `--run-dir` pointing to a run directory with `eval_20seed/` subdirectory
**Output:** Save PNGs to `{run-dir}/figures/`

**Style:** Use seaborn with `paper` context, `colorblind` palette. Font size 12.
Figure size suitable for two-column IEEE paper (3.5" wide for single, 7" for double).

### Task 3: Verify Pipeline End-to-End

Run the smoke test and verify:
```bash
python3 scripts/train.py --config configs/experiments/smoke_ue.json
python3 scripts/evaluate.py \
  --checkpoint results/runs/smoke_ue/checkpoints/gnn_dqn.pt \
  --out-dir results/runs/smoke_ue/eval_verify \
  --seeds 3
python3 scripts/summarize_evaluation.py results/runs/smoke_ue/eval_verify --no-fail
```

---

## Phase 2: After Codex Returns Results

### Task 4: Verify Codex Output
- Check all CSVs have 7 methods
- Verify no NaN values
- Confirm CI95 intervals are reasonable
- Run summarize script to check acceptance gates

### Task 5: Generate Final Figures
```bash
python3 scripts/generate_figures.py --run-dir results/runs/multiscenario_ue
```

### Task 6: Extract Key Numbers for Gemini
Format results into a clean summary that Gemini can drop into Chapter 4:
- Per-scenario performance table
- Aggregate improvement percentages
- Statistical significance indicators

---

## Phase 3: Final Polish

### Task 7: Code Cleanup
- Remove any debug prints
- Verify all docstrings are minimal and accurate
- Run final test suite
- Ensure git is clean

### Task 8: Generate Training Curve Figure
If training logs include loss/reward history, plot convergence curves.

---

## Decision: What NOT To Do
- Do NOT re-run the diagnostic (it's done, results are valid as diagnostic)
- Do NOT modify the model architecture (it's frozen for thesis)
- Do NOT change reward function or hyperparameters (locked in config)
- Do NOT start overnight training (that's Codex's job)
