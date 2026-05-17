"""Targeted adaptive eval: pokhara_dense_peakhour + kathmandu_real only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.handover_gnn_dqn.metrics.experiment as exp
from src.handover_gnn_dqn.rl.training import evaluate_and_write, load_gnn_checkpoint, son_config_from_dict
from src.handover_gnn_dqn.topology import get_training_scenarios, get_test_scenarios

CHECKPOINT = ROOT / "results/runs/multiscenario_ue_defense/checkpoints/resume/resume_ep0160.pt"
OUT_DIR = ROOT / "results/runs/adaptive_pokhara_kathmandu_20seeds"
SEEDS = 10
STEPS = 80
TARGET = {"pokhara_dense_peakhour", "kathmandu_real"}

def main():
    agent, meta, _ = load_gnn_checkpoint(CHECKPOINT)
    cfg = meta["config"]
    son_cfg = son_config_from_dict(dict(cfg.get("son_config", {})))
    seed = int(cfg.get("seed", 42))

    train = get_training_scenarios(seed=seed)
    test = get_test_scenarios(seed=seed + 57)
    scenarios = [s for s in train + test if s.name in TARGET]
    print(f"Running: {[s.name for s in scenarios]}")

    orig = exp.default_policy_factories
    def targeted_factories(*a, **kw):
        all_p = orig(*a, **kw)
        return {k: v for k, v in all_p.items() if k in ("a3_ttt", "son_gnn_dqn", "adaptive_son_gnn_dqn")}
    exp.default_policy_factories = targeted_factories

    seeds = [seed + 20_000 + i * 37 for i in range(SEEDS)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluate_and_write(
        scenarios, agent, OUT_DIR,
        feature_mode=cfg.get("feature_mode", "ue_only"),
        prb_available=bool(cfg.get("prb_available", cfg.get("feature_mode") != "ue_only")),
        steps=STEPS,
        seeds=seeds,
        son_config=son_cfg,
    )
    print(f"Done → {OUT_DIR}")

if __name__ == "__main__":
    main()
