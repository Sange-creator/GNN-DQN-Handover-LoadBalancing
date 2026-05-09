#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.models import DQNConfig
from handover_gnn_dqn.rl import (
    load_checkpoint_payload,
    make_env_from_scenario,
    save_checkpoint,
    train_multi_scenario,
    validate_checkpoint_metadata,
    write_history,
)
from handover_gnn_dqn.rl.training import evaluate_and_write
from handover_gnn_dqn.topology import Scenario, get_test_scenarios, get_training_scenarios


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def dqn_config(values: dict) -> DQNConfig:
    valid = {f.name for f in fields(DQNConfig)}
    return DQNConfig(**{k: v for k, v in values.items() if k in valid})


def select_scenarios(all_scenarios: list[Scenario], names: list[str] | None) -> list[Scenario]:
    if not names:
        return all_scenarios
    by_name = {s.name: s for s in all_scenarios}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown scenario names: {missing}")
    return [by_name[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN-DQN from a JSON experiment config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a compatible resume checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg.get("run_name", args.config.stem)
    out_dir = Path(cfg.get("out_dir", f"results/runs/{run_name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_mode = cfg.get("feature_mode", "ue_only")
    prb_available = bool(cfg.get("prb_available", feature_mode != "ue_only"))
    seed = int(cfg.get("seed", 42))
    episodes = int(cfg.get("episodes", 10))
    steps = int(cfg.get("steps_per_episode", 40))
    eval_seeds = int(cfg.get("eval_seeds", 0))

    train_scenarios = select_scenarios(get_training_scenarios(seed=seed), cfg.get("train_scenarios"))
    test_scenarios = select_scenarios(get_test_scenarios(seed=seed + 57), cfg.get("test_scenarios"))
    dqn_cfg = dqn_config(cfg.get("dqn", {}))
    sample_env = make_env_from_scenario(train_scenarios[0], feature_mode=feature_mode, prb_available=prb_available)
    resume_payload = None
    if args.resume is not None:
        resume_payload = load_checkpoint_payload(args.resume)
        validate_checkpoint_metadata(
            resume_payload["metadata"],
            expected_feature_profile=feature_mode,
            expected_feature_dim=sample_env.feature_dim,
        )
        if resume_payload["metadata"].get("checkpoint_kind") != "resume":
            raise ValueError(
                f"{args.resume} is a {resume_payload['metadata'].get('checkpoint_kind')!r} checkpoint; "
                "resume training requires a checkpoint from checkpoints/resume/"
            )
        saved_scenarios = resume_payload["metadata"].get("scenario_names", [])
        requested_scenarios = [s.name for s in train_scenarios]
        if saved_scenarios and saved_scenarios != requested_scenarios:
            raise ValueError(
                "Incompatible checkpoint scenarios: "
                f"{saved_scenarios!r} != {requested_scenarios!r}"
            )

    print(f"=== {run_name} ===")
    print(f"Feature profile: {feature_mode} (prb_available={prb_available})")
    print(f"Training scenarios: {[s.name for s in train_scenarios]}")
    print(f"Episodes: {episodes}, steps/episode: {steps}, seed={seed}")

    agent, history, max_cells = train_multi_scenario(
        train_scenarios,
        dqn_cfg,
        total_episodes=episodes,
        steps_per_episode=steps,
        feature_mode=feature_mode,
        prb_available=prb_available,
        seed=seed,
        verbose=True,
        resume_payload=resume_payload,
        checkpoint_dir=out_dir / "checkpoints" / "resume",
        checkpoint_every_episodes=int(cfg.get("checkpoint_every_episodes", 0)),
        checkpoint_include_replay=bool(cfg.get("checkpoint_include_replay", False)),
        checkpoint_config=cfg,
        cwd=ROOT,
    )

    best_row = max(history, key=lambda row: row.get("validation_score", float("-inf"))) if history else {}
    checkpoint = out_dir / "checkpoints" / "gnn_dqn.pt"
    save_checkpoint(
        agent,
        checkpoint,
        config=cfg,
        feature_dim=sample_env.feature_dim,
        max_cells=max_cells,
        scenario_names=[s.name for s in train_scenarios],
        history=history,
        cwd=ROOT,
        training_state={
            "episode_completed": len(history),
            "best_episode": int(best_row.get("episode", 0.0)),
            "best_score": float(best_row.get("validation_score", 0.0)),
        },
        checkpoint_kind="best_model",
    )
    write_history(history, out_dir / "history.json")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved checkpoint: {checkpoint}")

    if eval_seeds > 0:
        seeds = [seed + 10_000 + i * 37 for i in range(eval_seeds)]
        eval_dir = out_dir / "evaluation"
        evaluate_and_write(
            train_scenarios + test_scenarios,
            agent,
            eval_dir,
            feature_mode=feature_mode,
            prb_available=prb_available,
            steps=steps,
            seeds=seeds,
        )
        print(f"Wrote evaluation CSVs: {eval_dir}")


if __name__ == "__main__":
    main()
