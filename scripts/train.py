#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.rl import (
    load_checkpoint_payload,
    make_env_from_scenario,
    save_checkpoint,
    train_multi_scenario,
    validate_checkpoint_metadata,
    write_history,
)
from handover_gnn_dqn.rl.training import (
    dqn_config_from_dict,
    evaluate_and_write,
    son_config_from_dict,
    son_config_to_dict,
    train_flat_multi_scenario,
)
from handover_gnn_dqn.topology import (
    Scenario,
    get_stress_scenarios,
    get_test_scenarios,
    get_training_scenarios,
)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


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
    parser.add_argument(
        "--allow-existing-out-dir",
        action="store_true",
        help="Allow writing into a non-empty output directory. Prefer archiving old runs instead.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = cfg.get("run_name", args.config.stem)
    out_dir = Path(cfg.get("out_dir", f"results/runs/{run_name}"))
    if (
        out_dir.exists()
        and any(out_dir.iterdir())
        and args.resume is None
        and not args.allow_existing_out_dir
    ):
        raise SystemExit(
            f"{out_dir} already contains outputs. Archive or move it before a clean run, "
            "or pass --allow-existing-out-dir for an intentional overwrite/trial."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_mode = cfg.get("feature_mode", "ue_only")
    prb_available = bool(cfg.get("prb_available", feature_mode != "ue_only"))
    seed = int(cfg.get("seed", 42))
    episodes = int(cfg.get("episodes", 10))
    steps = int(cfg.get("steps_per_episode", 40))
    eval_seeds = int(cfg.get("eval_seeds", 0))

    base_train_scenarios = get_training_scenarios(seed=seed)
    base_test_scenarios = get_test_scenarios(seed=seed + 57)
    base_stress_scenarios = get_stress_scenarios(seed=seed + 137)
    # Stress scenarios are normally evaluation-only, but time-critical defense
    # runs may include capped stress variants in training. The evaluation still
    # uses disjoint seeds; do not claim scenario-family holdout when configured.
    train_scenarios = select_scenarios(base_train_scenarios + base_stress_scenarios, cfg.get("train_scenarios"))
    test_scenarios = select_scenarios(base_test_scenarios + base_stress_scenarios, cfg.get("test_scenarios"))
    dqn_cfg = dqn_config_from_dict(cfg.get("dqn", {}))

    # Validation pass config (held-out, epsilon=0). Validation scenarios may be a subset
    # of training scenarios or held-out test scenarios; either is acceptable as long as
    # the seed list is disjoint from training seeds.
    val_names = cfg.get("validation_scenarios")
    if val_names:
        # Allow validation scenarios to be drawn from either training or test pool
        all_known = {
            s.name: s
            for s in (train_scenarios + test_scenarios + base_train_scenarios + base_test_scenarios + base_stress_scenarios)
        }
        unknown = [n for n in val_names if n not in all_known]
        if unknown:
            raise ValueError(f"Unknown validation_scenarios: {unknown}")
        validation_scenarios = [all_known[n] for n in val_names]
    else:
        validation_scenarios = []
    n_val_seeds = int(cfg.get("validation_seeds", 0))
    validate_every_episodes = int(cfg.get("validate_every_episodes", 0))
    validation_steps = int(cfg.get("validation_steps", 0))
    # Validation seeds drawn from a disjoint range so they never overlap with training
    validation_seed_list = (
        [seed + 50_000 + i * 71 for i in range(n_val_seeds)] if n_val_seeds > 0 else []
    )
    # §9 new config fields
    validation_ue_cap = cfg.get("validation_ue_cap")
    validation_steps_override = cfg.get("validation_steps_override")
    skip_validation_epsilon_above = cfg.get("skip_validation_epsilon_above")
    early_stopping_min_episodes = int(cfg.get("early_stopping_min_episodes", 0))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    scenario_sampling_weights = cfg.get("scenario_sampling_weights")
    behavioral_clone_episodes = int(cfg.get("behavioral_clone_episodes", 0))
    flat_dqn_episodes = int(cfg.get("flat_dqn_episodes", 0))
    log_every_episodes = cfg.get("log_every_episodes")
    steps_per_episode_curriculum = cfg.get("steps_per_episode_curriculum")
    training_ue_caps = cfg.get("training_ue_caps")
    son_cfg = son_config_from_dict(cfg.get("son_config"))
    cfg["son_config"] = son_config_to_dict(son_cfg)

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

        # Warn if resume checkpoint omits replay buffer — biases first thousands of grad steps
        # to whichever scenario is sampled first after restart. Tag metadata so downstream tools
        # (summarize_evaluation, generate_figures) can flag the run as exploratory.
        has_replay = resume_payload.get("replay_state") is not None
        if has_replay:
            cfg["resume_mode"] = "warm_start_with_replay"
        else:
            cfg["resume_mode"] = "warm_start_no_replay"
            print("=" * 72, file=sys.stderr)
            print("WARNING: resume checkpoint does NOT contain replay buffer.", file=sys.stderr)
            print(f"         Source: {args.resume}", file=sys.stderr)
            print("         Network state restored, replay buffer is empty.", file=sys.stderr)
            print("         The next several thousand gradient steps will be biased toward", file=sys.stderr)
            print("         the first scenario sampled after resume. Results from this run", file=sys.stderr)
            print("         must be treated as EXPLORATORY and not cited as final evidence.", file=sys.stderr)
            print("         To do a clean resume: re-run with 'checkpoint_include_replay': true", file=sys.stderr)
            print("         in the config, then resume from a checkpoint saved with that flag.", file=sys.stderr)
            print("=" * 72, file=sys.stderr)

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
        validation_scenarios=validation_scenarios,
        validation_seeds=validation_seed_list,
        validate_every_episodes=validate_every_episodes,
        validation_steps=validation_steps,
        validation_ue_cap=validation_ue_cap,
        validation_steps_override=validation_steps_override,
        skip_validation_epsilon_above=skip_validation_epsilon_above,
        early_stopping_min_episodes=early_stopping_min_episodes,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        scenario_sampling_weights=scenario_sampling_weights,
        son_config=son_cfg,
        behavioral_clone_episodes=behavioral_clone_episodes,
        log_every_episodes=(
            int(log_every_episodes) if log_every_episodes is not None else None
        ),
        steps_per_episode_curriculum=steps_per_episode_curriculum,
        scenario_ue_caps=training_ue_caps,
    )

    # Prefer the honest held-out validation score for best-row selection.
    # Fall back to training-metric score for configs without validation.
    validation_was_used = validation_scenarios and validate_every_episodes > 0
    score_key = "val_holdout_validation_score" if validation_was_used else "validation_score"
    if validation_was_used:
        min_select_episode = max(6, episodes // 10 + 1)
        rows_with_score = [
            r for r in history
            if score_key in r and int(r.get("episode", 0)) >= min_select_episode
        ]
    else:
        rows_with_score = history
    best_row = (
        max(
            rows_with_score,
            key=lambda row: (
                row.get(score_key, float("-inf")),
                int(row.get("episode", 0)),
            ),
        )
        if rows_with_score
        else (history[-1] if history else {})
    )
    checkpoint = out_dir / "checkpoints" / "gnn_dqn.pt"
    best_score_for_metadata = float(
        best_row.get(score_key, best_row.get("validation_score", 0.0))
    )
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
            "episode_completed": int(best_row.get("episode", episodes)),
            "history_rows": len(history),
            "behavioral_clone_episodes": behavioral_clone_episodes,
            "best_episode": int(best_row.get("episode", 0.0)),
            "best_score": best_score_for_metadata,
            "best_score_source": score_key,
        },
        checkpoint_kind="best_model",
    )
    best_pt = out_dir / "checkpoints" / "best.pt"
    shutil.copy2(checkpoint, best_pt)
    write_history(history, out_dir / "history.json")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved checkpoint: {checkpoint} (also {best_pt})")
    print(
        f"Best episode: {int(best_row.get('episode', 0))} "
        f"({score_key}={best_score_for_metadata:+.3f})"
    )

    # --- Flat-DQN ablation (no GNN, same replay/scenarios) ---
    flat_agent = None
    if flat_dqn_episodes > 0:
        print(f"\n=== Flat-DQN ablation ({flat_dqn_episodes} episodes) ===")
        flat_agent, flat_history, _ = train_flat_multi_scenario(
            train_scenarios,
            dqn_cfg,
            total_episodes=flat_dqn_episodes,
            steps_per_episode=steps,
            feature_mode=feature_mode,
            prb_available=prb_available,
            seed=seed + 7000,
            verbose=True,
            scenario_sampling_weights=scenario_sampling_weights,
        )
        flat_ckpt = out_dir / "checkpoints" / "flat_dqn.pt"
        flat_ckpt.parent.mkdir(parents=True, exist_ok=True)
        import torch as _torch
        _torch.save({"state_dict": flat_agent.state_dict(), "history": flat_history}, flat_ckpt)
        print(f"Saved flat-DQN checkpoint: {flat_ckpt}")

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
            flat_agent=flat_agent,
            son_config=son_cfg,
        )
        print(f"Wrote evaluation CSVs: {eval_dir}")


if __name__ == "__main__":
    main()
