#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handover_gnn_dqn.rl import load_checkpoint_payload
from handover_gnn_dqn.rl.training import MODEL_VERSION, REWARD_VERSION, validate_checkpoint_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate GNN-DQN checkpoint metadata.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--feature-profile", default=None)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--reward-version", default=REWARD_VERSION)
    parser.add_argument("--allow-reward-version", action="append", default=[])
    args = parser.parse_args()

    payload = load_checkpoint_payload(args.checkpoint)
    metadata = payload["metadata"]
    validate_checkpoint_metadata(
        metadata,
        expected_feature_profile=args.feature_profile,
        expected_feature_dim=args.feature_dim,
        expected_model_version=args.model_version,
    )

    accepted_rewards = {args.reward_version, *args.allow_reward_version}
    reward_version = metadata.get("reward_version")
    if reward_version not in accepted_rewards:
        raise SystemExit(
            f"Incompatible checkpoint: reward_version={reward_version!r}, "
            f"expected one of {sorted(accepted_rewards)!r}"
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_kind": metadata.get("checkpoint_kind"),
        "model_version": metadata.get("model_version"),
        "reward_version": reward_version,
        "feature_profile": metadata.get("feature_profile"),
        "prb_available": metadata.get("prb_available"),
        "feature_dim": metadata.get("feature_dim"),
        "max_cells": metadata.get("max_cells"),
        "scenario_names": metadata.get("scenario_names"),
        "git_commit": metadata.get("git_commit"),
        "best_episode": metadata.get("best_episode"),
        "best_score": metadata.get("best_score"),
        "training_state": metadata.get("training_state"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
