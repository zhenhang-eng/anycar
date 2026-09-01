#!/usr/bin/env python3
"""Paired action-MSE versus trajectory-equivalent J16 supervision audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arm(directory: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    return (
        json.loads((directory / "summary.json").read_text()),
        np.load(directory / "oof_evaluation.npz", allow_pickle=False),
    )


def rows(data: np.lib.npyio.NpzFile) -> dict[str, tuple[np.ndarray, float]]:
    seeds = np.unique(data["seed"])
    if len(seeds) != 1 or int(seeds[0]) != 0:
        raise ValueError("this paired pilot expects exactly seed 0")
    return {
        key: (predicted, float(gain))
        for key, predicted, gain in zip(
            data["state_keys"].astype(str),
            data["predicted_knots"],
            data["actor_gain"],
        )
    }


def metrics(predicted: np.ndarray, target: np.ndarray, gain: np.ndarray,
            j_actor: np.ndarray, teacher_gain: np.ndarray,
            summary: dict) -> dict:
    flat_predicted = predicted.reshape(len(predicted), -1)
    flat_target = target.reshape(len(target), -1)
    correlations = [
        np.corrcoef(flat_predicted[:, dim], flat_target[:, dim])[0, 1]
        for dim in range(flat_predicted.shape[1])
    ]
    errors = np.abs(predicted - target)
    return {
        "h_train_median": float(summary["per_seed"]["0"]["h_train_median"]),
        "h_oof_pooled": float(summary["per_seed"]["0"]["h_oof_pooled"]),
        "correlation_median": float(np.median(correlations)),
        "absolute_action_error_median": float(np.median(errors)),
        "early_steering_error_median": float(np.median(errors[:, :3, 1])),
        "J_mean": float(np.mean(j_actor)),
        "J_median": float(np.median(j_actor)),
        "positive_gain_fraction": float(np.mean(gain > 0.0)),
        "gain_p05": float(np.quantile(gain, 0.05)),
        "gain_worst": float(np.min(gain)),
        "teacher_headroom": float(np.sum(teacher_gain)),
    }


def main() -> None:
    args = parse_args()
    labels = np.load(args.labels, allow_pickle=False)
    label_index = {
        key: index for index, key in enumerate(labels["episodes"].astype(str))
    }
    paths = {
        "action_mse": args.action,
        "trajectory_effect": args.trajectory,
        "trajectory_hybrid_0p05": args.hybrid,
    }
    summaries = {}
    data = {}
    arm_rows = {}
    for name, path in paths.items():
        summaries[name], data[name] = load_arm(path)
        arm_rows[name] = rows(data[name])
    keys = np.asarray(sorted(set.intersection(*(
        set(value) for value in arm_rows.values()
    ))))
    if len(keys) != len(labels["episodes"]):
        raise AssertionError("paired state set is not complete")
    indices = np.asarray([label_index[key] for key in keys])
    target = labels["label_knots"][indices]
    j_a0 = labels["j_a0"][indices]
    j_teacher = labels["j_teacher"][indices]
    teacher_gain = j_a0 - j_teacher
    episodes = np.asarray([key.split("#")[0] for key in keys])
    unique_episodes = np.unique(episodes)
    rng = np.random.default_rng(260820)

    per_arm = {}
    aligned = {}
    for name, values in arm_rows.items():
        predicted = np.stack([values[key][0] for key in keys])
        gain = np.asarray([values[key][1] for key in keys])
        j_actor = j_a0 - gain
        aligned[name] = (predicted, gain, j_actor)
        per_arm[name] = metrics(
            predicted, target, gain, j_actor, teacher_gain, summaries[name]
        )

    paired = {}
    base_cost = aligned["action_mse"][2]
    for name in ("trajectory_effect", "trajectory_hybrid_0p05"):
        delta = aligned[name][2] - base_cost
        episode_delta = np.asarray([
            np.mean(delta[episodes == episode]) for episode in unique_episodes
        ])
        bootstrap = np.asarray([
            np.mean(episode_delta[
                rng.integers(len(unique_episodes), size=len(unique_episodes))
            ])
            for _ in range(args.bootstrap)
        ])
        paired[name + "_minus_action_mse"] = {
            "J_delta_mean": float(np.mean(delta)),
            "J_delta_median": float(np.median(delta)),
            "lower_J_fraction": float(np.mean(delta < 0.0)),
            "episode_bootstrap_ci95_J_delta_mean": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        }

    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "TRAJECTORY_EQUIVALENT_LOSS_REDUCES_ACTION_MSE_FAILURE_"
            "BUT_FAILS_A0_BASELINE"
        ),
        "contract": {
            "split": "train-only episode-grouped three-fold OOF",
            "seed": 0,
            "batch_size": 256,
            "epochs": 120,
            "actor": "strict-clean no-anchor G-X",
            "target": "J16 best-found",
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "sources": {
            "labels": str(args.labels.resolve()),
            "labels_sha256": sha256(args.labels),
            **{
                name: {
                    "directory": str(path.resolve()),
                    "summary_sha256": sha256(path / "summary.json"),
                    "oof_sha256": sha256(path / "oof_evaluation.npz"),
                }
                for name, path in paths.items()
            },
        },
        "per_arm": per_arm,
        "paired_vs_action_mse": paired,
        "decision": (
            "Trajectory-equivalent supervision dramatically reduces the "
            "catastrophic action-MSE failure, confirming that action-basin "
            "equivalence matters. However both trajectory arms still have "
            "negative train and OOF recovery versus a0, so pointwise teacher-"
            "trajectory matching is not a deployable objective and is not "
            "expanded to three seeds."
        ),
        "next_test": (
            "If this direction continues, use the original differentiable J50 "
            "task objective (with warm/two-center safety only at evaluation), "
            "not another trajectory/action-loss weight sweep."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
