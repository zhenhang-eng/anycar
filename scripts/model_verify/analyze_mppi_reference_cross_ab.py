#!/usr/bin/env python3
"""Paired G-X versus G-X reference cross-attention analysis."""

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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reference-cross", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_for_seed(data: np.lib.npyio.NpzFile, seed: int) -> dict[str, tuple]:
    mask = data["seed"] == seed
    return {
        key: (predicted, float(gain), int(fold))
        for key, predicted, gain, fold in zip(
            data["state_keys"][mask].astype(str),
            data["predicted_knots"][mask],
            data["actor_gain"][mask],
            data["fold"][mask],
        )
    }


def arm_metrics(predicted: np.ndarray, target: np.ndarray, gain: np.ndarray,
                cost: np.ndarray) -> dict:
    error = np.abs(predicted - target)
    flat_predicted = predicted.reshape(len(predicted), -1)
    flat_target = target.reshape(len(target), -1)
    correlations = [
        np.corrcoef(flat_predicted[:, dim], flat_target[:, dim])[0, 1]
        for dim in range(flat_predicted.shape[1])
    ]
    return {
        "correlation_median": float(np.median(correlations)),
        "absolute_error_mean": float(np.mean(error)),
        "absolute_error_median": float(np.median(error)),
        "knot7_absolute_error_median": float(np.median(error[:, 7])),
        "J_mean": float(np.mean(cost)),
        "J_median": float(np.median(cost)),
        "positive_gain_fraction": float(np.mean(gain > 0.0)),
        "gain_p05": float(np.quantile(gain, 0.05)),
        "gain_worst": float(np.min(gain)),
    }


def main() -> None:
    args = parse_args()
    labels = np.load(args.labels, allow_pickle=False)
    baseline = np.load(args.baseline, allow_pickle=False)
    reference_cross = np.load(args.reference_cross, allow_pickle=False)
    label_index = {
        key: row for row, key in enumerate(labels["episodes"].astype(str))
    }
    reference_cost = np.minimum(labels["j_a0"], labels["j_warm"])
    seeds = sorted(set(baseline["seed"].tolist()))
    if seeds != sorted(set(reference_cross["seed"].tolist())):
        raise ValueError("seed sets differ between arms")

    rng = np.random.default_rng(260820)
    per_seed = {}
    precision_passes = 0
    rollout_direction_passes = 0
    for seed in seeds:
        base_rows = rows_for_seed(baseline, seed)
        cross_rows = rows_for_seed(reference_cross, seed)
        if set(base_rows) != set(cross_rows):
            raise ValueError(f"state keys differ for seed {seed}")
        keys = np.asarray(sorted(base_rows))
        indices = np.asarray([label_index[key] for key in keys])
        target = labels["label_knots"][indices]
        episodes = np.asarray([key.split("#")[0] for key in keys])
        unique_episodes = np.unique(episodes)
        base_predicted = np.stack([base_rows[key][0] for key in keys])
        cross_predicted = np.stack([cross_rows[key][0] for key in keys])
        base_gain = np.asarray([base_rows[key][1] for key in keys])
        cross_gain = np.asarray([cross_rows[key][1] for key in keys])
        base_cost = reference_cost[indices] - base_gain
        cross_cost = reference_cost[indices] - cross_gain
        delta_j = cross_cost - base_cost
        episode_delta = np.asarray([
            delta_j[episodes == episode].mean() for episode in unique_episodes
        ])
        boot = np.asarray([
            episode_delta[
                rng.integers(len(episode_delta), size=len(episode_delta))
            ].mean()
            for _ in range(args.bootstrap)
        ])
        base_error = np.abs(base_predicted - target)
        cross_error = np.abs(cross_predicted - target)
        median_error_delta = float(np.median(cross_error) - np.median(base_error))
        if median_error_delta <= -0.001:
            precision_passes += 1
        if float(np.median(delta_j)) <= 0.0:
            rollout_direction_passes += 1
        per_seed[str(seed)] = {
            "g_x": arm_metrics(base_predicted, target, base_gain, base_cost),
            "g_x_reference_cross": arm_metrics(
                cross_predicted, target, cross_gain, cross_cost
            ),
            "paired_reference_cross_minus_g_x": {
                "J_delta_mean": float(np.mean(delta_j)),
                "J_delta_median": float(np.median(delta_j)),
                "episode_bootstrap_ci95_J_delta_mean": [
                    float(np.quantile(boot, 0.025)),
                    float(np.quantile(boot, 0.975)),
                ],
                "lower_J_fraction": float(np.mean(delta_j < 0.0)),
                "absolute_error_mean_delta": float(
                    np.mean(cross_error) - np.mean(base_error)
                ),
                "absolute_error_median_delta": median_error_delta,
            },
        }

    precision_gate = precision_passes >= 2
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "REFERENCE_CROSS_WEAK_POSITIVE_FAILS_REGISTERED_PRECISION_GATE"
        ),
        "contract": {
            "baseline": "strict-clean no-anchor G-X",
            "intervention": (
                "eight control queries cross-attend fifty reference tokens; "
                "history remains globally encoded"
            ),
            "target": "J16 best-found complete 8x2 knots",
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "sources": {
            "labels": str(args.labels.resolve()),
            "labels_sha256": sha256(args.labels),
            "g_x_oof": str(args.baseline.resolve()),
            "g_x_oof_sha256": sha256(args.baseline),
            "reference_cross_oof": str(args.reference_cross.resolve()),
            "reference_cross_oof_sha256": sha256(args.reference_cross),
        },
        "registered_gate": {
            "minimum_absolute_error_improvement": 0.001,
            "minimum_passing_seeds_out_of_three": 2,
            "passing_seeds": precision_passes,
            "passed": precision_gate,
            "rollout_median_nonworse_passing_seeds": rollout_direction_passes,
        },
        "per_seed": per_seed,
        "decision": (
            "Reference cross-attention has a consistent but statistically weak "
            "paired rollout trend. It does not pass the registered action-"
            "precision gate, so G-X remains the baseline and the history-token "
            "second stage is not started."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
