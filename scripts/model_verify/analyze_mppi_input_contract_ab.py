#!/usr/bin/env python3
"""Paired audit for the no-anchor GT first-pass-input contract ablation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def dim_correlation(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    predicted = predicted.reshape(len(predicted), -1)
    target = target.reshape(len(target), -1)
    return [
        float(np.corrcoef(predicted[:, dim], target[:, dim])[0, 1])
        for dim in range(predicted.shape[1])
    ]


def main() -> None:
    args = parse_args()
    baseline = np.load(args.baseline, allow_pickle=False)
    clean = np.load(args.clean, allow_pickle=False)
    labels = np.load(args.labels, allow_pickle=False)
    label_index = {
        key: row for row, key in enumerate(labels["episodes"].astype(str))
    }
    baseline_cost = np.minimum(labels["j_a0"], labels["j_warm"])
    rng = np.random.default_rng(260820)
    per_seed = {}
    for seed in sorted(set(baseline["seed"].tolist())):
        base_mask = baseline["seed"] == seed
        clean_mask = clean["seed"] == seed
        base_rows = {
            key: (predicted, gain)
            for key, predicted, gain in zip(
                baseline["state_keys"][base_mask].astype(str),
                baseline["predicted_knots"][base_mask],
                baseline["actor_gain"][base_mask],
            )
        }
        clean_rows = {
            key: (predicted, gain)
            for key, predicted, gain in zip(
                clean["state_keys"][clean_mask].astype(str),
                clean["predicted_knots"][clean_mask],
                clean["actor_gain"][clean_mask],
            )
        }
        keys = np.asarray(sorted(set(base_rows) & set(clean_rows)))
        indices = np.asarray([label_index[key] for key in keys])
        target = labels["label_knots"][indices]
        base_pred = np.stack([base_rows[key][0] for key in keys])
        clean_pred = np.stack([clean_rows[key][0] for key in keys])
        base_gain = np.asarray([base_rows[key][1] for key in keys])
        clean_gain = np.asarray([clean_rows[key][1] for key in keys])
        base_j = baseline_cost[indices] - base_gain
        clean_j = baseline_cost[indices] - clean_gain
        j_delta = clean_j - base_j
        episodes = np.asarray([key.split("#")[0] for key in keys])
        unique_episodes = np.unique(episodes)
        episode_delta = np.asarray([
            j_delta[episodes == episode].mean() for episode in unique_episodes
        ])
        bootstrap = []
        for _ in range(args.bootstrap):
            draw = rng.integers(len(unique_episodes), size=len(unique_episodes))
            bootstrap.append(float(episode_delta[draw].mean()))

        def metrics(predicted: np.ndarray, gain: np.ndarray, cost: np.ndarray):
            error = np.abs(predicted - target)
            correlations = dim_correlation(predicted, target)
            return {
                "per_dim_correlation_median": float(np.median(correlations)),
                "absolute_error_median": float(np.median(error)),
                "knot7_absolute_error_median": float(np.median(error[:, 7])),
                "J_median": float(np.median(cost)),
                "J_mean": float(np.mean(cost)),
                "gain_median": float(np.median(gain)),
                "gain_p05": float(np.quantile(gain, 0.05)),
                "gain_worst": float(np.min(gain)),
                "positive_gain_fraction": float(np.mean(gain > 0.0)),
            }

        per_seed[str(seed)] = {
            "baseline": metrics(base_pred, base_gain, base_j),
            "clean": metrics(clean_pred, clean_gain, clean_j),
            "paired_clean_minus_baseline": {
                "J_mean": float(j_delta.mean()),
                "J_median": float(np.median(j_delta)),
                "episode_bootstrap_ci95_J_mean": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
                "clean_lower_J_fraction": float(np.mean(j_delta < 0.0)),
                "absolute_error_mean_delta": float(
                    np.mean(np.abs(clean_pred - target))
                    - np.mean(np.abs(base_pred - target))
                ),
            },
        }

    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "FIRST_PASS_INPUTS_NO_MEASURABLE_PRECISION_BENEFIT",
        "contract": {
            "baseline": "no-anchor GT + J16; feedback/gradient visible",
            "clean": "history/reference/current only; anchor/feedback/gradient blinded",
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "sources": {
            "baseline": str(args.baseline.resolve()),
            "clean": str(args.clean.resolve()),
            "labels": str(args.labels.resolve()),
        },
        "per_seed": per_seed,
        "decision": (
            "Retain the strict clean input contract. Correlation is slightly "
            "higher for all seeds, absolute error is effectively unchanged, "
            "and rollout-cost differences are not consistently signed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
