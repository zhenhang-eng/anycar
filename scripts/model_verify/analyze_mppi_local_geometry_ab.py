#!/usr/bin/env python3
"""Compare clean no-anchor GT with longitudinal/Frenet geometry arms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--x", type=Path, required=True)
    parser.add_argument("--frenet", type=Path, required=True)
    parser.add_argument("--xf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def rows_for_seed(data, seed):
    mask = data["seed"] == seed
    return {
        key: (predicted, float(gain))
        for key, predicted, gain in zip(
            data["state_keys"][mask].astype(str),
            data["predicted_knots"][mask],
            data["actor_gain"][mask],
        )
    }


def metrics(predicted, target, gain, cost):
    error = np.abs(predicted - target)
    flat_predicted = predicted.reshape(len(predicted), -1)
    flat_target = target.reshape(len(target), -1)
    correlations = [
        np.corrcoef(flat_predicted[:, dim], flat_target[:, dim])[0, 1]
        for dim in range(flat_predicted.shape[1])
    ]
    return {
        "correlation_median": float(np.median(correlations)),
        "absolute_error_median": float(np.median(error)),
        "knot7_absolute_error_median": float(np.median(error[:, 7])),
        "J_median": float(np.median(cost)),
        "J_mean": float(np.mean(cost)),
        "positive_gain_fraction": float(np.mean(gain > 0.0)),
        "gain_p05": float(np.quantile(gain, 0.05)),
        "gain_worst": float(np.min(gain)),
    }


def main() -> None:
    args = parse_args()
    labels = np.load(args.labels, allow_pickle=False)
    label_index = {
        key: row for row, key in enumerate(labels["episodes"].astype(str))
    }
    reference_cost = np.minimum(labels["j_a0"], labels["j_warm"])
    paths = {
        "clean": args.baseline,
        "g_x": args.x,
        "g_f": args.frenet,
        "g_xf": args.xf,
    }
    data = {
        name: np.load(path, allow_pickle=False) for name, path in paths.items()
    }
    seeds = sorted(set(data["clean"]["seed"].tolist()))
    rng = np.random.default_rng(260820)
    per_seed = {}
    for seed in seeds:
        seed_rows = {name: rows_for_seed(value, seed) for name, value in data.items()}
        keys = np.asarray(sorted(set.intersection(*(
            set(value) for value in seed_rows.values()
        ))))
        indices = np.asarray([label_index[key] for key in keys])
        target = labels["label_knots"][indices]
        episodes = np.asarray([key.split("#")[0] for key in keys])
        unique_episodes = np.unique(episodes)
        seed_result = {}
        for name, rows in seed_rows.items():
            predicted = np.stack([rows[key][0] for key in keys])
            gain = np.asarray([rows[key][1] for key in keys])
            cost = reference_cost[indices] - gain
            seed_result[name] = metrics(predicted, target, gain, cost)
            if name == "clean":
                continue
            clean_rows = seed_rows["clean"]
            clean_gain = np.asarray([clean_rows[key][1] for key in keys])
            clean_predicted = np.stack([clean_rows[key][0] for key in keys])
            clean_cost = reference_cost[indices] - clean_gain
            delta_j = cost - clean_cost
            episode_delta = np.asarray([
                delta_j[episodes == episode].mean()
                for episode in unique_episodes
            ])
            boot = np.asarray([
                episode_delta[
                    rng.integers(len(episode_delta), size=len(episode_delta))
                ].mean()
                for _ in range(args.bootstrap)
            ])
            seed_result[name]["paired_vs_clean"] = {
                "J_delta_mean": float(delta_j.mean()),
                "J_delta_median": float(np.median(delta_j)),
                "episode_bootstrap_ci95_J_delta_mean": [
                    float(np.quantile(boot, 0.025)),
                    float(np.quantile(boot, 0.975)),
                ],
                "lower_J_fraction": float(np.mean(delta_j < 0.0)),
                "absolute_error_mean_delta": float(
                    np.mean(np.abs(predicted - target))
                    - np.mean(np.abs(clean_predicted - target))
                ),
            }
        per_seed[str(seed)] = seed_result

    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "LOCAL_X_IMPROVES_PRECISION_FRENET_NOT_DECISIVE",
        "contract": {
            "actor_inputs": "history/reference/current plus arm-local geometry",
            "anchor_feedback_first_pass_gradient_visible": False,
            "target": "J16 best-found complete 8x2 knots",
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "sources": {name: str(path.resolve()) for name, path in paths.items()},
        "labels": str(args.labels.resolve()),
        "per_seed": per_seed,
        "decision": (
            "G-X is the simplest supported precision improvement. G-F alone "
            "is weak; G-XF improves action error but does not improve rollout "
            "tail consistently. Keep G-X as the next clean architecture base."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
