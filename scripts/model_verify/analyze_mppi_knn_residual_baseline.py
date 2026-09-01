#!/usr/bin/env python3
"""KNN residual-retrieval baseline for the A0/A1 routing decision.

Before choosing between A2 (early/late heads) and coverage expansion, answer
a cheaper question: can a plain nearest-train-neighbor retrieval of the
sigma-normalized label residual produce positive out-of-fold recovery on
episode-grouped splits? Phase 0.1b showed nearest-neighbor residual
coherence transfers across episodes (flip 0.112, k=1 cosine 0.494); this
baseline converts that label geometry into the same rollout-recovery metric
the actors are judged by.

- 3-fold episode-grouped split, speed-stratified (own deterministic split;
  not bit-identical to the A0b partition but the same discipline).
- Distance: block-equal IQR metric over six/reference/current/a0, fitted on
  the train fold only (the metric validated in Phase 0.1b).
- Prediction: the nearest train state's normalized label residual scaled by
  the query's own sigma; k=1 primary, k=3 mean secondary.
- Evaluation: real DBM rollouts; per-fold and pooled OOF recovery, P05/worst
  and distance stats. Teacher recovery shown for reference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import (
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
)

DEFAULT_LABELS = Path(
    "outputs/mppi_proposal/consensus64_labels_20260818_v1/labels.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/proximal_search_phase1b_20260818_v1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/knn_residual_baseline_20260818_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def robust_scale(block: np.ndarray) -> np.ndarray:
    quarter = np.quantile(block, [0.25, 0.75], axis=0)
    scale = quarter[1] - quarter[0]
    return np.where(scale > 1e-9, scale, np.std(block, axis=0) + 1e-9)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = np.load(args.labels, allow_pickle=False)
    manifest = json.loads(args.manifest.read_text())
    keys = [
        f"{row['episode']}#{row['snapshot']}" for row in manifest["states"]
    ]
    label_keys = [str(value) for value in labels["episodes"]]
    if label_keys != keys:
        raise AssertionError("labels/manifest order mismatch")

    loader_args = SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=args.repeat,
    )
    by_key = {
        f"{s['episode']}#{s['snapshot']}": s for s in load_states(loader_args)
    }
    states = [by_key[key] for key in keys]
    count = len(states)
    device = torch.device(args.device)
    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    low = np.tile(np.asarray(params.action_min, np.float32), 8)
    high = np.tile(np.asarray(params.action_max, np.float32), 8)

    label_knots = labels["label_knots"].reshape(count, -1).astype(np.float32)
    a0_flat = np.stack([s["a0"].reshape(-1) for s in states]).astype(np.float32)
    sigma_tiled = np.stack(
        [np.repeat(s["sigma"], 8) for s in states]
    ).astype(np.float32)
    normalized_label = (label_knots - a0_flat) / sigma_tiled
    j_a0 = labels["j_a0"].astype(np.float64)
    j16 = labels["j16"].astype(np.float64)

    blocks = {
        "six": np.stack([s["initial_state_six"] for s in states]).astype(np.float32),
        "reference": np.stack(
            [s["reference"].reshape(-1) for s in states]
        ).astype(np.float32),
        "control": np.stack([s["current_action"] for s in states]).astype(np.float32),
        "anchor": a0_flat,
    }
    episodes = np.asarray([s["episode"] for s in states])
    speeds = np.asarray([s["speed"] for s in states])
    unique_episodes, episode_index = np.unique(episodes, return_inverse=True)
    episode_speed = {}
    for position in range(count):
        episode_speed.setdefault(int(episode_index[position]), []).append(
            speeds[position]
        )
    episode_order = sorted(
        range(len(unique_episodes)),
        key=lambda index: (float(np.mean(episode_speed[index])), index),
    )
    fold_of_episode = {
        index: position % args.folds
        for position, index in enumerate(episode_order)
    }
    fold_of_state = np.asarray(
        [fold_of_episode[int(episode_index[i])] for i in range(count)]
    )

    def evaluate(centers: np.ndarray, rows: np.ndarray) -> np.ndarray:
        knots = torch.as_tensor(
            centers.reshape(-1, 8, 2)[:, None],
            dtype=torch.float32, device=device,
        )
        actions = interpolate_knots(knots, params.horizon)
        with torch.no_grad():
            return batched_cost(
                backend, weights, actions,
                torch.as_tensor(
                    np.stack([states[i]["initial_state_six"] for i in rows]),
                    dtype=torch.float32, device=device,
                ),
                torch.as_tensor(
                    np.stack([states[i]["current_action"] for i in rows]),
                    dtype=torch.float32, device=device,
                ),
                torch.as_tensor(
                    np.stack([states[i]["reference"] for i in rows]),
                    dtype=torch.float32, device=device,
                ),
            ).squeeze(-1).cpu().numpy().astype(np.float64)

    report = {"per_fold": {}, "k": {}}
    pooled_gain = {1: [], 3: []}
    pooled_denominator = 0.0
    nearest_distances = []
    for fold in range(args.folds):
        test_rows = np.flatnonzero(fold_of_state == fold)
        train_rows = np.flatnonzero(fold_of_state != fold)
        scales = {
            name: robust_scale(value[train_rows]).astype(np.float32)
            for name, value in blocks.items()
        }

        def embed(rows: np.ndarray) -> np.ndarray:
            pieces = []
            for name, value in blocks.items():
                scaled = value[rows] / scales[name]
                pieces.append(scaled / np.sqrt(scaled.shape[1]))
            return np.concatenate(pieces, axis=1).astype(np.float64)

        query = embed(test_rows)
        bank = embed(train_rows)
        squared = (
            np.sum(query * query, axis=1)[:, None]
            + np.sum(bank * bank, axis=1)[None, :]
            - 2.0 * (query @ bank.T)
        )
        np.maximum(squared, 0.0, out=squared)
        distances = np.sqrt(squared)
        order = np.argsort(distances, axis=1)
        fold_gain = {}
        for k in (1, 3):
            neighbor_rows = train_rows[order[:, :k]]
            neighbor_residual = normalized_label[neighbor_rows].mean(axis=1)
            centers = np.clip(
                a0_flat[test_rows] + neighbor_residual * sigma_tiled[test_rows],
                low, high,
            )
            costs = evaluate(centers, test_rows)
            gain = j_a0[test_rows] - costs
            denominator = float(np.sum(j_a0[test_rows] - j16[test_rows]))
            fold_gain[k] = {
                "oof_recovery": float(np.sum(gain) / denominator),
                "gain_p05": float(np.quantile(gain, 0.05)),
                "gain_worst": float(np.min(gain)),
                "regression_fraction": float(np.mean(gain < -1e-6)),
            }
            pooled_gain[k].extend(gain.tolist())
        pooled_denominator += float(np.sum(j_a0[test_rows] - j16[test_rows]))
        nearest = distances[np.arange(len(test_rows)), order[:, 0]]
        nearest_distances.extend(nearest.tolist())
        report["per_fold"][str(fold)] = {
            "test_states": int(len(test_rows)),
            "train_states": int(len(train_rows)),
            "knn": fold_gain,
            "nearest_distance_median": float(np.median(nearest)),
        }
        print(json.dumps(report["per_fold"][str(fold)]), flush=True)

    for k in (1, 3):
        gain = np.asarray(pooled_gain[k])
        report["k"][str(k)] = {
            "oof_recovery_pooled": float(np.sum(gain) / pooled_denominator),
            "gain_p05": float(np.quantile(gain, 0.05)),
            "gain_worst": float(np.min(gain)),
            "regression_fraction": float(np.mean(gain < -1e-6)),
        }
    report["nearest_distance_median_overall"] = float(
        np.median(np.asarray(nearest_distances))
    )
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "KNN_RESIDUAL_BASELINE_DIAGNOSIS_ACTOR_FROZEN",
        "sources": {
            "labels": str(args.labels),
            "manifest": str(args.manifest),
        },
        "protocol": {
            "states": count,
            "folds": args.folds,
            "split": (
                "own deterministic speed-stratified episode folds; not "
                "bit-identical to the A0b partition"
            ),
            "metric": "block-equal IQR over six/reference/control/a0, train-fold fitted",
            "prediction": "mean normalized label residual of k nearest train states",
        },
        "results": report,
        "routing_note": (
            "positive KNN OOF recovery implies the information transfers and "
            "the MLP/representation is the bottleneck; near-zero KNN OOF "
            "implies coverage is the bottleneck at this state count"
        ),
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"k1": report["k"]["1"], "k3": report["k"]["3"]}, indent=1))


if __name__ == "__main__":
    main()
