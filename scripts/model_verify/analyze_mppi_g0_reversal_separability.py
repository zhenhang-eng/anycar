#!/usr/bin/env python3
"""Zero-rollout separability diagnosis for g0 reversal pairs.

Question (review §11.29 follow-up): can raw observables separate neighbor
pairs whose true g0 directions flip, or do flips live in an indistinguishable
regime?

Method:
- Features per internal-selection context: raw observable channels only
  (initial_state_six, current block, reference summary, reference speed,
  clip flag, scenario one-hot, repeat index, absolute steering center,
  sampling sigma summary). Nothing derived from g0 labels enters predictors.
- Neighbor pairs: kNN (k=10) in the standardized physical+center space.
- Pair target: flip = cosine(g0_i, g0_j) < 0 using fresh-FD gradients.
- Pair predictors: absolute differences of observables, same-scenario /
  same-repeat flags, neighbor distance.
- Grouped 5-fold CV by episode; report per-fold AUC and accuracy at the
  base-rate threshold.
- Label-aware oracle (non-deployable): k-means on [g0 | observables], pair
  feature = same-cluster indicator, same grouped CV — an upper bound on what
  any routing based on labels+observables could achieve.
- Density check: distance to nearest same-direction vs nearest opposite-
  direction neighbor per context, by stratum.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
)


DEFAULT_INITIAL = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_FRESH = Path(
    "outputs/mppi_proposal/direct_critic_fresh_fd_b4_smallest_target_20260813_v2/"
    "fresh_fd_audit.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/g0_reversal_separability_20260814_v1/analysis.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--fresh-npz", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260814)
    parser.add_argument("--oracle-clusters", type=int, default=12)
    return parser.parse_args()


def build_features(
    data,
    fresh: dict[str, np.ndarray],
    manifest_rows: list[dict],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    fresh_context = fresh["context_index"].astype(np.int64)
    row_by_context = {
        int(row["context_index"]): row for row in manifest_rows
    }
    initial_six = data.initial_state_six[fresh_context]
    current = data.inputs[2][fresh_context]
    reference = data.inputs[1][fresh_context]
    reference_last = reference[:, -1, :]
    reference_mean = reference.mean(axis=1)
    reference_std = reference.std(axis=1)
    speed = np.asarray([[
        row_by_context[int(value)]["reference_speed_mps"]
        for value in fresh_context
    ]], np.float32).T
    clipped = np.asarray([[
        1.0 if row_by_context[int(value)]["clipped"] else 0.0
        for value in fresh_context
    ]], np.float32).T
    scenarios = sorted({row["scenario"] for row in manifest_rows})
    scenario_one_hot = np.asarray([
        [1.0 if row_by_context[int(value)]["scenario"] == name else 0.0
         for name in scenarios]
        for value in fresh_context
    ], np.float32)
    repeat_index = np.asarray([[
        float(row_by_context[int(value)]["repeat_index"])
        for value in fresh_context
    ]], np.float32).T
    center = fresh["actor_center"].astype(np.float32).reshape(len(fresh_context), -1)
    sigma = fresh["sigma"].astype(np.float32)
    sigma_summary = np.concatenate((
        sigma.mean(axis=1, keepdims=True),
        sigma.std(axis=1, keepdims=True),
        sigma,
    ), axis=1)
    blocks = [
        ("initial_state_six", initial_six),
        ("current", current),
        ("reference_last", reference_last),
        ("reference_mean", reference_mean),
        ("reference_std", reference_std),
        ("speed", speed),
        ("clipped", clipped),
        ("scenario", scenario_one_hot),
        ("repeat_index", repeat_index),
        ("center", center),
        ("sigma", sigma_summary),
    ]
    features = np.concatenate([block for _, block in blocks], axis=1)
    names: list[str] = []
    for block_name, block in blocks:
        width = block.shape[1]
        names.extend(
            f"{block_name}_{index}" if width > 1 else block_name
            for index in range(width)
        )
    gradient = fresh["gradient"].astype(np.float32)
    stratum = np.asarray([
        row_by_context[int(value)]["stratum_id"] for value in fresh_context
    ])
    return features.astype(np.float32), names, gradient, stratum


def pair_matrix(
    features: np.ndarray,
    names: list[str],
    gradient: np.ndarray,
    episodes: np.ndarray,
    neighbor_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(standardized)
    distances, indices = neighbors.kneighbors(standardized)
    cosine = np.asarray([
        (gradient @ gradient[other]) / (
            np.linalg.norm(gradient) * np.linalg.norm(gradient[other]) + 1e-12
        )
        for other in range(len(gradient))
    ])
    pair_set: set[tuple[int, int]] = set()
    for source in range(len(gradient)):
        for neighbor in indices[source][1:]:
            pair_set.add((min(source, int(neighbor)), max(source, int(neighbor))))
    left = np.asarray([pair[0] for pair in sorted(pair_set)], np.int64)
    right = np.asarray([pair[1] for pair in sorted(pair_set)], np.int64)
    differences = np.abs(features[left] - features[right])
    same_episode = (episodes[left] == episodes[right]).astype(np.float32)[:, None]
    pair_distance = np.linalg.norm(
        standardized[left] - standardized[right], axis=1, keepdims=True
    )
    pair_features = np.concatenate(
        (differences, same_episode, pair_distance), axis=1
    )
    pair_names = [f"abs_d_{name}" for name in names] + [
        "same_episode", "neighbor_distance",
    ]
    target = (cosine[left, right] < 0.0).astype(np.int64)
    return (
        pair_features.astype(np.float32),
        np.asarray(pair_names),
        target,
    )


def grouped_auc(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> dict:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    fold_assignment = {
        group: index % folds for index, group in enumerate(shuffled)
    }
    pair_fold = np.asarray([fold_assignment[g] for g in groups], np.int64)
    aucs, accuracies = [], []
    for fold in range(folds):
        train = pair_fold != fold
        test = ~train
        if len(np.unique(target[train])) < 2 or len(np.unique(target[test])) < 2:
            continue
        scaler = StandardScaler().fit(features[train])
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, random_state=seed + fold
        )
        model.fit(scaler.transform(features[train]), target[train])
        probability = model.predict_proba(scaler.transform(features[test]))[:, 1]
        aucs.append(float(roc_auc_score(target[test], probability)))
        base_rate = float(np.mean(target[train]))
        prediction = (probability >= base_rate).astype(np.int64)
        accuracies.append(float(np.mean(prediction == target[test])))
    return {
        "fold_auc": aucs,
        "median_auc": float(np.median(aucs)) if aucs else None,
        "median_accuracy": float(np.median(accuracies)) if accuracies else None,
        "base_rate": float(np.mean(target)),
    }


def linear_reference(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    pair_names: list[str],
    folds: int,
    seed: int,
) -> dict:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed + 1)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    fold_assignment = {
        group: index % folds for index, group in enumerate(shuffled)
    }
    pair_fold = np.asarray([fold_assignment[g] for g in groups], np.int64)
    weights = []
    aucs = []
    for fold in range(folds):
        train = pair_fold != fold
        test = ~train
        if len(np.unique(target[train])) < 2:
            continue
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(max_iter=2000, C=0.5)
        model.fit(scaler.transform(features[train]), target[train])
        probability = model.predict_proba(scaler.transform(features[test]))[:, 1]
        if len(np.unique(target[test])) > 1:
            aucs.append(float(roc_auc_score(target[test], probability)))
        weights.append(np.abs(model.coef_[0]))
    mean_weight = np.mean(weights, axis=0)
    order = np.argsort(mean_weight)[::-1][:15]
    return {
        "median_auc": float(np.median(aucs)) if aucs else None,
        "top_features": [
            {"name": pair_names[index], "mean_abs_weight": float(mean_weight[index])}
            for index in order
        ],
    }


def main() -> None:
    args = parse_args()
    initial = torch_load(args.initial_actor)
    alpha_payload = torch_load(Path(initial["base_alpha_checkpoint"]))
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial["labels"]), old_payload)
    fresh = dict(np.load(args.fresh_npz, allow_pickle=False))
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    features, names, gradient, stratum = build_features(data, fresh, rows)
    episodes = fresh["episode"].astype(str)
    pair_features, pair_names, target = pair_matrix(
        features, names, gradient, episodes, args.neighbors, args.seed
    )
    # Recompute pair endpoints deterministically for group keys.
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    neighbors = NearestNeighbors(n_neighbors=args.neighbors + 1).fit(standardized)
    _, indices = neighbors.kneighbors(standardized)
    pair_set: set[tuple[int, int]] = set()
    for source in range(len(gradient)):
        for neighbor in indices[source][1:]:
            pair_set.add((min(source, int(neighbor)), max(source, int(neighbor))))
    left = np.asarray([pair[0] for pair in sorted(pair_set)], np.int64)
    right = np.asarray([pair[1] for pair in sorted(pair_set)], np.int64)
    groups = np.asarray([
        episodes[int(a)] if episodes[int(a)] <= episodes[int(b)] else episodes[int(b)]
        for a, b in zip(left, right)
    ], dtype=object)
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "G0_REVERSAL_SEPARABILITY_DIAGNOSIS_ACTOR_FROZEN",
        "sources": {
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "fresh_fd": str(args.fresh_npz.resolve()),
            "fresh_fd_sha256": sha256_file(args.fresh_npz),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "counts": {
            "contexts": int(len(gradient)),
            "neighbor_pairs": int(len(target)),
        },
        "neighbor_flip": {
            "flip_fraction": float(np.mean(target)),
            "note": "fraction of kNN pairs with negative true g0 cosine",
        },
        "gradient_boosting": grouped_auc(
            pair_features, target, groups, args.folds, args.seed
        ),
        "logistic_reference": linear_reference(
            pair_features, target, groups, list(pair_names), args.folds, args.seed
        ),
    }
    # Density check: per-context distance to nearest same vs opposite label.
    same_distance, opposite_distance = [], []
    conflict_fraction = []
    for source in range(len(gradient)):
        neighbour_indices = indices[source][1:]
        similarities = [
            float(
                (gradient[source] @ gradient[other])
                / (np.linalg.norm(gradient[source])
                   * np.linalg.norm(gradient[other]) + 1e-12)
            )
            for other in neighbour_indices
        ]
        distances_source = [
            float(np.linalg.norm(standardized[source] - standardized[other]))
            for other in neighbour_indices
        ]
        positive = [d for d, s in zip(distances_source, similarities) if s >= 0]
        negative = [d for d, s in zip(distances_source, similarities) if s < 0]
        if positive:
            same_distance.append(min(positive))
        if negative:
            opposite_distance.append(min(negative))
        conflict_fraction.append(float(np.mean([s < 0 for s in similarities])))
    conflict = np.asarray(conflict_fraction)
    density = {
        "median_nearest_same_direction_distance": float(np.median(same_distance)),
        "median_nearest_opposite_direction_distance": float(
            np.median(opposite_distance)
        ),
        "conflict_fraction_by_stratum": {
            stratum_name: {
                "count": int(np.sum(stratum == stratum_name)),
                "median": float(np.median(conflict[stratum == stratum_name])),
                "p90": float(np.quantile(conflict[stratum == stratum_name], 0.90)),
            }
            for stratum_name in np.unique(stratum)
        },
    }
    result["density"] = density
    # Label-aware oracle bound (non-deployable): same-cluster indicator.
    oracle_features = np.concatenate(
        (StandardScaler().fit_transform(
            np.concatenate((features, gradient), axis=1)
        ),), axis=1
    )
    clusters = KMeans(
        n_clusters=args.oracle_clusters, random_state=args.seed, n_init=10
    ).fit_predict(oracle_features)
    same_cluster = (clusters[left] == clusters[right]).astype(np.float32)[:, None]
    oracle_input = np.concatenate((same_cluster, pair_features), axis=1)
    result["label_aware_oracle_non_deployable"] = {
        **grouped_auc(oracle_input, target, groups, args.folds, args.seed),
        "note": (
            "k-means on [observables | g0 labels]; routing on this is oracle "
            "only and cannot be deployed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "neighbor_flip_fraction": result["neighbor_flip"]["flip_fraction"],
        "gradient_boosting_median_auc": result["gradient_boosting"]["median_auc"],
        "logistic_median_auc": result["logistic_reference"]["median_auc"],
        "oracle_median_auc": result["label_aware_oracle_non_deployable"][
            "median_auc"
        ],
        "density": {
            "nearest_same": density["median_nearest_same_direction_distance"],
            "nearest_opposite": density["median_nearest_opposite_direction_distance"],
        },
        "conflict_by_stratum": {
            key: value["median"]
            for key, value in density["conflict_fraction_by_stratum"].items()
        },
    }, indent=2))


def torch_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    main()
