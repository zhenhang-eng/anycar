#!/usr/bin/env python3
"""Zero-rollout target/coherence diagnosis for Query absolute-center BC.

Only the qualified train-derived 132-candidate bank and independently validated
OOF predictions are read.  Formal validation/test, DBM, new Query rollouts, and
model updates are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "query_expected_road_fullrank_expansion_20260901_v3"
)
DEFAULT_PRETRAIN = REPO_ROOT / (
    "outputs/query_mppi/query_expected_road_absolute_pretrain_expansion_20260901_v3"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "outputs/query_mppi/query_expected_road_target_coherence_20260901_v2"
)
ELITE_RELATIVE_EPSILONS = (0.01, 0.02, 0.05)
PAIR_MARGIN_THRESHOLDS = (0.0, 0.01, 0.02, 0.05)
TOPK_COUNTS = (1, 5, 20, 60, 120, 360)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pretrain", type=Path, default=DEFAULT_PRETRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=29091)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    quarter = np.quantile(values, (0.25, 0.75), axis=0)
    scale = quarter[1] - quarter[0]
    standard = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(standard > 1e-8, standard, 1.0))
    return median, scale


def feature_blocks(data: dict[str, np.ndarray], include_warm: bool) -> list[np.ndarray]:
    blocks = [
        data["history"].reshape(len(data["state"]), -1),
        data["reference_ego"].reshape(len(data["state"]), -1),
        np.concatenate((data["state"][:, 3:5], data["current_action"]), axis=1),
    ]
    if include_warm:
        blocks.append(data["mean_knots_before"].reshape(len(data["state"]), -1))
    return blocks


def embed(
    blocks: list[np.ndarray], rows: np.ndarray, fit_rows: np.ndarray
) -> np.ndarray:
    pieces = []
    for block in blocks:
        center, scale = robust_scale(block[fit_rows])
        normalized = (block[rows] - center) / scale
        pieces.append(normalized / np.sqrt(block.shape[1]))
    return np.concatenate(pieces, axis=1).astype(np.float64)


def distance_matrix(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(query * query, axis=1)[:, None]
        + np.sum(bank * bank, axis=1)[None, :]
        - 2.0 * query @ bank.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def nested_neighbors(
    data: dict[str, np.ndarray], include_warm: bool, max_k: int = 360
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    count = len(data["state"])
    order = np.empty((count, max_k), dtype=np.int64)
    neighbor_distance = np.empty((count, max_k), dtype=np.float64)
    target_rank = np.empty(count, dtype=np.int64)
    input_target_spearman = np.empty(count, dtype=np.float64)
    target_min_by_k = {
        str(k): np.empty(count, dtype=np.float64) for k in TOPK_COUNTS
    }
    sigma = np.tile(np.asarray((0.25, 0.35), np.float64), 8)
    target = data["fullrank_teacher_knots"].reshape(count, -1) / sigma
    blocks = feature_blocks(data, include_warm)
    folds = np.asarray(data["fold_id"], np.int64)
    for fold in range(5):
        query_rows = np.flatnonzero(folds == fold)
        selection_fold = (fold + 1) % 5
        bank_rows = np.flatnonzero((folds != fold) & (folds != selection_fold))
        if len(query_rows) != 120 or len(bank_rows) != 360:
            raise AssertionError("expected 120-query/360-fit nested folds")
        query_embedding = embed(blocks, query_rows, bank_rows)
        bank_embedding = embed(blocks, bank_rows, bank_rows)
        input_distance = distance_matrix(query_embedding, bank_embedding)
        local_order = np.argsort(input_distance, axis=1)
        order[query_rows] = bank_rows[local_order[:, :max_k]]
        neighbor_distance[query_rows] = np.take_along_axis(
            input_distance, local_order[:, :max_k], axis=1
        )
        target_distance = np.sqrt(
            np.mean(
                np.square(target[query_rows, None] - target[bank_rows][None]),
                axis=2,
            )
        )
        for local_index, row in enumerate(query_rows):
            best_target_local = int(np.argmin(target_distance[local_index]))
            target_rank[row] = (
                int(np.flatnonzero(local_order[local_index] == best_target_local)[0]) + 1
            )
            input_target_spearman[row] = float(
                spearmanr(input_distance[local_index], target_distance[local_index]).statistic
            )
            for k in TOPK_COUNTS:
                selected = local_order[local_index, :k]
                target_min_by_k[str(k)][row] = float(
                    np.min(target_distance[local_index, selected])
                )
    extras = {
        "target_best_input_rank": target_rank,
        "input_target_spearman": input_target_spearman,
        **{f"target_min_top{k}": value for k, value in target_min_by_k.items()},
    }
    return order, neighbor_distance, extras


def aggregate_recovery(
    warm_cost: np.ndarray, teacher_cost: np.ndarray, evaluated_cost: np.ndarray
) -> float:
    denominator = float(np.sum(warm_cost - teacher_cost))
    return float(np.sum(warm_cost - evaluated_cost) / denominator)


def absolute_distance(
    left: np.ndarray, right: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    delta = (left.reshape(len(left), -1) - right.reshape(len(right), -1)) / sigma
    return np.sqrt(np.mean(np.square(delta), axis=1))


def set_metrics(
    data: dict[str, np.ndarray], neighbor: np.ndarray, epsilon: float
) -> dict:
    cost = np.asarray(data["candidate_cost"], np.float64)
    best = np.min(cost, axis=1)
    knots = np.asarray(data["candidate_knots"], np.float64).reshape(len(cost), 132, -1)
    sigma = np.tile(np.asarray((0.25, 0.35), np.float64), 8)
    knots = knots / sigma
    set_size = []
    semantic_jaccard = []
    semantic_transfer_cost = []
    absolute_min_pair = []
    absolute_chamfer = []
    absolute_symmetric_coverage_0p25 = []
    for row, other in enumerate(neighbor):
        current_set = np.flatnonzero(cost[row] <= best[row] * (1.0 + epsilon) + 1e-10)
        other_set = np.flatnonzero(cost[other] <= best[other] * (1.0 + epsilon) + 1e-10)
        set_size.append(len(current_set))
        current_semantic = set(current_set.tolist())
        other_semantic = set(other_set.tolist())
        semantic_jaccard.append(
            len(current_semantic & other_semantic)
            / max(len(current_semantic | other_semantic), 1)
        )
        semantic_transfer_cost.append(float(np.min(cost[row, other_set])))
        current_action = knots[row, current_set]
        other_action = knots[other, other_set]
        action_distance = np.sqrt(
            np.mean(
                np.square(current_action[:, None] - other_action[None]), axis=2
            )
        )
        absolute_min_pair.append(float(np.min(action_distance)))
        absolute_chamfer.append(
            float(
                0.5
                * (
                    np.mean(np.min(action_distance, axis=1))
                    + np.mean(np.min(action_distance, axis=0))
                )
            )
        )
        absolute_symmetric_coverage_0p25.append(
            float(
                min(
                    np.mean(np.min(action_distance, axis=1) <= 0.25),
                    np.mean(np.min(action_distance, axis=0) <= 0.25),
                )
            )
        )
    transferred = np.asarray(semantic_transfer_cost)
    warm = np.asarray(data["warm_direct_cost"], np.float64)
    teacher = np.asarray(data["fullrank_teacher_direct_cost"], np.float64)
    return {
        "relative_epsilon": epsilon,
        "set_size": stats(np.asarray(set_size)),
        "multi_member_fraction": float(np.mean(np.asarray(set_size) > 1)),
        "semantic_jaccard": stats(np.asarray(semantic_jaccard)),
        "semantic_transfer": {
            "aggregate_teacher_gain_recovery": aggregate_recovery(
                warm, teacher, transferred
            ),
            "beats_or_equals_warm_fraction": float(np.mean(transferred <= warm)),
            "regret_vs_teacher": stats(transferred - teacher),
        },
        "absolute_action_sigma_rms": {
            "minimum_pair": stats(np.asarray(absolute_min_pair)),
            "symmetric_chamfer": stats(np.asarray(absolute_chamfer)),
            "symmetric_coverage_within_0p25": stats(
                np.asarray(absolute_symmetric_coverage_0p25)
            ),
        },
    }


def pair_sign_metrics(data: dict[str, np.ndarray], neighbor: np.ndarray) -> dict:
    pair_cost = np.asarray(data["pair_cost"], np.float64)
    sign = np.argmin(pair_cost, axis=3)
    margin = np.abs(pair_cost[..., 0] - pair_cost[..., 1]) / np.maximum(
        np.minimum(pair_cost[..., 0], pair_cost[..., 1]), 1.0
    )
    result = {"relative_pair_margin": stats(margin)}
    for threshold in PAIR_MARGIN_THRESHOLDS:
        eligible = (margin >= threshold) & (margin[neighbor] >= threshold)
        result[f"both_margin_ge_{threshold:.2f}"] = {
            "pair_fraction": float(np.mean(eligible)),
            "sign_agreement": float(np.mean((sign == sign[neighbor])[eligible])),
        }
    return result


def neighbor_report(
    data: dict[str, np.ndarray], order: np.ndarray, distances: np.ndarray, extras: dict
) -> dict:
    neighbor = order[:, 0]
    count = len(neighbor)
    sigma = np.tile(np.asarray((0.25, 0.35), np.float64), 8)
    hard_index = np.asarray(data["fullrank_teacher_index"], np.int64)
    candidate_cost = np.asarray(data["candidate_cost"], np.float64)
    transferred_cost = candidate_cost[np.arange(count), hard_index[neighbor]]
    warm_cost = np.asarray(data["warm_direct_cost"], np.float64)
    teacher_cost = np.asarray(data["fullrank_teacher_direct_cost"], np.float64)
    report = {
        "nearest_input_distance": stats(distances[:, 0]),
        "nearest_metadata_match": {
            "speed": float(np.mean(data["speed_kph"] == data["speed_kph"][neighbor])),
            "variant": float(np.mean(data["variant_index"] == data["variant_index"][neighbor])),
            "speed_and_variant": float(
                np.mean(
                    (data["speed_kph"] == data["speed_kph"][neighbor])
                    & (data["variant_index"] == data["variant_index"][neighbor])
                )
            ),
            "control_step": float(
                np.mean(data["control_step"] == data["control_step"][neighbor])
            ),
        },
        "nearest_absolute_action_sigma_rms": {
            "fullrank_teacher": stats(
                absolute_distance(
                    data["fullrank_teacher_knots"],
                    data["fullrank_teacher_knots"][neighbor],
                    sigma,
                )
            ),
            "t0_teacher": stats(
                absolute_distance(
                    data["teacher_knots"], data["teacher_knots"][neighbor], sigma
                )
            ),
            "warm": stats(
                absolute_distance(
                    data["mean_knots_before"],
                    data["mean_knots_before"][neighbor],
                    sigma,
                )
            ),
        },
        "input_target_relationship": {
            "spearman": stats(extras["input_target_spearman"]),
            "target_best_input_rank": stats(extras["target_best_input_rank"]),
            "minimum_target_distance_by_input_topk": {
                str(k): stats(extras[f"target_min_top{k}"]) for k in TOPK_COUNTS
            },
        },
        "hard_argmin": {
            "semantic_index_agreement": float(
                np.mean(hard_index == hard_index[neighbor])
            ),
            "neighbor_semantic_transfer": {
                "aggregate_teacher_gain_recovery": aggregate_recovery(
                    warm_cost, teacher_cost, transferred_cost
                ),
                "beats_or_equals_warm_fraction": float(
                    np.mean(transferred_cost <= warm_cost)
                ),
                "regret_vs_teacher": stats(transferred_cost - teacher_cost),
            },
        },
        "near_optimal_sets": {
            f"relative_{epsilon:.2f}": set_metrics(data, neighbor, epsilon)
            for epsilon in ELITE_RELATIVE_EPSILONS
        },
        "local_pair_sign": pair_sign_metrics(data, neighbor),
    }
    return report


def same_cell_and_random_baselines(
    data: dict[str, np.ndarray], order: np.ndarray, repeats: int, seed: int
) -> dict:
    count = len(order)
    folds = np.asarray(data["fold_id"], np.int64)
    speed = np.asarray(data["speed_kph"])
    variant = np.asarray(data["variant_index"])
    teacher_index = np.asarray(data["fullrank_teacher_index"], np.int64)
    candidate_cost = np.asarray(data["candidate_cost"], np.float64)
    warm_cost = np.asarray(data["warm_direct_cost"], np.float64)
    teacher_cost = np.asarray(data["fullrank_teacher_direct_cost"], np.float64)
    sigma = np.tile(np.asarray((0.25, 0.35), np.float64), 8)
    teacher_knots = np.asarray(data["fullrank_teacher_knots"])
    same_cell_nearest = np.empty(count, np.int64)
    for row in range(count):
        fit = (folds != folds[row]) & (folds != (folds[row] + 1) % 5)
        match = fit & (speed == speed[row]) & (variant == variant[row])
        candidates = set(np.flatnonzero(match).tolist())
        same_cell_nearest[row] = next(
            int(candidate) for candidate in order[row] if int(candidate) in candidates
        )
    evaluated = candidate_cost[np.arange(count), teacher_index[same_cell_nearest]]
    result = {
        "input_nearest_same_cell": {
            "hard_index_agreement": float(
                np.mean(teacher_index == teacher_index[same_cell_nearest])
            ),
            "target_action_sigma_rms": stats(
                absolute_distance(
                    teacher_knots, teacher_knots[same_cell_nearest], sigma
                )
            ),
            "semantic_transfer_recovery": aggregate_recovery(
                warm_cost, teacher_cost, evaluated
            ),
            "semantic_transfer_beats_warm_fraction": float(
                np.mean(evaluated <= warm_cost)
            ),
        }
    }
    rng = np.random.default_rng(seed)
    random_recovery = []
    for _ in range(repeats):
        chosen = np.empty(count, np.int64)
        for row in range(count):
            fit = (folds != folds[row]) & (folds != (folds[row] + 1) % 5)
            candidates = np.flatnonzero(
                fit & (speed == speed[row]) & (variant == variant[row])
            )
            chosen[row] = int(rng.choice(candidates))
        random_cost = candidate_cost[np.arange(count), teacher_index[chosen]]
        random_recovery.append(
            aggregate_recovery(warm_cost, teacher_cost, random_cost)
        )
    result["random_same_cell_semantic_transfer_recovery"] = stats(
        np.asarray(random_recovery)
    )
    return result


def landscape_report(data: dict[str, np.ndarray], oof: dict[str, np.ndarray]) -> dict:
    candidate_cost = np.asarray(data["candidate_cost"], np.float64)
    ordered = np.sort(candidate_cost, axis=1)
    relative_gap = (ordered[:, 1] - ordered[:, 0]) / np.maximum(ordered[:, 0], 1.0)
    warm_cost = np.asarray(data["warm_direct_cost"], np.float64)
    t0_cost = np.asarray(data["teacher_direct_cost"], np.float64)
    fullrank_cost = np.asarray(data["fullrank_teacher_direct_cost"], np.float64)
    sigma = np.tile(np.asarray((0.25, 0.35), np.float64), 8)
    target = np.asarray(data["fullrank_teacher_knots"])
    actor_distance = {}
    for seed_index, seed in enumerate(oof["seeds"]):
        actor_distance[str(int(seed))] = stats(
            absolute_distance(oof["actor_knots"][seed_index], target, sigma)
        )
    elite_size = {}
    for epsilon in ELITE_RELATIVE_EPSILONS:
        count = np.sum(
            candidate_cost <= ordered[:, :1] * (1.0 + epsilon) + 1e-10,
            axis=1,
        )
        elite_size[f"relative_{epsilon:.2f}"] = {
            "size": stats(count),
            "multi_member_fraction": float(np.mean(count > 1)),
        }
    return {
        "best_second_relative_cost_gap": stats(relative_gap),
        "elite_sets": elite_size,
        "cost_headroom": {
            "warm_mean": float(np.mean(warm_cost)),
            "t0_mean": float(np.mean(t0_cost)),
            "fullrank_mean": float(np.mean(fullrank_cost)),
            "t0_fraction_of_warm_to_fullrank_headroom": aggregate_recovery(
                warm_cost, fullrank_cost, t0_cost
            ),
            "fullrank_gain_vs_t0": stats(t0_cost - fullrank_cost),
        },
        "target_action_sigma_rms": {
            "warm_to_fullrank": stats(
                absolute_distance(data["mean_knots_before"], target, sigma)
            ),
            "t0_to_fullrank": stats(
                absolute_distance(data["teacher_knots"], target, sigma)
            ),
            "oof_actor_to_fullrank_by_seed": actor_distance,
        },
    }


def grouped_report(
    data: dict[str, np.ndarray],
    oof: dict[str, np.ndarray],
    neighbor: np.ndarray,
    input_target_spearman: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    count = len(neighbor)
    sigma = np.tile(np.asarray((0.25, 0.35), np.float64), 8)
    target = data["fullrank_teacher_knots"].reshape(count, -1) / sigma
    hard_distance = np.sqrt(
        np.mean(np.square(target - target[neighbor]), axis=1)
    )
    cost = np.asarray(data["candidate_cost"], np.float64)
    best = np.min(cost, axis=1)
    knots = data["candidate_knots"].reshape(count, 132, -1) / sigma
    elite2_minimum = np.empty(count, np.float64)
    for row, other in enumerate(neighbor):
        current = np.flatnonzero(cost[row] <= best[row] * 1.02 + 1e-10)
        other_set = np.flatnonzero(cost[other] <= best[other] * 1.02 + 1e-10)
        action_distance = np.sqrt(
            np.mean(
                np.square(knots[row][current, None] - knots[other][None, other_set]),
                axis=2,
            )
        )
        elite2_minimum[row] = float(np.min(action_distance))
    pair_cost = np.asarray(data["pair_cost"], np.float64)
    pair_sign = np.argmin(pair_cost, axis=3)
    pair_margin = np.abs(pair_cost[..., 0] - pair_cost[..., 1]) / np.maximum(
        np.minimum(pair_cost[..., 0], pair_cost[..., 1]), 1.0
    )
    pair_eligible = (pair_margin >= 0.05) & (pair_margin[neighbor] >= 0.05)
    pair_agree = pair_sign == pair_sign[neighbor]
    warm_cost = np.asarray(data["warm_direct_cost"], np.float64)
    teacher_cost = np.asarray(data["fullrank_teacher_direct_cost"], np.float64)

    def summarize_mask(mask: np.ndarray) -> dict:
        hard_median = float(np.median(hard_distance[mask]))
        elite_median = float(np.median(elite2_minimum[mask]))
        eligible = pair_eligible[mask]
        actor_recovery = {}
        for seed_index, seed in enumerate(oof["seeds"]):
            actor_recovery[str(int(seed))] = aggregate_recovery(
                warm_cost[mask],
                teacher_cost[mask],
                oof["actor_direct_cost"][seed_index, mask],
            )
        return {
            "rows": int(np.sum(mask)),
            "hard_target_distance_sigma_rms": stats(hard_distance[mask]),
            "relative_2pct_set_minimum_pair_sigma_rms": stats(
                elite2_minimum[mask]
            ),
            "relative_2pct_set_distance_reduction_fraction": float(
                1.0 - elite_median / hard_median
            ),
            "input_target_spearman": stats(input_target_spearman[mask]),
            "robust_pair_margin_ge_5pct": {
                "pair_fraction": float(np.mean(eligible)),
                "sign_agreement": float(np.mean(pair_agree[mask][eligible])),
            },
            "oof_actor_teacher_gain_recovery_by_seed": actor_recovery,
        }

    result = {}
    for field in ("speed_kph", "variant_index", "fold_id"):
        result[field] = {
            str(int(value)): summarize_mask(np.asarray(data[field]) == value)
            for value in np.unique(data[field])
        }
    episode_028 = np.asarray(data["episode_id"]) == "episode_028"
    result["episode_028_pressure"] = {
        "episode_028": summarize_mask(episode_028),
        "all_other_episodes": summarize_mask(~episode_028),
    }
    arrays = {
        "strict_hard_target_distance_sigma_rms": hard_distance.astype(np.float32),
        "strict_relative_2pct_set_minimum_pair_sigma_rms": elite2_minimum.astype(
            np.float32
        ),
        "strict_pair_margin_ge_5pct": pair_eligible,
        "strict_pair_sign_agreement": pair_agree,
    }
    return result, arrays


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    pretrain = args.pretrain.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if args.random_repeats < 1:
        raise ValueError("--random-repeats must be positive")
    source_validation = json.loads((source / "validation.json").read_text())
    pretrain_validation = json.loads((pretrain / "validation.json").read_text())
    if source_validation["qualification"] != "QUERY_EXPECTED_ROAD_FULLRANK_PASS":
        raise AssertionError("source full-rank sidecar did not pass")
    if pretrain_validation["qualification"] != "QUERY_ABSOLUTE_PRETRAIN_INDEPENDENT_PASS":
        raise AssertionError("pretrain artifact did not pass independent validation")
    if source_validation.get("formal_validation_or_test_consumed", True):
        raise AssertionError("source consumed formal validation/test")
    if pretrain_validation.get("formal_validation_or_test_consumed", True):
        raise AssertionError("pretrain consumed formal validation/test")
    with np.load(source / "bank.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(pretrain / "oof_predictions.npz", allow_pickle=False) as archive:
        oof = {name: np.asarray(archive[name]) for name in archive.files}
    if len(data["state"]) != 600 or len(np.unique(data["episode_id"])) != 100:
        raise AssertionError("expected qualified 600-row/100-episode source")
    if not np.array_equal(data["row_index"], oof["row_index"]):
        raise AssertionError("OOF rows do not align with source bank")
    if not np.array_equal(np.argmin(data["candidate_cost"], axis=1), data["fullrank_teacher_index"]):
        raise AssertionError("full-rank teacher is not candidate argmin")

    spaces = {}
    saved_arrays = {}
    neighbor_orders = {}
    space_extras = {}
    for name, include_warm in (
        ("strict_no_anchor", False),
        ("diagnostic_plus_exact_warm", True),
    ):
        order, distances, extras = nested_neighbors(data, include_warm)
        neighbor_orders[name] = order
        space_extras[name] = extras
        spaces[name] = neighbor_report(data, order, distances, extras)
        saved_arrays[f"{name}_neighbor_index"] = order
        saved_arrays[f"{name}_neighbor_distance"] = distances.astype(np.float32)
        for extra_name, value in extras.items():
            saved_arrays[f"{name}_{extra_name}"] = value

    strict_nearest = neighbor_orders["strict_no_anchor"][:, 0]
    hard_distance = spaces["strict_no_anchor"][
        "nearest_absolute_action_sigma_rms"
    ]["fullrank_teacher"]["median"]
    elite2_min = spaces["strict_no_anchor"]["near_optimal_sets"][
        "relative_0.02"
    ]["absolute_action_sigma_rms"]["minimum_pair"]["median"]
    strict_spearman = spaces["strict_no_anchor"]["input_target_relationship"][
        "spearman"
    ]["median"]
    warm_spearman = spaces["diagnostic_plus_exact_warm"][
        "input_target_relationship"
    ]["spearman"]["median"]
    robust_pair = spaces["strict_no_anchor"]["local_pair_sign"][
        "both_margin_ge_0.05"
    ]["sign_agreement"]
    grouped, grouped_arrays = grouped_report(
        data,
        oof,
        neighbor_orders["strict_no_anchor"][:, 0],
        space_extras["strict_no_anchor"]["input_target_spearman"],
    )
    saved_arrays.update(grouped_arrays)
    analysis = {
        "qualification": "QUERY_TARGET_COHERENCE_DIAGNOSTIC_COMPLETE",
        "decision": "NO_JUSTIFICATION_FOR_SET_VALUED_ABSOLUTE_BC_RETRY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "pretrain": str(pretrain),
        "row_count": 600,
        "episode_count": 100,
        "protocol": {
            "neighbor_split": (
                "outer fold is query; (fold+1)%5 is selection; remaining three "
                "whole-episode folds form the 360-row fit bank"
            ),
            "strict_actor_inputs": [
                "history[250,7]",
                "expected-road reference_ego[51,4]",
                "current vx/yawrate/action",
            ],
            "diagnostic_only_input_ablation": "strict inputs plus exact mean_knots_before",
            "feature_metric": (
                "fit-only median/IQR standardization with std fallback and equal "
                "weight per feature block"
            ),
            "formal_validation_or_test_consumed": False,
            "new_query_or_dbm_rollouts": 0,
        },
        "candidate_landscape": landscape_report(data, oof),
        "spaces": spaces,
        "same_cell_baselines": same_cell_and_random_baselines(
            data, neighbor_orders["strict_no_anchor"], args.random_repeats, args.seed
        ),
        "grouped_strict_no_anchor": grouped,
        "routing_evidence": {
            "hard_argmin_absolute_distance_median_sigma_rms": hard_distance,
            "relative_2pct_set_minimum_pair_median_sigma_rms": elite2_min,
            "relative_2pct_set_distance_reduction_fraction": float(
                1.0 - elite2_min / hard_distance
            ),
            "strict_input_target_spearman_median": strict_spearman,
            "plus_warm_input_target_spearman_median": warm_spearman,
            "plus_warm_spearman_increment": warm_spearman - strict_spearman,
            "robust_pair_sign_agreement_margin_ge_5pct": robust_pair,
            "interpretation": (
                "The finite-bank hard index is unstable, but the 2% near-optimal "
                "set does not materially close the cross-episode absolute-action "
                "gap. Most absolute target error is the T0 center itself, not its "
                "small full-rank refinement. Exact warm contains diagnostic target "
                "information but is not authorized as a strict Actor input."
            ),
        },
    }
    output.mkdir(parents=True)
    diagnostics_path = output / "diagnostics.npz"
    np.savez_compressed(diagnostics_path, **saved_arrays)
    analysis_path = output / "analysis.json"
    dump_json(analysis_path, analysis)
    manifest = {
        "format_version": 1,
        "dataset_type": "query-expected-road-target-coherence-diagnostic",
        "qualification": analysis["qualification"],
        "decision": analysis["decision"],
        "created_utc": analysis["created_utc"],
        "source": str(source),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_validation_sha256": sha256(source / "validation.json"),
        "source_bank_sha256": sha256(source / "bank.npz"),
        "pretrain": str(pretrain),
        "pretrain_manifest_sha256": sha256(pretrain / "manifest.json"),
        "pretrain_validation_sha256": sha256(pretrain / "validation.json"),
        "pretrain_oof_sha256": sha256(pretrain / "oof_predictions.npz"),
        "analysis_sha256": sha256(analysis_path),
        "diagnostics_sha256": sha256(diagnostics_path),
        "analyzer_sha256": sha256(Path(__file__)),
        "formal_validation_or_test_consumed": False,
        "new_query_or_dbm_rollouts": 0,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps(analysis["routing_evidence"], indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
