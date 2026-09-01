#!/usr/bin/env python3
"""Audit repeat-chord excitation and a physical-only local linear-H oracle.

This is a zero-rollout diagnostic.  It uses internal train/validation repeat
chords to predict the action-gradient response of consumed internal-selection
fresh-FD pairs.  Formal validation and test data are never loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import TorchMPPIStructuredLocalQCritic
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_local_gradient_critic import repeat_partner_positions
from train_mppi_direct_trust_region_actor import load_actor_payload, load_dataset


DEFAULT_CRITIC = Path(
    "outputs/mppi_proposal/"
    "direct_local_gradient_critic_b4_smallest_target_20260813_v2"
)
DEFAULT_FRESH = Path(
    "outputs/mppi_proposal/"
    "direct_critic_fresh_fd_b4_smallest_target_20260813_v2"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/chord_geometry_local_oracle_20260814_v1"
)
FLAT_NAMES = tuple(
    f"knot_{k}_{channel}"
    for k in range(8)
    for channel in ("acceleration", "steering")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbor-counts", default="16,32,64,128")
    parser.add_argument("--ridge-grid", default="1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--response-scales", default="0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--state-temperature", type=float, default=0.05)
    parser.add_argument("--chord-d0", type=float, default=0.15)
    parser.add_argument("--chord-epsilon", type=float, default=0.02)
    parser.add_argument("--chord-w-max", type=float, default=4.0)
    parser.add_argument("--primary-neighbor-count", type=int, default=32)
    return parser.parse_args()


def distribution(value: np.ndarray) -> dict[str, float | None]:
    value = np.asarray(value, np.float64)
    value = value[np.isfinite(value)]
    if len(value) == 0:
        return {key: None for key in (
            "mean", "minimum", "p05", "p10", "p25", "median", "p75",
            "p90", "p95", "maximum",
        )}
    return {
        "mean": float(np.mean(value)),
        "minimum": float(np.min(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "p25": float(np.quantile(value, 0.25)),
        "median": float(np.median(value)),
        "p75": float(np.quantile(value, 0.75)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "maximum": float(np.max(value)),
    }


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    return np.sum(left * right, axis=-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-12
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def pair_left(partner: np.ndarray) -> np.ndarray:
    position = np.arange(len(partner))
    result = np.flatnonzero(position < partner)
    if len(result) * 2 != len(partner):
        raise AssertionError("repeat partner mapping is not exactly paired")
    return result


def fit_physical_features(
    data: Any, fit_context: np.ndarray, query_context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Standardize repeat-invariant blocks and concatenate unit block vectors."""
    fit_parts, query_parts, block_summary = [], [], {}
    for name, value in zip(("history", "reference", "current"), data.inputs[:3]):
        fit = np.asarray(value[fit_context], np.float32).reshape(len(fit_context), -1)
        query = np.asarray(value[query_context], np.float32).reshape(
            len(query_context), -1
        )
        mean = np.mean(fit, axis=0)
        std = np.std(fit, axis=0)
        active = std > 1e-5
        if not np.any(active):
            raise AssertionError(f"physical block {name} has no active feature")
        fit = (fit[:, active] - mean[active]) / std[active]
        query = (query[:, active] - mean[active]) / std[active]
        fit /= np.linalg.norm(fit, axis=1, keepdims=True) + 1e-12
        query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-12
        fit_parts.append(fit / math.sqrt(3.0))
        query_parts.append(query / math.sqrt(3.0))
        block_summary[name] = {
            "flat_dimension": int(value[fit_context].reshape(len(fit_context), -1).shape[1]),
            "active_standardized_dimension": int(np.sum(active)),
        }
    return (
        np.concatenate(fit_parts, axis=1).astype(np.float32),
        np.concatenate(query_parts, axis=1).astype(np.float32),
        block_summary,
    )


def transform_physical_features(
    data: Any,
    normalization_context: np.ndarray,
    bank_context: np.ndarray,
    query_context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Use only train contexts to fit normalization, even when the final bank
    # also includes internal validation episodes.
    bank_parts, query_parts, block_summary = [], [], {}
    for name, value in zip(("history", "reference", "current"), data.inputs[:3]):
        normalization = np.asarray(
            value[normalization_context], np.float32
        ).reshape(len(normalization_context), -1)
        bank = np.asarray(value[bank_context], np.float32).reshape(
            len(bank_context), -1
        )
        query = np.asarray(value[query_context], np.float32).reshape(
            len(query_context), -1
        )
        mean = np.mean(normalization, axis=0)
        std = np.std(normalization, axis=0)
        active = std > 1e-5
        bank = (bank[:, active] - mean[active]) / std[active]
        query = (query[:, active] - mean[active]) / std[active]
        bank /= np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12
        query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-12
        bank_parts.append(bank / math.sqrt(3.0))
        query_parts.append(query / math.sqrt(3.0))
        block_summary[name] = {
            "flat_dimension": int(value[bank_context].reshape(len(bank_context), -1).shape[1]),
            "active_standardized_dimension": int(np.sum(active)),
        }
    return (
        np.concatenate(bank_parts, axis=1).astype(np.float32),
        np.concatenate(query_parts, axis=1).astype(np.float32),
        block_summary,
    )


def pair_data(
    data: Any,
    labels: dict[str, np.ndarray],
    left: np.ndarray,
    partner: np.ndarray,
    maximum_residual_sigma: float,
    gradient: np.ndarray | None = None,
    center: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    right = partner[left]
    context = labels["context_index"].astype(np.int64)
    local_gradient = (
        labels["gradient_by_radius"][:, 0]
        if gradient is None else np.asarray(gradient, np.float32)
    )
    local_center = (
        labels["actor_center"] if center is None else np.asarray(center, np.float32)
    )
    local_sigma = (
        data.sigma[context] if sigma is None else np.asarray(sigma, np.float32)
    )
    delta_center_sigma = (
        local_center[right] - local_center[left]
    ) / local_sigma[left, None, :]
    delta_action = delta_center_sigma / maximum_residual_sigma
    return {
        "left": left,
        "right": right,
        "left_context": context[left],
        "right_context": context[right],
        "episode": labels["episode"][left],
        "left_gradient": local_gradient[left].astype(np.float32),
        "right_gradient": local_gradient[right].astype(np.float32),
        "delta_gradient": (
            local_gradient[right] - local_gradient[left]
        ).astype(np.float32),
        "delta_action": delta_action.reshape(len(left), -1).astype(np.float32),
        "distance_sigma": np.sqrt(np.mean(np.square(
            delta_center_sigma
        ), axis=(1, 2))).astype(np.float32),
    }


def fresh_pair_data(
    data: Any,
    labels: dict[str, np.ndarray],
    left: np.ndarray,
    partner: np.ndarray,
    fresh: dict[str, np.ndarray],
    maximum_residual_sigma: float,
) -> dict[str, np.ndarray]:
    lookup = {
        int(context): position
        for position, context in enumerate(fresh["context_index"])
    }
    label_positions = np.concatenate((left, partner[left]))
    missing = [
        int(labels["context_index"][position]) for position in label_positions
        if int(labels["context_index"][position]) not in lookup
    ]
    if missing:
        raise AssertionError(f"fresh artifact misses heldout contexts: {missing[:5]}")
    mapped = np.full(len(labels["context_index"]), -1, np.int64)
    for position in label_positions:
        mapped[position] = lookup[int(labels["context_index"][position])]
    fresh_gradient = np.zeros_like(labels["gradient_by_radius"][:, 0])
    fresh_center = np.zeros_like(labels["actor_center"])
    fresh_sigma = np.ones((len(labels["context_index"]), 2), np.float32)
    fresh_gradient[label_positions] = fresh["gradient"][mapped[label_positions]]
    fresh_center[label_positions] = fresh["actor_center"][mapped[label_positions]]
    fresh_sigma[label_positions] = fresh["sigma"][mapped[label_positions]]
    return pair_data(
        data, labels, left, partner, maximum_residual_sigma,
        gradient=fresh_gradient, center=fresh_center, sigma=fresh_sigma,
    )


def local_geometry(
    bank_delta_action: np.ndarray,
    similarity: np.ndarray,
    neighbor_counts: list[int],
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    maximum = max(neighbor_counts)
    if maximum > len(bank_delta_action):
        raise ValueError("neighbor count exceeds chord bank")
    order = np.argpartition(-similarity, maximum - 1, axis=1)[:, :maximum]
    local = np.take_along_axis(similarity, order, axis=1)
    sorted_local = np.argsort(-local, axis=1)
    order = np.take_along_axis(order, sorted_local, axis=1)
    result = {}
    orders = {}
    for count in neighbor_counts:
        index = order[:, :count]
        orders[count] = index
        numeric_rank, rank_1pct, rank_5pct = [], [], []
        stable_rank, entropy_rank, condition, minimum_ratio = [], [], [], []
        for row in index:
            action = bank_delta_action[row]
            action = action / (
                np.linalg.norm(action, axis=1, keepdims=True) + 1e-12
            )
            singular = np.linalg.svd(action, compute_uv=False)
            ratio = singular / max(float(singular[0]), 1e-12)
            numeric_rank.append(int(np.sum(ratio > 1e-6)))
            rank_1pct.append(int(np.sum(ratio > 0.01)))
            rank_5pct.append(int(np.sum(ratio > 0.05)))
            stable_rank.append(float(np.sum(np.square(singular)) / (
                np.square(singular[0]) + 1e-12
            )))
            probability = np.square(singular) / (
                np.sum(np.square(singular)) + 1e-12
            )
            entropy_rank.append(float(np.exp(-np.sum(
                probability * np.log(probability + 1e-12)
            ))))
            minimum_ratio.append(float(ratio[-1]))
            condition.append(float(1.0 / ratio[-1]) if ratio[-1] > 1e-8 else np.inf)
        result[str(count)] = {
            "query_count": int(len(index)),
            "numeric_rank": distribution(np.asarray(numeric_rank)),
            "rank_singular_ratio_gt_0_01": distribution(np.asarray(rank_1pct)),
            "rank_singular_ratio_gt_0_05": distribution(np.asarray(rank_5pct)),
            "stable_rank": distribution(np.asarray(stable_rank)),
            "entropy_effective_rank": distribution(np.asarray(entropy_rank)),
            "smallest_to_largest_singular_ratio": distribution(
                np.asarray(minimum_ratio)
            ),
            "finite_condition_number": distribution(np.asarray(condition)),
            "full_numeric_rank_fraction": float(
                np.mean(np.asarray(numeric_rank) == 16)
            ),
            "rank_ge_12_at_5pct_fraction": float(
                np.mean(np.asarray(rank_5pct) >= 12)
            ),
        }
    return result, orders


def component_excitation(delta_action: np.ndarray) -> dict[str, Any]:
    direction = delta_action / (
        np.linalg.norm(delta_action, axis=1, keepdims=True) + 1e-12
    )
    rows = []
    for index, name in enumerate(FLAT_NAMES):
        value = np.abs(direction[:, index])
        rows.append({
            "flat_index": index,
            "name": name,
            "mean_absolute_unit_direction": float(np.mean(value)),
            "median_absolute_unit_direction": float(np.median(value)),
            "fraction_abs_lt_0_02": float(np.mean(value < 0.02)),
            "fraction_abs_lt_0_05": float(np.mean(value < 0.05)),
        })
    steering_0_2 = np.asarray((1, 3, 5), np.int64)
    return {
        "per_component": rows,
        "front_steering_knot_0_2_energy_fraction": distribution(
            np.sum(np.square(direction[:, steering_0_2]), axis=1)
        ),
        "acceleration_energy_fraction": distribution(
            np.sum(np.square(direction[:, np.arange(0, 16, 2)]), axis=1)
        ),
        "steering_energy_fraction": distribution(
            np.sum(np.square(direction[:, np.arange(1, 16, 2)]), axis=1)
        ),
    }


def chord_weight(distance: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return np.minimum(
        args.chord_w_max,
        args.chord_d0 / np.maximum(distance, args.chord_epsilon),
    ).astype(np.float64)


def fit_local_response(
    bank_action: np.ndarray,
    bank_gradient: np.ndarray,
    bank_distance: np.ndarray,
    similarity: np.ndarray,
    query_action: np.ndarray,
    neighbor_count: int,
    ridge: float,
    temperature: float,
    args: argparse.Namespace,
    method: str,
) -> np.ndarray:
    order = np.argpartition(
        -similarity, neighbor_count - 1, axis=1
    )[:, :neighbor_count]
    result = np.zeros_like(query_action, dtype=np.float64)
    identity = np.eye(query_action.shape[1], dtype=np.float64)
    for index, neighbor in enumerate(order):
        local_similarity = similarity[index, neighbor].astype(np.float64)
        state_weight = np.exp(
            (local_similarity - np.max(local_similarity)) / temperature
        )
        weight = state_weight * chord_weight(bank_distance[neighbor], args)
        action = bank_action[neighbor].astype(np.float64)
        response = bank_gradient[neighbor].astype(np.float64)
        normal = action.T @ (weight[:, None] * action)
        scale = max(float(np.trace(normal) / len(normal)), 1e-8)
        cross = action.T @ (weight[:, None] * response)
        coefficient = np.linalg.solve(
            normal + ridge * scale * identity, cross
        )
        if method == "diagonal":
            coefficient = np.diag(np.diag(coefficient))
        elif method == "symmetric":
            coefficient = 0.5 * (coefficient + coefficient.T)
        elif method != "unconstrained":
            raise ValueError(method)
        result[index] = query_action[index] @ coefficient
    return result.astype(np.float32)


def oracle_metrics_subset(
    pair: dict[str, np.ndarray], predicted_delta: np.ndarray, mask: np.ndarray,
) -> dict[str, Any]:
    true_delta = pair["delta_gradient"][mask]
    left = pair["left_gradient"][mask]
    right = pair["right_gradient"][mask]
    predicted_delta = predicted_delta[mask]
    predicted_right = left + predicted_delta
    predicted_left = right - predicted_delta
    cross_cosine = np.concatenate((
        cosine_rows(predicted_right, right),
        cosine_rows(predicted_left, left),
    ))
    true_response = cosine_rows(left, right)
    reversal = true_response < 0.0
    predicted_response = np.concatenate((
        cosine_rows(left, predicted_right),
        cosine_rows(right, predicted_left),
    ))
    reversal_orientation = np.concatenate((reversal, reversal))
    delta_ratio = np.linalg.norm(predicted_delta, axis=1) / (
        np.linalg.norm(true_delta, axis=1) + 1e-12
    )
    delta_cosine = cosine_rows(predicted_delta, true_delta)
    result = {
        "pair_count": int(len(left)),
        "true_reversal_pair_count": int(np.sum(reversal)),
        "delta_gradient_cosine": distribution(delta_cosine),
        "delta_gradient_norm_ratio": distribution(delta_ratio),
        "cross_target_cosine": distribution(cross_cosine),
        "cross_target_positive_fraction": float(np.mean(cross_cosine > 0.0)),
        "true_reversal_flip_recall": (
            float(np.mean(predicted_response[reversal_orientation] < 0.0))
            if np.any(reversal) else 1.0
        ),
        "nonreversal_false_flip_fraction": float(np.mean(
            predicted_response[~reversal_orientation] < 0.0
        )),
    }
    return result


def oracle_metrics(pair: dict[str, np.ndarray], predicted_delta: np.ndarray) -> dict[str, Any]:
    distance = pair["distance_sigma"]
    result = oracle_metrics_subset(
        pair, predicted_delta, np.ones(len(distance), dtype=bool)
    )
    result["by_chord_distance"] = {
        "small_le_0_15": oracle_metrics_subset(
            pair, predicted_delta, distance <= 0.15
        ),
        "medium_0_15_0_30": oracle_metrics_subset(
            pair, predicted_delta, (distance > 0.15) & (distance <= 0.30)
        ),
        "long_gt_0_30": oracle_metrics_subset(
            pair, predicted_delta, distance > 0.30
        ),
    }
    return result


def selection_score(metrics: dict[str, Any]) -> float:
    # Local-H is only expected to be valid inside the G0-validated 0.15 sigma
    # radius.  Medium/long chords remain extrapolation diagnostics.
    local = metrics["by_chord_distance"]["small_le_0_15"]
    norm_median = local["delta_gradient_norm_ratio"]["median"]
    return float(
        local["cross_target_cosine"]["median"]
        + 0.5 * local["cross_target_cosine"]["p10"]
        + 0.5 * local["true_reversal_flip_recall"]
        - 0.1 * abs(math.log(max(float(norm_median), 1e-8)))
    )


def repeat_semantics(
    data: Any,
    pair: dict[str, np.ndarray],
    fresh: dict[str, np.ndarray],
) -> dict[str, Any]:
    left, right = pair["left_context"], pair["right_context"]
    physical = {}
    for name, value in zip(("history", "reference", "current"), data.inputs[:3]):
        delta = np.asarray(value[left]) - np.asarray(value[right])
        physical[name] = {
            "maximum_abs_repeat_difference": float(np.max(np.abs(delta))),
            "rms_repeat_difference": distribution(np.sqrt(np.mean(
                np.square(delta), axis=tuple(range(1, delta.ndim))
            ))),
        }
    nuisance = {}
    for name, value in zip(
        ("guided_anchor", "feedback", "gradient_context"), data.inputs[3:]
    ):
        delta = np.asarray(value[left]) - np.asarray(value[right])
        nuisance[name] = distribution(np.sqrt(np.mean(
            np.square(delta), axis=tuple(range(1, delta.ndim))
        )))
    fresh_lookup = {
        int(context): position
        for position, context in enumerate(fresh["context_index"])
    }
    fresh_left = np.asarray([fresh_lookup[int(x)] for x in left], np.int64)
    fresh_right = np.asarray([fresh_lookup[int(x)] for x in right], np.int64)
    for name in ("alpha_center", "actor_center"):
        delta = (
            fresh[name][fresh_left] - fresh[name][fresh_right]
        ) / fresh["sigma"][fresh_left, None, :]
        nuisance[f"{name}_sigma_rms"] = distribution(np.sqrt(np.mean(
            np.square(delta), axis=(1, 2)
        )))
    parameters = list(inspect.signature(
        TorchMPPIStructuredLocalQCritic.local_parameters
    ).parameters)
    return {
        "repeat_invariant_physical_blocks": physical,
        "repeat_varying_first_pass_blocks": nuisance,
        "structured_local_parameters_signature": parameters,
        "absolute_actor_center_explicit_in_local_parameters": (
            "reference_action" in parameters or "absolute_center" in parameters
        ),
        "semantic_finding": (
            "physical blocks are repeat-invariant, but the current Structured "
            "Critic conditions Q0/g0/H on repeat-varying anchor/feedback/gradient "
            "context and does not explicitly receive the Actor absolute center"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    neighbor_counts = [int(value) for value in args.neighbor_counts.split(",")]
    ridge_grid = [float(value) for value in args.ridge_grid.split(",")]
    response_scales = [float(value) for value in args.response_scales.split(",")]
    if args.primary_neighbor_count not in neighbor_counts:
        raise ValueError("primary neighbor count must be in neighbor-counts")
    if args.state_temperature <= 0 or min(ridge_grid) <= 0 or min(response_scales) <= 0:
        raise ValueError("temperature and ridge must be positive")
    args.output_dir.mkdir(parents=True)

    critic_summary_path = args.critic_dir / "summary.json"
    fresh_path = args.fresh_dir / "fresh_fd_audit.npz"
    fresh_validation_path = args.fresh_dir / "validation_summary.json"
    critic_summary = json.loads(critic_summary_path.read_text())
    fresh_validation = json.loads(fresh_validation_path.read_text())
    if fresh_validation["qualification"] != "PASS":
        raise AssertionError("fresh-FD source is not independently validated")
    if critic_summary.get("contract", {}).get("formal_validation_loaded", False):
        raise AssertionError("critic source loaded formal validation")
    if critic_summary.get("contract", {}).get("test_loaded", False):
        raise AssertionError("critic source loaded test")

    initial_actor = Path(critic_summary["initial_actor"])
    initial_payload = torch.load(initial_actor, map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial_payload["labels"]), old_payload)
    labels_path = args.critic_dir / "local_forward_labels.npz"
    labels = load_npz(labels_path)
    fresh = load_npz(fresh_path)
    if not np.allclose(labels["probe_radii_sigma"], (0.05, 0.10, 0.20)):
        raise AssertionError("local label radii changed")
    partner = repeat_partner_positions(data, labels)
    left = pair_left(partner)
    episode = labels["episode"][left]
    train_set = set(critic_summary["split"]["train_episodes"])
    validation_set = set(critic_summary["split"]["internal_validation_episodes"])
    heldout_set = set(critic_summary["split"]["heldout_episodes"])
    train_left = left[np.asarray([value in train_set for value in episode])]
    validation_left = left[np.asarray([value in validation_set for value in episode])]
    heldout_left = left[np.asarray([value in heldout_set for value in episode])]
    if len(train_left) * 2 != 4590 or len(validation_left) * 2 != 1110:
        raise AssertionError("internal split context counts changed")
    if len(heldout_left) * 2 != 600:
        raise AssertionError("heldout context count changed")

    maximum_residual_sigma = float(initial_payload["maximum_residual_sigma"])
    train_pair = pair_data(
        data, labels, train_left, partner, maximum_residual_sigma
    )
    validation_pair = pair_data(
        data, labels, validation_left, partner, maximum_residual_sigma
    )
    heldout_pair = fresh_pair_data(
        data, labels, heldout_left, partner, fresh, maximum_residual_sigma
    )

    train_feature, validation_feature, feature_blocks = fit_physical_features(
        data, train_pair["left_context"], validation_pair["left_context"]
    )
    validation_similarity = validation_feature @ train_feature.T
    validation_geometry, _ = local_geometry(
        train_pair["delta_action"], validation_similarity, neighbor_counts
    )

    grid = []
    methods = ("diagonal", "symmetric", "unconstrained")
    for method in methods:
        for count in neighbor_counts:
            for ridge in ridge_grid:
                raw_prediction = fit_local_response(
                    train_pair["delta_action"], train_pair["delta_gradient"],
                    train_pair["distance_sigma"], validation_similarity,
                    validation_pair["delta_action"], count, ridge,
                    args.state_temperature, args, method,
                )
                for response_scale in response_scales:
                    metrics = oracle_metrics(
                        validation_pair, response_scale * raw_prediction
                    )
                    grid.append({
                        "method": method,
                        "neighbor_count": count,
                        "ridge": ridge,
                        "response_scale": response_scale,
                        "selection_score": selection_score(metrics),
                        "metrics": metrics,
                    })
    selected = {}
    for method in methods:
        selected[method] = max(
            (row for row in grid if row["method"] == method),
            key=lambda row: row["selection_score"],
        )

    fit_left = np.concatenate((train_left, validation_left))
    fit_pair = pair_data(
        data, labels, fit_left, partner, maximum_residual_sigma
    )
    fit_feature, heldout_feature, final_feature_blocks = transform_physical_features(
        data, train_pair["left_context"], fit_pair["left_context"],
        heldout_pair["left_context"],
    )
    heldout_similarity = heldout_feature @ fit_feature.T
    heldout_geometry, heldout_orders = local_geometry(
        fit_pair["delta_action"], heldout_similarity, neighbor_counts
    )
    heldout_oracle, heldout_predictions = {}, {}
    for method, choice in selected.items():
        raw_prediction = fit_local_response(
            fit_pair["delta_action"], fit_pair["delta_gradient"],
            fit_pair["distance_sigma"], heldout_similarity,
            heldout_pair["delta_action"], int(choice["neighbor_count"]),
            float(choice["ridge"]), args.state_temperature, args, method,
        )
        prediction = float(choice["response_scale"]) * raw_prediction
        heldout_predictions[method] = prediction
        heldout_oracle[method] = {
            "selected_on_internal_validation": {
                "neighbor_count": int(choice["neighbor_count"]),
                "ridge": float(choice["ridge"]),
                "response_scale": float(choice["response_scale"]),
                "selection_score": float(choice["selection_score"]),
                "metrics": choice["metrics"],
            },
            "heldout_fresh_fd": oracle_metrics(heldout_pair, prediction),
        }

    symmetric = heldout_oracle["symmetric"]["heldout_fresh_fd"]
    unconstrained = heldout_oracle["unconstrained"]["heldout_fresh_fd"]
    symmetric_small = symmetric["by_chord_distance"]["small_le_0_15"]
    unconstrained_small = unconstrained["by_chord_distance"]["small_le_0_15"]
    primary_geometry = heldout_geometry[str(args.primary_neighbor_count)]
    geometry_well_excited = bool(
        primary_geometry["rank_ge_12_at_5pct_fraction"] >= 0.80
        and primary_geometry["smallest_to_largest_singular_ratio"]["median"] >= 0.02
    )
    symmetric_oracle_pass = bool(
        symmetric_small["cross_target_cosine"]["median"] >= 0.90
        and symmetric_small["cross_target_cosine"]["p10"] >= 0.0
        and symmetric_small["true_reversal_flip_recall"] >= 0.50
    )
    unconstrained_only = bool(
        not symmetric_oracle_pass
        and unconstrained_small["cross_target_cosine"]["median"] >= 0.90
        and unconstrained_small["cross_target_cosine"]["p10"] >= 0.0
        and unconstrained_small["true_reversal_flip_recall"] >= 0.50
    )
    if symmetric_oracle_pass:
        qualification = "LOCAL_H_ORACLE_SMALL_CHORD_PASS_EXISTING_DATA_SUFFICIENT"
        route = "clean state-conditioned scalar-Q/H representation and training"
    elif unconstrained_only:
        qualification = "UNCONSTRAINED_RESPONSE_ONLY_LOCAL_SCALAR_Q_MISMATCH"
        route = "audit locality/label consistency before collecting more action probes"
    else:
        qualification = "LOCAL_H_ORACLE_FAIL_EXACT_STATE_RANK1_TARGETED_PROBES_JUSTIFIED"
        route = "collect same-state small-radius directionally independent DBM probes"

    semantics = repeat_semantics(data, heldout_pair, fresh)
    analysis = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "recommended_route": route,
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "sources": {
            "critic_summary": str(critic_summary_path.resolve()),
            "critic_summary_sha256": sha256_file(critic_summary_path),
            "labels": str(labels_path.resolve()),
            "labels_sha256": sha256_file(labels_path),
            "fresh_fd": str(fresh_path.resolve()),
            "fresh_fd_sha256": sha256_file(fresh_path),
            "fresh_validation": str(fresh_validation_path.resolve()),
            "fresh_validation_sha256": sha256_file(fresh_validation_path),
            "initial_actor": str(initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(initial_actor),
        },
        "contract": {
            "new_dbm_rollouts": 0,
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "neighbor_selection": "repeat-invariant history/reference/current only",
            "heldout_target": "consumed internal-selection fresh-FD",
            "leave_one_pair_out_by_episode_split": True,
            "oracle_receives_query_endpoint_gradient": (
                "yes; this isolates H/action-response recovery and is not deployable"
            ),
        },
        "counts": {
            "train_pair": int(len(train_left)),
            "internal_validation_pair": int(len(validation_left)),
            "fit_pair_after_selection": int(len(fit_left)),
            "heldout_pair": int(len(heldout_left)),
        },
        "coordinate": {
            "flat_action_names": FLAT_NAMES,
            "front_steering_knot_0_2_flat_indices": [1, 3, 5],
            "delta_action": "(center_right-center_left)/(maximum_residual_sigma*sigma)",
            "maximum_residual_sigma": maximum_residual_sigma,
        },
        "semantic_audit": semantics,
        "physical_feature_blocks": final_feature_blocks,
        "chord_distance_sigma": {
            "train": distribution(train_pair["distance_sigma"]),
            "internal_validation": distribution(validation_pair["distance_sigma"]),
            "heldout": distribution(heldout_pair["distance_sigma"]),
        },
        "component_excitation": {
            "fit_bank": component_excitation(fit_pair["delta_action"]),
            "heldout": component_excitation(heldout_pair["delta_action"]),
        },
        "local_geometry": {
            "exact_physical_state": {
                "chord_count_per_state": 1,
                "maximum_action_direction_rank": 1,
                "finding": (
                    "full local rank below is obtained only by pooling chords "
                    "from different physical states"
                ),
            },
            "thresholds": {
                "geometry_well_excited": (
                    "at K=primary, >=80% queries have >=12 singular values "
                    "above 5% of s_max and median s_min/s_max >=0.02"
                ),
                "primary_neighbor_count": args.primary_neighbor_count,
            },
            "geometry_well_excited": geometry_well_excited,
            "internal_validation": validation_geometry,
            "heldout": heldout_geometry,
        },
        "oracle": {
            "state_temperature": args.state_temperature,
            "chord_weighting": {
                "d0": args.chord_d0,
                "epsilon": args.chord_epsilon,
                "w_max": args.chord_w_max,
            },
            "selection_grid": grid,
            "response_scale_grid": response_scales,
            "selected_and_heldout": heldout_oracle,
            "symmetric_oracle_gate": {
                "scope": "heldout chords with physical sigma RMS <= 0.15",
                "median_ge_0_90": symmetric_small["cross_target_cosine"]["median"] >= 0.90,
                "p10_ge_0": symmetric_small["cross_target_cosine"]["p10"] >= 0.0,
                "reversal_recall_ge_0_50": symmetric_small["true_reversal_flip_recall"] >= 0.50,
                "all_passed": symmetric_oracle_pass,
            },
        },
    }
    analysis_path = args.output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "local_oracle_audit.npz",
        heldout_context=heldout_pair["left_context"],
        heldout_episode=heldout_pair["episode"],
        heldout_delta_action=heldout_pair["delta_action"],
        heldout_distance_sigma=heldout_pair["distance_sigma"],
        heldout_delta_gradient=heldout_pair["delta_gradient"],
        heldout_left_gradient=heldout_pair["left_gradient"],
        heldout_right_gradient=heldout_pair["right_gradient"],
        heldout_similarity=heldout_similarity,
        fit_delta_action=fit_pair["delta_action"],
        primary_neighbor_index=heldout_orders[args.primary_neighbor_count],
        diagonal_predicted_delta=heldout_predictions["diagonal"],
        symmetric_predicted_delta=heldout_predictions["symmetric"],
        unconstrained_predicted_delta=heldout_predictions["unconstrained"],
    )
    (args.output_dir / "README.md").write_text(
        "# Chord geometry / local-H oracle audit\n\n"
        f"Qualification: `{qualification}`.\n\n"
        "Zero new DBM rollouts; Actor remained frozen. See `analysis.json`.\n"
    )
    print(json.dumps({
        "qualification": qualification,
        "counts": analysis["counts"],
        "geometry_well_excited": geometry_well_excited,
        "symmetric_oracle": symmetric,
        "unconstrained_oracle": unconstrained,
        "output": str(analysis_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
