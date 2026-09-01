#!/usr/bin/env python3
"""Audit zero-rollout g0 learnability with repeat-clean KNN semantics.

The audit selects deterministic KNN aggregation on internal validation and
evaluates once on consumed internal-selection fresh-FD labels.  It compares a
repeat-invariant physical-state metric with the same metric augmented by an
explicit absolute Actor center.  A label-aware oracle-best neighbor is reported
only as non-deployable headroom.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
DEFAULT_HARD = Path(
    "outputs/mppi_proposal/"
    "direct_local_critic_hard_state_attribution_20260813_v2"
)
DEFAULT_H_ORACLE = Path(
    "outputs/mppi_proposal/chord_geometry_local_oracle_20260814_v3"
)
DEFAULT_STRUCTURED = Path(
    "outputs/mppi_proposal/structured_local_q_full_20260814_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/g0_learnability_audit_20260814_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--hard-dir", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--h-oracle-dir", type=Path, default=DEFAULT_H_ORACLE)
    parser.add_argument("--structured-dir", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbor-counts", default="1,2,4,8,16,32,64")
    parser.add_argument("--temperatures", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--action-weights", default="0.25,0.5,0.75")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    return np.sum(left * right, axis=-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-12
    )


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    return value / (np.linalg.norm(value, axis=1, keepdims=True) + 1e-12)


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


def gradient_metrics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mask is not None:
        prediction, target = prediction[mask], target[mask]
    cosine = cosine_rows(prediction, target)
    ratio = np.linalg.norm(prediction, axis=1) / (
        np.linalg.norm(target, axis=1) + 1e-12
    )
    return {
        "count": int(len(target)),
        "cosine": distribution(cosine),
        "positive_fraction": float(np.mean(cosine > 0.0)),
        "norm_ratio": distribution(ratio),
        "norm_ratio_below_0_5_fraction": float(np.mean(ratio < 0.5)),
        "norm_ratio_above_2_fraction": float(np.mean(ratio > 2.0)),
    }


def standardized_unit_block(
    normalization: np.ndarray, bank: np.ndarray, query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    normalization = np.asarray(normalization, np.float32).reshape(len(normalization), -1)
    bank = np.asarray(bank, np.float32).reshape(len(bank), -1)
    query = np.asarray(query, np.float32).reshape(len(query), -1)
    mean, std = np.mean(normalization, axis=0), np.std(normalization, axis=0)
    active = std > 1e-5
    bank = (bank[:, active] - mean[active]) / std[active]
    query = (query[:, active] - mean[active]) / std[active]
    bank = normalize_rows(bank)
    query = normalize_rows(query)
    return bank, query, {
        "flat_dimension": int(normalization.shape[1]),
        "active_dimension": int(np.sum(active)),
    }


def similarity_components(
    data: Any,
    normalization_context: np.ndarray,
    bank_context: np.ndarray,
    query_context: np.ndarray,
    normalization_action: np.ndarray,
    bank_action: np.ndarray,
    query_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    physical = None
    blocks = {}
    for name, value in zip(("history", "reference", "current"), data.inputs[:3]):
        bank, query, info = standardized_unit_block(
            value[normalization_context], value[bank_context], value[query_context]
        )
        local = query @ bank.T
        physical = local / 3.0 if physical is None else physical + local / 3.0
        blocks[name] = info
    action_bank, action_query, action_info = standardized_unit_block(
        normalization_action, bank_action, query_action
    )
    blocks["absolute_actor_center"] = action_info
    return (
        np.asarray(physical, np.float32),
        np.asarray(action_query @ action_bank.T, np.float32),
        blocks,
    )


def sorted_neighbors(similarity: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argpartition(-similarity, maximum - 1, axis=1)[:, :maximum]
    local = np.take_along_axis(similarity, order, axis=1)
    sort = np.argsort(-local, axis=1)
    return (
        np.take_along_axis(order, sort, axis=1),
        np.take_along_axis(local, sort, axis=1),
    )


def knn_prediction(
    order: np.ndarray,
    similarity: np.ndarray,
    bank_gradient: np.ndarray,
    count: int,
    temperature: float,
    aggregation: str,
) -> np.ndarray:
    index = order[:, :count]
    local_similarity = similarity[:, :count].astype(np.float64)
    weight = np.exp(
        (local_similarity - local_similarity[:, :1]) / temperature
    )
    weight /= np.sum(weight, axis=1, keepdims=True)
    neighbor = bank_gradient[index].astype(np.float64)
    if aggregation == "raw_mean":
        prediction = np.sum(weight[:, :, None] * neighbor, axis=1)
    elif aggregation == "unit_mean_rescaled":
        unit = neighbor / (
            np.linalg.norm(neighbor, axis=2, keepdims=True) + 1e-12
        )
        direction = np.sum(weight[:, :, None] * unit, axis=1)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
        magnitude = np.sum(
            weight * np.linalg.norm(neighbor, axis=2), axis=1
        )
        prediction = direction * magnitude[:, None]
    else:
        raise ValueError(aggregation)
    return prediction.astype(np.float32)


def selection_score(metrics: dict[str, Any]) -> float:
    cosine = metrics["cosine"]
    norm = float(metrics["norm_ratio"]["median"])
    return float(
        cosine["median"] + 0.75 * cosine["p10"]
        + 0.10 * metrics["positive_fraction"]
        - 0.10 * abs(math.log(max(norm, 1e-8)))
    )


def neighbor_diagnostics(
    order: np.ndarray,
    similarity: np.ndarray,
    bank_gradient: np.ndarray,
    target: np.ndarray,
    counts: tuple[int, ...],
    hard: np.ndarray,
) -> tuple[dict[str, Any], dict[int, np.ndarray], dict[int, np.ndarray]]:
    bank_unit = normalize_rows(bank_gradient)
    target_unit = normalize_rows(target)
    result, best_arrays, coherence_arrays = {}, {}, {}
    for count in counts:
        neighbor = bank_unit[order[:, :count]]
        mean = np.mean(neighbor, axis=1)
        coherence = np.linalg.norm(mean, axis=1)
        individual = np.sum(neighbor * target_unit[:, None], axis=2)
        best = np.max(individual, axis=1)
        mean_cosine = cosine_rows(mean, target_unit)
        pairwise = (
            count * np.square(coherence) - 1.0
        ) / max(count - 1, 1)
        if count == 1:
            pairwise = np.ones_like(coherence)
        result[str(count)] = {
            "neighbor_similarity": distribution(similarity[:, count - 1]),
            "mean_direction_cosine": distribution(mean_cosine),
            "hard_mean_direction_cosine": distribution(mean_cosine[hard]),
            "label_coherence": distribution(coherence),
            "hard_label_coherence": distribution(coherence[hard]),
            "neighbor_pairwise_label_cosine": distribution(pairwise),
            "oracle_best_cosine": distribution(best),
            "hard_oracle_best_cosine": distribution(best[hard]),
            "oracle_best_positive_fraction": float(np.mean(best > 0.0)),
            "hard_oracle_best_positive_fraction": float(np.mean(best[hard] > 0.0)),
        }
        best_arrays[count] = best.astype(np.float32)
        coherence_arrays[count] = coherence.astype(np.float32)
    return result, best_arrays, coherence_arrays


def by_speed_scenario(
    prediction: np.ndarray,
    target: np.ndarray,
    speed: np.ndarray,
    scenario: np.ndarray,
    clipped: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for value in sorted(np.unique(speed)):
        mask = np.isclose(speed, value)
        result[f"speed_{float(value):.1f}"] = gradient_metrics(
            prediction, target, mask
        )
    for value in sorted(np.unique(scenario)):
        mask = scenario == value
        result[f"scenario_{value}"] = gradient_metrics(prediction, target, mask)
    for value in (False, True):
        mask = clipped == value
        result[f"clipped_{str(value).lower()}"] = gradient_metrics(
            prediction, target, mask
        )
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    neighbor_counts = [int(value) for value in args.neighbor_counts.split(",")]
    temperatures = [float(value) for value in args.temperatures.split(",")]
    action_weights = [float(value) for value in args.action_weights.split(",")]
    maximum_neighbors = max(max(neighbor_counts), 20)

    critic_summary_path = args.critic_dir / "summary.json"
    labels_path = args.critic_dir / "local_forward_labels.npz"
    fresh_path = args.fresh_dir / "fresh_fd_audit.npz"
    fresh_validation_path = args.fresh_dir / "validation_summary.json"
    hard_path = args.hard_dir / "hard_state_analysis.npz"
    h_oracle_path = args.h_oracle_dir / "local_oracle_audit.npz"
    h_oracle_analysis_path = args.h_oracle_dir / "analysis.json"
    structured_path = args.structured_dir / "summary.json"
    critic_summary = json.loads(critic_summary_path.read_text())
    fresh_validation = json.loads(fresh_validation_path.read_text())
    if fresh_validation["qualification"] != "PASS":
        raise AssertionError("fresh-FD source is not independently validated")
    h_oracle_analysis = json.loads(h_oracle_analysis_path.read_text())
    if h_oracle_analysis["contract"]["formal_validation_loaded"]:
        raise AssertionError("H oracle loaded formal validation")

    initial_actor = Path(critic_summary["initial_actor"])
    initial_payload = torch.load(initial_actor, map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial_payload["labels"]), old_payload)
    labels, fresh = load_npz(labels_path), load_npz(fresh_path)
    hard_data, h_oracle = load_npz(hard_path), load_npz(h_oracle_path)
    if not np.allclose(labels["probe_radii_sigma"], (0.05, 0.10, 0.20)):
        raise AssertionError("local labels changed")
    partner = repeat_partner_positions(data, labels)
    if not np.array_equal(partner[partner], np.arange(len(partner))):
        raise AssertionError("repeat mapping changed")

    train_set = set(critic_summary["split"]["train_episodes"])
    validation_set = set(critic_summary["split"]["internal_validation_episodes"])
    heldout_set = set(critic_summary["split"]["heldout_episodes"])
    train_position = np.flatnonzero(np.asarray([
        value in train_set for value in labels["episode"]
    ]))
    validation_position = np.flatnonzero(np.asarray([
        value in validation_set for value in labels["episode"]
    ]))
    heldout_position = np.flatnonzero(np.asarray([
        value in heldout_set for value in labels["episode"]
    ]))
    if tuple(map(len, (train_position, validation_position, heldout_position))) != (
        4590, 1110, 600
    ):
        raise AssertionError("split counts changed")
    context = labels["context_index"].astype(np.int64)
    train_context, validation_context = context[train_position], context[validation_position]
    heldout_context = fresh["context_index"].astype(np.int64)
    if set(heldout_context.tolist()) != set(context[heldout_position].tolist()):
        raise AssertionError("fresh heldout contexts changed")
    fresh_lookup = {int(value): index for index, value in enumerate(heldout_context)}
    heldout_order = np.asarray([
        fresh_lookup[int(value)] for value in context[heldout_position]
    ], np.int64)
    heldout_context = heldout_context[heldout_order]
    heldout_target = fresh["gradient"][heldout_order].astype(np.float32)
    heldout_action = fresh["actor_center"][heldout_order].astype(np.float32)
    heldout_episode = fresh["episode"][heldout_order]
    heldout_speed = fresh["reference_speed"][heldout_order]
    heldout_scenario = fresh["scenario"][heldout_order]
    heldout_clipped = fresh["clipped_context"][heldout_order]
    if not np.array_equal(heldout_episode, labels["episode"][heldout_position]):
        raise AssertionError("heldout episode order reconstruction failed")

    train_gradient = labels["gradient_by_radius"][train_position, 0].astype(np.float32)
    validation_gradient = labels["gradient_by_radius"][validation_position, 0].astype(np.float32)
    train_action = labels["actor_center"][train_position].astype(np.float32)
    validation_action = labels["actor_center"][validation_position].astype(np.float32)
    validation_physical, validation_action_similarity, feature_blocks = similarity_components(
        data, train_context, train_context, validation_context,
        train_action, train_action, validation_action,
    )

    validation_candidates = []
    spaces = {
        "physical_only": [0.0],
        "physical_plus_absolute_action": action_weights,
    }
    for space, weights in spaces.items():
        for action_weight in weights:
            similarity = (
                (1.0 - action_weight) * validation_physical
                + action_weight * validation_action_similarity
            )
            order, local_similarity = sorted_neighbors(similarity, maximum_neighbors)
            for count in neighbor_counts:
                local_temperatures = (temperatures[:1] if count == 1 else temperatures)
                for temperature in local_temperatures:
                    for aggregation in ("raw_mean", "unit_mean_rescaled"):
                        prediction = knn_prediction(
                            order, local_similarity, train_gradient,
                            count, temperature, aggregation,
                        )
                        metrics = gradient_metrics(prediction, validation_gradient)
                        validation_candidates.append({
                            "space": space,
                            "action_weight": action_weight,
                            "neighbor_count": count,
                            "temperature": temperature,
                            "aggregation": aggregation,
                            "selection_score": selection_score(metrics),
                            "metrics": metrics,
                        })
    selected = {
        space: max(
            (row for row in validation_candidates if row["space"] == space),
            key=lambda row: row["selection_score"],
        )
        for space in spaces
    }

    fit_position = np.concatenate((train_position, validation_position))
    fit_context = context[fit_position]
    fit_action = labels["actor_center"][fit_position].astype(np.float32)
    fit_gradient = labels["gradient_by_radius"][fit_position, 0].astype(np.float32)
    heldout_physical, heldout_action_similarity, final_feature_blocks = similarity_components(
        data, train_context, fit_context, heldout_context,
        train_action, fit_action, heldout_action,
    )
    hard_lookup = {
        int(value): bool(flag)
        for value, flag in zip(hard_data["heldout_context"], hard_data["hard"])
    }
    hard = np.asarray([hard_lookup[int(value)] for value in heldout_context], bool)
    current_ensemble_lookup = {
        int(value): gradient
        for value, gradient in zip(
            hard_data["heldout_context"], hard_data["ensemble"]
        )
    }
    current_ensemble = np.asarray([
        current_ensemble_lookup[int(value)] for value in heldout_context
    ], np.float32)

    heldout_result, predictions, diagnostic_arrays = {}, {}, {}
    for space, choice in selected.items():
        action_weight = float(choice["action_weight"])
        similarity = (
            (1.0 - action_weight) * heldout_physical
            + action_weight * heldout_action_similarity
        )
        order, local_similarity = sorted_neighbors(similarity, maximum_neighbors)
        prediction = knn_prediction(
            order, local_similarity, fit_gradient,
            int(choice["neighbor_count"]), float(choice["temperature"]),
            str(choice["aggregation"]),
        )
        predictions[space] = prediction
        diagnostic_counts = tuple(sorted(set((1, 5, 20, int(choice["neighbor_count"])))))
        diagnostics, best, coherence = neighbor_diagnostics(
            order, local_similarity, fit_gradient, heldout_target,
            diagnostic_counts, hard,
        )
        selected_count = int(choice["neighbor_count"])
        heldout_result[space] = {
            "selected_on_internal_validation": choice,
            "heldout": {
                "all": gradient_metrics(prediction, heldout_target),
                "current_critic_hard": gradient_metrics(
                    prediction, heldout_target, hard
                ),
                "current_critic_nonhard": gradient_metrics(
                    prediction, heldout_target, ~hard
                ),
                "by_regime": by_speed_scenario(
                    prediction, heldout_target, heldout_speed,
                    heldout_scenario, heldout_clipped,
                ),
            },
            "neighbor_diagnostics": diagnostics,
        }
        diagnostic_arrays[space] = {
            "order": order,
            "similarity": local_similarity,
            "oracle_best_20": best[20],
            "coherence_20": coherence[20],
            "oracle_best_selected": best[selected_count],
            "coherence_selected": coherence[selected_count],
        }

    action_result = heldout_result["physical_plus_absolute_action"]
    action_metrics = action_result["heldout"]["all"]
    oracle20 = action_result["neighbor_diagnostics"]["20"]["oracle_best_cosine"]
    if action_metrics["cosine"]["p10"] >= 0.0:
        qualification = "G0_KNN_HELDOUT_TAIL_PASS_EXISTING_DATA_LEARNABLE"
        route = "semantic-clean Critic retraining before targeted state expansion"
    elif oracle20["p10"] >= 0.0:
        qualification = "G0_INFORMATION_PRESENT_BUT_FIXED_NEIGHBOR_RULE_FAILS_TAIL"
        route = (
            "semantic-clean state+absolute-action representation/training and "
            "target H probes for contexts with joint g0/H failure"
        )
    else:
        qualification = "G0_LOCAL_LABEL_COVERAGE_INSUFFICIENT"
        route = "prioritize new labels for g0-uncovered states before H-only probes"

    # Align the independently audited H-only failure with both repeat contexts.
    label_position_by_context = {
        int(value): index for index, value in enumerate(context)
    }
    h_left_context = h_oracle["heldout_context"].astype(np.int64)
    h_left_position = np.asarray([
        label_position_by_context[int(value)] for value in h_left_context
    ], np.int64)
    h_right_position = partner[h_left_position]
    h_right_context = context[h_right_position]
    h_prediction = h_oracle["symmetric_predicted_delta"]
    h_left_gradient = h_oracle["heldout_left_gradient"]
    h_right_gradient = h_oracle["heldout_right_gradient"]
    h_cross_left = cosine_rows(
        h_left_gradient + h_prediction, h_right_gradient
    )
    h_cross_right = cosine_rows(
        h_right_gradient - h_prediction, h_left_gradient
    )
    h_bad_lookup = {}
    for left_value, right_value, left_bad, right_bad in zip(
        h_left_context, h_right_context, h_cross_left < 0.0, h_cross_right < 0.0
    ):
        h_bad_lookup[int(left_value)] = bool(left_bad)
        h_bad_lookup[int(right_value)] = bool(right_bad)
    h_bad = np.asarray([h_bad_lookup[int(value)] for value in heldout_context], bool)
    g0_prediction = predictions["physical_plus_absolute_action"]
    g0_cosine = cosine_rows(g0_prediction, heldout_target)
    oracle_best_20 = diagnostic_arrays[
        "physical_plus_absolute_action"
    ]["oracle_best_20"]
    coherence_20 = diagnostic_arrays[
        "physical_plus_absolute_action"
    ]["coherence_20"]
    g0_bad = g0_cosine < 0.0
    g0_uncovered = oracle_best_20 < 0.0
    mixed = coherence_20 < 0.30
    routing = {
        "g0_knn_negative_count": int(np.sum(g0_bad)),
        "g0_oracle_best20_negative_count": int(np.sum(g0_uncovered)),
        "g0_mixed_neighbor_count": int(np.sum(mixed)),
        "h_oracle_cross_negative_count": int(np.sum(h_bad)),
        "joint_g0_knn_and_h_oracle_negative_count": int(np.sum(g0_bad & h_bad)),
        "current_critic_hard_and_g0_knn_negative_count": int(np.sum(hard & g0_bad)),
        "current_critic_hard_but_g0_oracle_best20_positive_count": int(np.sum(
            hard & (oracle_best_20 >= 0.0)
        )),
    }

    structured = json.loads(structured_path.read_text())
    d_r2_recall = [
        record["fresh_fd"]["chord_bins"]["small_le_0_15"][
            "reversal_flip_recall"
        ] for record in structured["arms"]["D_R2"]["records"]
    ]
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
            "hard_state": str(hard_path.resolve()),
            "hard_state_sha256": sha256_file(hard_path),
            "h_oracle": str(h_oracle_path.resolve()),
            "h_oracle_sha256": sha256_file(h_oracle_path),
            "h_oracle_analysis": str(h_oracle_analysis_path.resolve()),
            "h_oracle_analysis_sha256": sha256_file(h_oracle_analysis_path),
            "structured_summary": str(structured_path.resolve()),
            "structured_summary_sha256": sha256_file(structured_path),
            "initial_actor": str(initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(initial_actor),
        },
        "contract": {
            "new_dbm_rollouts": 0,
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "hyperparameter_selection": "internal-validation only",
            "heldout_target": "consumed internal-selection fresh-FD",
            "physical_blocks": ["history", "reference", "current"],
            "explicit_action": "absolute Actor center knots; no alpha/feedback/gradient context",
            "oracle_best_uses_target_label": True,
            "oracle_best_is_deployable": False,
        },
        "counts": {
            "train_context": int(len(train_position)),
            "internal_validation_context": int(len(validation_position)),
            "fit_context_after_selection": int(len(fit_position)),
            "heldout_context": int(len(heldout_position)),
            "current_critic_hard_context": int(np.sum(hard)),
        },
        "feature_blocks": final_feature_blocks,
        "selection_grid": {
            "neighbor_counts": neighbor_counts,
            "temperatures": temperatures,
            "action_weights": action_weights,
            "aggregation": ["raw_mean", "unit_mean_rescaled"],
            "records": validation_candidates,
        },
        "spaces": heldout_result,
        "current_critic_baseline": {
            "ensemble": gradient_metrics(current_ensemble, heldout_target),
            "structured_D_R2_small_chord_reversal_recall_by_seed": d_r2_recall,
            "structured_D_R2_small_chord_reversal_recall_median": float(
                np.median(d_r2_recall)
            ),
            "H_only_oracle_small_chord_reversal_recall": h_oracle_analysis[
                "oracle"
            ]["selected_and_heldout"]["symmetric"]["heldout_fresh_fd"][
                "by_chord_distance"
            ]["small_le_0_15"]["true_reversal_flip_recall"],
        },
        "collection_priority_routing": routing,
    }
    analysis_path = args.output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    manifest = []
    physical_ordinal_lookup = {
        int(value): int(ordinal)
        for value, ordinal in zip(
            hard_data["heldout_context"], hard_data["physical_snapshot_ordinal"]
        )
    }
    repeat_lookup = {
        int(value): int(repeat)
        for value, repeat in zip(
            hard_data["heldout_context"], hard_data["repeat_index"]
        )
    }
    for index, value in enumerate(heldout_context):
        manifest.append({
            "context_index": int(value),
            "episode": str(heldout_episode[index]),
            "physical_snapshot_ordinal": physical_ordinal_lookup[int(value)],
            "repeat_index": repeat_lookup[int(value)],
            "reference_speed_mps": float(heldout_speed[index]),
            "scenario": str(heldout_scenario[index]),
            "clipped": bool(heldout_clipped[index]),
            "current_critic_hard": bool(hard[index]),
            "g0_knn_cosine": float(g0_cosine[index]),
            "g0_knn_negative": bool(g0_bad[index]),
            "g0_oracle_best20_cosine": float(oracle_best_20[index]),
            "g0_oracle_best20_negative": bool(g0_uncovered[index]),
            "g0_neighbor_coherence20": float(coherence_20[index]),
            "g0_mixed_neighbors": bool(mixed[index]),
            "h_oracle_cross_negative": bool(h_bad[index]),
            "joint_g0_h_failure": bool(g0_bad[index] and h_bad[index]),
        })
    manifest_path = args.output_dir / "g0_priority_manifest.json"
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "source_analysis": str(analysis_path.resolve()),
        "rows": manifest,
    }, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "g0_learnability_audit.npz",
        heldout_context=heldout_context,
        heldout_episode=heldout_episode,
        heldout_target=heldout_target,
        current_critic_hard=hard,
        current_critic_ensemble=current_ensemble,
        physical_only_prediction=predictions["physical_only"],
        action_aware_prediction=g0_prediction,
        physical_only_neighbor_index=diagnostic_arrays["physical_only"]["order"],
        action_aware_neighbor_index=diagnostic_arrays[
            "physical_plus_absolute_action"
        ]["order"],
        action_aware_oracle_best20=oracle_best_20,
        action_aware_coherence20=coherence_20,
        h_oracle_cross_negative=h_bad,
        fit_gradient=fit_gradient,
    )
    (args.output_dir / "README.md").write_text(
        "# g0 learnability audit\n\n"
        f"Qualification: `{qualification}`.\n\n"
        "Zero DBM rollouts; Actor frozen. See `analysis.json`.\n"
    )
    print(json.dumps({
        "qualification": qualification,
        "recommended_route": route,
        "selected": {
            space: {
                key: value for key, value in choice.items()
                if key != "metrics"
            } for space, choice in selected.items()
        },
        "heldout": {
            space: value["heldout"]["all"]
            for space, value in heldout_result.items()
        },
        "routing": routing,
        "output": str(analysis_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
