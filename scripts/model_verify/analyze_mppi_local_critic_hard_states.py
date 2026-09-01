#!/usr/bin/env python3
"""Attribute full-data local-Critic hard states without new DBM rollouts.

The analysis freezes the Actor and B4 Critics, consumes only internal-fit and
internal-selection contexts, and separates low density, learned-representation
aliasing, clipping, component-local failure, and within-episode transitions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import (
    TorchMPPIActorCenteredLocalCritic,
    TorchMPPIDeterministicCenterActor,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_CRITIC = Path(
    "outputs/mppi_proposal/direct_local_gradient_critic_b4_smallest_target_20260813_v2"
)
DEFAULT_FRESH = Path(
    "outputs/mppi_proposal/direct_critic_fresh_fd_b4_smallest_target_20260813_v2"
)
DEFAULT_PREVIOUS = Path(
    "outputs/mppi_proposal/direct_local_critic_negative_tail_20260813_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_local_critic_hard_state_attribution_20260813_v2"
)
BLOCK_NAMES = (
    "history", "reference", "current", "anchor", "feedback",
    "gradient_context", "actor_action",
)

# Pre-registered before the v2 attribution is generated.  These thresholds are
# routing flags, not mutually exclusive causal verdicts.
ATTRIBUTION_THRESHOLDS = {
    "hard_critic_cosine_lt": 0.0,
    "label_mismatch_cosine_lt": 0.90,
    "label_warning_cosine_lt": 0.95,
    "low_density_percentile_lte": 0.10,
    "mixed_neighbor_coherence_lt": 0.30,
    "wrong_coherent_mean_label_cosine_lt": -0.50,
    "wrong_coherent_neighbor_coherence_gt": 0.50,
    "within_state_repeat_cosine_lt": 0.50,
    "adjacent_physical_state_cosine_lt": 0.50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--previous-dir", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbor-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    return np.sum(left * right, axis=-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-12
    )


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    return value / (np.linalg.norm(value, axis=1, keepdims=True) + 1e-12)


def distribution(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, np.float64)
    if len(value) == 0:
        return {key: float("nan") for key in (
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


def binary_summary(value: np.ndarray, hard: np.ndarray) -> dict[str, Any]:
    return {
        "hard": distribution(value[hard]),
        "nonhard": distribution(value[~hard]),
        "hard_minus_nonhard_median": float(
            np.median(value[hard]) - np.median(value[~hard])
        ),
    }


def percentile_rank(value: np.ndarray) -> np.ndarray:
    """Return an empirical [0, 1] rank where zero is the least dense row."""
    value = np.asarray(value, np.float64)
    if len(value) <= 1:
        return np.ones(len(value), np.float32)
    order = np.argsort(value, kind="stable")
    result = np.empty(len(value), np.float32)
    result[order] = np.arange(len(value), dtype=np.float32) / (len(value) - 1)
    return result


def neighbor_row_metrics(
    order: np.ndarray,
    train_gradient: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    normalized_train = normalize_rows(train_gradient)
    normalized_target = normalize_rows(target)
    neighbor = normalized_train[order]
    mean_direction = np.mean(neighbor, axis=1)
    coherence = np.linalg.norm(mean_direction, axis=1)
    individual = np.sum(neighbor * normalized_target[:, None], axis=2)
    return {
        "mean_label_cosine": cosine_rows(mean_direction, normalized_target).astype(
            np.float32
        ),
        "coherence": coherence.astype(np.float32),
        "pairwise_label_cosine_mean": (
            (order.shape[1] * np.square(coherence) - 1.0)
            / max(order.shape[1] - 1, 1)
        ).astype(np.float32),
        "best_label_cosine": np.max(individual, axis=1).astype(np.float32),
        "worst_label_cosine": np.min(individual, axis=1).astype(np.float32),
    }


@torch.no_grad()
def actor_embeddings(
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[torch.Tensor, ...],
    positions: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    result = []
    for start in range(0, len(positions), batch_size):
        local = torch.from_numpy(positions[start : start + batch_size]).to(device)
        result.append(actor.encoder(*(value[local] for value in inputs)).cpu())
    return torch.cat(result)


@torch.no_grad()
def critic_features_and_gradients(
    model: TorchMPPIActorCenteredLocalCritic,
    inputs: tuple[torch.Tensor, ...],
    positions: np.ndarray,
    anchor_action: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    features, gradients = [], []
    for start in range(0, len(positions), batch_size):
        local_np = positions[start : start + batch_size]
        local = torch.from_numpy(local_np).to(device)
        anchor = torch.from_numpy(
            anchor_action[start : start + len(local_np)]
        ).to(device)
        state = model.encoder(*(value[local] for value in inputs))
        action = model.anchor_action_encoder(anchor.flatten(1))
        feature = model.fusion(torch.cat((state, action), dim=1))
        gradient = model.gradient_head(feature).reshape(-1, 16)
        features.append(feature.cpu())
        gradients.append(gradient.cpu().numpy())
    return torch.cat(features), np.concatenate(gradients)


def standardized_similarity(
    train: torch.Tensor,
    heldout: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    train = train.to(device=device, dtype=torch.float32).flatten(1)
    heldout = heldout.to(device=device, dtype=torch.float32).flatten(1)
    mean = train.mean(0)
    std = train.std(0).clamp_min(1e-4)
    train = F.normalize((train - mean) / std, dim=1)
    heldout = F.normalize((heldout - mean) / std, dim=1)
    return (heldout @ train.T).cpu().numpy().astype(np.float32)


def neighbor_analysis(
    similarity: np.ndarray,
    train_gradient: np.ndarray,
    target: np.ndarray,
    hard: np.ndarray,
    count: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    order = np.argpartition(-similarity, count - 1, axis=1)[:, :count]
    local_similarity = np.take_along_axis(similarity, order, axis=1)
    sorted_order = np.argsort(-local_similarity, axis=1)
    order = np.take_along_axis(order, sorted_order, axis=1)
    local_similarity = np.take_along_axis(similarity, order, axis=1)
    normalized_train = normalize_rows(train_gradient)
    normalized_target = normalize_rows(target)
    neighbor = normalized_train[order]
    individual = np.sum(neighbor * normalized_target[:, None], axis=2)
    by_count = {}
    for k in (1, 5, count):
        mean_direction = np.mean(neighbor[:, :k], axis=1)
        cosine = cosine_rows(mean_direction, normalized_target)
        coherence = np.linalg.norm(mean_direction, axis=1)
        pairwise_mean = (k * np.square(coherence) - 1.0) / max(k - 1, 1)
        if k == 1:
            pairwise_mean = np.ones_like(coherence)
        by_count[str(k)] = {
            "mean_direction_cosine": distribution(cosine),
            "hard_mean_direction_cosine": distribution(cosine[hard]),
            "positive_fraction": float(np.mean(cosine > 0.0)),
            "hard_positive_fraction": float(np.mean(cosine[hard] > 0.0)),
            "neighbor_direction_coherence": distribution(coherence),
            "hard_neighbor_direction_coherence": distribution(coherence[hard]),
            "neighbor_pairwise_cosine_mean": distribution(pairwise_mean),
            "hard_neighbor_pairwise_cosine_mean": distribution(pairwise_mean[hard]),
            "best_neighbor_cosine": distribution(np.max(individual[:, :k], axis=1)),
            "hard_best_neighbor_cosine": distribution(
                np.max(individual[hard, :k], axis=1)
            ),
        }
    density = local_similarity[:, 0]
    boundaries = np.quantile(density, (0.25, 0.50, 0.75))
    density_bin = np.digitize(density, boundaries, right=True)
    quartiles = {}
    for index in range(4):
        mask = density_bin == index
        quartiles[f"q{index + 1}"] = {
            "count": int(np.sum(mask)),
            "similarity": distribution(density[mask]),
            "hard_count": int(np.sum(hard & mask)),
            "hard_fraction": float(np.mean(hard[mask])),
        }
    return ({
        "top1_similarity": distribution(density),
        "top1_similarity_hard_vs_nonhard": binary_summary(density, hard),
        "top5_similarity_mean_hard_vs_nonhard": binary_summary(
            np.mean(local_similarity[:, :5], axis=1), hard
        ),
        "density_quartiles": quartiles,
        "neighbors": by_count,
    }, order, local_similarity)


def component_analysis(
    prediction: np.ndarray, target: np.ndarray, hard: np.ndarray,
) -> dict[str, Any]:
    raw_prediction = np.asarray(prediction, np.float64)
    raw_target = np.asarray(target, np.float64)
    prediction = normalize_rows(raw_prediction)
    target = normalize_rows(raw_target)
    alignment = prediction * target
    sign_equal = np.sign(raw_prediction) == np.sign(raw_target)
    prediction_norm = np.linalg.norm(raw_prediction, axis=1)
    target_norm = np.linalg.norm(raw_target, axis=1)
    norm_ratio = prediction_norm / (target_norm + 1e-12)
    cosine = cosine_rows(raw_prediction, raw_target)
    parallel_scale = np.sum(raw_prediction * raw_target, axis=1) / (
        np.square(target_norm) + 1e-12
    )
    parallel = parallel_scale[:, None] * raw_target
    perpendicular_ratio = np.linalg.norm(raw_prediction - parallel, axis=1) / (
        target_norm + 1e-12
    )
    rows = []
    for knot in range(8):
        for action in range(2):
            index = 2 * knot + action
            rows.append({
                "flat_index": index,
                "knot": knot,
                "action_channel": "acceleration" if action == 0 else "steering",
                "alignment_mean_all": float(np.mean(alignment[:, index])),
                "alignment_mean_hard": float(np.mean(alignment[hard, index])),
                "alignment_median_hard": float(np.median(alignment[hard, index])),
                "sign_accuracy_all": float(np.mean(sign_equal[:, index])),
                "sign_accuracy_hard": float(np.mean(sign_equal[hard, index])),
                "target_abs_direction_mean_hard": float(
                    np.mean(np.abs(target[hard, index]))
                ),
            })
    channel_rows = {}
    for name, indices in (
        ("acceleration", np.arange(0, 16, 2)),
        ("steering", np.arange(1, 16, 2)),
    ):
        local = np.sum(alignment[:, indices], axis=1)
        negative_mass = np.sum(np.maximum(-alignment[hard][:, indices], 0.0))
        total_negative_mass = np.sum(np.maximum(-alignment[hard], 0.0))
        channel_rows[name] = {
            "alignment_hard": distribution(local[hard]),
            "negative_alignment_mass_fraction": float(
                negative_mass / max(total_negative_mass, 1e-12)
            ),
        }
    top_component = {}
    magnitude_order = np.argsort(-np.abs(raw_target), axis=1)
    for count in (1, 3, 5):
        index = magnitude_order[:, :count]
        local_sign = np.take_along_axis(sign_equal, index, axis=1)
        fraction = np.mean(local_sign, axis=1)
        top_component[str(count)] = {
            "sign_accuracy_fraction": distribution(fraction),
            "hard_sign_accuracy_fraction": distribution(fraction[hard]),
            "hard_all_sign_correct_fraction": float(np.mean(np.all(local_sign[hard], axis=1))),
            "hard_all_sign_flipped_fraction": float(np.mean(~np.any(local_sign[hard], axis=1))),
        }
    return {
        "norm_and_direction": {
            "prediction_norm_hard_vs_nonhard": binary_summary(prediction_norm, hard),
            "target_norm_hard_vs_nonhard": binary_summary(target_norm, hard),
            "prediction_to_target_norm_ratio_hard_vs_nonhard": binary_summary(
                norm_ratio, hard
            ),
            "cosine_hard_vs_nonhard": binary_summary(cosine, hard),
        },
        "parallel_perpendicular": {
            "parallel_scale_hard_vs_nonhard": binary_summary(parallel_scale, hard),
            "perpendicular_to_target_norm_ratio_hard_vs_nonhard": binary_summary(
                perpendicular_ratio, hard
            ),
            "negative_parallel_scale_count": int(np.sum(parallel_scale < 0.0)),
            "hard_negative_parallel_scale_fraction": float(
                np.mean(parallel_scale[hard] < 0.0)
            ),
        },
        "top_true_magnitude_components": top_component,
        "per_component": rows,
        "by_action_channel": channel_rows,
        "hard_negative_alignment_component_count": distribution(
            np.sum(alignment[hard] < 0.0, axis=1)
        ),
    }


def physical_state_ordinals(
    episode: np.ndarray,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physical = np.full(len(episode), -1, np.int64)
    repeat = np.full(len(episode), -1, np.int64)
    context = np.full(len(episode), -1, np.int64)
    for value in np.unique(episode):
        local = np.flatnonzero(episode == value)
        group_index = -1
        previous = None
        repeat_index = 0
        for context_index, row in enumerate(local):
            state = initial_state[row]
            if previous is None or not np.allclose(
                state, previous, rtol=0.0, atol=1e-6
            ):
                group_index += 1
                repeat_index = 0
                previous = state
            else:
                repeat_index += 1
            physical[row] = group_index
            repeat[row] = repeat_index
            context[row] = context_index
    if np.any(physical < 0) or np.any(repeat < 0) or np.any(context < 0):
        raise AssertionError("failed to assign episode ordinals")
    return physical, repeat, context


def label_self_check(
    stored: np.ndarray,
    fresh: np.ndarray,
    hard: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    cosine = cosine_rows(stored, fresh).astype(np.float32)
    norm_ratio = (
        np.linalg.norm(stored, axis=1) / (np.linalg.norm(fresh, axis=1) + 1e-12)
    ).astype(np.float32)
    mismatch = cosine < ATTRIBUTION_THRESHOLDS["label_mismatch_cosine_lt"]
    warning = cosine < ATTRIBUTION_THRESHOLDS["label_warning_cosine_lt"]
    summary = {
        "stored_label_radius_sigma": 0.05,
        "cosine": {
            "all": distribution(cosine),
            "hard": distribution(cosine[hard]),
            "nonhard": distribution(cosine[~hard]),
        },
        "norm_ratio": {
            "all": distribution(norm_ratio),
            "hard": distribution(norm_ratio[hard]),
            "nonhard": distribution(norm_ratio[~hard]),
        },
        "threshold_overlap": {
            "cosine_lt_0.95": {
                "count": int(np.sum(warning)),
                "hard_count": int(np.sum(warning & hard)),
                "hard_coverage_fraction": float(np.mean(warning[hard])),
            },
            "cosine_lt_0.90": {
                "count": int(np.sum(mismatch)),
                "hard_count": int(np.sum(mismatch & hard)),
                "hard_coverage_fraction": float(np.mean(mismatch[hard])),
            },
            "cosine_lt_0": {
                "count": int(np.sum(cosine < 0.0)),
                "hard_count": int(np.sum((cosine < 0.0) & hard)),
                "hard_coverage_fraction": float(np.mean(cosine[hard] < 0.0)),
            },
        },
    }
    return summary, cosine, norm_ratio


def transition_analysis(
    episode: np.ndarray,
    initial_state: np.ndarray,
    target: np.ndarray,
    actor_action: np.ndarray,
    hard: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    normalized_target = normalize_rows(target)
    repeat_cosine = np.full(len(episode), np.nan, np.float32)
    temporal_cosine = np.full(len(episode), np.nan, np.float32)
    temporal_action_rms = np.full(len(episode), np.nan, np.float32)
    repeat_pairs = 0
    state_groups = 0
    for value in np.unique(episode):
        local = np.flatnonzero(episode == value)
        groups: list[np.ndarray] = []
        start = 0
        while start < len(local):
            same = [local[start]]
            end = start + 1
            while end < len(local) and np.allclose(
                initial_state[local[end]], initial_state[local[start]],
                rtol=0.0, atol=1e-6,
            ):
                same.append(local[end])
                end += 1
            groups.append(np.asarray(same, np.int64))
            start = end
        state_groups += len(groups)
        group_direction = []
        group_action = []
        for group in groups:
            direction = normalize_rows(np.mean(normalized_target[group], axis=0)[None])[0]
            group_direction.append(direction)
            group_action.append(np.mean(actor_action[group], axis=0))
            if len(group) == 2:
                value_cosine = float(np.sum(
                    normalized_target[group[0]] * normalized_target[group[1]]
                ))
                repeat_cosine[group] = value_cosine
                repeat_pairs += 1
        for index, group in enumerate(groups):
            adjacent_cosine, adjacent_action = [], []
            for other in (index - 1, index + 1):
                if 0 <= other < len(groups):
                    adjacent_cosine.append(float(np.sum(
                        group_direction[index] * group_direction[other]
                    )))
                    adjacent_action.append(float(np.sqrt(np.mean(np.square(
                        group_action[index] - group_action[other]
                    )))))
            if adjacent_cosine:
                temporal_cosine[group] = min(adjacent_cosine)
                temporal_action_rms[group] = max(adjacent_action)
    valid_repeat = np.isfinite(repeat_cosine)
    valid_temporal = np.isfinite(temporal_cosine)
    high_transition = valid_temporal & (temporal_cosine < 0.5)
    low_transition = valid_temporal & ~high_transition
    return ({
        "scenario_label_is_episode_static": True,
        "physical_state_group_count": state_groups,
        "two_repeat_pair_count": repeat_pairs,
        "within_state_repeat_gradient_cosine": distribution(
            repeat_cosine[valid_repeat][::2]
        ),
        "within_state_repeat_hard_vs_nonhard": binary_summary(
            repeat_cosine[valid_repeat], hard[valid_repeat]
        ),
        "adjacent_physical_state_gradient_cosine_hard_vs_nonhard": binary_summary(
            temporal_cosine[valid_temporal], hard[valid_temporal]
        ),
        "adjacent_physical_state_actor_action_rms_hard_vs_nonhard": binary_summary(
            temporal_action_rms[valid_temporal], hard[valid_temporal]
        ),
        "high_gradient_transition": {
            "definition": "minimum adjacent physical-state gradient cosine < 0.5",
            "count": int(np.sum(high_transition)),
            "hard_fraction": float(np.mean(hard[high_transition])),
        },
        "low_gradient_transition": {
            "count": int(np.sum(low_transition)),
            "hard_fraction": float(np.mean(hard[low_transition])),
        },
    }, repeat_cosine, temporal_cosine)


def cross_table(
    hard: np.ndarray,
    clipped: np.ndarray,
    density: np.ndarray,
    speed: np.ndarray,
    scenario: np.ndarray,
) -> dict[str, Any]:
    boundaries = np.quantile(density, (0.25, 0.50, 0.75))
    density_bin = np.digitize(density, boundaries, right=True)
    clipping_density = {}
    for clip_value in (False, True):
        for quartile in range(4):
            mask = (clipped == clip_value) & (density_bin == quartile)
            clipping_density[f"{'clipped' if clip_value else 'unclipped'}_q{quartile+1}"] = {
                "count": int(np.sum(mask)),
                "hard_count": int(np.sum(hard & mask)),
                "hard_fraction": float(np.mean(hard[mask])) if np.any(mask) else None,
            }
    speed_scenario = {}
    for one_speed in sorted(np.unique(speed)):
        for one_scenario in sorted(np.unique(scenario)):
            mask = np.isclose(speed, one_speed) & (scenario == one_scenario)
            if np.any(mask):
                speed_scenario[f"{float(one_speed):.1f}/{one_scenario}"] = {
                    "count": int(np.sum(mask)),
                    "hard_count": int(np.sum(hard & mask)),
                    "hard_fraction": float(np.mean(hard[mask])),
                    "clipped_count": int(np.sum(clipped & mask)),
                }
    return {
        "clipping_density": clipping_density,
        "speed_scenario": speed_scenario,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    critic_summary_path = args.critic_dir / "summary.json"
    fresh_summary_path = args.fresh_dir / "summary.json"
    fresh_validation_path = args.fresh_dir / "validation_summary.json"
    previous_path = args.previous_dir / "analysis.json"
    critic_summary = json.loads(critic_summary_path.read_text())
    fresh_validation = json.loads(fresh_validation_path.read_text())
    previous = json.loads(previous_path.read_text())
    if fresh_validation["qualification"] != "PASS":
        raise AssertionError("fresh-FD source is not independently validated")
    if previous["qualification"] != "STATE_TO_GRADIENT_REPRESENTATION_AMBIGUITY_PILOT":
        raise AssertionError("unexpected previous negative-tail artifact")

    initial_actor_path = Path(critic_summary["initial_actor"])
    initial_payload = torch.load(initial_actor_path, map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial_payload["labels"]), old_payload)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]), args.batch_size, device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    actor = TorchMPPIDeterministicCenterActor(
        float(initial_payload["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()

    label_path = args.critic_dir / "local_forward_labels.npz"
    fresh_npz_path = args.fresh_dir / "fresh_fd_audit.npz"
    with np.load(label_path, allow_pickle=False) as archive:
        label = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.allclose(label["probe_radii_sigma"], (0.05, 0.10, 0.20)):
        raise AssertionError("stored local-label radii changed")
    train_episode = set(critic_summary["split"]["train_episodes"])
    train_mask = np.asarray([value in train_episode for value in label["episode"]])
    train_context = label["context_index"][train_mask]
    train_gradient = label["gradient_by_radius"][train_mask, 0]
    train_action = label["actor_action"][train_mask]

    with np.load(fresh_npz_path, allow_pickle=False) as archive:
        heldout_context = np.asarray(archive["context_index"], np.int64)
        heldout_episode = np.asarray(archive["episode"])
        speed = np.asarray(archive["reference_speed"], np.float32)
        scenario = np.asarray(archive["scenario"])
        clipped = np.asarray(archive["clipped_context"], bool)
        target = np.asarray(archive["gradient"], np.float32)
        critic_gradient = np.asarray(archive["critic_gradient"], np.float32)
    if set(heldout_episode) != set(critic_summary["split"]["heldout_episodes"]):
        raise AssertionError("heldout episode mapping changed")
    ensemble = np.mean(critic_gradient, axis=0)
    ensemble_cosine = cosine_rows(ensemble, target)
    hard = ensemble_cosine < 0.0
    if int(np.sum(hard)) != previous["critic_sign_agreement"]["ensemble_negative_count"]:
        raise AssertionError("hard-state count changed")

    label_lookup = {
        int(value): position for position, value in enumerate(label["context_index"])
    }
    train_label_position = np.asarray(
        [label_lookup[int(x)] for x in train_context], np.int64
    )
    heldout_label_position = np.asarray(
        [label_lookup[int(x)] for x in heldout_context], np.int64
    )
    stored_smallest_gradient = label["gradient_by_radius"][heldout_label_position, 0]
    label_check, label_fresh_cosine, label_fresh_norm_ratio = label_self_check(
        stored_smallest_gradient, target, hard
    )
    label_check["critic_cosine_vs_label_fresh_cosine_pearson"] = float(
        np.corrcoef(ensemble_cosine, label_fresh_cosine)[0, 1]
    )
    all_context = np.concatenate((train_context, heldout_context))
    all_anchor = np.concatenate((
        train_action, label["actor_action"][heldout_label_position]
    ))
    actor_feature = actor_embeddings(
        actor, inputs, all_context, args.batch_size, device
    )
    actor_feature = torch.cat((actor_feature, torch.from_numpy(all_anchor.reshape(len(all_anchor), -1))), dim=1)
    actor_similarity = standardized_similarity(
        actor_feature[:len(train_context)], actor_feature[len(train_context):], device
    )

    critic_similarities = []
    critic_replay_errors = []
    for checkpoint_path in critic_summary["checkpoints"]:
        payload = torch.load(checkpoint_path, map_location="cpu")
        model = TorchMPPIActorCenteredLocalCritic(dropout=0.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        feature, replay_gradient = critic_features_and_gradients(
            model, inputs, all_context, all_anchor, args.batch_size, device
        )
        critic_replay_errors.append(float(np.max(np.abs(
            replay_gradient[len(train_context):]
            - critic_gradient[len(critic_replay_errors)]
        ))))
        critic_similarities.append(standardized_similarity(
            feature[:len(train_context)], feature[len(train_context):], device
        ))
    critic_similarity = np.mean(critic_similarities, axis=0)

    raw_similarity = np.zeros_like(actor_similarity)
    raw_block_similarity = {}
    raw_values = list(inputs) + [torch.from_numpy(label["actor_action"]).to(device)]
    for name, value in zip(BLOCK_NAMES, raw_values):
        if name == "actor_action":
            train_value = value[torch.from_numpy(train_label_position).to(device)]
            heldout_value = value[torch.from_numpy(heldout_label_position).to(device)]
        else:
            train_value = value[torch.from_numpy(train_context).to(device)]
            heldout_value = value[torch.from_numpy(heldout_context).to(device)]
        local = standardized_similarity(train_value, heldout_value, device)
        raw_block_similarity[name] = local
        raw_similarity += local / len(BLOCK_NAMES)

    spaces = {}
    top_order = {}
    top_similarity = {}
    neighbor_rows = {}
    density_percentiles = {}
    for name, similarity in (
        ("raw_equal_block", raw_similarity),
        ("actor_encoder", actor_similarity),
        ("critic_ensemble_representation", critic_similarity),
    ):
        spaces[name], top_order[name], top_similarity[name] = neighbor_analysis(
            similarity, train_gradient, target, hard, args.neighbor_count
        )
        neighbor_rows[name] = neighbor_row_metrics(
            top_order[name], train_gradient, target
        )
        density_percentiles[name] = percentile_rank(top_similarity[name][:, 0])
    spaces["critic_individual_top1_density"] = {
        f"critic_{index}": binary_summary(np.max(value, axis=1), hard)
        for index, value in enumerate(critic_similarities)
    }

    pair_audit = {}
    normalized_train = normalize_rows(train_gradient)
    normalized_target = normalize_rows(target)
    actor_neighbor = top_order["actor_encoder"]
    actor_neighbor_cosine = np.sum(
        normalized_train[actor_neighbor] * normalized_target[:, None], axis=2
    )
    good_index = actor_neighbor[np.arange(len(actor_neighbor)), np.argmax(actor_neighbor_cosine, axis=1)]
    bad_index = actor_neighbor[np.arange(len(actor_neighbor)), np.argmin(actor_neighbor_cosine, axis=1)]
    for block in BLOCK_NAMES:
        similarity = raw_block_similarity[block]
        good = similarity[np.arange(len(similarity)), good_index]
        bad = similarity[np.arange(len(similarity)), bad_index]
        pair_audit[block] = {
            "hard_good_similarity": distribution(good[hard]),
            "hard_bad_similarity": distribution(bad[hard]),
            "hard_good_minus_bad_similarity": distribution((good - bad)[hard]),
            "hard_good_closer_fraction": float(np.mean(good[hard] > bad[hard])),
        }

    components = component_analysis(ensemble, target, hard)
    heldout_initial_state = data.initial_state_six[heldout_context]
    transitions, repeat_cosine, temporal_cosine = transition_analysis(
        heldout_episode, heldout_initial_state, target,
        label["actor_action"][heldout_label_position], hard,
    )
    physical_ordinal, repeat_index, context_ordinal = physical_state_ordinals(
        heldout_episode, heldout_initial_state
    )
    tables = {
        name: cross_table(hard, clipped, similarity[:, 0], speed, scenario)
        for name, similarity in top_similarity.items()
    }

    prediction_norm = np.linalg.norm(ensemble, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    prediction_norm_ratio = prediction_norm / (target_norm + 1e-12)
    parallel_scale = np.sum(ensemble * target, axis=1) / (
        np.square(target_norm) + 1e-12
    )
    parallel = parallel_scale[:, None] * target
    perpendicular_ratio = np.linalg.norm(ensemble - parallel, axis=1) / (
        target_norm + 1e-12
    )

    route_flags: dict[str, np.ndarray] = {
        "label_mismatch": label_fresh_cosine
        < ATTRIBUTION_THRESHOLDS["label_mismatch_cosine_lt"],
        "label_warning": (
            (label_fresh_cosine
             < ATTRIBUTION_THRESHOLDS["label_warning_cosine_lt"])
            & (label_fresh_cosine
               >= ATTRIBUTION_THRESHOLDS["label_mismatch_cosine_lt"])
        ),
        "boundary_clipped": clipped,
        "cold_start_physical_snapshot": physical_ordinal == 0,
        "within_state_repeat_transition": np.isfinite(repeat_cosine)
        & (repeat_cosine
           < ATTRIBUTION_THRESHOLDS["within_state_repeat_cosine_lt"]),
        "adjacent_physical_state_transition": np.isfinite(temporal_cosine)
        & (temporal_cosine
           < ATTRIBUTION_THRESHOLDS["adjacent_physical_state_cosine_lt"]),
    }
    for space in ("raw_equal_block", "actor_encoder", "critic_ensemble_representation"):
        route_flags[f"low_density_{space}"] = (
            density_percentiles[space]
            <= ATTRIBUTION_THRESHOLDS["low_density_percentile_lte"]
        )
        route_flags[f"mixed_neighbors_{space}"] = (
            neighbor_rows[space]["coherence"]
            < ATTRIBUTION_THRESHOLDS["mixed_neighbor_coherence_lt"]
        )
    route_flags["wrong_coherent_critic_branch"] = (
        (neighbor_rows["critic_ensemble_representation"]["mean_label_cosine"]
         < ATTRIBUTION_THRESHOLDS["wrong_coherent_mean_label_cosine_lt"])
        & (neighbor_rows["critic_ensemble_representation"]["coherence"]
           > ATTRIBUTION_THRESHOLDS[
               "wrong_coherent_neighbor_coherence_gt"
           ])
    )

    context_rows = []
    for row in range(len(hard)):
        row_neighbors = {}
        for space in ("raw_equal_block", "actor_encoder", "critic_ensemble_representation"):
            neighbor = top_order[space][row]
            cosine = np.sum(
                normalized_train[neighbor] * normalized_target[row, None], axis=1
            )
            selections = {
                "top1": 0,
                "best_label": int(np.argmax(cosine)),
                "worst_label": int(np.argmin(cosine)),
            }
            row_neighbors[space] = {}
            for role, local_position in selections.items():
                train_position = int(neighbor[local_position])
                context = int(train_context[train_position])
                row_neighbors[space][role] = {
                    "train_context_index": context,
                    "train_episode": str(data.episodes[context]),
                    "representation_similarity": float(
                        top_similarity[space][row, local_position]
                    ),
                    "label_cosine_to_fresh_target": float(cosine[local_position]),
                }
        flags = {name: bool(value[row]) for name, value in route_flags.items()}
        context_rows.append({
            "heldout_row": int(row),
            "context_index": int(heldout_context[row]),
            "episode": str(heldout_episode[row]),
            "physical_snapshot_ordinal": int(physical_ordinal[row]),
            "repeat_index": int(repeat_index[row]),
            "context_ordinal": int(context_ordinal[row]),
            "reference_speed_mps": float(speed[row]),
            "scenario": str(scenario[row]),
            "clipped": bool(clipped[row]),
            "hard": bool(hard[row]),
            "ensemble_cosine": float(ensemble_cosine[row]),
            "ensemble_norm": float(prediction_norm[row]),
            "target_norm": float(target_norm[row]),
            "ensemble_to_target_norm_ratio": float(prediction_norm_ratio[row]),
            "parallel_scale": float(parallel_scale[row]),
            "perpendicular_to_target_norm_ratio": float(perpendicular_ratio[row]),
            "stored_0p05_label_vs_fresh_cosine": float(label_fresh_cosine[row]),
            "stored_0p05_label_vs_fresh_norm_ratio": float(
                label_fresh_norm_ratio[row]
            ),
            "per_critic_cosine": [
                float(cosine_rows(value[row:row+1], target[row:row+1])[0])
                for value in critic_gradient
            ],
            "within_state_repeat_gradient_cosine": (
                float(repeat_cosine[row]) if np.isfinite(repeat_cosine[row]) else None
            ),
            "adjacent_physical_state_gradient_cosine": (
                float(temporal_cosine[row]) if np.isfinite(temporal_cosine[row]) else None
            ),
            "knn_diagnostics": {
                space: {
                    "top1_similarity": float(top_similarity[space][row, 0]),
                    "top1_density_percentile": float(
                        density_percentiles[space][row]
                    ),
                    "top20_mean_label_cosine": float(
                        neighbor_rows[space]["mean_label_cosine"][row]
                    ),
                    "top20_label_coherence": float(
                        neighbor_rows[space]["coherence"][row]
                    ),
                    "top20_pairwise_label_cosine_mean": float(
                        neighbor_rows[space]["pairwise_label_cosine_mean"][row]
                    ),
                    "top20_best_label_cosine": float(
                        neighbor_rows[space]["best_label_cosine"][row]
                    ),
                }
                for space in (
                    "raw_equal_block", "actor_encoder",
                    "critic_ensemble_representation",
                )
            },
            "route_flags": flags,
            "route_tags": sorted(name for name, value in flags.items() if value),
            "neighbors": row_neighbors,
        })
    hard_rows = [row for row in context_rows if row["hard"]]

    route_summary = {}
    for name, value in route_flags.items():
        hard_rate_with = float(np.mean(hard[value])) if np.any(value) else None
        hard_rate_without = float(np.mean(hard[~value])) if np.any(~value) else None
        route_summary[name] = {
            "all_count": int(np.sum(value)),
            "all_fraction": float(np.mean(value)),
            "hard_count": int(np.sum(value & hard)),
            "hard_coverage_fraction": float(np.mean(value[hard])),
            "hard_rate_with_flag": hard_rate_with,
            "hard_rate_without_flag": hard_rate_without,
            "hard_rate_lift": (
                float(hard_rate_with / hard_rate_without)
                if hard_rate_with is not None
                and hard_rate_without is not None
                and hard_rate_without > 0.0 else None
            ),
        }

    context_manifest_path = args.output_dir / "context_attribution_manifest.json"
    context_manifest_path.write_text(json.dumps({
        "format_version": 2,
        "role": "consumed-split multi-cause diagnostic manifest; not an oracle policy",
        "hard_definition": "B4 ensemble cosine versus fresh FD < 0",
        "thresholds_pre_registered": ATTRIBUTION_THRESHOLDS,
        "routing_semantics": "flags are non-exclusive; no single-cause adjudication",
        "context_count": len(context_rows),
        "hard_count": len(hard_rows),
        "rows": context_rows,
    }, indent=2) + "\n")
    hard_manifest_path = args.output_dir / "hard_state_manifest.json"
    hard_manifest_path.write_text(json.dumps({
        "format_version": 2,
        "role": "consumed-split diagnostic manifest; not an oracle policy",
        "hard_definition": "B4 ensemble cosine versus fresh FD < 0",
        "thresholds_pre_registered": ATTRIBUTION_THRESHOLDS,
        "routing_semantics": "flags are non-exclusive; no single-cause adjudication",
        "hard_count": len(hard_rows),
        "rows": hard_rows,
    }, indent=2) + "\n")

    npz_path = args.output_dir / "hard_state_analysis.npz"
    np.savez_compressed(
        npz_path,
        heldout_context=heldout_context,
        heldout_episode=heldout_episode,
        hard=hard,
        clipped=clipped,
        speed=speed,
        scenario=scenario,
        target=target,
        stored_smallest_gradient=stored_smallest_gradient,
        label_fresh_cosine=label_fresh_cosine,
        label_fresh_norm_ratio=label_fresh_norm_ratio,
        critic_gradient=critic_gradient,
        ensemble=ensemble,
        ensemble_cosine=ensemble_cosine,
        train_context=train_context,
        train_gradient=train_gradient,
        repeat_cosine=repeat_cosine,
        temporal_cosine=temporal_cosine,
        physical_snapshot_ordinal=physical_ordinal,
        repeat_index=repeat_index,
        context_ordinal=context_ordinal,
        prediction_norm_ratio=prediction_norm_ratio,
        parallel_scale=parallel_scale,
        perpendicular_ratio=perpendicular_ratio,
        **{
            f"{space}_top_index": top_order[space]
            for space in ("raw_equal_block", "actor_encoder", "critic_ensemble_representation")
        },
        **{
            f"{space}_top_similarity": top_similarity[space]
            for space in ("raw_equal_block", "actor_encoder", "critic_ensemble_representation")
        },
        **{
            f"{space}_density_percentile": density_percentiles[space]
            for space in ("raw_equal_block", "actor_encoder", "critic_ensemble_representation")
        },
        **{
            f"{space}_{metric}": value
            for space, metrics in neighbor_rows.items()
            for metric, value in metrics.items()
        },
        **{f"route_{name}": value for name, value in route_flags.items()},
    )

    summary = {
        "format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HARD_STATE_ZERO_ROLLOUT_ATTRIBUTION_COMPLETE",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "sources": {
            "critic_summary": str(critic_summary_path.resolve()),
            "critic_summary_sha256": sha256_file(critic_summary_path),
            "fresh_summary": str(fresh_summary_path.resolve()),
            "fresh_summary_sha256": sha256_file(fresh_summary_path),
            "fresh_validation": str(fresh_validation_path.resolve()),
            "fresh_validation_sha256": sha256_file(fresh_validation_path),
            "previous_analysis": str(previous_path.resolve()),
            "previous_analysis_sha256": sha256_file(previous_path),
            "local_forward_labels": str(label_path.resolve()),
            "local_forward_labels_sha256": sha256_file(label_path),
            "fresh_fd_audit": str(fresh_npz_path.resolve()),
            "fresh_fd_audit_sha256": sha256_file(fresh_npz_path),
        },
        "counts": {
            "train_context": int(len(train_context)),
            "heldout_context": int(len(heldout_context)),
            "hard_context": int(np.sum(hard)),
            "hard_fraction": float(np.mean(hard)),
        },
        "critic_gradient_replay_max_abs_error": critic_replay_errors,
        "label_self_check": label_check,
        "component_decomposition": components,
        "representation_spaces": spaces,
        "raw_input_good_vs_bad_actor_neighbor": pair_audit,
        "transition_analysis": transitions,
        "routing": {
            "thresholds_pre_registered": ATTRIBUTION_THRESHOLDS,
            "semantics": "non-exclusive multi-cause flags",
            "flag_summary": route_summary,
        },
        "cross_tables": tables,
        "artifacts": {
            "context_attribution_manifest": str(context_manifest_path.resolve()),
            "context_attribution_manifest_sha256": sha256_file(
                context_manifest_path
            ),
            "hard_state_manifest": str(hard_manifest_path.resolve()),
            "hard_state_manifest_sha256": sha256_file(hard_manifest_path),
            "analysis_npz": str(npz_path.resolve()),
            "analysis_npz_sha256": sha256_file(npz_path),
        },
        "interpretation_limits": [
            "All KNN and best/worst neighbor choices are consumed-split diagnostics, not deployable selectors.",
            "Low coherence in a learned representation does not prove raw inputs are intrinsically ambiguous.",
            "Scenario labels are episode-static; regime analysis therefore uses same-state repeats and adjacent physical snapshots.",
            "Episode context rows contain two repeats per physical snapshot; physical_snapshot_ordinal and repeat_index must not be conflated.",
            "Pre-registered route flags are non-exclusive diagnostic partitions, not causal proof.",
            "Tiny-set memorization passed previously, but does not establish episode-heldout generalization.",
        ],
        "contract": {
            "new_dbm_rollouts": 0,
            "actor_updated": False,
            "critic_updated": False,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "explicit_gradient_head_already_present": True,
        },
    }
    analysis_path = args.output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Local-Critic hard-state attribution\n\n"
        "Qualification: `HARD_STATE_ZERO_ROLLOUT_ATTRIBUTION_COMPLETE`.\n\n"
        "This is a consumed-split, zero-new-rollout diagnostic. Actor and Critics "
        "remain frozen; read analysis.json and hard_state_manifest.json before "
        "routing targeted data or representation work.\n"
    )
    print(json.dumps({
        "qualification": summary["qualification"],
        "counts": summary["counts"],
        "critic_gradient_replay_max_abs_error": critic_replay_errors,
        "label_self_check": label_check,
        "route_summary": route_summary,
        "representation_spaces": {
            name: {
                "top1": value["top1_similarity_hard_vs_nonhard"],
                "top20_hard": value["neighbors"][str(args.neighbor_count)]["hard_mean_direction_cosine"],
                "top20_hard_coherence": value["neighbors"][str(args.neighbor_count)]["hard_neighbor_direction_coherence"],
            }
            for name, value in spaces.items() if "neighbors" in value
        },
        "transition_analysis": transitions,
    }, indent=2))


if __name__ == "__main__":
    main()
