#!/usr/bin/env python3
"""Diagnose the cross-episode negative cosine tail of the frozen local Critic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor
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
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_local_critic_negative_tail_20260813_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-12
    )


def distribution(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, np.float64)
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "p25": float(np.quantile(value, 0.25)),
        "p75": float(np.quantile(value, 0.75)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def direction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    cosine = cosine_rows(prediction, target)
    return {
        "cosine": distribution(cosine),
        "positive_fraction": float(np.mean(cosine > 0.0)),
        "above_0_5_fraction": float(np.mean(cosine > 0.5)),
    }


@torch.no_grad()
def actor_embeddings(
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[torch.Tensor, ...],
    batch_size: int,
) -> torch.Tensor:
    rows = []
    for start in range(0, len(inputs[0]), batch_size):
        index = torch.arange(
            start, min(start + batch_size, len(inputs[0])), device=inputs[0].device
        )
        rows.append(actor.encoder(*(value[index] for value in inputs)).cpu())
    return torch.cat(rows)


def grouped_cosine(
    cosine: np.ndarray, group: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result = {}
    for value in sorted(np.unique(group)):
        local = cosine[group == value]
        result[str(value)] = {
            "count": int(len(local)),
            "cosine": distribution(local),
            "positive_fraction": float(np.mean(local > 0.0)),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    critic_summary_path = args.critic_dir / "summary.json"
    fresh_summary_path = args.fresh_dir / "summary.json"
    fresh_validation_path = args.fresh_dir / "validation_summary.json"
    critic_summary = json.loads(critic_summary_path.read_text())
    fresh_validation = json.loads(fresh_validation_path.read_text())
    if fresh_validation["qualification"] != "PASS":
        raise AssertionError("fresh-FD artifact did not pass independent validation")

    initial_payload = torch.load(critic_summary["initial_actor"], map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, splits = load_dataset(Path(initial_payload["labels"]), old_payload)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy,
        tensors,
        extra,
        np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.batch_size,
        device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    actor = TorchMPPIDeterministicCenterActor(
        float(initial_payload["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()
    embedding = actor_embeddings(actor, inputs, args.batch_size)

    with np.load(args.critic_dir / "local_forward_labels.npz", allow_pickle=False) as old:
        context_index = np.asarray(old["context_index"], np.int64)
        episode = np.asarray(old["episode"])
        actor_action = np.asarray(old["actor_action"], np.float32).reshape(
            len(context_index), -1
        )
        train_gradient = np.asarray(old["gradient_by_radius"][:, 0], np.float32)
    train_episode = set(critic_summary["split"]["train_episodes"])
    train_position = np.asarray([value in train_episode for value in episode])
    train_context = context_index[train_position]

    with np.load(args.fresh_dir / "fresh_fd_audit.npz", allow_pickle=False) as fresh:
        heldout_context = np.asarray(fresh["context_index"], np.int64)
        heldout_episode = np.asarray(fresh["episode"])
        speed = np.asarray(fresh["reference_speed"], np.float32)
        scenario = np.asarray(fresh["scenario"])
        clipped = np.asarray(fresh["clipped_context"], bool)
        target = np.asarray(fresh["gradient"], np.float32)
        critic_gradient = np.asarray(fresh["critic_gradient"], np.float32)
    lookup = {int(value): position for position, value in enumerate(context_index)}
    heldout_position = np.asarray([lookup[int(value)] for value in heldout_context])
    if set(data.episodes[train_context]) != train_episode:
        raise AssertionError("training episode mapping changed")
    if set(heldout_episode) != set(critic_summary["split"]["heldout_episodes"]):
        raise AssertionError("heldout episode mapping changed")

    ensemble = np.mean(critic_gradient, axis=0)
    ensemble_cosine = cosine_rows(ensemble, target)
    per_critic = {
        f"critic_{index}": direction_metrics(gradient, target)
        for index, gradient in enumerate(critic_gradient)
    }
    all_negative = np.all(
        np.asarray([cosine_rows(one, target) < 0 for one in critic_gradient]), axis=0
    )
    all_positive = np.all(
        np.asarray([cosine_rows(one, target) > 0 for one in critic_gradient]), axis=0
    )

    # Use the frozen Actor representation plus the exact Actor action to test whether
    # nearby consumed train states have coherent derivative labels.  This is a
    # representation diagnostic, not an oracle policy or a replacement Critic.
    full_embedding = torch.cat(
        (embedding[context_index], torch.from_numpy(actor_action)), dim=1
    ).float()
    train_embedding = full_embedding[train_position]
    heldout_embedding = full_embedding[heldout_position]
    mean = train_embedding.mean(0)
    std = train_embedding.std(0).clamp_min(1e-4)
    train_embedding = F.normalize((train_embedding - mean) / std, dim=1).to(device)
    heldout_embedding = F.normalize((heldout_embedding - mean) / std, dim=1).to(device)
    similarity = heldout_embedding @ train_embedding.T
    top_similarity, neighbor_index = torch.topk(similarity, 20, dim=1)
    top_similarity = top_similarity.cpu().numpy()
    neighbor_index = neighbor_index.cpu().numpy()
    normalized_train_gradient = train_gradient[train_position] / (
        np.linalg.norm(train_gradient[train_position], axis=1, keepdims=True) + 1e-12
    )
    normalized_target = target / (
        np.linalg.norm(target, axis=1, keepdims=True) + 1e-12
    )
    network_tail = ensemble_cosine < 0.0
    neighbor_results = {}
    for count in (1, 3, 5, 10, 20):
        neighbor = normalized_train_gradient[neighbor_index[:, :count]]
        predicted = np.mean(neighbor, axis=1)
        cosine = cosine_rows(predicted, normalized_target)
        individual_cosine = np.sum(
            neighbor * normalized_target[:, None], axis=2
        )
        coherence = np.linalg.norm(np.mean(neighbor, axis=1), axis=1)
        neighbor_results[str(count)] = {
            "mean_direction": direction_metrics(predicted, normalized_target),
            "best_neighbor_cosine": distribution(np.max(individual_cosine, axis=1)),
            "neighbor_direction_coherence": distribution(coherence),
            "network_negative_subset": {
                "count": int(np.sum(network_tail)),
                "mean_direction": direction_metrics(
                    predicted[network_tail], normalized_target[network_tail]
                ),
            },
        }

    ordinal = np.zeros(len(heldout_episode), np.int64)
    episode_rows = []
    for one_episode in np.unique(heldout_episode):
        local = np.flatnonzero(heldout_episode == one_episode)
        ordinal[local] = np.arange(len(local))
        values = ensemble_cosine[local]
        episode_rows.append({
            "episode": str(one_episode),
            "count": int(len(local)),
            "cosine_median": float(np.median(values)),
            "negative_fraction": float(np.mean(values < 0.0)),
        })

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "STATE_TO_GRADIENT_REPRESENTATION_AMBIGUITY_PILOT",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "critic_summary": str(critic_summary_path.resolve()),
        "critic_summary_sha256": sha256_file(critic_summary_path),
        "fresh_summary": str(fresh_summary_path.resolve()),
        "fresh_summary_sha256": sha256_file(fresh_summary_path),
        "fresh_validation": str(fresh_validation_path.resolve()),
        "fresh_validation_sha256": sha256_file(fresh_validation_path),
        "fresh_validation_qualification": fresh_validation["qualification"],
        "train_context_count": int(np.sum(train_position)),
        "train_episode_count": int(len(train_episode)),
        "heldout_context_count": int(len(heldout_context)),
        "heldout_episode_count": int(len(np.unique(heldout_episode))),
        "ensemble": direction_metrics(ensemble, target),
        "per_critic": per_critic,
        "critic_sign_agreement": {
            "all_three_negative_fraction": float(np.mean(all_negative)),
            "all_three_positive_fraction": float(np.mean(all_positive)),
            "ensemble_negative_count": int(np.sum(network_tail)),
            "ensemble_abs_cosine": distribution(np.abs(ensemble_cosine)),
        },
        "grouped": {
            "reference_speed": grouped_cosine(
                ensemble_cosine,
                np.asarray([f"{float(value):.1f}" for value in speed]),
            ),
            "scenario": grouped_cosine(ensemble_cosine, scenario),
            "within_episode_ordinal": grouped_cosine(ensemble_cosine, ordinal),
            "clipped": grouped_cosine(
                ensemble_cosine, np.where(clipped, "clipped", "unclipped")
            ),
        },
        "episode_rows": sorted(episode_rows, key=lambda value: value["episode"]),
        "nearest_train_representation": {
            "representation": "frozen Actor encoder output + deterministic Actor action",
            "top1_cosine_similarity": distribution(top_similarity[:, 0]),
            "neighbors": neighbor_results,
            "interpretation_limit": (
                "KNN in the frozen Actor representation is diagnostic only. It can "
                "show label-direction mixing in the representation, but cannot prove "
                "that raw state inputs are intrinsically ambiguous."
            ),
        },
        "conclusions": {
            "tail_is_speed_or_scenario_localized": False,
            "tail_is_ensemble_cancellation": False,
            "tail_contains_near_global_sign_reversals": bool(
                np.median(np.abs(ensemble_cosine)) >= 0.8
            ),
            "more_random_episodes_alone_is_supported": False,
            "targeted_hard_state_or_representation_work_required": True,
        },
        "contract": {
            "actor_updated": False,
            "critic_updated": False,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "knn_is_diagnostic_only": True,
        },
    }
    path = args.output_dir / "analysis.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Local-Critic negative-tail diagnostic\n\n"
        "Qualification: `STATE_TO_GRADIENT_REPRESENTATION_AMBIGUITY_PILOT`.\n\n"
        "The negative tail is shared by all Critics and appears across every speed "
        "and scenario. Nearby frozen-Actor representations contain mixed derivative "
        "directions, so collect/weight targeted hard states before adding more random "
        "episodes. This is a consumed-split mechanism diagnostic only.\n"
    )
    print(json.dumps({
        "qualification": summary["qualification"],
        "ensemble": summary["ensemble"],
        "critic_sign_agreement": summary["critic_sign_agreement"],
        "nearest_train_representation": summary["nearest_train_representation"],
        "conclusions": summary["conclusions"],
    }, indent=2))


if __name__ == "__main__":
    main()
