#!/usr/bin/env python3
"""Train H0/D/D+R1/D+R2 structured local-Q Critics with a frozen Actor."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import (
    TorchMPPIDeterministicCenterActor,
    TorchMPPISemanticStructuredLocalQCritic,
    TorchMPPIStructuredLocalQCritic,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_local_gradient_critic import (
    correlation,
    cosine_rows,
    norm_metrics,
    repeat_partner_positions,
)
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_INITIAL = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_LABELS = Path(
    "outputs/mppi_proposal/direct_local_gradient_critic_b4_smallest_target_20260813_v2/"
    "local_forward_labels.npz"
)
DEFAULT_FRESH = Path(
    "outputs/mppi_proposal/direct_critic_fresh_fd_b4_smallest_target_20260813_v2/"
    "fresh_fd_audit.npz"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/structured_local_q_critic_20260814_v1"
)
ARMS = {
    "H0": (False, 0),
    "D": (True, 0),
    "D_R1": (True, 1),
    "D_R2": (True, 2),
}
SEMANTIC_ARMS = {
    "PA": {
        "hessian_enabled": True, "low_rank": 2,
        "include_feedback": False, "same_center_invariance": False,
    },
    "PAF": {
        "hessian_enabled": True, "low_rank": 2,
        "include_feedback": True, "same_center_invariance": False,
    },
    "PAF_INV": {
        "hessian_enabled": True, "low_rank": 2,
        "include_feedback": True, "same_center_invariance": True,
    },
}


def arm_configuration(arm: str) -> dict[str, Any]:
    if arm in ARMS:
        enabled, rank = ARMS[arm]
        return {
            "semantic_clean": False,
            "hessian_enabled": enabled,
            "low_rank": rank,
            "include_feedback": True,
            "same_center_invariance": False,
        }
    if arm in SEMANTIC_ARMS:
        return {"semantic_clean": True, **SEMANTIC_ARMS[arm]}
    raise ValueError(f"unknown arm: {arm}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--labels-npz", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--fresh-npz", type=Path, default=DEFAULT_FRESH)
    parser.add_argument(
        "--targeted-labels-npz", type=Path, default=None,
        help=(
            "Validated same-state multidirectional response labels. Only rows "
            "whose pilot_role is 'target' are used for training; matched easy "
            "controls remain unseen diagnostics."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arms", default="H0,D,D_R1,D_R2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--selection-min-epoch", type=int, default=80)
    parser.add_argument("--scheduler-min-epoch", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hessian-scale", type=float, default=256.0)
    parser.add_argument("--value-weight", type=float, default=0.20)
    parser.add_argument("--gradient-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.50)
    parser.add_argument("--probe-value-weight", type=float, default=0.25)
    parser.add_argument("--chord-weight", type=float, default=5.0)
    parser.add_argument("--norm-weight", type=float, default=0.20)
    parser.add_argument("--targeted-value-weight", type=float, default=0.20)
    parser.add_argument("--targeted-gradient-weight", type=float, default=1.0)
    parser.add_argument("--targeted-cosine-weight", type=float, default=0.50)
    parser.add_argument("--targeted-chord-weight", type=float, default=5.0)
    parser.add_argument("--targeted-norm-weight", type=float, default=0.20)
    parser.add_argument(
        "--same-center-invariance-weight", type=float, default=0.25,
        help=(
            "Weight for PAF_INV same-physical-state/same-absolute-action "
            "Q/g0/H consistency. Ignored by all other arms."
        ),
    )
    parser.add_argument("--chord-d0", type=float, default=0.15)
    parser.add_argument("--chord-w-max", type=float, default=4.0)
    parser.add_argument("--chord-epsilon", type=float, default=0.02)
    parser.add_argument("--small-chord-sigma", type=float, default=0.15)
    parser.add_argument("--medium-chord-sigma", type=float, default=0.30)
    parser.add_argument(
        "--overfit-pairs", type=int, default=0,
        help="Use this many train repeat pairs as both train and validation.",
    )
    parser.add_argument(
        "--overfit-reversal-balanced", action="store_true",
        help="Choose half true-gradient reversal pairs in tiny overfit mode.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def subset_first_axis(
    arrays: dict[str, np.ndarray], positions: np.ndarray
) -> dict[str, np.ndarray]:
    count = len(arrays["context_index"])
    return {
        key: (value[positions] if value.ndim > 0 and len(value) == count else value)
        for key, value in arrays.items()
    }


def location_partner(location_count: int) -> np.ndarray:
    """Pair plus/minus locations within each radius/direction block."""
    if location_count != 77:
        raise AssertionError(f"expected 77 targeted locations, got {location_count}")
    partner = np.arange(location_count, dtype=np.int64)
    for plus_start, minus_start in ((1, 20), (39, 58)):
        partner[plus_start : plus_start + 19] = np.arange(
            minus_start, minus_start + 19
        )
        partner[minus_start : minus_start + 19] = np.arange(
            plus_start, plus_start + 19
        )
    return partner


def prepare_targeted_training(
    targeted: dict[str, np.ndarray], role: str = "target"
) -> dict[str, np.ndarray]:
    rows = np.flatnonzero(targeted["pilot_role"] == role)
    if role == "target" and len(rows) != 79:
        raise AssertionError(f"expected 79 targeted training contexts, got {len(rows)}")
    location_count = targeted["outer_centers"].shape[1]
    partner = location_partner(location_count)
    context = np.repeat(targeted["context_index"][rows], location_count)
    absolute_action = targeted["outer_centers"][rows].reshape(-1, 8, 2)
    local_action = targeted["outer_actions"][rows].reshape(-1, 8, 2)
    value = targeted["transformed_reward"][rows, :, 0].reshape(-1)
    gradient = targeted["local_gradient"][rows].reshape(-1, 16)
    partner_action = targeted["outer_actions"][rows][:, partner].reshape(-1, 8, 2)
    partner_gradient = targeted["local_gradient"][rows][:, partner].reshape(-1, 16)
    return {
        "source_row": np.repeat(rows, location_count),
        "location": np.tile(np.arange(location_count), len(rows)),
        "context_index": context.astype(np.int64),
        "absolute_action": absolute_action.astype(np.float32),
        "local_action": local_action.astype(np.float32),
        "value": value.astype(np.float32),
        "gradient": gradient.astype(np.float32),
        "partner_action": partner_action.astype(np.float32),
        "partner_gradient": partner_gradient.astype(np.float32),
    }


def parameters_at_absolute_actions(
    model: TorchMPPIStructuredLocalQCritic,
    inputs: tuple[torch.Tensor, ...],
    context: torch.Tensor,
    absolute_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model.local_parameters(
        inputs[0][context],
        inputs[1][context],
        inputs[2][context],
        absolute_action,
        inputs[4][context],
        inputs[5][context],
    )


@torch.no_grad()
def targeted_metrics(
    model: TorchMPPIStructuredLocalQCritic,
    inputs: tuple[torch.Tensor, ...],
    targeted: dict[str, np.ndarray],
    rows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    location_count = targeted["outer_centers"].shape[1]
    partner = location_partner(location_count)
    context = np.repeat(targeted["context_index"][rows], location_count)
    absolute_action = targeted["outer_centers"][rows].reshape(-1, 8, 2)
    local_action = targeted["outer_actions"][rows].reshape(-1, 8, 2)
    target = targeted["local_gradient"][rows].reshape(-1, 16)
    partner_target = targeted["local_gradient"][rows][:, partner].reshape(-1, 16)
    partner_action = targeted["outer_actions"][rows][:, partner].reshape(-1, 8, 2)
    predicted, hessians = [], []
    model.eval()
    for start in range(0, len(context), batch_size):
        stop = min(start + batch_size, len(context))
        local_context = torch.from_numpy(context[start:stop]).to(device)
        local_absolute = torch.from_numpy(absolute_action[start:stop]).to(device)
        _, gradient, hessian = parameters_at_absolute_actions(
            model, inputs, local_context, local_absolute
        )
        predicted.append(gradient.cpu().numpy())
        hessians.append(hessian.cpu().numpy())
    predicted = np.concatenate(predicted).astype(np.float32)
    hessians = np.concatenate(hessians).astype(np.float32)
    delta = (partner_action - local_action).reshape(len(context), 16)
    predicted_partner = predicted + np.einsum("bij,bj->bi", hessians, delta)
    true_response = cosine_rows(target, partner_target)
    predicted_response = cosine_rows(predicted, predicted_partner)
    reversal = true_response < 0.0
    ratio = np.linalg.norm(predicted, axis=1) / (
        np.linalg.norm(target, axis=1) + 1e-12
    )
    direct_cosine = cosine_rows(predicted, target)
    return {
        "context_count": int(len(rows)),
        "location_count": int(len(context)),
        "gradient_cosine_median": float(np.median(direct_cosine)),
        "gradient_cosine_p10": float(np.quantile(direct_cosine, 0.10)),
        "gradient_norm_ratio_median": float(np.median(ratio)),
        "gradient_norm_ratio_p90": float(np.quantile(ratio, 0.90)),
        "reversal_count": int(np.sum(reversal)),
        "reversal_flip_recall": (
            float(np.mean(predicted_response[reversal] < 0.0))
            if np.any(reversal) else 1.0
        ),
        "hessian_symmetry_max_abs_error": float(np.max(np.abs(
            hessians - np.swapaxes(hessians, 1, 2)
        ))),
    }


def partner_positions(data: Any, labels: dict[str, np.ndarray]) -> np.ndarray:
    return repeat_partner_positions(data, labels)


def cross_actions(
    data: Any,
    labels: dict[str, np.ndarray],
    partner: np.ndarray,
    alpha_center: np.ndarray,
    maximum_residual_sigma: float,
) -> np.ndarray:
    context = labels["context_index"]
    paired_center = labels["actor_center"][partner]
    return (
        (paired_center - alpha_center[context])
        / (maximum_residual_sigma * data.sigma[context, None, :])
    ).astype(np.float32)


def chord_distance_sigma(
    data: Any, labels: dict[str, np.ndarray], partner: np.ndarray
) -> np.ndarray:
    context = labels["context_index"]
    delta = (
        labels["actor_center"][partner] - labels["actor_center"]
    ) / data.sigma[context, None, :]
    return np.sqrt(np.mean(np.square(delta), axis=(1, 2))).astype(np.float32)


def chord_weights(distance: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    raw = np.minimum(
        args.chord_w_max,
        args.chord_d0 / np.maximum(distance, args.chord_epsilon),
    )
    return (raw / np.mean(raw)).astype(np.float32)


def batch_parameters(
    model: TorchMPPIStructuredLocalQCritic,
    inputs: tuple[torch.Tensor, ...],
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model.local_parameters(*(value[context] for value in inputs))


def same_absolute_parameters(
    model: TorchMPPIStructuredLocalQCritic,
    inputs: tuple[torch.Tensor, ...],
    source_context: torch.Tensor,
    nuisance_context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate another first-pass context at the source absolute action."""
    return model.local_parameters(
        inputs[0][nuisance_context],
        inputs[1][nuisance_context],
        inputs[2][nuisance_context],
        inputs[3][source_context],
        inputs[4][nuisance_context],
        inputs[5][nuisance_context],
    )


def initialize_semantic_encoder_from_actor(
    model: TorchMPPISemanticStructuredLocalQCritic,
    actor: TorchMPPIDeterministicCenterActor,
) -> None:
    """Reuse compatible frozen-Actor feature blocks without semantic aliasing."""
    model.encoder.history_encoder.load_state_dict(
        actor.encoder.history_encoder.state_dict(), strict=True
    )
    model.encoder.reference_encoder.load_state_dict(
        actor.encoder.reference_encoder.state_dict(), strict=True
    )
    model.encoder.current_encoder.load_state_dict(
        actor.encoder.current_encoder.state_dict(), strict=True
    )
    model.encoder.absolute_action_encoder.load_state_dict(
        actor.encoder.anchor_encoder.state_dict(), strict=True
    )
    if model.encoder.feedback_encoder is not None:
        model.encoder.feedback_encoder.load_state_dict(
            actor.encoder.feedback_encoder.state_dict(), strict=True
        )


@torch.no_grad()
def predict_parameters(
    model: TorchMPPIStructuredLocalQCritic,
    inputs: tuple[torch.Tensor, ...],
    context: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, gradients, hessians = [], [], []
    model.eval()
    for start in range(0, len(context), batch_size):
        local = torch.from_numpy(context[start : start + batch_size]).to(device)
        q0, g0, h = batch_parameters(model, inputs, local)
        values.append(q0.cpu().numpy())
        gradients.append(g0.cpu().numpy())
        hessians.append(h.cpu().numpy())
    return (
        np.concatenate(values).astype(np.float32),
        np.concatenate(gradients).astype(np.float32),
        np.concatenate(hessians).astype(np.float32),
    )


def distribution(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, np.float64)
    return {
        "minimum": float(np.min(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "maximum": float(np.max(value)),
        "mean": float(np.mean(value)),
    }


def structured_metrics(
    model: TorchMPPIStructuredLocalQCritic,
    inputs: tuple[torch.Tensor, ...],
    labels: dict[str, np.ndarray],
    positions: np.ndarray,
    partner: np.ndarray,
    cross_action: np.ndarray,
    distance: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    context = labels["context_index"][positions]
    value, gradient, hessian = predict_parameters(
        model, inputs, context, args.evaluation_batch_size, device
    )
    target = labels["gradient"][positions]
    cosine = cosine_rows(gradient, target)
    own_action = labels["actor_action"][positions]
    delta = (cross_action[positions] - own_action).reshape(len(positions), -1)
    cross_gradient = gradient + np.einsum("bij,bj->bi", hessian, delta)
    partner_target = labels["gradient"][partner[positions]]
    cross_target_cosine = cosine_rows(cross_gradient, partner_target)
    true_response = cosine_rows(target, partner_target)
    predicted_response = cosine_rows(gradient, cross_gradient)
    reversal = true_response < 0.0

    local_for_global = {int(pos): local for local, pos in enumerate(positions)}
    nuisance = []
    for local, pos in enumerate(positions):
        paired = int(partner[pos])
        if paired in local_for_global:
            nuisance.append(cosine_rows(
                gradient[local : local + 1],
                cross_gradient[local_for_global[paired] : local_for_global[paired] + 1],
            )[0])
    nuisance_array = np.asarray(nuisance, np.float32)

    probe_action = labels["actions"][positions, 0]
    probe_target = labels["transformed_reward"][positions, 0]
    probe_delta = probe_action - own_action[:, None]
    flat_delta = probe_delta.reshape(len(positions), len(probe_delta[0]), -1)
    probe_prediction = (
        value[:, None]
        + np.einsum("bi,bki->bk", gradient, flat_delta)
        + 0.5 * np.einsum("bki,bij,bkj->bk", flat_delta, hessian, flat_delta)
    )
    pair_true = probe_target[:, 1:17] - probe_target[:, 17:33]
    pair_pred = probe_prediction[:, 1:17] - probe_prediction[:, 17:33]
    meaningful = np.abs(pair_true) >= 0.005

    norm = norm_metrics(gradient, target)
    norm_ratio = np.linalg.norm(gradient, axis=1) / (
        np.linalg.norm(target, axis=1) + 1e-12
    )
    absolute_log = np.abs(np.log(norm_ratio + 1e-12))
    result: dict[str, Any] = {
        "count": int(len(positions)),
        "gradient_cosine_median": float(np.median(cosine)),
        "gradient_cosine_p10": float(np.quantile(cosine, 0.10)),
        "gradient_positive_fraction": float(np.mean(cosine > 0.0)),
        **norm,
        "gradient_norm_ratio_below_0_5_fraction": float(np.mean(norm_ratio < 0.5)),
        "gradient_norm_ratio_above_2_fraction": float(np.mean(norm_ratio > 2.0)),
        "gradient_abs_log_norm_ratio_p90": float(np.quantile(absolute_log, 0.90)),
        "probe_value_rmse": float(np.sqrt(np.mean(np.square(
            probe_prediction - probe_target
        )))),
        "probe_pair_sign_accuracy": float(np.mean(
            np.sign(pair_pred[meaningful]) == np.sign(pair_true[meaningful])
        )),
        "cross_target_cosine_median": float(np.median(cross_target_cosine)),
        "cross_target_cosine_p10": float(np.quantile(cross_target_cosine, 0.10)),
        "true_reversal_count": int(np.sum(reversal) // 2),
        "true_reversal_flip_recall": (
            float(np.mean(predicted_response[reversal] < 0.0))
            if np.any(reversal) else 1.0
        ),
        "predicted_response_cosine": distribution(predicted_response),
        "same_center_nuisance_cosine": (
            distribution(nuisance_array) if len(nuisance_array) else None
        ),
        "hessian_symmetry_max_abs_error": float(np.max(np.abs(
            hessian - np.swapaxes(hessian, 1, 2)
        ))),
        "hessian_eigenvalue": distribution(np.linalg.eigvalsh(hessian)),
    }
    bins = {
        "small_le_0_15": distance[positions] <= args.small_chord_sigma,
        "medium_0_15_0_30": (
            (distance[positions] > args.small_chord_sigma)
            & (distance[positions] <= args.medium_chord_sigma)
        ),
        "long_gt_0_30": distance[positions] > args.medium_chord_sigma,
    }
    result["chord_bins"] = {}
    for name, mask in bins.items():
        local_reversal = reversal & mask
        result["chord_bins"][name] = {
            "context_count": int(np.sum(mask)),
            "pair_count": int(np.sum(mask) // 2),
            "reversal_pair_count": int(np.sum(local_reversal) // 2),
            "response_cosine_median": (
                float(np.median(predicted_response[mask])) if np.any(mask) else None
            ),
            "reversal_flip_recall": (
                float(np.mean(predicted_response[local_reversal] < 0.0))
                if np.any(local_reversal) else None
            ),
        }
    return result


def fresh_metrics(
    model: TorchMPPIStructuredLocalQCritic,
    inputs: tuple[torch.Tensor, ...],
    fresh: dict[str, np.ndarray],
    maximum_residual_sigma: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    context = fresh["context_index"].astype(np.int64)
    _, gradient, hessian = predict_parameters(
        model, inputs, context, args.evaluation_batch_size, device
    )
    target = fresh["gradient"].astype(np.float32)
    cosine = cosine_rows(gradient, target)
    episode = fresh["episode"]
    partner = np.empty(len(context), np.int64)
    for value in np.unique(episode):
        local = np.flatnonzero(episode == value)
        for start in range(0, len(local), 2):
            left, right = local[start : start + 2]
            partner[left], partner[right] = right, left
    center = fresh["actor_center"].astype(np.float32)
    sigma = fresh["sigma"].astype(np.float32)
    delta = (
        (center[partner] - center)
        / (maximum_residual_sigma * sigma[:, None, :])
    ).reshape(len(context), -1)
    cross_gradient = gradient + np.einsum("bij,bj->bi", hessian, delta)
    true_response = cosine_rows(target, target[partner])
    predicted_response = cosine_rows(gradient, cross_gradient)
    reversal = true_response < 0.0
    nuisance = cosine_rows(gradient, cross_gradient[partner])
    distance = np.sqrt(np.mean(np.square(
        (center[partner] - center) / sigma[:, None, :]
    ), axis=(1, 2)))
    norm = norm_metrics(gradient, target)
    ratio = np.linalg.norm(gradient, axis=1) / (
        np.linalg.norm(target, axis=1) + 1e-12
    )
    result: dict[str, Any] = {
        "count": int(len(context)),
        "gradient_cosine_median": float(np.median(cosine)),
        "gradient_cosine_p10": float(np.quantile(cosine, 0.10)),
        "gradient_positive_fraction": float(np.mean(cosine > 0.0)),
        **norm,
        "gradient_norm_ratio_below_0_5_fraction": float(np.mean(ratio < 0.5)),
        "gradient_norm_ratio_above_2_fraction": float(np.mean(ratio > 2.0)),
        "gradient_abs_log_norm_ratio_p90": float(np.quantile(
            np.abs(np.log(ratio + 1e-12)), 0.90
        )),
        "true_reversal_pair_count": int(np.sum(reversal) // 2),
        "true_reversal_flip_recall": float(np.mean(
            predicted_response[reversal] < 0.0
        )),
        "same_center_nuisance_cosine": distribution(nuisance),
        "response_cosine": distribution(predicted_response),
    }
    result["chord_bins"] = {}
    for name, mask in {
        "small_le_0_15": distance <= args.small_chord_sigma,
        "medium_0_15_0_30": (
            (distance > args.small_chord_sigma)
            & (distance <= args.medium_chord_sigma)
        ),
        "long_gt_0_30": distance > args.medium_chord_sigma,
    }.items():
        local_reversal = reversal & mask
        result["chord_bins"][name] = {
            "pair_count": int(np.sum(mask) // 2),
            "reversal_pair_count": int(np.sum(local_reversal) // 2),
            "reversal_flip_recall": (
                float(np.mean(predicted_response[local_reversal] < 0.0))
                if np.any(local_reversal) else None
            ),
        }
    return result


def train_one(
    arm: str,
    seed: int,
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[torch.Tensor, ...],
    labels: dict[str, np.ndarray],
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
    partner: np.ndarray,
    cross_action: np.ndarray,
    distance: np.ndarray,
    weight: np.ndarray,
    scales: dict[str, np.ndarray | float],
    targeted_train: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TorchMPPIStructuredLocalQCritic, dict[str, Any]]:
    set_seed(seed)
    config = arm_configuration(arm)
    if config["semantic_clean"]:
        model = TorchMPPISemanticStructuredLocalQCritic(
            include_feedback=config["include_feedback"],
            low_rank=config["low_rank"],
            dropout=args.dropout,
            hessian_scale=args.hessian_scale,
            hessian_enabled=config["hessian_enabled"],
        ).to(device)
        initialize_semantic_encoder_from_actor(model, actor)
    else:
        model = TorchMPPIStructuredLocalQCritic(
            low_rank=config["low_rank"],
            dropout=args.dropout,
            hessian_scale=args.hessian_scale,
            hessian_enabled=config["hessian_enabled"],
        ).to(device)
        model.encoder.load_state_dict(actor.encoder.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.35, patience=12, min_lr=3e-7
    )
    rng = np.random.default_rng(260814 + seed)
    own_action = torch.from_numpy(labels["actor_action"]).to(device)
    target_value = torch.from_numpy(labels["value"]).to(device)
    target_gradient = torch.from_numpy(labels["gradient"]).to(device)
    probe_action = torch.from_numpy(labels["actions"][:, 0]).to(device)
    probe_target = torch.from_numpy(labels["transformed_reward"][:, 0]).to(device)
    cross_action_t = torch.from_numpy(cross_action).to(device)
    partner_t = torch.from_numpy(partner).to(device)
    chord_weight_t = torch.from_numpy(weight).to(device)
    gradient_scale = torch.from_numpy(
        np.asarray(scales["gradient"], np.float32)
    ).to(device)
    chord_scale = torch.from_numpy(
        np.asarray(scales["chord"], np.float32)
    ).to(device)
    value_scale = float(scales["value"])
    probe_scale = float(scales["probe"])
    if targeted_train is not None:
        targeted_context = torch.from_numpy(
            targeted_train["context_index"]
        ).to(device)
        targeted_absolute_action = torch.from_numpy(
            targeted_train["absolute_action"]
        ).to(device)
        targeted_local_action = torch.from_numpy(
            targeted_train["local_action"]
        ).to(device)
        targeted_value = torch.from_numpy(targeted_train["value"]).to(device)
        targeted_gradient = torch.from_numpy(
            targeted_train["gradient"]
        ).to(device)
        targeted_partner_action = torch.from_numpy(
            targeted_train["partner_action"]
        ).to(device)
        targeted_partner_gradient = torch.from_numpy(
            targeted_train["partner_gradient"]
        ).to(device)
    best_score, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_positions)
        losses = {key: [] for key in (
            "total", "value", "gradient", "cosine", "probe", "chord", "norm",
            "same_center_invariance", "targeted_total", "targeted_value",
            "targeted_gradient", "targeted_cosine", "targeted_chord",
            "targeted_norm",
        )}
        for start in range(0, len(order), args.batch_size):
            pos_np = order[start : start + args.batch_size]
            pos = torch.from_numpy(pos_np).to(device)
            context = torch.from_numpy(labels["context_index"][pos_np]).to(device)
            q0, g0, hessian = batch_parameters(model, inputs, context)
            tv, tg = target_value[pos], target_gradient[pos]
            value_loss = F.smooth_l1_loss(
                (q0 - tv) / value_scale, torch.zeros_like(q0), beta=0.5
            )
            gradient_loss = F.smooth_l1_loss(
                (g0 - tg) / gradient_scale, torch.zeros_like(g0), beta=0.5
            )
            cosine_loss = (1.0 - F.cosine_similarity(g0, tg, dim=1)).mean()
            delta_probe = probe_action[pos] - own_action[pos, None]
            predicted_probe = model.local_value(
                q0[:, None], g0[:, None], hessian[:, None], delta_probe
            )
            probe_loss = F.smooth_l1_loss(
                (predicted_probe - probe_target[pos]) / probe_scale,
                torch.zeros_like(predicted_probe), beta=0.5,
            )
            delta_chord = cross_action_t[pos] - own_action[pos]
            predicted_delta_gradient = torch.einsum(
                "bij,bj->bi", hessian, delta_chord.flatten(1)
            )
            target_delta_gradient = target_gradient[partner_t[pos]] - tg
            per_component = F.smooth_l1_loss(
                (predicted_delta_gradient - target_delta_gradient) / chord_scale,
                torch.zeros_like(predicted_delta_gradient), beta=0.5,
                reduction="none",
            ).mean(1)
            chord_loss = torch.sum(
                chord_weight_t[pos] * per_component
            ) / torch.sum(chord_weight_t[pos])
            log_norm = torch.log(torch.linalg.vector_norm(g0, dim=1) + 1e-4)
            target_log_norm = torch.log(torch.linalg.vector_norm(tg, dim=1) + 1e-4)
            norm_loss = F.smooth_l1_loss(
                log_norm, target_log_norm, beta=0.5
            )
            if config["same_center_invariance"]:
                nuisance_context = torch.from_numpy(
                    labels["context_index"][partner[pos_np]]
                ).to(device)
                nuisance_q0, nuisance_g0, nuisance_hessian = same_absolute_parameters(
                    model, inputs, context, nuisance_context
                )
                invariant_value = F.smooth_l1_loss(
                    (q0 - nuisance_q0) / value_scale,
                    torch.zeros_like(q0), beta=0.5,
                )
                invariant_gradient = F.smooth_l1_loss(
                    (g0 - nuisance_g0) / gradient_scale,
                    torch.zeros_like(g0), beta=0.5,
                )
                invariant_hessian = F.smooth_l1_loss(
                    (hessian - nuisance_hessian) / args.hessian_scale,
                    torch.zeros_like(hessian), beta=0.5,
                )
                invariance_loss = (
                    0.10 * invariant_value
                    + invariant_gradient
                    + 0.10 * invariant_hessian
                )
            else:
                invariance_loss = q0.sum() * 0.0
            loss = (
                args.value_weight * value_loss
                + args.gradient_weight * gradient_loss
                + args.cosine_weight * cosine_loss
                + args.probe_value_weight * probe_loss
                + args.chord_weight * chord_loss
                + args.norm_weight * norm_loss
                + args.same_center_invariance_weight * invariance_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, local in (
                ("total", loss), ("value", value_loss),
                ("gradient", gradient_loss), ("cosine", cosine_loss),
                ("probe", probe_loss), ("chord", chord_loss),
                ("norm", norm_loss),
                ("same_center_invariance", invariance_loss),
            ):
                losses[name].append(float(local.detach()))
        if targeted_train is not None:
            targeted_order = rng.permutation(len(targeted_train["context_index"]))
            for start in range(0, len(targeted_order), args.batch_size):
                local_np = targeted_order[start : start + args.batch_size]
                local = torch.from_numpy(local_np).to(device)
                tq0, tg0, thessian = parameters_at_absolute_actions(
                    model,
                    inputs,
                    targeted_context[local],
                    targeted_absolute_action[local],
                )
                target_value_local = targeted_value[local]
                target_gradient_local = targeted_gradient[local]
                targeted_value_loss = F.smooth_l1_loss(
                    (tq0 - target_value_local) / value_scale,
                    torch.zeros_like(tq0),
                    beta=0.5,
                )
                targeted_gradient_loss = F.smooth_l1_loss(
                    (tg0 - target_gradient_local) / gradient_scale,
                    torch.zeros_like(tg0),
                    beta=0.5,
                )
                targeted_cosine_loss = (
                    1.0 - F.cosine_similarity(
                        tg0, target_gradient_local, dim=1
                    )
                ).mean()
                targeted_delta = (
                    targeted_partner_action[local]
                    - targeted_local_action[local]
                ).flatten(1)
                targeted_predicted_delta = torch.einsum(
                    "bij,bj->bi", thessian, targeted_delta
                )
                targeted_true_delta = (
                    targeted_partner_gradient[local] - target_gradient_local
                )
                targeted_chord_loss = F.smooth_l1_loss(
                    (targeted_predicted_delta - targeted_true_delta) / chord_scale,
                    torch.zeros_like(targeted_predicted_delta),
                    beta=0.5,
                )
                targeted_norm_loss = F.smooth_l1_loss(
                    torch.log(torch.linalg.vector_norm(tg0, dim=1) + 1e-4),
                    torch.log(
                        torch.linalg.vector_norm(target_gradient_local, dim=1)
                        + 1e-4
                    ),
                    beta=0.5,
                )
                targeted_loss = (
                    args.targeted_value_weight * targeted_value_loss
                    + args.targeted_gradient_weight * targeted_gradient_loss
                    + args.targeted_cosine_weight * targeted_cosine_loss
                    + args.targeted_chord_weight * targeted_chord_loss
                    + args.targeted_norm_weight * targeted_norm_loss
                )
                optimizer.zero_grad(set_to_none=True)
                targeted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                for name, local_loss in (
                    ("targeted_total", targeted_loss),
                    ("targeted_value", targeted_value_loss),
                    ("targeted_gradient", targeted_gradient_loss),
                    ("targeted_cosine", targeted_cosine_loss),
                    ("targeted_chord", targeted_chord_loss),
                    ("targeted_norm", targeted_norm_loss),
                ):
                    losses[name].append(float(local_loss.detach()))
        validation = structured_metrics(
            model, inputs, labels, validation_positions, partner, cross_action,
            distance, args, device,
        )
        score = (
            1.0 - validation["gradient_cosine_median"]
            + 0.5 * (1.0 - validation["gradient_cosine_p10"])
            + 0.20 * validation["gradient_abs_log_norm_ratio_median"]
            + 0.25 * (1.0 - validation["cross_target_cosine_median"])
            + 0.50 * (1.0 - validation["true_reversal_flip_recall"])
            + 0.05 * validation["probe_value_rmse"]
        )
        if epoch >= args.scheduler_min_epoch and not args.overfit_pairs:
            scheduler.step(score)
        eligible = (
            epoch >= args.selection_min_epoch
            and 0.5 <= validation["gradient_norm_ratio_median"] <= 2.0
        )
        if eligible and score < best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        elif eligible:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "checkpoint_eligible": eligible,
                "score": float(score),
                "loss": {
                    key: (float(np.mean(value)) if value else None)
                    for key, value in losses.items()
                },
                "validation": validation,
            }
            history.append(row)
            print(json.dumps({"arm": arm, "seed": seed, **row}), flush=True)
        if eligible and stale >= args.patience:
            break
    if best_state is None:
        # Tiny overfit can legitimately reach its best before the normal epoch gate.
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        best_score = score
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "arm": arm,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "best_validation_score": float(best_score),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    valid_arms = set(ARMS) | set(SEMANTIC_ARMS)
    if any(arm not in valid_arms for arm in arms):
        raise ValueError(f"unknown arms: {arms}")
    semantic_run = all(arm in SEMANTIC_ARMS for arm in arms)
    if not semantic_run and any(arm in SEMANTIC_ARMS for arm in arms):
        raise ValueError("legacy and semantic-clean arms cannot share one run")
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(
        args.chord_d0, args.chord_w_max, args.chord_epsilon,
        args.small_chord_sigma, args.medium_chord_sigma,
    ) <= 0:
        raise ValueError("chord parameters must be positive")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    initial_payload = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, splits = load_dataset(Path(initial_payload["labels"]), old_payload)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.evaluation_batch_size, device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    actor = TorchMPPIDeterministicCenterActor(
        float(initial_payload["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()

    labels = load_npz(args.labels_npz)
    labels["gradient"] = labels["gradient_by_radius"][:, 0].astype(np.float32)
    labels["value"] = labels["transformed_reward"][:, 0, 0].astype(np.float32)
    context = labels["context_index"].astype(np.int64)
    if semantic_run:
        actor_center_by_context = np.zeros(
            (len(data.episodes), 8, 2), dtype=np.float32
        )
        assigned = np.zeros(len(data.episodes), dtype=bool)
        for position, value in enumerate(context):
            if assigned[value] and not np.array_equal(
                actor_center_by_context[value], labels["actor_center"][position]
            ):
                raise AssertionError("one context maps to multiple absolute Actor centers")
            actor_center_by_context[value] = labels["actor_center"][position]
            assigned[value] = True
        if not np.all(assigned):
            missing = np.flatnonzero(~assigned)
            raise AssertionError(f"semantic action map misses contexts: {missing[:5]}")
        semantic_inputs = list(inputs)
        semantic_inputs[3] = torch.from_numpy(actor_center_by_context).to(device)
        inputs = tuple(semantic_inputs)
    partner = partner_positions(data, labels)
    if semantic_run:
        paired_context = context[partner]
        for block_index, name in enumerate(("history", "reference", "current")):
            maximum = float(np.max(np.abs(
                data.inputs[block_index][context]
                - data.inputs[block_index][paired_context]
            )))
            if maximum != 0.0:
                raise AssertionError(
                    f"semantic physical block {name} is not repeat invariant: {maximum}"
                )
    cross_action = cross_actions(
        data, labels, partner, alpha_center,
        float(initial_payload["maximum_residual_sigma"]),
    )
    distance = chord_distance_sigma(data, labels, partner)
    weight = chord_weights(distance, args)

    fit_positions = np.flatnonzero(np.isin(
        data.episodes[context], splits["internal_fit"]
    ))
    heldout_positions = np.flatnonzero(np.isin(
        data.episodes[context], splits["internal_selection"]
    ))
    fit_episodes = np.unique(data.episodes[context[fit_positions]])
    split_rng = np.random.default_rng(260812)
    shuffled = fit_episodes.copy()
    split_rng.shuffle(shuffled)
    validation_episodes = set(shuffled[: max(1, len(shuffled) // 5)])
    validation_positions = fit_positions[np.asarray([
        episode in validation_episodes
        for episode in data.episodes[context[fit_positions]]
    ])]
    train_positions = fit_positions[np.asarray([
        episode not in validation_episodes
        for episode in data.episodes[context[fit_positions]]
    ])]
    if args.overfit_pairs:
        pair_left = train_positions[train_positions < partner[train_positions]]
        if args.overfit_reversal_balanced:
            response = cosine_rows(
                labels["gradient"][pair_left],
                labels["gradient"][partner[pair_left]],
            )
            reversal = pair_left[response < 0.0]
            regular = pair_left[response >= 0.0]
            reversal_count = min(len(reversal), args.overfit_pairs // 2)
            chosen = np.concatenate((
                reversal[:reversal_count],
                regular[: args.overfit_pairs - reversal_count],
            ))
        else:
            chosen = pair_left[: args.overfit_pairs]
        train_positions = np.sort(np.concatenate((chosen, partner[chosen])))
        validation_positions = train_positions.copy()

    scales: dict[str, np.ndarray | float] = {
        "value": max(float(np.std(labels["value"][train_positions])), 0.05),
        "gradient": np.maximum(
            np.std(labels["gradient"][train_positions], axis=0), 0.05
        ).astype(np.float32),
        "probe": max(float(np.std(
            labels["transformed_reward"][train_positions, 0]
        )), 0.05),
        "chord": np.maximum(np.std(
            labels["gradient"][partner[train_positions]]
            - labels["gradient"][train_positions], axis=0
        ), 0.05).astype(np.float32),
    }
    fresh = load_npz(args.fresh_npz)
    if semantic_run:
        fresh_context = fresh["context_index"].astype(np.int64)
        center_error = float(np.max(np.abs(
            actor_center_by_context[fresh_context] - fresh["actor_center"]
        )))
        if center_error > 1e-6:
            raise AssertionError(
                f"fresh/semantic absolute Actor center mismatch: {center_error}"
            )
    targeted = None
    targeted_train = None
    targeted_train_rows = np.empty(0, np.int64)
    targeted_control_rows = np.empty(0, np.int64)
    fresh_targeted_positions = np.empty(0, np.int64)
    fresh_unseen_positions = np.arange(len(fresh["context_index"]), dtype=np.int64)
    targeted_validation = None
    if args.targeted_labels_npz is not None:
        if not semantic_run:
            raise ValueError("targeted response labels require semantic-clean arms")
        targeted_validation_path = (
            args.targeted_labels_npz.parent / "validation_summary.json"
        )
        targeted_validation = json.loads(targeted_validation_path.read_text())
        if targeted_validation["qualification"] != (
            "TARGETED_LOCAL_RESPONSE_LABELS_VALIDATED"
        ):
            raise AssertionError("targeted response labels are not independently validated")
        if sha256_file(args.targeted_labels_npz) != targeted_validation["arrays_sha256"]:
            raise AssertionError("targeted response label hash changed after validation")
        targeted = load_npz(args.targeted_labels_npz)
        targeted_train_rows = np.flatnonzero(targeted["pilot_role"] == "target")
        targeted_control_rows = np.flatnonzero(
            targeted["pilot_role"] == "matched_easy_control"
        )
        if len(targeted_train_rows) != 79 or len(targeted_control_rows) != 21:
            raise AssertionError("target/control targeted-response count mismatch")
        targeted_train = prepare_targeted_training(targeted, role="target")
        targeted_context = targeted["context_index"].astype(np.int64)
        center_error = float(np.max(np.abs(
            actor_center_by_context[targeted_context]
            - targeted["outer_centers"][:, 0]
        )))
        if center_error > 1e-6:
            raise AssertionError(
                f"targeted/semantic absolute Actor center mismatch: {center_error}"
            )
        fresh_context = fresh["context_index"].astype(np.int64)
        consumed_context = set(
            targeted["context_index"][targeted_train_rows].astype(np.int64).tolist()
        )
        consumed_episode = set(
            fresh["episode"][np.asarray(
                [int(value) in consumed_context for value in fresh_context], bool
            )].tolist()
        )
        fresh_targeted_positions = np.asarray(
            [
                position for position, value in enumerate(fresh["episode"])
                if value in consumed_episode
            ],
            np.int64,
        )
        fresh_unseen_positions = np.asarray(
            [
                position for position, value in enumerate(fresh["episode"])
                if value not in consumed_episode
            ],
            np.int64,
        )
        if (
            len(fresh_targeted_positions) + len(fresh_unseen_positions) != 600
            or len(fresh_targeted_positions) % 2
            or len(fresh_unseen_positions) % 2
        ):
            raise AssertionError("fresh complete-episode partition mismatch")
    records, checkpoints = [], []
    per_arm: dict[str, Any] = {}
    for arm in arms:
        arm_records = []
        for seed in seeds:
            arm_config = arm_configuration(arm)
            model, training = train_one(
                arm, seed, actor, inputs, labels, train_positions,
                validation_positions, partner, cross_action, distance, weight,
                scales, targeted_train, args, device,
            )
            path = args.output_dir / f"structured_local_q_{arm}_seed{seed}.pt"
            torch.save({
                "format_version": 1,
                "model_class": (
                    "TorchMPPISemanticStructuredLocalQCritic"
                    if arm_config["semantic_clean"]
                    else "TorchMPPIStructuredLocalQCritic"
                ),
                "model_state_dict": model.state_dict(),
                "arm": arm,
                "hessian_enabled": arm_config["hessian_enabled"],
                "low_rank": arm_config["low_rank"],
                "include_feedback": arm_config["include_feedback"],
                "same_center_invariance": arm_config["same_center_invariance"],
                "hessian_scale": args.hessian_scale,
                "initial_actor": str(args.initial_actor.resolve()),
                "initial_actor_sha256": sha256_file(args.initial_actor),
                "labels": str(args.labels_npz.resolve()),
                "labels_sha256": sha256_file(args.labels_npz),
                "targeted_labels": (
                    str(args.targeted_labels_npz.resolve())
                    if args.targeted_labels_npz is not None else None
                ),
                "targeted_labels_sha256": (
                    sha256_file(args.targeted_labels_npz)
                    if args.targeted_labels_npz is not None else None
                ),
                "chord_weighting": {
                    "distance_coordinate": "RMS((center1-center0)/deployment_sigma)",
                    "d0": args.chord_d0,
                    "w_max": args.chord_w_max,
                    "epsilon": args.chord_epsilon,
                    "normalization": "divide by global mean raw weight",
                    "small_boundary_sigma": args.small_chord_sigma,
                    "medium_boundary_sigma": args.medium_chord_sigma,
                },
                "training": training,
                "contract": {
                    "actor_frozen": True,
                    "generic_action_encoder": False,
                    "semantic_clean": arm_config["semantic_clean"],
                    "absolute_actor_center_explicit": arm_config["semantic_clean"],
                    "gradient_context_consumed": not arm_config["semantic_clean"],
                    "analytic_dbm_gradient": False,
                    "formal_validation_loaded": False,
                    "test_loaded": False,
                    "targeted_response_training": targeted_train is not None,
                    "targeted_response_role": (
                        "target_only" if targeted_train is not None else None
                    ),
                },
            }, path)
            validation = structured_metrics(
                model, inputs, labels, validation_positions, partner,
                cross_action, distance, args, device,
            )
            heldout = structured_metrics(
                model, inputs, labels, heldout_positions, partner,
                cross_action, distance, args, device,
            )
            fresh_all_result = fresh_metrics(
                model, inputs, fresh,
                float(initial_payload["maximum_residual_sigma"]), args, device,
            )
            fresh_result = fresh_all_result
            fresh_targeted_result = None
            fresh_unseen_result = fresh_all_result
            targeted_train_result = None
            targeted_control_result = None
            if targeted is not None:
                fresh_targeted_result = fresh_metrics(
                    model, inputs,
                    subset_first_axis(fresh, fresh_targeted_positions),
                    float(initial_payload["maximum_residual_sigma"]), args, device,
                )
                fresh_unseen_result = fresh_metrics(
                    model, inputs,
                    subset_first_axis(fresh, fresh_unseen_positions),
                    float(initial_payload["maximum_residual_sigma"]), args, device,
                )
                # Only complete episodes never consumed by targeted training can be used
                # as the mechanism gate in this run. It is still not a formal
                # validation set and cannot authorize Actor updates.
                fresh_result = fresh_unseen_result
                targeted_train_result = targeted_metrics(
                    model, inputs, targeted, targeted_train_rows,
                    args.evaluation_batch_size, device,
                )
                targeted_control_result = targeted_metrics(
                    model, inputs, targeted, targeted_control_rows,
                    args.evaluation_batch_size, device,
                )
            small_recall = fresh_result["chord_bins"]["small_le_0_15"][
                "reversal_flip_recall"
            ]
            gates = {
                "fresh_cosine_median_ge_0_70": (
                    fresh_result["gradient_cosine_median"] >= 0.70
                ),
                "fresh_cosine_p10_ge_0": fresh_result["gradient_cosine_p10"] >= 0.0,
                "fresh_norm_median_ge_0_50": (
                    fresh_result["gradient_norm_ratio_median"] >= 0.50
                ),
                "fresh_norm_median_le_2": (
                    fresh_result["gradient_norm_ratio_median"] <= 2.0
                ),
                "small_chord_flip_recall_ge_0_50": (
                    small_recall is not None and small_recall >= 0.50
                ),
                "same_center_p10_ge_0": (
                    fresh_result["same_center_nuisance_cosine"]["p10"] >= 0.0
                ),
                "symmetry_error_le_1e_6": (
                    heldout["hessian_symmetry_max_abs_error"] <= 1e-6
                ),
            }
            record = {
                "arm": arm,
                "seed": seed,
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": sha256_file(path),
                "training": training,
                "internal_validation": validation,
                "heldout_0_05_labels": heldout,
                "fresh_fd": fresh_result,
                "fresh_fd_all_600_mixed_seen": fresh_all_result,
                "fresh_fd_targeted_episode_seen": fresh_targeted_result,
                "fresh_fd_unseen_complete_episode": fresh_unseen_result,
                "targeted_response_train_79_seen": targeted_train_result,
                "targeted_response_control_21_unseen": targeted_control_result,
                "gates": gates,
                "all_gates_passed": all(gates.values()),
            }
            records.append(record)
            arm_records.append(record)
            checkpoints.append(str(path.resolve()))
        pass_count = sum(row["all_gates_passed"] for row in arm_records)
        per_arm[arm] = {
            "seed_count": len(arm_records),
            "complete_gate_pass_count": int(pass_count),
            "two_of_three_complete_gate_pass": (
                pass_count >= 2 if len(arm_records) == 3 else False
            ),
            "records": arm_records,
        }

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            (
                "TARGETED_STRUCTURED_LOCAL_Q_MECHANISM_GATE_PASS_ACTOR_STILL_FROZEN"
                if any(
                    value["two_of_three_complete_gate_pass"]
                    for value in per_arm.values()
                )
                else "TARGETED_STRUCTURED_LOCAL_Q_CRITIC_FAIL_ACTOR_FROZEN"
            )
            if targeted is not None
            else (
                "STRUCTURED_LOCAL_Q_MECHANISM_GATE_PASS_ACTOR_STILL_FROZEN"
                if any(
                    value["two_of_three_complete_gate_pass"]
                    for value in per_arm.values()
                )
                else "STRUCTURED_LOCAL_Q_CRITIC_FAIL_ACTOR_FROZEN"
            )
        ),
        "actor_update_performed": False,
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "checkpoints": checkpoints,
        "training_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "sources": {
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "labels": str(args.labels_npz.resolve()),
            "labels_sha256": sha256_file(args.labels_npz),
            "fresh_fd": str(args.fresh_npz.resolve()),
            "fresh_fd_sha256": sha256_file(args.fresh_npz),
            "targeted_labels": (
                str(args.targeted_labels_npz.resolve())
                if args.targeted_labels_npz is not None else None
            ),
            "targeted_labels_sha256": (
                sha256_file(args.targeted_labels_npz)
                if args.targeted_labels_npz is not None else None
            ),
            "targeted_validation": (
                str((args.targeted_labels_npz.parent / "validation_summary.json").resolve())
                if args.targeted_labels_npz is not None else None
            ),
            "targeted_validation_sha256": (
                sha256_file(args.targeted_labels_npz.parent / "validation_summary.json")
                if args.targeted_labels_npz is not None else None
            ),
        },
        "counts": {
            "train_context": int(len(train_positions)),
            "internal_validation_context": int(len(validation_positions)),
            "heldout_context": int(len(heldout_positions)),
            "targeted_train_state": int(len(targeted_train_rows)),
            "targeted_unseen_control_state": int(len(targeted_control_rows)),
            "targeted_train_location": (
                int(len(targeted_train["context_index"]))
                if targeted_train is not None else 0
            ),
            "fresh_fd_seen_state": int(len(fresh_targeted_positions)),
            "fresh_fd_unseen_state": int(len(fresh_unseen_positions)),
        },
        "chord_weighting": {
            "distance_coordinate": "RMS((center1-center0)/deployment_sigma)",
            "d0": args.chord_d0,
            "w_max": args.chord_w_max,
            "epsilon": args.chord_epsilon,
            "normalization": "divide by global mean raw weight",
            "weight_distribution": distribution(weight),
            "distance_distribution": distribution(distance),
            "small_boundary_sigma": args.small_chord_sigma,
            "medium_boundary_sigma": args.medium_chord_sigma,
        },
        "scales": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in scales.items()
        },
        "arms": per_arm,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "new_dbm_rollouts": 0,
            "ensemble_is_qualification_gate": False,
            "seed_gate": "at least 2/3 complete per-seed gates; bootstrap noninferiority pending validator",
            "semantic_clean_run": semantic_run,
            "absolute_actor_center_explicit": semantic_run,
            "same_center_invariance_semantics": (
                "same physical state and same absolute action; first-pass feedback may differ"
                if semantic_run else None
            ),
            "targeted_response_training": targeted is not None,
            "targeted_train_role": "target_only" if targeted is not None else None,
            "matched_easy_control_consumed_for_training": False,
            "fresh_gate_partition": (
                "complete episodes not consumed by targeted-response training; mechanism only"
                if targeted is not None else "all 600 fresh-FD contexts"
            ),
            "targeted_run_formal_qualification_allowed": False,
        },
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Structured Local-Q Critic pilot\n\n"
        f"Qualification: `{summary['qualification']}`. Actor remained frozen.\n"
    )
    print(json.dumps({
        "qualification": summary["qualification"],
        "counts": summary["counts"],
        "arms": {
            arm: {
                "complete_gate_pass_count": value["complete_gate_pass_count"],
                "two_of_three_complete_gate_pass": value["two_of_three_complete_gate_pass"],
            }
            for arm, value in per_arm.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
