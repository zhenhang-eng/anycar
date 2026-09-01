#!/usr/bin/env python3
"""Train a frozen-Actor, explicit full-16D local Critic from DBM forward rewards.

The Actor and its input contract are unchanged.  For every context, deterministic
antithetic Hadamard probes at several radii provide a complete local gradient label
in normalized Actor-action coordinates.  The Critic predicts V, all 16 gradient
components, and one radial curvature.  No analytic DBM gradient is used and this
script never updates the Actor.
"""

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
    TorchMPPIAbsoluteCenterLocalCritic,
    TorchMPPIActorCenteredLocalCritic,
    TorchMPPIDeterministicCenterActor,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import (
    actor_inputs,
    hadamard_directions,
    make_base_policy,
    residual_outputs,
)
from train_mppi_direct_response_slope_ac import centers_to_actions, transformed
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_INITIAL = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_local_gradient_critic_20260812_v1"
)
ACTION_DIM = 16
DIRECTION_COUNT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--labels-npz",
        type=Path,
        default=None,
        help="Reuse a previously collected local_forward_labels.npz instead of DBM rollout.",
    )
    parser.add_argument("--probe-radii-sigma", default="0.05,0.10,0.20")
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--action-conditioning",
        choices=("relative_residual", "absolute_center"),
        default="relative_residual",
        help=(
            "Coordinate passed to the Critic action encoder. The local Taylor "
            "coordinate remains the normalized residual in both modes."
        ),
    )
    parser.add_argument("--value-weight", type=float, default=0.20)
    parser.add_argument("--gradient-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.50)
    parser.add_argument(
        "--cosine-warmup-epochs",
        type=int,
        default=0,
        help="Use zero cosine weight for this many epochs.",
    )
    parser.add_argument(
        "--cosine-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp cosine weight after warmup; zero enables it immediately.",
    )
    parser.add_argument(
        "--magnitude-weight",
        type=float,
        default=0.0,
        help="Weight for per-context log gradient-norm calibration loss.",
    )
    parser.add_argument("--magnitude-epsilon", type=float, default=1e-4)
    parser.add_argument("--curvature-weight", type=float, default=0.05)
    parser.add_argument("--bank-value-weight", type=float, default=0.25)
    parser.add_argument(
        "--pair-delta-weight",
        type=float,
        default=0.0,
        help="Weight for same-state smallest-radius Q(+) - Q(-) regression.",
    )
    parser.add_argument(
        "--pair-ranking-weight",
        type=float,
        default=0.0,
        help="Weight for meaningful smallest-radius antithetic sign ranking.",
    )
    parser.add_argument(
        "--cross-anchor-gradient-weight",
        type=float,
        default=0.0,
        help=(
            "Supervise each repeat context at its paired context's absolute "
            "Actor center using the paired numerical gradient label."
        ),
    )
    parser.add_argument(
        "--cross-anchor-cosine-weight",
        type=float,
        default=0.0,
        help="Direction-loss weight for paired cross-anchor gradient targets.",
    )
    parser.add_argument(
        "--cross-anchor-delta-weight",
        type=float,
        default=0.0,
        help=(
            "Match the predicted and labelled gradient change between the two "
            "absolute centers of one physical snapshot."
        ),
    )
    parser.add_argument(
        "--cross-anchor-invariance-weight",
        type=float,
        default=0.0,
        help=(
            "Force paired first-pass contexts to predict the same gradient at "
            "the same absolute center; removes feedback/context shortcuts."
        ),
    )
    parser.add_argument(
        "--selection-cross-anchor-weight",
        type=float,
        default=0.0,
        help="Checkpoint-score weight for cross-anchor target cosine.",
    )
    parser.add_argument(
        "--selection-action-flip-weight",
        type=float,
        default=0.0,
        help="Checkpoint-score weight for true repeat-gradient flip recall.",
    )
    parser.add_argument(
        "--bank-mode",
        choices=("all", "smallest", "off"),
        default="all",
        help="Which finite-radius probes are allowed to drive bank reconstruction.",
    )
    parser.add_argument(
        "--gradient-target",
        choices=("combined", "smallest"),
        default="combined",
        help="Use the old multi-radius fit or only the smallest-radius derivative label.",
    )
    parser.add_argument(
        "--selection-norm-weight",
        type=float,
        default=0.0,
        help="Checkpoint-score weight for median absolute log predicted/target norm ratio.",
    )
    parser.add_argument(
        "--selection-cosine-p10-weight",
        type=float,
        default=0.0,
        help="Checkpoint-score weight for the negative-tail cosine penalty (1-p10)/2.",
    )
    parser.add_argument(
        "--selection-min-epoch",
        type=int,
        default=1,
        help="Do not select/early-stop a checkpoint before this epoch.",
    )
    parser.add_argument(
        "--selection-min-norm-ratio",
        type=float,
        default=0.0,
        help="Hard checkpoint eligibility floor on median predicted/target norm ratio.",
    )
    parser.add_argument(
        "--selection-max-norm-ratio",
        type=float,
        default=float("inf"),
        help="Hard checkpoint eligibility ceiling on median predicted/target norm ratio.",
    )
    parser.add_argument(
        "--scheduler-min-epoch",
        type=int,
        default=1,
        help="Do not reduce the learning rate before this epoch.",
    )
    parser.add_argument(
        "--gradient-head-weight-decay",
        type=float,
        default=None,
        help="Optional AdamW decay for gradient_head; defaults to global weight decay.",
    )
    parser.add_argument("--meaningful-delta", type=float, default=0.005)
    parser.add_argument("--max-fit-contexts", type=int, default=0)
    parser.add_argument("--max-selection-contexts", type=int, default=0)
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


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    )


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def cosine_weight_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    """Delay scale-invariant direction loss until component loss establishes norm."""
    if epoch <= args.cosine_warmup_epochs:
        return 0.0
    if args.cosine_ramp_epochs <= 0:
        return float(args.cosine_weight)
    progress = min(
        1.0,
        (epoch - args.cosine_warmup_epochs) / float(args.cosine_ramp_epochs),
    )
    return float(args.cosine_weight) * progress


def optimizer_for_model(
    model: TorchMPPIActorCenteredLocalCritic,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    if args.gradient_head_weight_decay is None:
        return torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    gradient_head = list(model.gradient_head.parameters())
    gradient_ids = {id(parameter) for parameter in gradient_head}
    other = [
        parameter for parameter in model.parameters() if id(parameter) not in gradient_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": other, "weight_decay": args.weight_decay},
            {
                "params": gradient_head,
                "weight_decay": args.gradient_head_weight_decay,
            },
        ],
        lr=args.learning_rate,
    )


def configure_gradient_target(
    labels: dict[str, np.ndarray],
    mode: str,
) -> tuple[dict[str, np.ndarray], str]:
    configured = dict(labels)
    configured["combined_gradient"] = np.asarray(labels["gradient"], np.float32)
    if mode == "combined":
        source = "combined multi-radius least-squares fit"
    elif mode == "smallest":
        configured["gradient"] = np.asarray(
            labels["gradient_by_radius"][:, 0], np.float32
        )
        source = "smallest stored probe radius only"
    else:
        raise ValueError(f"unsupported gradient target: {mode}")
    return configured, source


def norm_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    predicted_norm = np.linalg.norm(prediction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    ratio = (predicted_norm + epsilon) / (target_norm + epsilon)
    absolute_log_ratio = np.abs(np.log(ratio))
    return {
        "gradient_predicted_norm_median": float(np.median(predicted_norm)),
        "gradient_target_norm_median": float(np.median(target_norm)),
        "gradient_norm_ratio_of_medians": float(
            np.median(predicted_norm) / max(float(np.median(target_norm)), epsilon)
        ),
        "gradient_norm_ratio_median": float(np.median(ratio)),
        "gradient_norm_ratio_p10": float(np.quantile(ratio, 0.10)),
        "gradient_norm_ratio_p90": float(np.quantile(ratio, 0.90)),
        "gradient_abs_log_norm_ratio_median": float(np.median(absolute_log_ratio)),
        "gradient_abs_log_norm_ratio_p90": float(
            np.quantile(absolute_log_ratio, 0.90)
        ),
        "gradient_norm_correlation": correlation(predicted_norm, target_norm),
    }


def build_probe_centers(
    center: np.ndarray,
    sigma: np.ndarray,
    directions: np.ndarray,
    radius: float,
) -> np.ndarray:
    offset = radius * sigma[:, None, None, :] * directions[None]
    positive = np.clip(center[:, None] + offset, -1.0, 1.0)
    negative = np.clip(center[:, None] - offset, -1.0, 1.0)
    return np.concatenate((center[:, None], positive, negative), axis=1).astype(
        np.float32
    )


def fit_local_parameters(
    actions: np.ndarray,
    transformed_reward: np.ndarray,
    anchor_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit gradient and scalar curvature for each context independently."""
    gradient, curvature, ranks = [], [], []
    for action, reward, anchor in zip(actions, transformed_reward, anchor_action):
        delta = (action - anchor[None]).reshape(len(action), ACTION_DIM)
        design = np.concatenate(
            (delta, 0.5 * np.sum(np.square(delta), axis=1, keepdims=True)), axis=1
        )
        design = design[1:]
        target = reward[1:] - reward[0]
        ranks.append(int(np.linalg.matrix_rank(design)))
        coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
        gradient.append(coefficient[:ACTION_DIM])
        curvature.append(coefficient[ACTION_DIM])
    return (
        np.asarray(gradient, np.float32),
        np.asarray(curvature, np.float32),
        np.asarray(ranks, np.int64),
    )


def collect_labels(
    actor: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    data: Any,
    tensors: dict[str, Any],
    index: np.ndarray,
    alpha_center: np.ndarray,
    base_cost: np.ndarray,
    maximum_residual_sigma: float,
    radii: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    directions = hadamard_directions()
    actor_action, actor_center = residual_outputs(
        actor, inputs, index, args.evaluation_batch_size, device
    )
    radius_actions, radius_costs, radius_transformed = [], [], []
    radius_gradients, radius_curvatures, radius_ranks = [], [], []
    for radius in radii:
        actions_parts, costs_parts = [], []
        for start in range(0, len(index), args.rollout_batch_size):
            local = index[start : start + args.rollout_batch_size]
            row = slice(start, start + len(local))
            bank = build_probe_centers(
                actor_center[row], data.sigma[local], directions, float(radius)
            )
            flat_center = bank.reshape(-1, 8, 2)
            flat_index = np.repeat(local, 1 + 2 * DIRECTION_COUNT)
            cost = direct_cost(
                flat_center,
                data,
                tensors,
                flat_index,
                args.evaluation_batch_size,
                device,
            ).reshape(len(local), 1 + 2 * DIRECTION_COUNT)
            actions_parts.append(
                centers_to_actions(
                    bank,
                    alpha_center[local],
                    data.sigma[local],
                    maximum_residual_sigma,
                )
            )
            costs_parts.append(cost.astype(np.float32))
        action = np.concatenate(actions_parts)
        cost = np.concatenate(costs_parts)
        reward = base_cost[index, None] - cost
        z = transformed(reward, args.reward_scale)
        gradient, curvature, rank = fit_local_parameters(action, z, actor_action)
        radius_actions.append(action)
        radius_costs.append(cost)
        radius_transformed.append(z)
        radius_gradients.append(gradient)
        radius_curvatures.append(curvature)
        radius_ranks.append(rank)

    actions = np.stack(radius_actions, axis=1)
    costs = np.stack(radius_costs, axis=1)
    z = np.stack(radius_transformed, axis=1)
    gradients = np.stack(radius_gradients, axis=1)
    curvatures = np.stack(radius_curvatures, axis=1)
    ranks = np.stack(radius_ranks, axis=1)
    combined_action = actions.reshape(len(index), -1, 8, 2)
    combined_z = z.reshape(len(index), -1)
    # Each radius repeats the anchor.  Repeated zero rows are harmless and make
    # the saved bank easy to independently replay.
    combined_gradient, combined_curvature, combined_rank = fit_local_parameters(
        combined_action, combined_z, actor_action
    )
    return {
        "context_index": index.astype(np.int64),
        "actor_action": actor_action.astype(np.float32),
        "actor_center": actor_center.astype(np.float32),
        "actions": actions.astype(np.float32),
        "cost": costs.astype(np.float32),
        "transformed_reward": z.astype(np.float32),
        "gradient_by_radius": gradients.astype(np.float32),
        "curvature_by_radius": curvatures.astype(np.float32),
        "design_rank_by_radius": ranks,
        "gradient": combined_gradient.astype(np.float32),
        "curvature": combined_curvature.astype(np.float32),
        "combined_design_rank": combined_rank,
        "value": z[:, 0, 0].astype(np.float32),
    }


def label_consistency(labels: dict[str, np.ndarray]) -> dict[str, Any]:
    gradients = labels["gradient_by_radius"]
    pair_cosine = []
    for left in range(gradients.shape[1]):
        for right in range(left + 1, gradients.shape[1]):
            pair_cosine.append(cosine_rows(gradients[:, left], gradients[:, right]))
    cosine = np.concatenate(pair_cosine) if pair_cosine else np.ones(len(gradients))
    return {
        "cross_radius_cosine_median": float(np.median(cosine)),
        "cross_radius_cosine_p10": float(np.quantile(cosine, 0.10)),
        "cross_radius_cosine_positive_fraction": float(np.mean(cosine > 0.0)),
        "minimum_radius_design_rank": int(labels["design_rank_by_radius"].min()),
        "minimum_combined_design_rank": int(labels["combined_design_rank"].min()),
        "gradient_norm_mean": float(np.linalg.norm(labels["gradient"], axis=1).mean()),
        "gradient_norm_median": float(np.median(np.linalg.norm(labels["gradient"], axis=1))),
    }


def batch_parameters(
    model: TorchMPPIActorCenteredLocalCritic,
    inputs: tuple[torch.Tensor, ...],
    anchor_action: torch.Tensor,
    context_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model.local_parameters(
        *(value[context_index] for value in inputs), anchor_action
    )


def repeat_partner_positions(
    data: Any,
    labels: dict[str, np.ndarray],
) -> np.ndarray:
    """Pair the two first-pass contexts at every identical physical snapshot."""
    context = np.asarray(labels["context_index"], np.int64)
    episode = data.episodes[context]
    state = data.initial_state_six[context]
    partner = np.full(len(context), -1, np.int64)
    for value in np.unique(episode):
        local = np.flatnonzero(episode == value)
        start = 0
        while start < len(local):
            group = [int(local[start])]
            end = start + 1
            while end < len(local) and np.allclose(
                state[local[end]], state[local[start]], rtol=0.0, atol=1e-6
            ):
                group.append(int(local[end]))
                end += 1
            if len(group) != 2:
                raise AssertionError(
                    f"expected two repeat contexts for {value}, got {len(group)}"
                )
            left, right = group
            for source, target in ((left, right), (right, left)):
                if not np.array_equal(
                    data.current_action[context[source]],
                    data.current_action[context[target]],
                ):
                    raise AssertionError("repeat current action changed")
                if not np.array_equal(
                    data.direct_reference[context[source]],
                    data.direct_reference[context[target]],
                ):
                    raise AssertionError("repeat reference changed")
                partner[source] = target
            start = end
    if np.any(partner < 0) or not np.array_equal(partner[partner], np.arange(len(partner))):
        raise AssertionError("repeat partner mapping is incomplete or asymmetric")
    return partner


def cross_anchor_actions(
    data: Any,
    labels: dict[str, np.ndarray],
    partner: np.ndarray,
    alpha_center: np.ndarray,
    maximum_residual_sigma: float,
) -> np.ndarray:
    context = np.asarray(labels["context_index"], np.int64)
    paired_center = np.asarray(labels["actor_center"], np.float32)[partner]
    return (
        (paired_center - alpha_center[context])
        / (
            maximum_residual_sigma
            * data.sigma[context, None]
        )
    ).astype(np.float32)


@torch.no_grad()
def metrics(
    models: list[TorchMPPIActorCenteredLocalCritic],
    inputs: tuple[torch.Tensor, ...],
    labels: dict[str, np.ndarray],
    positions: np.ndarray,
    device: torch.device,
    batch_size: int,
    meaningful_delta: float,
    partner_position: np.ndarray | None = None,
    cross_action: np.ndarray | None = None,
) -> dict[str, Any]:
    global_index = labels["context_index"][positions]
    anchor_all = torch.from_numpy(labels["actor_action"]).to(device)
    prediction_value, prediction_gradient, prediction_curvature = [], [], []
    for start in range(0, len(positions), batch_size):
        pos = positions[start : start + batch_size]
        idx = torch.from_numpy(global_index[start : start + len(pos)]).to(device)
        anchor = torch.from_numpy(labels["actor_action"][pos]).to(device)
        outputs = [
            model.local_parameters(*(value[idx] for value in inputs), anchor)
            for model in models
        ]
        prediction_value.append(torch.stack([x[0] for x in outputs]).mean(0).cpu().numpy())
        prediction_gradient.append(torch.stack([x[1] for x in outputs]).mean(0).flatten(1).cpu().numpy())
        prediction_curvature.append(torch.stack([x[2] for x in outputs]).mean(0).cpu().numpy())
    value = np.concatenate(prediction_value)
    gradient = np.concatenate(prediction_gradient)
    curvature = np.concatenate(prediction_curvature)
    target_gradient = labels["gradient"][positions]
    cosine = cosine_rows(gradient, target_gradient)

    closest = int(np.argmin(np.abs(RADII_FOR_METRICS - 0.10)))
    actions = labels["actions"][positions, closest]
    delta = actions - labels["actor_action"][positions, None]
    predicted_bank = (
        value[:, None]
        + np.sum(gradient[:, None] * delta.reshape(len(delta), len(delta[0]), -1), axis=2)
        + 0.5 * curvature[:, None] * np.sum(np.square(delta), axis=(2, 3))
    )
    truth = labels["transformed_reward"][positions, closest]
    true_difference = truth[:, 1:17] - truth[:, 17:33]
    predicted_difference = predicted_bank[:, 1:17] - predicted_bank[:, 17:33]
    meaningful = np.abs(true_difference) >= meaningful_delta
    selected = np.argmax(predicted_bank, axis=1)
    cost = labels["cost"][positions, closest]
    regret = cost[np.arange(len(cost)), selected] - cost.min(axis=1)
    result = {
        "count": int(len(positions)),
        "gradient_cosine_median": float(np.median(cosine)),
        "gradient_cosine_p10": float(np.quantile(cosine, 0.10)),
        "gradient_cosine_positive_fraction": float(np.mean(cosine > 0.0)),
        "gradient_cosine_above_0_5_fraction": float(np.mean(cosine > 0.5)),
        "gradient_correlation": correlation(gradient, target_gradient),
        "gradient_rmse": float(np.sqrt(np.mean(np.square(gradient - target_gradient)))),
        "probe_delta_correlation": correlation(predicted_difference, true_difference),
        "probe_delta_sign_accuracy": float(np.mean(
            np.sign(predicted_difference[meaningful]) == np.sign(true_difference[meaningful])
        )),
        "probe_meaningful_pair_count": int(meaningful.sum()),
        "probe_mean_argmax_regret": float(np.mean(regret)),
        "probe_argmax_regret_p95": float(np.quantile(regret, 0.95)),
        "value_rmse": float(np.sqrt(np.mean(np.square(value - labels["value"][positions])))),
        "curvature_rmse": float(np.sqrt(np.mean(np.square(curvature - labels["curvature"][positions])))),
        **norm_metrics(gradient, target_gradient),
    }
    if partner_position is None or cross_action is None:
        return result

    local_lookup = {int(position): index for index, position in enumerate(positions)}
    valid_positions = np.asarray([
        int(position) for position in positions
        if int(partner_position[position]) in local_lookup
    ], np.int64)
    if len(valid_positions) == 0:
        raise AssertionError("no complete repeat pair in metric split")
    valid_local = np.asarray(
        [local_lookup[int(position)] for position in valid_positions], np.int64
    )
    partner_positions = partner_position[valid_positions]
    partner_local = np.asarray(
        [local_lookup[int(position)] for position in partner_positions], np.int64
    )
    cross_parts = []
    for start in range(0, len(valid_positions), batch_size):
        pos = valid_positions[start : start + batch_size]
        idx = torch.from_numpy(labels["context_index"][pos]).to(device)
        anchor = torch.from_numpy(cross_action[pos]).to(device)
        outputs = [
            model.local_parameters(*(value[idx] for value in inputs), anchor)[1]
            for model in models
        ]
        cross_parts.append(
            torch.stack(outputs).mean(0).flatten(1).cpu().numpy()
        )
    cross_gradient = np.concatenate(cross_parts)
    own_gradient = gradient[valid_local]
    own_target = labels["gradient"][valid_positions]
    paired_target = labels["gradient"][partner_positions]
    cross_target_cosine = cosine_rows(cross_gradient, paired_target)
    true_response = cosine_rows(own_target, paired_target)
    predicted_response = cosine_rows(own_gradient, cross_gradient)
    true_reversal = true_response < 0.0
    # The partner row's cross prediction is the same absolute center evaluated
    # under the other first-pass feedback/context.
    nuisance_cosine = cosine_rows(
        own_gradient,
        cross_gradient[partner_local],
    )
    result["cross_anchor"] = {
        "count": int(len(valid_positions)),
        "pair_count": int(len(valid_positions) // 2),
        "target_cosine_median": float(np.median(cross_target_cosine)),
        "target_cosine_p10": float(np.quantile(cross_target_cosine, 0.10)),
        "target_positive_fraction": float(np.mean(cross_target_cosine > 0.0)),
        "target_norm": norm_metrics(cross_gradient, paired_target),
        "true_response_cosine_median": float(np.median(true_response)),
        "true_reversal_count": int(np.sum(true_reversal) // 2),
        "predicted_response_cosine_median": float(np.median(predicted_response)),
        "predicted_response_cosine_p10": float(np.quantile(predicted_response, 0.10)),
        "true_reversal_flip_recall": (
            float(np.mean(predicted_response[true_reversal] < 0.0))
            if np.any(true_reversal) else 1.0
        ),
        "same_absolute_center_nuisance_cosine_median": float(
            np.median(nuisance_cosine)
        ),
        "same_absolute_center_nuisance_cosine_p10": float(
            np.quantile(nuisance_cosine, 0.10)
        ),
    }
    return result


def train_one(
    seed: int,
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[torch.Tensor, ...],
    labels: dict[str, np.ndarray],
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
    scales: dict[str, np.ndarray | float],
    partner_position: np.ndarray,
    cross_action: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TorchMPPIActorCenteredLocalCritic, dict[str, Any]]:
    set_seed(seed)
    rng = np.random.default_rng(seed + 260812)
    if args.action_conditioning == "absolute_center":
        model = TorchMPPIAbsoluteCenterLocalCritic(
            args.dropout, actor.maximum_delta_sigma
        ).to(device)
    else:
        model = TorchMPPIActorCenteredLocalCritic(args.dropout).to(device)
    model.encoder.load_state_dict(actor.encoder.state_dict(), strict=True)
    optimizer = optimizer_for_model(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.35, patience=12, min_lr=3e-7
    )
    anchor_all = torch.from_numpy(labels["actor_action"]).to(device)
    target_value = torch.from_numpy(labels["value"]).to(device)
    target_gradient = torch.from_numpy(labels["gradient"]).to(device)
    target_curvature = torch.from_numpy(labels["curvature"]).to(device)
    if args.bank_mode == "all":
        bank_actions_np = labels["actions"].reshape(len(labels["actions"]), -1, 8, 2)
        bank_target_np = labels["transformed_reward"].reshape(len(labels["actions"]), -1)
    elif args.bank_mode == "smallest":
        bank_actions_np = labels["actions"][:, 0]
        bank_target_np = labels["transformed_reward"][:, 0]
    elif args.bank_mode == "off":
        bank_actions_np = labels["actions"][:, 0, :1]
        bank_target_np = labels["transformed_reward"][:, 0, :1]
    else:
        raise ValueError(f"unsupported bank mode: {args.bank_mode}")
    action_bank = torch.from_numpy(bank_actions_np).to(device)
    target_bank = torch.from_numpy(bank_target_np).to(device)
    pair_actions = torch.from_numpy(labels["actions"][:, 0]).to(device)
    pair_target = torch.from_numpy(
        labels["transformed_reward"][:, 0, 1:17]
        - labels["transformed_reward"][:, 0, 17:33]
    ).to(device)
    value_scale = float(scales["value"])
    gradient_scale = torch.from_numpy(np.asarray(scales["gradient"], np.float32)).to(device)
    curvature_scale = float(scales["curvature"])
    bank_scale = float(scales["bank"])
    pair_delta_scale = torch.from_numpy(
        np.asarray(scales["pair_delta"], np.float32)
    ).to(device)
    partner_all = torch.from_numpy(partner_position).to(device)
    cross_action_all = torch.from_numpy(cross_action).to(device)
    best_score, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_positions)
        loss_rows: dict[str, list[float]] = {
            key: []
            for key in (
                "total", "value", "gradient", "cosine", "magnitude",
                "curvature", "bank", "pair_delta", "pair_ranking",
                "cross_anchor_gradient", "cross_anchor_cosine",
                "cross_anchor_delta", "cross_anchor_invariance",
            )
        }
        epoch_cosine_weight = cosine_weight_for_epoch(args, epoch)
        for start in range(0, len(order), args.batch_size):
            pos_np = order[start : start + args.batch_size]
            pos = torch.from_numpy(pos_np).to(device)
            idx = torch.from_numpy(labels["context_index"][pos_np]).to(device)
            pv, pg, pc = batch_parameters(model, inputs, anchor_all[pos], idx)
            pg = pg.flatten(1)
            tv, tg, tc = target_value[pos], target_gradient[pos], target_curvature[pos]
            value_loss = F.smooth_l1_loss((pv - tv) / value_scale, torch.zeros_like(pv), beta=0.5)
            gradient_loss = F.smooth_l1_loss(
                (pg - tg) / gradient_scale, torch.zeros_like(pg), beta=0.5
            )
            cosine_loss = (1.0 - F.cosine_similarity(pg, tg, dim=1)).mean()
            prediction_norm = torch.linalg.vector_norm(pg, dim=1)
            target_norm = torch.linalg.vector_norm(tg, dim=1)
            log_norm_difference = torch.log(
                prediction_norm + args.magnitude_epsilon
            ) - torch.log(target_norm + args.magnitude_epsilon)
            magnitude_loss = F.smooth_l1_loss(
                log_norm_difference,
                torch.zeros_like(log_norm_difference),
                beta=0.5,
            )
            curvature_loss = F.smooth_l1_loss(
                (pc - tc) / curvature_scale, torch.zeros_like(pc), beta=0.5
            )
            delta = action_bank[pos] - anchor_all[pos, None]
            predicted_bank = model.local_value(
                pv[:, None], pg.reshape(-1, 1, 8, 2), pc[:, None], delta
            )
            if args.bank_mode == "off" or args.bank_value_weight == 0.0:
                bank_loss = torch.zeros((), dtype=pv.dtype, device=device)
            else:
                bank_loss = F.smooth_l1_loss(
                    (predicted_bank - target_bank[pos]) / bank_scale,
                    torch.zeros_like(predicted_bank), beta=0.5,
                )
            pair_action_delta = pair_actions[pos] - anchor_all[pos, None]
            predicted_pair_bank = model.local_value(
                pv[:, None], pg.reshape(-1, 1, 8, 2), pc[:, None],
                pair_action_delta,
            )
            predicted_pair_delta = (
                predicted_pair_bank[:, 1:17] - predicted_pair_bank[:, 17:33]
            )
            pair_delta_loss = F.smooth_l1_loss(
                (predicted_pair_delta - pair_target[pos]) / pair_delta_scale,
                torch.zeros_like(predicted_pair_delta),
                beta=0.5,
            )
            meaningful_pair = torch.abs(pair_target[pos]) >= args.meaningful_delta
            if meaningful_pair.any():
                signed_prediction = (
                    torch.sign(pair_target[pos][meaningful_pair])
                    * predicted_pair_delta[meaningful_pair]
                    / pair_delta_scale.expand_as(predicted_pair_delta)[meaningful_pair]
                )
                pair_ranking_loss = F.softplus(-signed_prediction).mean()
            else:
                pair_ranking_loss = torch.zeros((), dtype=pv.dtype, device=device)
            cross_partner = partner_all[pos]
            _, cross_pg, _ = batch_parameters(
                model, inputs, cross_action_all[pos], idx
            )
            cross_pg = cross_pg.flatten(1)
            cross_tg = target_gradient[cross_partner]
            cross_anchor_gradient_loss = F.smooth_l1_loss(
                (cross_pg - cross_tg) / gradient_scale,
                torch.zeros_like(cross_pg),
                beta=0.5,
            )
            cross_anchor_cosine_loss = (
                1.0 - F.cosine_similarity(cross_pg, cross_tg, dim=1)
            ).mean()
            cross_anchor_delta_loss = F.smooth_l1_loss(
                ((cross_pg - pg) - (cross_tg - tg)) / gradient_scale,
                torch.zeros_like(cross_pg),
                beta=0.5,
            )
            partner_idx = torch.from_numpy(
                labels["context_index"][partner_position[pos_np]]
            ).to(device)
            _, same_center_pg, _ = batch_parameters(
                model,
                inputs,
                cross_action_all[cross_partner],
                partner_idx,
            )
            same_center_pg = same_center_pg.flatten(1)
            cross_anchor_invariance_loss = F.smooth_l1_loss(
                (same_center_pg - pg) / gradient_scale,
                torch.zeros_like(pg),
                beta=0.5,
            )
            loss = (
                args.value_weight * value_loss
                + args.gradient_weight * gradient_loss
                + epoch_cosine_weight * cosine_loss
                + args.magnitude_weight * magnitude_loss
                + args.curvature_weight * curvature_loss
                + args.bank_value_weight * bank_loss
                + args.pair_delta_weight * pair_delta_loss
                + args.pair_ranking_weight * pair_ranking_loss
                + args.cross_anchor_gradient_weight
                * cross_anchor_gradient_loss
                + args.cross_anchor_cosine_weight
                * cross_anchor_cosine_loss
                + args.cross_anchor_delta_weight
                * cross_anchor_delta_loss
                + args.cross_anchor_invariance_weight
                * cross_anchor_invariance_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in (
                ("total", loss), ("value", value_loss),
                ("gradient", gradient_loss), ("cosine", cosine_loss),
                ("magnitude", magnitude_loss), ("curvature", curvature_loss),
                ("bank", bank_loss),
                ("pair_delta", pair_delta_loss),
                ("pair_ranking", pair_ranking_loss),
                ("cross_anchor_gradient", cross_anchor_gradient_loss),
                ("cross_anchor_cosine", cross_anchor_cosine_loss),
                ("cross_anchor_delta", cross_anchor_delta_loss),
                ("cross_anchor_invariance", cross_anchor_invariance_loss),
            ):
                loss_rows[key].append(float(value.detach()))
        model.eval()
        validation = metrics(
            [model], inputs, labels, validation_positions, device,
            args.evaluation_batch_size, args.meaningful_delta,
            partner_position, cross_action,
        )
        score = (
            1.0 - validation["gradient_cosine_median"]
            + 0.05 * validation["probe_mean_argmax_regret"]
            + args.selection_norm_weight
            * validation["gradient_abs_log_norm_ratio_median"]
            + args.selection_cosine_p10_weight
            * 0.5 * (1.0 - validation["gradient_cosine_p10"])
            + args.selection_cross_anchor_weight
            * (1.0 - validation["cross_anchor"]["target_cosine_median"])
            + args.selection_action_flip_weight
            * (1.0 - validation["cross_anchor"]["true_reversal_flip_recall"])
        )
        if epoch >= args.scheduler_min_epoch:
            scheduler.step(score)
        checkpoint_eligible = (
            epoch >= args.selection_min_epoch
            and validation["gradient_norm_ratio_median"]
            >= args.selection_min_norm_ratio
            and validation["gradient_norm_ratio_median"]
            <= args.selection_max_norm_ratio
        )
        if checkpoint_eligible:
            if score < best_score - 1e-5:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        if epoch == 1 or epoch % 10 == 0:
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(loss_rows["total"])),
                "train_loss_components": {
                    key: float(np.mean(value)) for key, value in loss_rows.items()
                },
                "effective_cosine_weight": epoch_cosine_weight,
                "checkpoint_eligible": checkpoint_eligible,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation_score": score,
                **validation,
            }
            history.append(row)
            print(json.dumps({"seed": seed, **row}), flush=True)
        if checkpoint_eligible and stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_score": best_score,
        "history": history,
    }


RADII_FOR_METRICS = np.asarray([], np.float32)


def main() -> None:
    global RADII_FOR_METRICS
    args = parse_args()
    if args.cosine_warmup_epochs < 0 or args.cosine_ramp_epochs < 0:
        raise ValueError("cosine warmup/ramp epochs must be nonnegative")
    if args.selection_min_epoch < 1 or args.scheduler_min_epoch < 1:
        raise ValueError("selection/scheduler minimum epoch must be positive")
    if not 0 <= args.selection_min_norm_ratio <= args.selection_max_norm_ratio:
        raise ValueError("invalid checkpoint norm-ratio eligibility interval")
    if args.magnitude_weight < 0 or args.selection_norm_weight < 0:
        raise ValueError("magnitude and norm-selection weights must be nonnegative")
    if args.pair_delta_weight < 0 or args.pair_ranking_weight < 0:
        raise ValueError("pair-loss weights must be nonnegative")
    if min(
        args.cross_anchor_gradient_weight,
        args.cross_anchor_cosine_weight,
        args.cross_anchor_delta_weight,
        args.cross_anchor_invariance_weight,
        args.selection_cross_anchor_weight,
        args.selection_action_flip_weight,
    ) < 0:
        raise ValueError("cross-anchor weights must be nonnegative")
    if args.magnitude_epsilon <= 0:
        raise ValueError("magnitude epsilon must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    radii = np.asarray([float(x) for x in args.probe_radii_sigma.split(",")], np.float32)
    if len(radii) < 2 or np.any(radii <= 0):
        raise ValueError("at least two positive probe radii are required")
    RADII_FOR_METRICS = radii
    device = torch.device(args.device)

    initial_payload = torch.load(args.initial_actor, map_location="cpu")
    alpha_path = Path(initial_payload["base_alpha_checkpoint"])
    alpha_payload = torch.load(alpha_path, map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    labels_path = Path(initial_payload["labels"])
    data, _, splits = load_dataset(labels_path, old_payload)
    fit_index = np.flatnonzero(np.isin(data.episodes, splits["internal_fit"]))
    selection_index = np.flatnonzero(np.isin(data.episodes, splits["internal_selection"]))
    if args.max_fit_contexts:
        fit_index = fit_index[: args.max_fit_contexts]
    if args.max_selection_contexts:
        selection_index = selection_index[: args.max_selection_contexts]
    if set(data.episodes[fit_index]) & set(data.episodes[selection_index]):
        raise AssertionError("fit/selection episode leakage")
    all_index = np.concatenate((fit_index, selection_index))

    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]), args.evaluation_batch_size, device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    maximum_residual_sigma = float(initial_payload["maximum_residual_sigma"])
    actor = TorchMPPIDeterministicCenterActor(maximum_residual_sigma, dropout=0.0).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()
    base_cost = direct_cost(
        alpha_center, data, tensors, np.arange(len(data.episodes)),
        args.evaluation_batch_size, device,
    )

    label_source: dict[str, Any]
    if args.labels_npz is None:
        print(
            f"collecting labels for {len(all_index)} contexts x {len(radii)} radii",
            flush=True,
        )
        local = collect_labels(
            actor, inputs, data, tensors, all_index, alpha_center, base_cost,
            maximum_residual_sigma, radii, args, device,
        )
        label_source = {"mode": "fresh_dbm_rollout"}
    else:
        source_path = args.labels_npz.resolve()
        with np.load(source_path, allow_pickle=False) as archive:
            source_radii = np.asarray(archive["probe_radii_sigma"], np.float32)
            source_context = np.asarray(archive["context_index"], np.int64)
            source_local = {
                key: np.asarray(archive[key])
                for key in archive.files
                if key not in {"probe_radii_sigma", "episode"}
            }
        if not np.array_equal(source_radii, radii):
            raise AssertionError(
                f"stored radii {source_radii.tolist()} != requested {radii.tolist()}"
            )
        source_position = {int(value): pos for pos, value in enumerate(source_context)}
        try:
            take = np.asarray([source_position[int(value)] for value in all_index])
        except KeyError as error:
            raise AssertionError(f"precomputed labels miss context {error.args[0]}") from error
        local = {
            key: value[take]
            if isinstance(value, np.ndarray) and len(value) == len(source_context)
            else value
            for key, value in source_local.items()
        }
        if not np.array_equal(local["context_index"], all_index):
            raise AssertionError("precomputed labels do not reconstruct requested context order")
        label_source = {
            "mode": "reused_precomputed",
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        }
        print(
            f"reusing {len(all_index)} contexts x {len(radii)} radii from {source_path}",
            flush=True,
        )
    if int(local["design_rank_by_radius"].min()) < ACTION_DIM + 1:
        raise AssertionError("a per-radius local quadratic design is not full rank")
    if int(local["combined_design_rank"].min()) < ACTION_DIM + 1:
        raise AssertionError("a combined local quadratic design is not full rank")
    local, gradient_target_source = configure_gradient_target(
        local, args.gradient_target
    )
    partner_position = repeat_partner_positions(data, local)
    cross_action = cross_anchor_actions(
        data, local, partner_position, alpha_center, maximum_residual_sigma
    )
    cross_action_absolute_maximum = float(np.max(np.abs(cross_action)))
    np.savez_compressed(
        args.output_dir / "local_forward_labels.npz",
        probe_radii_sigma=radii,
        episode=data.episodes[all_index],
        **local,
    )
    fit_positions = np.arange(len(fit_index), dtype=np.int64)
    selection_positions = np.arange(len(fit_index), len(all_index), dtype=np.int64)
    fit_episodes = np.unique(data.episodes[fit_index])
    split_rng = np.random.default_rng(260812)
    shuffled = fit_episodes.copy()
    split_rng.shuffle(shuffled)
    validation_episodes = set(shuffled[: max(1, len(shuffled) // 5)])
    fit_episode_for_position = data.episodes[fit_index]
    validation_positions = fit_positions[
        np.asarray([x in validation_episodes for x in fit_episode_for_position])
    ]
    train_positions = fit_positions[
        np.asarray([x not in validation_episodes for x in fit_episode_for_position])
    ]
    if len(train_positions) == 0 or len(validation_positions) == 0:
        raise AssertionError("empty train or internal-validation split")

    scales: dict[str, np.ndarray | float] = {
        "value": max(float(np.std(local["value"][train_positions])), 0.05),
        "gradient": np.maximum(np.std(local["gradient"][train_positions], axis=0), 0.05),
        "curvature": max(float(np.std(local["curvature"][train_positions])), 0.05),
        "bank": max(float(np.std(local["transformed_reward"][train_positions])), 0.05),
        "pair_delta": np.maximum(
            np.std(
                local["transformed_reward"][train_positions, 0, 1:17]
                - local["transformed_reward"][train_positions, 0, 17:33],
                axis=0,
            ),
            0.01,
        ),
    }
    models, training = [], []
    checkpoint_paths = []
    for seed in [int(x) for x in args.seeds.split(",")]:
        model, record = train_one(
            seed, actor, inputs, local, train_positions, validation_positions,
            scales, partner_position, cross_action, args, device,
        )
        path = args.output_dir / f"local_critic_seed{seed}.pt"
        model_class = type(model).__name__
        torch.save({
            "format_version": 1,
            "model_class": model_class,
            "model_state_dict": model.state_dict(),
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "maximum_residual_sigma": maximum_residual_sigma,
            "action_conditioning": args.action_conditioning,
            "probe_radii_sigma": radii.tolist(),
            "reward_scale": args.reward_scale,
            "gradient_target": args.gradient_target,
            "gradient_target_source": gradient_target_source,
            "bank_mode": args.bank_mode,
            "cross_anchor": {
                "gradient_weight": args.cross_anchor_gradient_weight,
                "cosine_weight": args.cross_anchor_cosine_weight,
                "delta_weight": args.cross_anchor_delta_weight,
                "invariance_weight": args.cross_anchor_invariance_weight,
                "cross_action_absolute_maximum": cross_action_absolute_maximum,
            },
            "scales": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in scales.items()
            },
            "training": record,
            "contract": {
                "actor_frozen": True,
                "actor_new_probe_input": False,
                "analytic_dbm_gradient": False,
                "full_action_dimension": ACTION_DIM,
                "formal_validation_loaded": False,
                "test_loaded": False,
            },
        }, path)
        models.append(model)
        training.append(record)
        checkpoint_paths.append(str(path.resolve()))

    train_metrics = metrics(
        models, inputs, local, train_positions, device,
        args.evaluation_batch_size, args.meaningful_delta,
        partner_position, cross_action,
    )
    validation_metrics = metrics(
        models, inputs, local, validation_positions, device,
        args.evaluation_batch_size, args.meaningful_delta,
        partner_position, cross_action,
    )
    heldout_metrics = metrics(
        models, inputs, local, selection_positions, device,
        args.evaluation_batch_size, args.meaningful_delta,
        partner_position, cross_action,
    )
    fit_consistency = label_consistency({
        key: value[fit_positions] if isinstance(value, np.ndarray) and len(value) == len(all_index) else value
        for key, value in local.items()
    })
    heldout_consistency = label_consistency({
        key: value[selection_positions] if isinstance(value, np.ndarray) and len(value) == len(all_index) else value
        for key, value in local.items()
    })
    gates = {
        "label_cross_radius_cosine_median_ge_0_70": heldout_consistency["cross_radius_cosine_median"] >= 0.70,
        "heldout_gradient_cosine_median_ge_0_40": heldout_metrics["gradient_cosine_median"] >= 0.40,
        "heldout_gradient_positive_fraction_ge_0_70": heldout_metrics["gradient_cosine_positive_fraction"] >= 0.70,
        "heldout_pair_sign_ge_0_65": heldout_metrics["probe_delta_sign_accuracy"] >= 0.65,
        "heldout_argmax_regret_le_4": heldout_metrics["probe_mean_argmax_regret"] <= 4.0,
        "heldout_gradient_norm_ratio_ge_0_50": heldout_metrics["gradient_norm_ratio_median"] >= 0.50,
        "heldout_gradient_norm_ratio_le_2_00": heldout_metrics["gradient_norm_ratio_median"] <= 2.00,
        "heldout_gradient_cosine_p10_ge_0": heldout_metrics["gradient_cosine_p10"] >= 0.0,
        "heldout_cross_anchor_target_cosine_median_ge_0_70": (
            heldout_metrics["cross_anchor"]["target_cosine_median"] >= 0.70
        ),
        "heldout_action_response_flip_recall_ge_0_50": (
            heldout_metrics["cross_anchor"]["true_reversal_flip_recall"] >= 0.50
        ),
        "heldout_same_center_nuisance_cosine_median_ge_0_90": (
            heldout_metrics["cross_anchor"][
                "same_absolute_center_nuisance_cosine_median"
            ] >= 0.90
        ),
    }
    passed = all(gates.values())
    if not passed and train_metrics["gradient_cosine_median"] >= 0.70:
        diagnosis = "FIT_PASS_HELDOUT_FAIL_STATE_TO_GRADIENT_GENERALIZATION"
    elif not passed:
        diagnosis = "FIT_OR_LABEL_FAIL_OPTIMIZATION_OR_LOCAL_TARGET"
    else:
        diagnosis = "FROZEN_CRITIC_GATE_PASS_ACTOR_STEP_AUTHORIZED"
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "frozen-Actor explicit full-16D local Critic; action conditioning="
            f"{args.action_conditioning}"
        ),
        "qualification": diagnosis,
        "actor_update_performed": False,
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "initial_actor": str(args.initial_actor.resolve()),
        "initial_actor_sha256": sha256_file(args.initial_actor),
        "checkpoints": checkpoint_paths,
        "fit_context_count": int(len(fit_index)),
        "train_context_count": int(len(train_positions)),
        "internal_validation_context_count": int(len(validation_positions)),
        "heldout_context_count": int(len(selection_positions)),
        "probe_radii_sigma": radii.tolist(),
        "label_source": label_source,
        "gradient_target_source": gradient_target_source,
        "repeat_pair_count": int(len(partner_position) // 2),
        "cross_anchor_action_absolute_maximum": cross_action_absolute_maximum,
        "fit_label_consistency": fit_consistency,
        "heldout_label_consistency": heldout_consistency,
        "train_metrics": train_metrics,
        "internal_validation_metrics": validation_metrics,
        "heldout_metrics": heldout_metrics,
        "gates": gates,
        "all_gates_passed": passed,
        "generic_scalar_q_baseline": {
            "heldout_pair_sign_accuracy": 0.5235,
            "heldout_delta_correlation": -0.011,
            "heldout_argmax_regret": 18.84,
            "source": "direct_response_slope_critic_only_20260811_v1",
        },
        "training": training,
        "training_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "split": {
            "train_episodes": sorted(set(data.episodes[fit_index]) - validation_episodes),
            "internal_validation_episodes": sorted(validation_episodes),
            "heldout_episodes": sorted(set(data.episodes[selection_index])),
        },
        "contract": {
            "actor_input_changed": False,
            "actor_output": "one unique deterministic 8x2 residual center",
            "critic_gradient_dimension": ACTION_DIM,
            "probe_usage": "training/evaluation labels only",
            "analytic_dbm_gradient": False,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "gradient_target": args.gradient_target,
            "bank_mode": args.bank_mode,
            "repeat_pair_supervision": (
                "same physical state, paired absolute Actor centers; "
                "no same-state same-gradient assumption"
            ),
            "critic_action_conditioning": args.action_conditioning,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Full-16D actor-centered local Critic\n\n"
        f"Qualification: `{diagnosis}`. Actor was frozen and was not updated.\n\n"
        f"Heldout gradient median cosine: {heldout_metrics['gradient_cosine_median']:.4f}; "
        f"pair sign: {heldout_metrics['probe_delta_sign_accuracy']:.4f}; "
        f"33-probe regret: {heldout_metrics['probe_mean_argmax_regret']:.4f}.\n"
    )
    print(json.dumps({
        "qualification": diagnosis,
        "heldout_label_consistency": heldout_consistency,
        "train_metrics": train_metrics,
        "internal_validation_metrics": validation_metrics,
        "heldout_metrics": heldout_metrics,
        "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
