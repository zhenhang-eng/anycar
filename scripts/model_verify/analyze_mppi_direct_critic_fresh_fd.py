#!/usr/bin/env python3
"""Audit learned local-Critic action gradients with fresh DBM finite differences.

This is a diagnosis-only experiment.  It freezes the deterministic Actor and the
three explicit local Critics, uses only the consumed ``internal_selection`` split,
and evaluates a deterministic random orthogonal direction bank that was not used
to create the Critic labels.  No Actor/Critic parameter is updated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPIActorCenteredLocalCritic,
    TorchMPPIDeterministicCenterActor,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import (
    actor_inputs,
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
import train_mppi_direct_local_gradient_critic as local_train


DEFAULT_CRITIC_DIR = Path(
    "outputs/mppi_proposal/direct_local_gradient_critic_20260812_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_critic_fresh_fd_20260813_v1"
)
ACTION_DIM = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-radii-sigma", default="0.01,0.02,0.04")
    parser.add_argument("--direction-seed", type=int, default=260813)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--meaningful-gradient-norm", type=float, default=0.10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def distribution(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, np.float64)
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def gradient_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    meaningful_norm: float,
) -> dict[str, Any]:
    cosine = cosine_rows(prediction, target)
    target_norm = np.linalg.norm(target, axis=1)
    prediction_norm = np.linalg.norm(prediction, axis=1)
    active = target_norm >= meaningful_norm
    norm_ratio = prediction_norm / np.maximum(target_norm, 1e-8)
    result: dict[str, Any] = {
        "count": int(len(target)),
        "cosine": distribution(cosine),
        "cosine_positive_fraction": float(np.mean(cosine > 0.0)),
        "cosine_above_0_5_fraction": float(np.mean(cosine > 0.5)),
        "correlation": correlation(prediction, target),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "prediction_norm": distribution(prediction_norm),
        "target_norm": distribution(target_norm),
        "norm_ratio": distribution(norm_ratio),
        "meaningful_norm_threshold": float(meaningful_norm),
        "meaningful_count": int(np.sum(active)),
    }
    if np.any(active):
        active_cosine = cosine[active]
        result["meaningful_cosine"] = distribution(active_cosine)
        result["meaningful_cosine_positive_fraction"] = float(
            np.mean(active_cosine > 0.0)
        )
        result["meaningful_cosine_above_0_5_fraction"] = float(
            np.mean(active_cosine > 0.5)
        )
    return result


def fresh_orthogonal_directions(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(ACTION_DIM, ACTION_DIM))
    orthogonal, _ = np.linalg.qr(matrix)
    directions = orthogonal.T
    # Canonicalize the arbitrary QR signs to keep the bank stable across runs.
    for row in directions:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return directions.reshape(ACTION_DIM, 8, 2).astype(np.float32)


def build_probe_centers(
    actor_center: np.ndarray,
    sigma: np.ndarray,
    directions: np.ndarray,
    radius_sigma: float,
) -> np.ndarray:
    offset = radius_sigma * sigma[:, None, None, :] * directions[None]
    positive = np.clip(actor_center[:, None] + offset, -1.0, 1.0)
    negative = np.clip(actor_center[:, None] - offset, -1.0, 1.0)
    return np.concatenate(
        (actor_center[:, None], positive, negative), axis=1
    ).astype(np.float32)


def collect_fresh_fd(
    data: Any,
    tensors: dict[str, Any],
    context_index: np.ndarray,
    actor_action: np.ndarray,
    actor_center: np.ndarray,
    alpha_center: np.ndarray,
    base_cost: np.ndarray,
    maximum_residual_sigma: float,
    directions: np.ndarray,
    radii: np.ndarray,
    reward_scale: float,
    rollout_batch_size: int,
    evaluation_batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    all_center, all_action, all_cost, all_z = [], [], [], []
    all_gradient, all_curvature, all_rank = [], [], []
    for radius in radii:
        center_parts, action_parts, cost_parts = [], [], []
        for start in range(0, len(context_index), rollout_batch_size):
            local = context_index[start : start + rollout_batch_size]
            row = slice(start, start + len(local))
            bank = build_probe_centers(
                actor_center[row], data.sigma[local], directions, float(radius)
            )
            flat_center = bank.reshape(-1, 8, 2)
            flat_index = np.repeat(local, 1 + 2 * ACTION_DIM)
            cost = direct_cost(
                flat_center, data, tensors, flat_index,
                evaluation_batch_size, device,
            ).reshape(len(local), 1 + 2 * ACTION_DIM)
            center_parts.append(bank)
            action_parts.append(centers_to_actions(
                bank, alpha_center[local], data.sigma[local],
                maximum_residual_sigma,
            ))
            cost_parts.append(cost.astype(np.float32))
        centers = np.concatenate(center_parts)
        actions = np.concatenate(action_parts)
        costs = np.concatenate(cost_parts)
        rewards = base_cost[context_index, None] - costs
        z = transformed(rewards, reward_scale)
        gradient, curvature, rank = local_train.fit_local_parameters(
            actions, z, actor_action
        )
        all_center.append(centers)
        all_action.append(actions)
        all_cost.append(costs)
        all_z.append(z)
        all_gradient.append(gradient)
        all_curvature.append(curvature)
        all_rank.append(rank)

    centers = np.stack(all_center, axis=1)
    actions = np.stack(all_action, axis=1)
    costs = np.stack(all_cost, axis=1)
    z = np.stack(all_z, axis=1)
    gradient_by_radius = np.stack(all_gradient, axis=1)
    curvature_by_radius = np.stack(all_curvature, axis=1)
    rank_by_radius = np.stack(all_rank, axis=1)
    combined_gradient, combined_curvature, combined_rank = (
        local_train.fit_local_parameters(
            actions.reshape(len(context_index), -1, 8, 2),
            z.reshape(len(context_index), -1),
            actor_action,
        )
    )
    return {
        "centers": centers.astype(np.float32),
        "actions": actions.astype(np.float32),
        "cost": costs.astype(np.float32),
        "transformed_reward": z.astype(np.float32),
        "gradient_by_radius": gradient_by_radius.astype(np.float32),
        "curvature_by_radius": curvature_by_radius.astype(np.float32),
        "design_rank_by_radius": rank_by_radius.astype(np.int64),
        "gradient": combined_gradient.astype(np.float32),
        "curvature": combined_curvature.astype(np.float32),
        "combined_design_rank": combined_rank.astype(np.int64),
    }


@torch.no_grad()
def critic_gradients(
    models: list[TorchMPPIActorCenteredLocalCritic],
    inputs: tuple[torch.Tensor, ...],
    context_index: np.ndarray,
    actor_action: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, gradients, curvatures = [], [], []
    for model in models:
        model_values, model_gradients, model_curvatures = [], [], []
        for start in range(0, len(context_index), batch_size):
            local_np = context_index[start : start + batch_size]
            local = torch.from_numpy(local_np).to(device)
            anchor = torch.from_numpy(
                actor_action[start : start + len(local_np)]
            ).to(device)
            value, gradient, curvature = model.local_parameters(
                *(one[local] for one in inputs), anchor
            )
            model_values.append(value.cpu().numpy())
            model_gradients.append(gradient.flatten(1).cpu().numpy())
            model_curvatures.append(curvature.cpu().numpy())
        values.append(np.concatenate(model_values))
        gradients.append(np.concatenate(model_gradients))
        curvatures.append(np.concatenate(model_curvatures))
    return (
        np.asarray(values, np.float32),
        np.asarray(gradients, np.float32),
        np.asarray(curvatures, np.float32),
    )


def autograd_gradient_check(
    models: list[TorchMPPIActorCenteredLocalCritic],
    inputs: tuple[torch.Tensor, ...],
    context_index: np.ndarray,
    actor_action: np.ndarray,
    device: torch.device,
    count: int = 32,
) -> dict[str, Any]:
    chosen = context_index[: min(count, len(context_index))]
    index = torch.from_numpy(chosen).to(device)
    errors = []
    for model in models:
        anchor = torch.from_numpy(actor_action[: len(chosen)]).to(device)
        query_action = anchor.detach().clone().requires_grad_(True)
        value, analytic, _ = model.local_parameters(
            *(one[index] for one in inputs), anchor
        )
        prediction = model(
            *(one[index] for one in inputs), anchor, query_action
        )
        autograd = torch.autograd.grad(prediction.sum(), query_action)[0]
        errors.append(float(torch.max(torch.abs(autograd - analytic)).detach().cpu()))
        if not torch.isfinite(value).all():
            raise AssertionError("non-finite critic value in autograd check")
    return {
        "context_count": int(len(chosen)),
        "per_critic_max_abs_error": errors,
        "maximum_abs_error": float(max(errors, default=0.0)),
    }


def grouped_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    group: np.ndarray,
    meaningful_norm: float,
) -> dict[str, Any]:
    result = {}
    for value in sorted(np.unique(group)):
        mask = group == value
        result[str(value)] = gradient_metrics(
            prediction[mask], target[mask], meaningful_norm
        )
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    radii = np.asarray(
        [float(value) for value in args.probe_radii_sigma.split(",")], np.float32
    )
    if len(radii) < 2 or np.any(radii <= 0):
        raise ValueError("at least two positive radii are required")
    summary_path = args.critic_dir / "summary.json"
    critic_summary = json.loads(summary_path.read_text())
    initial_actor_path = Path(critic_summary["initial_actor"])
    initial_payload = torch.load(initial_actor_path, map_location="cpu")
    alpha_path = Path(initial_payload["base_alpha_checkpoint"])
    alpha_payload = torch.load(alpha_path, map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    labels_path = Path(initial_payload["labels"])
    data, _, splits = load_dataset(labels_path, old_payload)
    context_index = np.flatnonzero(
        np.isin(data.episodes, splits["internal_selection"])
    )
    expected_episodes = set(critic_summary["split"]["heldout_episodes"])
    if set(data.episodes[context_index]) != expected_episodes:
        raise AssertionError("internal-selection episode contract changed")
    if len(context_index) != int(critic_summary["heldout_context_count"]):
        raise AssertionError("internal-selection context count changed")

    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.evaluation_batch_size, device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    maximum_residual_sigma = float(initial_payload["maximum_residual_sigma"])
    actor = TorchMPPIDeterministicCenterActor(
        maximum_residual_sigma, dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()
    actor_action, actor_center = residual_outputs(
        actor, inputs, context_index, args.evaluation_batch_size, device
    )
    effective_anchor = centers_to_actions(
        actor_center[:, None], alpha_center[context_index],
        data.sigma[context_index], maximum_residual_sigma,
    )[:, 0]
    anchor_projection_error = np.max(
        np.abs(effective_anchor - actor_action), axis=(1, 2)
    )
    base_cost = direct_cost(
        alpha_center, data, tensors, np.arange(len(data.episodes)),
        args.evaluation_batch_size, device,
    )
    directions = fresh_orthogonal_directions(args.direction_seed)
    flat_directions = directions.reshape(ACTION_DIM, ACTION_DIM)
    orthogonality_error = float(np.max(np.abs(
        flat_directions @ flat_directions.T - np.eye(ACTION_DIM)
    )))
    if orthogonality_error > 1e-5:
        raise AssertionError("fresh direction bank is not orthonormal")

    print(
        f"fresh FD: {len(context_index)} contexts x {len(radii)} radii "
        f"x {1 + 2 * ACTION_DIM} candidates",
        flush=True,
    )
    fresh = collect_fresh_fd(
        data, tensors, context_index, actor_action, actor_center, alpha_center,
        base_cost, maximum_residual_sigma, directions, radii,
        float(critic_summary["training_arguments"]["reward_scale"]),
        args.rollout_batch_size, args.evaluation_batch_size, device,
    )
    if int(np.min(fresh["design_rank_by_radius"])) < ACTION_DIM + 1:
        raise AssertionError("fresh per-radius design is rank deficient")
    if int(np.min(fresh["combined_design_rank"])) < ACTION_DIM + 1:
        raise AssertionError("fresh combined design is rank deficient")

    models = []
    checkpoint_paths = [Path(value) for value in critic_summary["checkpoints"]]
    for checkpoint_path in checkpoint_paths:
        payload = torch.load(checkpoint_path, map_location="cpu")
        model = TorchMPPIActorCenteredLocalCritic(dropout=0.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
    critic_value, critic_gradient, critic_curvature = critic_gradients(
        models, inputs, context_index, actor_action,
        args.evaluation_batch_size, device,
    )
    ensemble_gradient = np.mean(critic_gradient, axis=0)

    old = np.load(args.critic_dir / "local_forward_labels.npz", allow_pickle=False)
    old_index = np.asarray(old["context_index"], np.int64)
    old_lookup = {int(value): position for position, value in enumerate(old_index)}
    old_position = np.asarray([old_lookup[int(value)] for value in context_index])
    old_gradient = np.asarray(old["gradient"], np.float32)[old_position]
    old_gradient_by_radius = np.asarray(
        old["gradient_by_radius"], np.float32
    )[old_position]
    old_radii = np.asarray(old["probe_radii_sigma"], np.float32)
    old.close()

    fresh_target = fresh["gradient"]
    critic_vs_fresh = {
        f"critic_{position}": gradient_metrics(
            gradient, fresh_target, args.meaningful_gradient_norm
        )
        for position, gradient in enumerate(critic_gradient)
    }
    critic_vs_fresh["ensemble_mean"] = gradient_metrics(
        ensemble_gradient, fresh_target, args.meaningful_gradient_norm
    )
    critic_vs_old = {
        f"critic_{position}": gradient_metrics(
            gradient, old_gradient, args.meaningful_gradient_norm
        )
        for position, gradient in enumerate(critic_gradient)
    }
    critic_vs_old["ensemble_mean"] = gradient_metrics(
        ensemble_gradient, old_gradient, args.meaningful_gradient_norm
    )
    old_vs_fresh = gradient_metrics(
        old_gradient, fresh_target, args.meaningful_gradient_norm
    )
    old_by_radius_vs_fresh = {}
    for old_radius_index, old_radius in enumerate(old_radii):
        radius_row = {}
        for fresh_radius_index, fresh_radius in enumerate(radii):
            radius_row[f"fresh_{float(fresh_radius):.2f}"] = gradient_metrics(
                old_gradient_by_radius[:, old_radius_index],
                fresh["gradient_by_radius"][:, fresh_radius_index],
                args.meaningful_gradient_norm,
            )
        radius_row["fresh_combined"] = gradient_metrics(
            old_gradient_by_radius[:, old_radius_index], fresh_target,
            args.meaningful_gradient_norm,
        )
        old_by_radius_vs_fresh[f"training_{float(old_radius):.2f}"] = radius_row

    cross_radius = {}
    for left in range(len(radii)):
        for right in range(left + 1, len(radii)):
            name = f"{float(radii[left]):.2f}_vs_{float(radii[right]):.2f}"
            cross_radius[name] = gradient_metrics(
                fresh["gradient_by_radius"][:, left],
                fresh["gradient_by_radius"][:, right],
                args.meaningful_gradient_norm,
            )
    critic_pairwise = {}
    for left in range(len(models)):
        for right in range(left + 1, len(models)):
            name = f"critic_{left}_vs_{right}"
            critic_pairwise[name] = gradient_metrics(
                critic_gradient[left], critic_gradient[right],
                args.meaningful_gradient_norm,
            )

    # Fraction of requested +/- physical perturbations altered by hard bounds.
    requested_delta = (
        radii[None, :, None, None, None]
        * data.sigma[context_index, None, None, None, :]
        * np.concatenate((
            np.zeros((1, 8, 2), np.float32), directions, -directions
        ), axis=0)[None, None]
    )
    requested_center = actor_center[:, None, None] + requested_delta
    clipped_element = np.abs(fresh["centers"] - requested_center) > 2e-6
    clipped_context = np.any(clipped_element, axis=(1, 2, 3, 4))

    speed = data.reference_speed[context_index]
    scenario = data.scenario[context_index]
    npz_path = args.output_dir / "fresh_fd_audit.npz"
    np.savez_compressed(
        npz_path,
        context_index=context_index.astype(np.int64),
        episode=data.episodes[context_index],
        reference_speed=speed.astype(np.float32),
        scenario=scenario,
        probe_radii_sigma=radii,
        direction_seed=np.asarray(args.direction_seed, np.int64),
        directions=directions,
        alpha_center=alpha_center[context_index].astype(np.float32),
        base_cost=base_cost[context_index].astype(np.float32),
        sigma=data.sigma[context_index].astype(np.float32),
        actor_action=actor_action.astype(np.float32),
        actor_center=actor_center.astype(np.float32),
        effective_anchor=effective_anchor.astype(np.float32),
        anchor_projection_error=anchor_projection_error.astype(np.float32),
        old_gradient=old_gradient.astype(np.float32),
        old_gradient_by_radius=old_gradient_by_radius.astype(np.float32),
        old_probe_radii_sigma=old_radii,
        critic_value=critic_value,
        critic_gradient=critic_gradient,
        critic_curvature=critic_curvature,
        clipped_context=clipped_context,
        **fresh,
    )

    autograd_check = autograd_gradient_check(
        models, inputs, context_index, actor_action, device
    )
    primary = critic_vs_fresh["ensemble_mean"]
    gates = {
        "fresh_cross_radius_median_cosine_ge_0_85": all(
            value["cosine"]["median"] >= 0.85 for value in cross_radius.values()
        ),
        "smallest_training_radius_vs_fresh_median_cosine_ge_0_90": (
            old_by_radius_vs_fresh[
                f"training_{float(old_radii[0]):.2f}"
            ]["fresh_combined"]["cosine"]["median"] >= 0.90
        ),
        "critic_vs_fresh_median_cosine_ge_0_70": (
            primary["cosine"]["median"] >= 0.70
        ),
        "critic_vs_fresh_p10_cosine_ge_0": primary["cosine"]["p10"] >= 0.0,
        "critic_pairwise_median_cosine_ge_0_80": all(
            value["cosine"]["median"] >= 0.80
            for value in critic_pairwise.values()
        ),
        "autograd_matches_gradient_head": autograd_check["maximum_abs_error"] <= 1e-6,
    }
    if (
        gates["fresh_cross_radius_median_cosine_ge_0_85"]
        and gates["smallest_training_radius_vs_fresh_median_cosine_ge_0_90"]
        and not gates["critic_vs_fresh_median_cosine_ge_0_70"]
    ):
        qualification = "REWARD_GRADIENT_STABLE_CRITIC_GENERALIZATION_FAIL"
    elif not gates["fresh_cross_radius_median_cosine_ge_0_85"]:
        qualification = "FRESH_FD_RADIUS_SENSITIVE"
    elif not gates["smallest_training_radius_vs_fresh_median_cosine_ge_0_90"]:
        qualification = "TRAIN_PROBE_TO_FRESH_DIRECTION_TRANSFER_FAIL"
    elif all(gates.values()):
        qualification = "FRESH_FD_GRADIENT_GATE_PASS"
    else:
        qualification = "MIXED_FRESH_FD_DIAGNOSIS"

    direction_hash = hashlib.sha256(directions.tobytes()).hexdigest()
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "fresh orthogonal finite-difference audit of frozen local Critics",
        "qualification": qualification,
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "artifact": str(npz_path.resolve()),
        "artifact_sha256": sha256_file(npz_path),
        "critic_summary": str(summary_path.resolve()),
        "critic_summary_sha256": sha256_file(summary_path),
        "initial_actor": str(initial_actor_path.resolve()),
        "initial_actor_sha256": sha256_file(initial_actor_path),
        "critic_checkpoints": [str(path.resolve()) for path in checkpoint_paths],
        "critic_checkpoint_sha256": [sha256_file(path) for path in checkpoint_paths],
        "context_count": int(len(context_index)),
        "split": "consumed_train/internal_selection",
        "formal_validation_loaded": False,
        "test_loaded": False,
        "probe_radii_sigma": radii.tolist(),
        "training_probe_radii_sigma": critic_summary["probe_radii_sigma"],
        "direction_seed": args.direction_seed,
        "direction_sha256": direction_hash,
        "direction_orthogonality_max_abs_error": orthogonality_error,
        "dbm_rollout_count": int(len(context_index) * len(radii) * (1 + 2 * ACTION_DIM)),
        "design_minimum_rank_by_radius": int(np.min(fresh["design_rank_by_radius"])),
        "combined_design_minimum_rank": int(np.min(fresh["combined_design_rank"])),
        "anchor_projection_error": distribution(anchor_projection_error),
        "anchor_projection_changed_context_fraction": float(np.mean(anchor_projection_error > 2e-6)),
        "clipped_probe_context_fraction": float(np.mean(clipped_context)),
        "fresh_cross_radius": cross_radius,
        "old_training_label_vs_fresh_fd": old_vs_fresh,
        "old_training_label_by_radius_vs_fresh_fd": old_by_radius_vs_fresh,
        "critic_vs_fresh_fd": critic_vs_fresh,
        "critic_vs_old_training_label": critic_vs_old,
        "critic_pairwise_action_gradient": critic_pairwise,
        "grouped_ensemble_vs_fresh_fd": {
            "reference_speed": grouped_metrics(
                ensemble_gradient, fresh_target,
                np.asarray([f"{float(value):.1f}" for value in speed]),
                args.meaningful_gradient_norm,
            ),
            "scenario": grouped_metrics(
                ensemble_gradient, fresh_target, scenario,
                args.meaningful_gradient_norm,
            ),
        },
        "autograd_gradient_check": autograd_check,
        "gates": gates,
        "contract": {
            "actor_frozen": True,
            "critics_frozen": True,
            "analytic_dbm_gradient": False,
            "fresh_direction_bank_not_used_for_training": True,
            "fresh_radii_not_used_for_training": not bool(
                set(np.round(radii, 8)).intersection(
                    set(np.round(critic_summary["probe_radii_sigma"], 8))
                )
            ),
            "reward": "asinh((J_alpha_base - J_direct(center)) / 5)",
            "action_coordinate": "normalized 8x2 Actor residual",
        },
    }
    summary_file = args.output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Fresh finite-difference Critic gradient audit\n\n"
        f"Qualification: `{qualification}`. Actor and Critics were frozen.\n\n"
        f"Fresh FD cross-radius cosine median range: "
        f"{min(v['cosine']['median'] for v in cross_radius.values()):.4f}--"
        f"{max(v['cosine']['median'] for v in cross_radius.values()):.4f}.\n\n"
        f"Old label vs fresh FD median cosine: "
        f"{old_vs_fresh['cosine']['median']:.4f}.\n\n"
        f"Critic ensemble vs fresh FD median/P10 cosine: "
        f"{primary['cosine']['median']:.4f} / {primary['cosine']['p10']:.4f}.\n"
    )
    print(json.dumps({
        "qualification": qualification,
        "fresh_cross_radius": cross_radius,
        "old_training_label_vs_fresh_fd": old_vs_fresh,
        "old_training_label_by_radius_vs_fresh_fd": old_by_radius_vs_fresh,
        "critic_vs_fresh_fd": critic_vs_fresh,
        "critic_pairwise_action_gradient": critic_pairwise,
        "gates": gates,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
