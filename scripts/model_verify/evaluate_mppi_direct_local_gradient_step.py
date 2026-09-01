#!/usr/bin/env python3
"""Validate frozen full-16D Critic gradients with real small-step DBM rollouts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy, residual_outputs
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_CRITIC = Path(
    "outputs/mppi_proposal/direct_local_gradient_critic_20260812_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_local_gradient_step_eval_20260812_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--radii-sigma", default="0,0.005,0.01,0.02,0.03,0.05,0.075,0.10"
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


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


def policy_metrics(base_cost: np.ndarray, cost: np.ndarray) -> dict[str, Any]:
    gain = np.asarray(base_cost) - np.asarray(cost)
    return {
        "cost": distribution(cost),
        "gain": distribution(gain),
        "wins": int(np.sum(gain > 1e-6)),
        "losses": int(np.sum(gain < -1e-6)),
        "ties": int(np.sum(np.abs(gain) <= 1e-6)),
        "win_fraction": float(np.mean(gain > 1e-6)),
        "regression_fraction": float(np.mean(gain < -1e-6)),
    }


@torch.no_grad()
def predict_gradients(
    models: list[TorchMPPIActorCenteredLocalCritic],
    inputs: tuple[torch.Tensor, ...],
    context_index: np.ndarray,
    actor_action: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    per_model = [[] for _ in models]
    for start in range(0, len(context_index), batch_size):
        one = context_index[start : start + batch_size]
        index = torch.from_numpy(one).to(device)
        anchor = torch.from_numpy(actor_action[start : start + len(one)]).to(device)
        for model_index, model in enumerate(models):
            _, gradient, _ = model.local_parameters(
                *(value[index] for value in inputs), anchor
            )
            per_model[model_index].append(gradient.flatten(1).cpu().numpy())
    stack = np.stack([np.concatenate(parts) for parts in per_model], axis=0)
    mean = stack.mean(axis=0)
    rms = np.sqrt(np.mean(np.square(mean), axis=1, keepdims=True))
    direction = mean / np.maximum(rms, 1e-8)
    unit = stack / np.maximum(np.linalg.norm(stack, axis=2, keepdims=True), 1e-8)
    mean_unit = unit.mean(axis=0)
    agreement = np.mean(np.sum(unit * mean_unit[None], axis=2), axis=0)
    component_std = stack.std(axis=0)
    relative_std = np.sqrt(np.mean(np.square(component_std), axis=1)) / np.maximum(
        rms[:, 0], 1e-8
    )
    return direction.astype(np.float32), agreement.astype(np.float32), relative_std.astype(np.float32)


def evaluate_signed_line(
    actor_action: np.ndarray,
    direction: np.ndarray,
    alpha_center: np.ndarray,
    sigma: np.ndarray,
    maximum_residual_sigma: float,
    signed_radii: np.ndarray,
    data: Any,
    tensors: dict[str, Any],
    context_index: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_delta = (
        signed_radii[None, :, None]
        / maximum_residual_sigma
        * direction[:, None]
    )
    action = np.clip(
        actor_action.reshape(len(actor_action), 1, 16) + normalized_delta,
        -1.0,
        1.0,
    ).reshape(len(actor_action), len(signed_radii), 8, 2).astype(np.float32)
    center = np.clip(
        alpha_center[context_index, None]
        + action
        * maximum_residual_sigma
        * sigma[:, None, None, :],
        -1.0,
        1.0,
    ).astype(np.float32)
    flat_index = np.repeat(context_index, len(signed_radii))
    cost = direct_cost(
        center.reshape(-1, 8, 2), data, tensors, flat_index, batch_size, device
    ).reshape(len(context_index), len(signed_radii))
    effective_delta = (
        center - alpha_center[context_index, None]
    ) / (maximum_residual_sigma * sigma[:, None, None, :]) - actor_action[:, None]
    effective_radius = np.sqrt(np.mean(np.square(effective_delta.reshape(
        len(actor_action), len(signed_radii), 16
    )), axis=2)) * maximum_residual_sigma
    return action, center, cost.astype(np.float32), effective_radius.astype(np.float32)


def group_metrics(
    base: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    line_oracle: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for value in sorted(np.unique(values)):
        mask = values == value
        result[str(value)] = {
            "count": int(mask.sum()),
            "positive": policy_metrics(base[mask], positive[mask]),
            "negative": policy_metrics(base[mask], negative[mask]),
            "positive_line_oracle": policy_metrics(base[mask], line_oracle[mask]),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    radii = np.asarray([float(x) for x in args.radii_sigma.split(",")], np.float32)
    if radii[0] != 0 or np.any(np.diff(radii) <= 0):
        raise ValueError("radii must start at zero and increase strictly")
    signed_radii = np.concatenate((-radii[:0:-1], radii)).astype(np.float32)
    zero_index = len(radii) - 1
    device = torch.device(args.device)

    critic_summary = json.loads((args.critic_dir / "summary.json").read_text())
    initial_actor_path = Path(critic_summary["initial_actor"])
    initial_payload = torch.load(initial_actor_path, map_location="cpu")
    alpha_path = Path(initial_payload["base_alpha_checkpoint"])
    alpha_payload = torch.load(alpha_path, map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    labels_path = Path(initial_payload["labels"])
    data, _, _ = load_dataset(labels_path, old_payload)
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
    actor = TorchMPPIDeterministicCenterActor(maximum_residual_sigma, dropout=0.0).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()
    models = []
    for path in critic_summary["checkpoints"]:
        payload = torch.load(path, map_location="cpu")
        model = TorchMPPIActorCenteredLocalCritic(dropout=0.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        models.append(model)

    with np.load(args.critic_dir / "local_forward_labels.npz", allow_pickle=False) as archive:
        context_index = np.asarray(archive["context_index"], np.int64)
        label_episode = np.asarray(archive["episode"]).astype(str)
    episodes = data.episodes[context_index].astype(str)
    if not np.array_equal(label_episode, episodes):
        raise AssertionError("local-label episode mapping changed")
    validation_set = set(critic_summary["split"]["internal_validation_episodes"])
    heldout_set = set(critic_summary["split"]["heldout_episodes"])
    validation_positions = np.flatnonzero(np.isin(episodes, sorted(validation_set)))
    heldout_positions = np.flatnonzero(np.isin(episodes, sorted(heldout_set)))
    if validation_set & heldout_set:
        raise AssertionError("internal-validation/heldout episode leakage")
    if len(validation_positions) != int(critic_summary["internal_validation_context_count"]):
        raise AssertionError("internal-validation context count changed")
    if len(heldout_positions) != int(critic_summary["heldout_context_count"]):
        raise AssertionError("heldout context count changed")

    used_positions = np.concatenate((validation_positions, heldout_positions))
    used_index = context_index[used_positions]
    actor_action, actor_center = residual_outputs(
        actor, inputs, used_index, args.evaluation_batch_size, device
    )
    direction, agreement, relative_std = predict_gradients(
        models, inputs, used_index, actor_action,
        args.evaluation_batch_size, device,
    )
    action, center, cost, effective_radius = evaluate_signed_line(
        actor_action, direction, alpha_center, data.sigma[used_index],
        maximum_residual_sigma, signed_radii, data, tensors, used_index,
        args.evaluation_batch_size, device,
    )
    base_error = float(np.max(np.abs(cost[:, zero_index] - direct_cost(
        actor_center, data, tensors, used_index, args.evaluation_batch_size, device
    ))))
    validation_count = len(validation_positions)
    val_cost = cost[:validation_count]
    held_cost = cost[validation_count:]
    positive_indices = np.arange(zero_index, len(signed_radii))
    validation_positive_mean = val_cost[:, positive_indices].mean(axis=0)
    chosen_local = int(np.argmin(validation_positive_mean))
    chosen_index = int(positive_indices[chosen_local])
    chosen_radius = float(signed_radii[chosen_index])
    negative_index = int(np.argmin(np.abs(signed_radii + chosen_radius)))
    base = held_cost[:, zero_index]
    positive = held_cost[:, chosen_index]
    negative = held_cost[:, negative_index]
    positive_line_oracle = held_cost[:, positive_indices].min(axis=1)
    bidirectional_oracle = held_cost.min(axis=1)
    sign_oracle = np.minimum(positive, negative)
    held_global = used_index[validation_count:]

    mechanism = {
        "positive_mean_gain_gt_zero": float(np.mean(base - positive)) > 0.0,
        "positive_mean_cost_below_negative": float(np.mean(positive)) < float(np.mean(negative)),
        "positive_wins_gt_losses": int(np.sum(base > positive + 1e-6)) > int(np.sum(base < positive - 1e-6)),
    }
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "frozen full-16D local-Critic signed small-step DBM validation",
        "qualification": "LOCAL_GRADIENT_STEP_MECHANISM_PASS" if all(mechanism.values()) else "LOCAL_GRADIENT_STEP_MECHANISM_FAIL",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "critic_dir": str(args.critic_dir.resolve()),
        "critic_summary_sha256": sha256_file(args.critic_dir / "summary.json"),
        "initial_actor": str(initial_actor_path.resolve()),
        "initial_actor_sha256": sha256_file(initial_actor_path),
        "validation_context_count": int(validation_count),
        "heldout_context_count": int(len(heldout_positions)),
        "radii_sigma": radii.tolist(),
        "signed_radii_sigma": signed_radii.tolist(),
        "selected_radius_sigma": chosen_radius,
        "selected_radius_source": "minimum positive-direction mean cost on fit-internal-validation only",
        "validation_positive_mean_cost": {
            str(float(radius)): float(value)
            for radius, value in zip(radii, validation_positive_mean)
        },
        "heldout": {
            "actor_base": policy_metrics(base, base),
            "fixed_positive_step": policy_metrics(base, positive),
            "same_radius_negative_control": policy_metrics(base, negative),
            "same_radius_sign_oracle": policy_metrics(base, sign_oracle),
            "positive_line_oracle": policy_metrics(base, positive_line_oracle),
            "bidirectional_line_oracle": policy_metrics(base, bidirectional_oracle),
            "positive_vs_negative_mean_cost_delta": float(np.mean(negative - positive)),
            "ensemble_direction_agreement": distribution(agreement[validation_count:]),
            "ensemble_relative_gradient_std": distribution(relative_std[validation_count:]),
        },
        "grouped_heldout": {
            "reference_speed": group_metrics(
                base, positive, negative, positive_line_oracle,
                data.reference_speed[held_global],
            ),
            "scenario": group_metrics(
                base, positive, negative, positive_line_oracle,
                data.scenario[held_global],
            ),
        },
        "mechanism_gates": mechanism,
        "base_replay_max_abs_error": base_error,
        "maximum_effective_radius_error": float(np.max(np.abs(
            effective_radius - np.abs(signed_radii)[None]
        ))),
        "contract": {
            "actor_updated": False,
            "critic_updated": False,
            "radius_selected_on_heldout": False,
            "analytic_dbm_gradient": False,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
    }
    np.savez_compressed(
        args.output_dir / "step_eval.npz",
        context_index=used_index,
        episode=episodes[used_positions],
        split=np.asarray(
            ["internal_validation"] * validation_count
            + ["internal_selection"] * len(heldout_positions)
        ),
        signed_radii_sigma=signed_radii,
        actor_action=actor_action,
        gradient_direction=direction.reshape(-1, 8, 2),
        ensemble_direction_agreement=agreement,
        ensemble_relative_gradient_std=relative_std,
        action=action,
        center=center,
        cost=cost,
        effective_radius_sigma=effective_radius,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Frozen local-gradient step validation\n\n"
        f"Qualification: `{summary['qualification']}`. "
        f"The radius `{chosen_radius:.3f} sigma` was selected only on internal-validation.\n\n"
        f"Heldout base/positive/negative mean cost: "
        f"{np.mean(base):.6f}/{np.mean(positive):.6f}/{np.mean(negative):.6f}.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
