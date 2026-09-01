#!/usr/bin/env python3
"""Evaluate rollout-feedback-guided 16-D Direct-Actor exploration.

The experiment is deliberately train-only.  It freezes the selected residual
Actor and uses only the disjoint ``internal_selection`` episodes.  For every
context, a shared first pass evaluates the Actor center and all 16 antithetic
Hadamard directions with the deterministic DBM direct objective.  The resulting
costs and complete weighted trajectory residuals define two forward-only local
response models:

* scalar cost response: ``delta_cost ~= delta_u @ gradient``;
* trajectory response: ``delta_residual ~= delta_u @ response`` followed by a
  damped Gauss--Newton direction.

The second pass evaluates bounded line candidates along the cost, trajectory,
and blended directions.  A same-budget blind baseline instead evaluates fixed
pseudorandom antithetic directions around the best first-pass center.  Every
reported score is a real DBM rollout cost; fitted values never count as results.
No DBM analytic gradient is used, so the outer algorithm is Query compatible.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor
from generate_dbm_direct_gt_validation import interpolate_knots
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import (
    actor_inputs,
    distribution,
    hadamard_directions,
    load_j16,
    make_base_policy,
    residual_outputs,
)
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_ACTOR = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_feedback_guided_exploration_20260811_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-radius-sigma", type=float, default=0.10)
    parser.add_argument(
        "--second-radii-sigma", default="0.025,0.05,0.10,0.15,0.25,0.40"
    )
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument("--maximum-gn-step-sigma", type=float, default=0.40)
    parser.add_argument("--blind-radius-sigma", type=float, default=0.10)
    parser.add_argument("--blind-seed", type=int, default=160811)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-contexts", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalize_rms(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(torch.mean(vector.square(), dim=1, keepdim=True)).clamp_min(1e-8)
    return vector / norm


def stable_solve(matrix: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(matrix, right)
    except torch.linalg.LinAlgError:
        return torch.linalg.lstsq(matrix, right).solution


@torch.no_grad()
def rollout_cost_residuals(
    centers: np.ndarray,
    data: Any,
    tensors: dict[str, Any],
    context_index: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact direct cost and a residual whose squared norm is that cost."""
    params = TorchMPPIParams(**data.mppi_params)
    weights = TorchMPPICostWeights(**data.cost_weights)
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**data.dbm_params))
    center = torch.from_numpy(np.asarray(centers, np.float32)).to(device)
    index = torch.from_numpy(np.asarray(context_index, np.int64)).to(device)
    batch, samples = center.shape[:2]
    actions = interpolate_knots(center, params.horizon)
    flat_actions = actions.reshape(batch * samples, params.horizon, 2)
    initial = tensors["initial_state_six"][index]
    initial = initial[:, None].expand(-1, samples, -1).reshape(batch * samples, 6)
    full = backend.rollout_full_state_differentiable(initial, flat_actions)
    trajectory = full[..., [0, 1, 2, 3, 5]].reshape(
        batch, samples, params.horizon, 5
    )
    reference = tensors["direct_reference"][index]
    current_action = tensors["current_action"][index]

    fields = [
        math.sqrt(weights.position)
        * (trajectory[..., :2] - reference[:, None, :, :2]),
        math.sqrt(weights.yaw)
        * torch.atan2(
            torch.sin(trajectory[..., 2] - reference[:, None, :, 2]),
            torch.cos(trajectory[..., 2] - reference[:, None, :, 2]),
        ).unsqueeze(-1),
        math.sqrt(weights.vx)
        * (trajectory[..., 3] - reference[:, None, :, 3]).unsqueeze(-1),
    ]
    if reference.shape[-1] == 5 and weights.yawrate != 0:
        fields.append(
            math.sqrt(weights.yawrate)
            * (trajectory[..., 4] - reference[:, None, :, 4]).unsqueeze(-1)
        )
    previous = torch.cat(
        (
            current_action[:, None, None, :].expand(-1, samples, 1, -1),
            actions[:, :, :-1],
        ),
        dim=2,
    )
    rate = actions - previous
    fields.extend(
        (
            math.sqrt(weights.acceleration_rate) * rate[..., 0:1],
            math.sqrt(weights.steering_rate) * rate[..., 1:2],
        )
    )
    residual = torch.cat(fields, dim=-1).reshape(batch, samples, -1)
    cost = residual.square().sum(dim=-1)
    return cost.cpu().numpy().astype(np.float32), residual.cpu().numpy().astype(np.float32)


def first_pass_bank(
    base: np.ndarray,
    sigma: np.ndarray,
    directions: np.ndarray,
    radius: float,
) -> np.ndarray:
    delta = radius * sigma[:, None, None, :] * directions[None]
    return np.clip(
        np.concatenate((base[:, None], base[:, None] + delta, base[:, None] - delta), axis=1),
        -1.0,
        1.0,
    ).astype(np.float32)


def fit_response_directions(
    centers: np.ndarray,
    base: np.ndarray,
    sigma: np.ndarray,
    cost: np.ndarray,
    residual: np.ndarray,
    fit_ridge: float,
    step_damping: float,
    maximum_gn_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    batch = len(base)
    x = torch.from_numpy(
        ((centers - base[:, None]) / sigma[:, None, None, :]).reshape(batch, len(centers[0]), 16)
    ).float()
    one_cost = torch.from_numpy(cost).float()
    one_residual = torch.from_numpy(residual).float()
    delta_cost = one_cost - one_cost[:, :1]
    delta_residual = one_residual - one_residual[:, :1]

    # Robust localization keeps extreme nonlinear probes from dominating while
    # retaining their direction/sign information.
    scale = torch.quantile(delta_cost.abs(), 0.5, dim=1, keepdim=True).clamp_min(0.25)
    weight = 1.0 / (1.0 + (delta_cost.abs() / scale).square())
    sqrt_weight = torch.sqrt(weight / weight.mean(dim=1, keepdim=True)).unsqueeze(-1)
    wx = x * sqrt_weight
    identity = torch.eye(16).expand(batch, -1, -1)
    lhs = wx.transpose(1, 2) @ wx + fit_ridge * identity
    cost_gradient = stable_solve(
        lhs, wx.transpose(1, 2) @ (delta_cost.unsqueeze(-1) * sqrt_weight)
    ).squeeze(-1)
    response = stable_solve(
        lhs, wx.transpose(1, 2) @ (delta_residual * sqrt_weight)
    )

    base_residual = one_residual[:, 0]
    gn_lhs = response @ response.transpose(1, 2) + step_damping * identity
    gn_rhs = (response @ base_residual.unsqueeze(-1)).squeeze(-1)
    gn_step = -stable_solve(gn_lhs, gn_rhs.unsqueeze(-1)).squeeze(-1)
    gn_rho = torch.sqrt(torch.mean(gn_step.square(), dim=1)).clamp_min(1e-8)
    gn_step = gn_step * torch.minimum(
        torch.ones_like(gn_rho), torch.full_like(gn_rho, maximum_gn_step) / gn_rho
    ).unsqueeze(1)

    cost_direction = normalize_rms(-cost_gradient)
    trajectory_direction = normalize_rms(gn_step)
    cosine = torch.sum(cost_direction * trajectory_direction, dim=1) / 16.0
    aligned_trajectory = torch.where(
        (cosine < 0).unsqueeze(1), -trajectory_direction, trajectory_direction
    )
    blended_direction = normalize_rms(cost_direction + aligned_trajectory)

    predicted_cost = torch.sum(x * cost_gradient[:, None], dim=-1)
    cost_fit_error = torch.sqrt(
        torch.sum(weight * (predicted_cost - delta_cost).square(), dim=1)
        / torch.sum(weight * delta_cost.square(), dim=1).clamp_min(1e-8)
    )
    predicted_residual = x @ response
    residual_fit_error = torch.linalg.vector_norm(
        (predicted_residual - delta_residual) * sqrt_weight, dim=(1, 2)
    ) / torch.linalg.vector_norm(delta_residual * sqrt_weight, dim=(1, 2)).clamp_min(1e-8)
    diagnostics = {
        "cost_trajectory_direction_cosine": cosine.numpy(),
        "cost_relative_fit_error": cost_fit_error.numpy(),
        "trajectory_relative_fit_error": residual_fit_error.numpy(),
        "gn_step_rho_source_sigma": gn_rho.numpy(),
    }
    return (
        cost_direction.numpy().astype(np.float32),
        trajectory_direction.numpy().astype(np.float32),
        blended_direction.numpy().astype(np.float32),
        diagnostics,
    )


def line_bank(
    base: np.ndarray,
    sigma: np.ndarray,
    direction: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    raw = base[:, None] + radii[None, :, None, None] * sigma[:, None, None, :] * direction[:, None].reshape(-1, 1, 8, 2)
    return np.clip(raw, -1.0, 1.0).astype(np.float32)


def blind_bank(
    first_best: np.ndarray,
    sigma: np.ndarray,
    candidate_count: int,
    radius: float,
    seed: int,
) -> np.ndarray:
    if candidate_count % 2:
        raise ValueError("blind candidate count must be even")
    rng = np.random.default_rng(seed)
    batch = len(first_best)
    pair_count = candidate_count // 2
    raw = rng.standard_normal((batch, pair_count, 16)).astype(np.float32)
    raw /= np.sqrt(np.mean(raw * raw, axis=2, keepdims=True)).clip(1e-8)
    direction = raw.reshape(batch, pair_count, 8, 2)
    delta = radius * sigma[:, None, None, :] * direction
    return np.clip(
        np.concatenate((first_best[:, None] + delta, first_best[:, None] - delta), axis=1),
        -1.0,
        1.0,
    ).astype(np.float32)


def paired_metrics(base: np.ndarray, cost: np.ndarray) -> dict[str, Any]:
    gain = base - cost
    return {
        "cost": distribution(cost),
        "gain_vs_actor": distribution(gain),
        "actor_beaten_fraction": float(np.mean(gain > 1e-6)),
        "actor_regression_fraction": float(np.mean(gain < -1e-6)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    radii = np.asarray([float(value) for value in args.second_radii_sigma.split(",")], np.float32)
    if len(radii) == 0 or np.any(radii <= 0):
        raise ValueError("second radii must be positive")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    actor_payload = torch.load(args.actor, map_location="cpu")
    if actor_payload.get("actor_class") != "TorchMPPIDeterministicCenterActor":
        raise AssertionError("requested checkpoint is not a deterministic residual Actor")
    labels = Path(actor_payload["labels"])
    alpha_path = Path(actor_payload["base_alpha_checkpoint"])
    alpha_payload = torch.load(alpha_path, map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, splits = load_dataset(labels, old_payload)
    selection_episodes = list(splits["internal_selection"])
    index = np.flatnonzero(np.isin(data.episodes, selection_episodes))
    if args.max_contexts:
        index = index[: args.max_contexts]
    if not len(index):
        raise AssertionError("empty internal-selection split")
    if set(data.episodes[index]) & set(splits.get("validation", [])):
        raise AssertionError("formal validation leaked into the experiment")

    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy,
        tensors,
        extra,
        np.arange(len(data.episodes)),
        float(actor_payload["base_move_threshold"]),
        args.batch_size,
        device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    actor = TorchMPPIDeterministicCenterActor(
        float(actor_payload["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(actor_payload["actor_state_dict"], strict=True)
    actor.eval()
    _, actor_center_all = residual_outputs(
        actor, inputs, np.arange(len(data.episodes)), args.batch_size, device
    )
    actor_center = actor_center_all[index]
    actor_cost = direct_cost(actor_center, data, tensors, index, args.batch_size, device)
    saved_actor_mean = float(actor_payload["internal_selection_metrics"]["direct_cost"]["mean"])
    if not args.max_contexts and abs(float(actor_cost.mean()) - saved_actor_mean) > 5e-5:
        raise AssertionError("frozen Actor cost does not reproduce its checkpoint")

    directions = hadamard_directions()
    sigma = data.sigma[index]
    first_best_parts: list[np.ndarray] = []
    blind_best_parts: list[np.ndarray] = []
    cost_best_parts: list[np.ndarray] = []
    trajectory_best_parts: list[np.ndarray] = []
    response_best_parts: list[np.ndarray] = []
    diagnostic_parts: dict[str, list[np.ndarray]] = {}
    rank_values: list[int] = []
    residual_cost_error = 0.0

    for start in range(0, len(index), args.batch_size):
        stop = min(start + args.batch_size, len(index))
        one_index = index[start:stop]
        one_base = actor_center[start:stop]
        one_sigma = sigma[start:stop]
        first_centers = first_pass_bank(
            one_base, one_sigma, directions, args.first_radius_sigma
        )
        first_cost, first_residual = rollout_cost_residuals(
            first_centers, data, tensors, one_index, device
        )
        residual_cost_error = max(
            residual_cost_error,
            float(np.max(np.abs(first_cost[:, 0] - actor_cost[start:stop]))),
        )
        for one_centers, base, one_source_sigma in zip(first_centers, one_base, one_sigma):
            design = ((one_centers - base) / one_source_sigma[None, None, :]).reshape(33, 16)
            rank_values.append(int(np.linalg.matrix_rank(design[1:])))

        cost_direction, trajectory_direction, blended_direction, diagnostics = (
            fit_response_directions(
                first_centers,
                one_base,
                one_sigma,
                first_cost,
                first_residual,
                args.fit_ridge,
                args.step_damping,
                args.maximum_gn_step_sigma,
            )
        )
        for name, value in diagnostics.items():
            diagnostic_parts.setdefault(name, []).append(value)

        cost_centers = line_bank(one_base, one_sigma, cost_direction, radii)
        trajectory_centers = line_bank(one_base, one_sigma, trajectory_direction, radii)
        blended_centers = line_bank(one_base, one_sigma, blended_direction, radii)
        response_centers = np.concatenate(
            (cost_centers, trajectory_centers, blended_centers), axis=1
        )
        response_cost, _ = rollout_cost_residuals(
            response_centers, data, tensors, one_index, device
        )
        cost_line_cost = response_cost[:, : len(radii)]
        trajectory_line_cost = response_cost[:, len(radii): 2 * len(radii)]

        row = np.arange(stop - start)
        first_argmin = np.argmin(first_cost, axis=1)
        one_first_best_center = first_centers[row, first_argmin]
        one_first_best = first_cost[row, first_argmin]
        blind_centers = blind_bank(
            one_first_best_center,
            one_sigma,
            response_centers.shape[1],
            args.blind_radius_sigma,
            args.blind_seed + start,
        )
        blind_cost, _ = rollout_cost_residuals(
            blind_centers, data, tensors, one_index, device
        )

        first_best_parts.append(one_first_best)
        blind_best_parts.append(np.minimum(one_first_best, blind_cost.min(axis=1)))
        cost_best_parts.append(np.minimum(one_first_best, cost_line_cost.min(axis=1)))
        trajectory_best_parts.append(
            np.minimum(one_first_best, trajectory_line_cost.min(axis=1))
        )
        response_best_parts.append(np.minimum(one_first_best, response_cost.min(axis=1)))
        print(
            f"[{stop:04d}/{len(index):04d}] actor/first/blind/response="
            f"{actor_cost[:stop].mean():.6f}/"
            f"{np.concatenate(first_best_parts).mean():.6f}/"
            f"{np.concatenate(blind_best_parts).mean():.6f}/"
            f"{np.concatenate(response_best_parts).mean():.6f}",
            flush=True,
        )

    values = {
        "actor": actor_cost,
        "first_pass_best": np.concatenate(first_best_parts),
        "blind_two_pass_best": np.concatenate(blind_best_parts),
        "cost_response_best": np.concatenate(cost_best_parts),
        "trajectory_response_best": np.concatenate(trajectory_best_parts),
        "combined_response_best": np.concatenate(response_best_parts),
    }
    diagnostics = {name: np.concatenate(parts) for name, parts in diagnostic_parts.items()}
    j16_center, j16_cost_all, j16_hashes = load_j16(
        labels,
        [Path(path) for path in actor_payload["j16_summaries"]],
        len(data.episodes),
    )
    del j16_center
    values["j16_best_found"] = j16_cost_all[index]

    metrics = {name: paired_metrics(actor_cost, cost) for name, cost in values.items()}
    first_best = values["first_pass_best"]
    for name in (
        "blind_two_pass_best", "cost_response_best",
        "trajectory_response_best", "combined_response_best",
    ):
        incremental = first_best - values[name]
        metrics[name]["incremental_gain_vs_first_pass"] = distribution(incremental)
        metrics[name]["second_pass_improvement_fraction"] = float(np.mean(incremental > 1e-6))
    grouped: dict[str, Any] = {}
    for speed in sorted(np.unique(data.reference_speed[index])):
        mask = np.isclose(data.reference_speed[index], speed)
        grouped[f"{float(speed):.1f}"] = {
            "context_count": int(np.sum(mask)),
            **{f"{name}_cost_mean": float(np.mean(cost[mask])) for name, cost in values.items()},
        }

    context_path = args.output_dir / "per_context.npz"
    np.savez_compressed(
        context_path,
        context_index=index,
        episode=data.episodes[index],
        reference_speed=data.reference_speed[index],
        scenario=data.scenario[index],
        **values,
        **diagnostics,
    )
    with (args.output_dir / "per_context.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "context_index", "episode", "reference_speed", "scenario",
                *values.keys(), *diagnostics.keys(),
            ],
        )
        writer.writeheader()
        for row in range(len(index)):
            writer.writerow({
                "context_index": int(index[row]),
                "episode": str(data.episodes[index[row]]),
                "reference_speed": float(data.reference_speed[index[row]]),
                "scenario": str(data.scenario[index[row]]),
                **{name: float(value[row]) for name, value in values.items()},
                **{name: float(value[row]) for name, value in diagnostics.items()},
            })

    labels_plot = [
        ("Actor", "actor", "#4c78a8"),
        ("First pass", "first_pass_best", "#9c9c9c"),
        ("Blind 2-pass", "blind_two_pass_best", "#f58518"),
        ("Cost response", "cost_response_best", "#e45756"),
        ("Trajectory response", "trajectory_response_best", "#54a24b"),
        ("Combined response", "combined_response_best", "#2ca02c"),
        ("J16 best-found", "j16_best_found", "#9467bd"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    means = [float(values[key].mean()) for _, key, _ in labels_plot]
    axes[0, 0].bar(
        np.arange(len(labels_plot)), means, color=[color for _, _, color in labels_plot]
    )
    axes[0, 0].set_xticks(np.arange(len(labels_plot)), [name for name, _, _ in labels_plot], rotation=25, ha="right")
    axes[0, 0].set_ylabel("Mean direct cost (lower is better)")
    axes[0, 0].set_title("Same fixed internal-selection contexts")
    axes[0, 0].bar_label(axes[0, 0].containers[0], fmt="%.3f", fontsize=8)

    for name, key, color in labels_plot[1:-1]:
        gain = actor_cost - values[key]
        ordered = np.sort(gain)
        axes[0, 1].plot(
            ordered, np.linspace(0, 1, len(ordered)), label=name, color=color
        )
    axes[0, 1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[0, 1].set_xlabel("Direct-cost gain versus Actor")
    axes[0, 1].set_ylabel("Empirical CDF")
    axes[0, 1].set_title("Per-context improvement distribution")
    axes[0, 1].legend(fontsize=8)

    speed_values = sorted(np.unique(data.reference_speed[index]))
    for name, key, color in labels_plot[:6]:
        axes[1, 0].plot(
            speed_values,
            [float(values[key][np.isclose(data.reference_speed[index], speed)].mean()) for speed in speed_values],
            marker="o", label=name, color=color,
        )
    axes[1, 0].set_xlabel("Reference speed (m/s)")
    axes[1, 0].set_ylabel("Mean direct cost")
    axes[1, 0].set_title("Reference-speed groups")
    axes[1, 0].legend(fontsize=8)

    blind_increment = first_best - values["blind_two_pass_best"]
    response_increment = first_best - values["combined_response_best"]
    axes[1, 1].scatter(
        blind_increment, response_increment,
        c=data.reference_speed[index], cmap="viridis", s=14, alpha=0.65,
    )
    bound = float(max(np.quantile(np.abs(blind_increment), 0.99), np.quantile(np.abs(response_increment), 0.99), 1e-3))
    axes[1, 1].plot([-bound, bound], [-bound, bound], "k--", linewidth=1)
    axes[1, 1].set_xlim(-0.05 * bound, bound)
    axes[1, 1].set_ylim(-0.05 * bound, bound)
    axes[1, 1].set_xlabel("Blind second-pass gain over first pass")
    axes[1, 1].set_ylabel("Response second-pass gain over first pass")
    axes[1, 1].set_title("Equal second-pass budget; above diagonal favors response")
    figure.suptitle(
        "Forward-rollout feedback-guided 16-D exploration (DBM, no analytic gradient)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "feedback_guided_exploration.png", dpi=180)
    figure.savefig(args.output_dir / "feedback_guided_exploration.svg")
    plt.close(figure)

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "two-pass forward-rollout cost/trajectory-response exploration",
        "qualification": "INTERNAL_SELECTION_MECHANISM_ONLY",
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "actor": str(args.actor.resolve()),
        "actor_sha256": sha256_file(args.actor),
        "labels": str(labels.resolve()),
        "split": "internal_selection only",
        "context_count": int(len(index)),
        "episode_count": int(len(set(data.episodes[index]))),
        "candidate_budget": {
            "shared_first_pass": 33,
            "blind_second_pass": int(3 * len(radii)),
            "combined_response_second_pass": int(3 * len(radii)),
            "total_per_strategy": int(33 + 3 * len(radii)),
        },
        "arguments": {
            **vars(args),
            "actor": str(args.actor),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "second_radii_sigma": radii.tolist(),
        },
        "metrics": metrics,
        "grouped_by_reference_speed": grouped,
        "diagnostics": {
            name: distribution(value) for name, value in diagnostics.items()
        },
        "invariants": {
            "minimum_first_pass_design_rank": int(min(rank_values)),
            "actor_vs_first_center_cost_max_abs_error": residual_cost_error,
            "analytic_dbm_gradient": False,
            "every_reported_cost_is_forward_replayed": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "j16_summary_sha256": j16_hashes,
        },
        "test_policy": "formal validation and test remain sealed",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
