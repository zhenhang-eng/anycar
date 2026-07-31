#!/usr/bin/env python3
"""Two-round, trajectory-error-guided sampling on a fixed DBM snapshot.

The first round treats batched DBM rollouts as local response experiments.  A
ridge-regression model maps normalized knot perturbations to the complete
weighted trajectory residual used by the existing MPPI cost.  A damped
Gauss-Newton step then supplies the center for a second, narrower round.

Only forward rollouts are required.  The DBM remains under ``torch.no_grad()``,
and the same outer algorithm can later be used with a Query or ONNX backend.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import (
    TorchMPPIController,
    TorchMPPICostWeights,
    TorchMPPIParams,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/snapshot.npz"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/"
    "trajectory_error_guided_dbm_step0340"
)


@dataclass
class RolloutEvaluation:
    knots: torch.Tensor
    actions: torch.Tensor
    trajectories: torch.Tensor
    cost: torch.Tensor
    components: Dict[str, torch.Tensor]
    residuals: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--round-samples", type=int, default=128)
    parser.add_argument(
        "--local-noise-scale",
        type=float,
        default=0.10,
        help="Second-round sigma as a fraction of the original MPPI sigma.",
    )
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument(
        "--max-standardized-step",
        type=float,
        default=1.0,
        help="Per-knot trust region measured in original noise sigmas.",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_antithetic_candidates(
    center: torch.Tensor,
    sigma: torch.Tensor,
    count: int,
    generator: torch.Generator,
    second_anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return an exact-size set with a center, pairs, and one optional anchor."""
    if count < 4 or count % 2:
        raise ValueError("round-samples must be an even integer of at least 4")
    pair_count = (count - 2) // 2
    perturbation = torch.randn(
        pair_count,
        *center.shape,
        generator=generator,
        device=center.device,
    ) * sigma
    if second_anchor is None:
        second_anchor = center + torch.randn(
            center.shape,
            generator=generator,
            device=center.device,
        ) * sigma
    candidates = torch.cat(
        (
            center[None],
            second_anchor[None],
            center[None] + perturbation,
            center[None] - perturbation,
        ),
        dim=0,
    )
    return candidates.clamp(-1.0, 1.0)


def weighted_residuals(
    controller: TorchMPPIController,
    trajectories: torch.Tensor,
    actions: torch.Tensor,
    reference: torch.Tensor,
    current_action: torch.Tensor,
) -> torch.Tensor:
    """Build a residual whose squared norm exactly matches the MPPI cost."""
    weights = controller.cost_weights
    fields = [
        math.sqrt(weights.position)
        * (trajectories[..., 0:2] - reference[None, :, 0:2]),
        math.sqrt(weights.yaw)
        * controller._wrapped_angle_difference(
            trajectories[..., 2], reference[None, :, 2]
        ).unsqueeze(-1),
        math.sqrt(weights.vx)
        * (trajectories[..., 3] - reference[None, :, 3]).unsqueeze(-1),
    ]
    if reference.shape[1] == 5 and weights.yawrate != 0:
        fields.append(
            math.sqrt(weights.yawrate)
            * (trajectories[..., 4] - reference[None, :, 4]).unsqueeze(-1)
        )

    previous_action = torch.cat(
        (
            current_action.expand(actions.shape[0], 1, -1),
            actions[:, :-1],
        ),
        dim=1,
    )
    action_rate = actions - previous_action
    fields.extend(
        (
            math.sqrt(weights.acceleration_rate) * action_rate[..., 0:1],
            math.sqrt(weights.steering_rate) * action_rate[..., 1:2],
        )
    )
    return torch.cat(fields, dim=-1).reshape(actions.shape[0], -1)


@torch.no_grad()
def evaluate_knots(
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    knots: torch.Tensor,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> RolloutEvaluation:
    actions = controller._interpolate_knots(knots)
    trajectories = backend(history, initial_state, current_action, actions).to(
        controller.device
    )
    components = controller.trajectory_cost_components(
        trajectories, actions, reference, current_action
    )
    cost = sum(components.values())
    residuals = weighted_residuals(
        controller, trajectories, actions, reference, current_action
    )
    return RolloutEvaluation(
        knots=knots,
        actions=actions,
        trajectories=trajectories,
        cost=cost,
        components=components,
        residuals=residuals,
    )


def fit_guided_center(
    first_round: RolloutEvaluation,
    warm_start: torch.Tensor,
    sigma: torch.Tensor,
    fit_ridge: float,
    step_damping: float,
    max_standardized_step: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Fit the empirical response and return a trust-region GN center."""
    sample_count = first_round.knots.shape[0]
    normalized_delta = ((first_round.knots - warm_start) / sigma).reshape(
        sample_count, -1
    )
    baseline_residual = first_round.residuals[0]
    residual_delta = first_round.residuals - baseline_residual

    # Smooth cost localization retains directional information from poor
    # samples without allowing extreme nonlinear rollouts to dominate the fit.
    cost_scale = torch.quantile(first_round.cost, 0.5).clamp_min(1e-6)
    fit_weight = torch.exp(
        -(first_round.cost - first_round.cost.min()) / cost_scale
    )
    sqrt_weight = torch.sqrt(fit_weight / fit_weight.mean()).unsqueeze(1)
    weighted_input = normalized_delta * sqrt_weight
    weighted_output = residual_delta * sqrt_weight

    dimension = normalized_delta.shape[1]
    identity = torch.eye(
        dimension, dtype=normalized_delta.dtype, device=normalized_delta.device
    )
    response = torch.linalg.solve(
        weighted_input.T @ weighted_input + fit_ridge * identity,
        weighted_input.T @ weighted_output,
    )
    standardized_step = -torch.linalg.solve(
        response @ response.T + step_damping * identity,
        response @ baseline_residual,
    )
    unclipped_step = standardized_step.clone()
    standardized_step = standardized_step.clamp(
        -max_standardized_step, max_standardized_step
    )
    guided_center = (
        warm_start + standardized_step.reshape_as(warm_start) * sigma
    ).clamp(-1.0, 1.0)

    predicted_delta = normalized_delta @ response
    fit_error = torch.linalg.vector_norm(
        (predicted_delta - residual_delta) * sqrt_weight
    )
    target_norm = torch.linalg.vector_norm(residual_delta * sqrt_weight).clamp_min(
        1e-12
    )
    diagnostics = {
        "cost_localization_scale": float(cost_scale),
        "relative_weighted_fit_error": float(fit_error / target_norm),
        "unclipped_step_l2": float(torch.linalg.vector_norm(unclipped_step)),
        "clipped_step_l2": float(torch.linalg.vector_norm(standardized_step)),
        "unclipped_step_max_abs": float(unclipped_step.abs().max()),
        "clipped_step_max_abs": float(standardized_step.abs().max()),
    }
    return guided_center, standardized_step, response, diagnostics


def concatenate_evaluations(
    first: RolloutEvaluation, second: RolloutEvaluation
) -> RolloutEvaluation:
    names = set(first.components) | set(second.components)
    if set(first.components) != set(second.components):
        raise ValueError("rounds returned different cost components")
    return RolloutEvaluation(
        knots=torch.cat((first.knots, second.knots)),
        actions=torch.cat((first.actions, second.actions)),
        trajectories=torch.cat((first.trajectories, second.trajectories)),
        cost=torch.cat((first.cost, second.cost)),
        components={
            name: torch.cat((first.components[name], second.components[name]))
            for name in names
        },
        residuals=torch.cat((first.residuals, second.residuals)),
    )


def summarize(
    evaluation: RolloutEvaluation, temperature: float
) -> dict[str, object]:
    cost = evaluation.cost
    weight = torch.softmax(-(cost - cost.min()) / temperature, dim=0)
    best_index = int(cost.argmin())
    return {
        "candidate_count": len(cost),
        "best_index": best_index,
        "best_cost": float(cost[best_index]),
        "mean_cost": float(cost.mean()),
        "median_cost": float(torch.quantile(cost, 0.5)),
        "p95_cost": float(torch.quantile(cost, 0.95)),
        "max_cost": float(cost.max()),
        "effective_sample_size": float(1.0 / weight.square().sum()),
        "count_cost_lt_5": int((cost < 5.0).sum()),
        "count_cost_lt_10": int((cost < 10.0).sum()),
        "count_cost_lt_20": int((cost < 20.0).sum()),
        "count_cost_lt_50": int((cost < 50.0).sum()),
        "best_cost_components": {
            name: float(value[best_index])
            for name, value in evaluation.components.items()
        },
    }


def cpu_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def save_evaluation_arrays(
    output_path: Path,
    first: RolloutEvaluation,
    second: RolloutEvaluation,
    combined: RolloutEvaluation,
    guided_center: torch.Tensor,
    standardized_step: torch.Tensor,
    response: torch.Tensor,
    guided_weighted_action_sequence: torch.Tensor,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "guided_center_knots": cpu_numpy(guided_center),
        "standardized_guidance_step": cpu_numpy(standardized_step),
        "empirical_response": cpu_numpy(response),
        "guided_weighted_action_sequence": cpu_numpy(
            guided_weighted_action_sequence
        ),
    }
    for prefix, evaluation in (
        ("first", first),
        ("second", second),
        ("combined", combined),
    ):
        arrays[f"{prefix}_knots"] = cpu_numpy(evaluation.knots)
        arrays[f"{prefix}_actions"] = cpu_numpy(evaluation.actions)
        arrays[f"{prefix}_trajectories"] = cpu_numpy(evaluation.trajectories)
        arrays[f"{prefix}_cost"] = cpu_numpy(evaluation.cost)
        arrays[f"{prefix}_residuals"] = cpu_numpy(evaluation.residuals)
        for name, value in evaluation.components.items():
            arrays[f"{prefix}_cost_{name}"] = cpu_numpy(value)
    np.savez_compressed(output_path, **arrays)


def style_axis(axis) -> None:
    axis.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_comparison(
    output_dir: Path,
    snapshot: np.lib.npyio.NpzFile,
    first: RolloutEvaluation,
    second: RolloutEvaluation,
    combined: RolloutEvaluation,
    summary: dict[str, object],
) -> None:
    baseline_cost = snapshot["cost"]
    baseline_trajectory = snapshot["predicted_trajectories"]
    baseline_actions = snapshot["sampled_action_sequences"]
    reference = snapshot["reference"]
    state = snapshot["initial_state"]
    current_action = snapshot["current_action"]
    first_np = cpu_numpy(first.trajectories)
    second_np = cpu_numpy(second.trajectories)
    first_actions = cpu_numpy(first.actions)
    second_actions = cpu_numpy(second.actions)
    combined_cost = cpu_numpy(combined.cost)
    second_cost = cpu_numpy(second.cost)
    best_baseline = int(np.argmin(baseline_cost))
    best_guided = int(np.argmin(combined_cost))
    best_guided_trajectory = cpu_numpy(combined.trajectories[best_guided])
    best_guided_actions = cpu_numpy(combined.actions[best_guided])
    warm_trajectory = first_np[0]
    warm_actions = first_actions[0]
    time_axis = (np.arange(50) + 1) * 0.05

    colors = {
        "baseline_ensemble": "#888888",
        "guided_ensemble": "#56B4E9",
        "warm": "#009E73",
        "baseline": "#D55E00",
        "guided": "#0072B2",
        "reference": "#111111",
    }

    def add_candidate_ensemble(
        axis,
        baseline_values: np.ndarray,
        guided_values: np.ndarray,
    ) -> None:
        axis.add_collection(
            LineCollection(
                [
                    np.column_stack((time_axis, values))
                    for values in baseline_values
                ],
                colors=colors["baseline_ensemble"],
                linewidths=0.42,
                alpha=0.07,
                label="baseline candidates",
            )
        )
        axis.add_collection(
            LineCollection(
                [
                    np.column_stack((time_axis, values))
                    for values in guided_values
                ],
                colors=colors["guided_ensemble"],
                linewidths=0.50,
                alpha=0.13,
                label="guided round 2",
            )
        )
        axis.autoscale()

    def add_representative_lines(
        axis,
        warm_values: np.ndarray,
        baseline_values: np.ndarray,
        guided_values: np.ndarray,
        reference_values: np.ndarray | None = None,
    ) -> None:
        if reference_values is not None:
            axis.plot(
                time_axis,
                reference_values,
                color=colors["reference"],
                linestyle="--",
                linewidth=2.0,
                label="reference",
            )
        axis.plot(
            time_axis,
            warm_values,
            color=colors["warm"],
            linewidth=1.7,
            label="warm-start",
        )
        axis.plot(
            time_axis,
            baseline_values,
            color=colors["baseline"],
            linewidth=1.9,
            label="baseline best",
        )
        axis.plot(
            time_axis,
            guided_values,
            color=colors["guided"],
            linewidth=2.2,
            label="guided best",
        )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 190,
        }
    )
    figure, axes = plt.subplots(
        3, 3, figsize=(16, 13.2), constrained_layout=True
    )

    # Row 1: predicted path and complete action sequences.
    axis = axes[0, 0]
    baseline_paths = [
        np.vstack((state[None, :2], trajectory[:, :2]))
        for trajectory in baseline_trajectory
    ]
    guided_paths = [
        np.vstack((state[None, :2], trajectory[:, :2]))
        for trajectory in second_np
    ]
    axis.add_collection(
        LineCollection(
            baseline_paths,
            colors=colors["baseline_ensemble"],
            linewidths=0.42,
            alpha=0.07,
            label="baseline candidates",
        )
    )
    axis.add_collection(
        LineCollection(
            guided_paths,
            colors=colors["guided_ensemble"],
            linewidths=0.50,
            alpha=0.13,
            label="guided round 2",
        )
    )
    axis.plot(
        reference[:, 0],
        reference[:, 1],
        color=colors["reference"],
        linestyle="--",
        linewidth=2.2,
        label="reference",
    )
    baseline_path = np.vstack(
        (state[None, :2], baseline_trajectory[best_baseline, :, :2])
    )
    warm_path = np.vstack((state[None, :2], warm_trajectory[:, :2]))
    guided_path = np.vstack((state[None, :2], best_guided_trajectory[:, :2]))
    axis.plot(
        warm_path[:, 0],
        warm_path[:, 1],
        color=colors["warm"],
        linewidth=1.7,
        label="warm-start",
    )
    axis.plot(
        baseline_path[:, 0],
        baseline_path[:, 1],
        color=colors["baseline"],
        linewidth=2,
        label=f"baseline best ({baseline_cost[best_baseline]:.3f})",
    )
    axis.plot(
        guided_path[:, 0],
        guided_path[:, 1],
        color=colors["guided"],
        linewidth=2.2,
        label=f"guided best ({combined_cost[best_guided]:.3f})",
    )
    axis.scatter(
        state[0],
        state[1],
        marker="*",
        color=colors["reference"],
        s=90,
        label="fixed state",
    )
    axis.autoscale()
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title("DBM-predicted XY trajectories")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.legend(ncol=2)
    style_axis(axis)

    for axis, action_index, title, ylabel in (
        (
            axes[0, 1],
            0,
            "Candidate acceleration sequences",
            "normalized acceleration",
        ),
        (
            axes[0, 2],
            1,
            "Candidate steering sequences",
            "normalized steering",
        ),
    ):
        add_candidate_ensemble(
            axis,
            baseline_actions[:, :, action_index],
            second_actions[:, :, action_index],
        )
        add_representative_lines(
            axis,
            warm_actions[:, action_index],
            baseline_actions[best_baseline, :, action_index],
            best_guided_actions[:, action_index],
        )
        axis.scatter(
            [0],
            [current_action[action_index]],
            marker="*",
            color=colors["reference"],
            s=48,
            label="current action",
            zorder=5,
        )
        axis.axhline(-1, color="#777777", linestyle=":", linewidth=0.9)
        axis.axhline(1, color="#777777", linestyle=":", linewidth=0.9)
        axis.set_title(title)
        axis.set_xlabel("prediction time [s]")
        axis.set_ylabel(ylabel)
        axis.set_ylim(-1.08, 1.08)
        axis.legend(ncol=2)
        style_axis(axis)

    # Row 2: all predicted state channels shown in the earlier sampling plot.
    baseline_yaw = np.stack(
        [
            np.unwrap(np.concatenate(([state[2]], trajectory[:, 2])))[1:]
            for trajectory in baseline_trajectory
        ]
    )
    guided_yaw = np.stack(
        [
            np.unwrap(np.concatenate(([state[2]], trajectory[:, 2])))[1:]
            for trajectory in second_np
        ]
    )
    warm_yaw = np.unwrap(
        np.concatenate(([state[2]], warm_trajectory[:, 2]))
    )[1:]
    baseline_best_yaw = np.unwrap(
        np.concatenate(([state[2]], baseline_trajectory[best_baseline, :, 2]))
    )[1:]
    guided_best_yaw = np.unwrap(
        np.concatenate(([state[2]], best_guided_trajectory[:, 2]))
    )[1:]
    reference_yaw_full = np.unwrap(reference[:, 2])
    reference_yaw = reference_yaw_full[1:]

    axis = axes[1, 0]
    add_candidate_ensemble(axis, baseline_yaw, guided_yaw)
    add_representative_lines(
        axis,
        warm_yaw,
        baseline_best_yaw,
        guided_best_yaw,
        reference_yaw,
    )
    axis.scatter(
        [0],
        [state[2]],
        marker="*",
        color=colors["reference"],
        s=48,
        label="current state",
        zorder=5,
    )
    axis.set_title("Predicted yaw")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("unwrapped yaw [rad]")
    axis.legend(ncol=2)
    style_axis(axis)

    axis = axes[1, 1]
    add_candidate_ensemble(
        axis, baseline_trajectory[:, :, 3], second_np[:, :, 3]
    )
    add_representative_lines(
        axis,
        warm_trajectory[:, 3],
        baseline_trajectory[best_baseline, :, 3],
        best_guided_trajectory[:, 3],
        reference[1:, 3],
    )
    axis.scatter(
        [0],
        [state[3]],
        marker="*",
        color=colors["reference"],
        s=48,
        label="current state",
        zorder=5,
    )
    axis.set_title("Predicted longitudinal speed")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("vx [m/s]")
    axis.legend(ncol=2)
    style_axis(axis)

    axis = axes[1, 2]
    reference_yawrate = np.diff(reference_yaw_full) / 0.05
    add_candidate_ensemble(
        axis, baseline_trajectory[:, :, 4], second_np[:, :, 4]
    )
    add_representative_lines(
        axis,
        warm_trajectory[:, 4],
        baseline_trajectory[best_baseline, :, 4],
        best_guided_trajectory[:, 4],
        reference_yawrate,
    )
    axis.scatter(
        [0],
        [state[4]],
        marker="*",
        color=colors["reference"],
        s=48,
        label="current state",
        zorder=5,
    )
    axis.set_title("Predicted yaw rate")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("yaw rate [rad/s]")
    axis.legend(ncol=2)
    style_axis(axis)

    # Row 3: comparison-only statistics.
    axis = axes[2, 0]
    rank_baseline = np.arange(1, len(baseline_cost) + 1)
    rank_combined = np.arange(1, len(combined_cost) + 1)
    rank_round = np.arange(1, len(second_cost) + 1)
    axis.plot(
        rank_baseline,
        np.sort(baseline_cost),
        color="#777777",
        linewidth=2,
        label="baseline: 256 broad",
    )
    axis.plot(
        rank_combined,
        np.sort(combined_cost),
        color=colors["guided"],
        linewidth=2,
        label="guided: 128 + 128",
    )
    axis.plot(
        rank_round,
        np.sort(second_cost),
        color=colors["warm"],
        linestyle="--",
        linewidth=1.6,
        label="guided round 2 only",
    )
    axis.set_yscale("log")
    axis.set_title("Comparison — equal-budget cost rank")
    axis.set_xlabel("candidate rank")
    axis.set_ylabel("total cost (log scale)")
    axis.legend()
    style_axis(axis)

    axis = axes[2, 1]

    def tracking_stage_cost(trajectory: np.ndarray) -> np.ndarray:
        position = 5.0 * np.square(
            trajectory[:, :, :2] - reference[None, 1:, :2]
        ).sum(axis=-1)
        yaw_delta = trajectory[:, :, 2] - reference[None, 1:, 2]
        yaw_delta = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
        yaw = 5.0 * np.square(yaw_delta)
        speed = np.square(trajectory[:, :, 3] - reference[None, 1:, 3])
        return position + yaw + speed

    warm_stage = tracking_stage_cost(first_np[0:1])[0]
    baseline_stage = tracking_stage_cost(
        baseline_trajectory[best_baseline : best_baseline + 1]
    )[0]
    guided_stage = tracking_stage_cost(best_guided_trajectory[None])[0]
    axis.plot(
        time_axis, warm_stage, color=colors["warm"], label="warm-start"
    )
    axis.plot(
        time_axis,
        baseline_stage,
        color=colors["baseline"],
        label="baseline best",
    )
    axis.plot(
        time_axis,
        guided_stage,
        color=colors["guided"],
        linewidth=2,
        label="guided best",
    )
    axis.set_yscale("log")
    axis.set_title("Comparison — tracking cost over horizon")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("position + yaw + speed cost")
    axis.legend()
    style_axis(axis)

    axis = axes[2, 2]
    thresholds = np.array([5, 10, 20, 50])
    baseline_counts = [(baseline_cost < threshold).sum() for threshold in thresholds]
    guided_counts = [(combined_cost < threshold).sum() for threshold in thresholds]
    position = np.arange(len(thresholds))
    width = 0.36
    baseline_bars = axis.bar(
        position - width / 2,
        baseline_counts,
        width,
        color="#999999",
        label="baseline",
    )
    guided_bars = axis.bar(
        position + width / 2,
        guided_counts,
        width,
        color="#0072B2",
        label="guided",
    )
    axis.bar_label(baseline_bars, padding=2, fontsize=8)
    axis.bar_label(guided_bars, padding=2, fontsize=8)
    axis.set_xticks(position, [f"cost < {value}" for value in thresholds])
    axis.set_title("Comparison — useful candidates")
    axis.set_ylabel("candidate count out of 256")
    axis.legend()
    style_axis(axis)

    baseline_summary = summary["baseline"]
    guided_summary = summary["combined"]
    figure.suptitle(
        "Fixed clean DBM snapshot — trajectory-error-guided sampling\n"
        f"best cost {baseline_summary['best_cost']:.3f} → "
        f"{guided_summary['best_cost']:.3f}; "
        f"median {baseline_summary['median_cost']:.1f} → "
        f"{guided_summary['median_cost']:.1f}; "
        f"cost<10: {baseline_summary['count_cost_lt_10']} → "
        f"{guided_summary['count_cost_lt_10']}",
        fontsize=14,
    )
    png_path = output_dir / "trajectory_error_guided_sampling.png"
    svg_path = output_dir / "trajectory_error_guided_sampling.svg"
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)


def snapshot_baseline_summary(
    snapshot: np.lib.npyio.NpzFile,
    temperature: float,
) -> dict[str, object]:
    cost = torch.from_numpy(snapshot["cost"])
    weight = torch.softmax(-(cost - cost.min()) / temperature, dim=0)
    best_index = int(cost.argmin())
    component_names = [
        key.removeprefix("cost_")
        for key in snapshot.files
        if key.startswith("cost_")
    ]
    return {
        "candidate_count": len(cost),
        "best_index": best_index,
        "best_cost": float(cost[best_index]),
        "mean_cost": float(cost.mean()),
        "median_cost": float(torch.quantile(cost, 0.5)),
        "p95_cost": float(torch.quantile(cost, 0.95)),
        "max_cost": float(cost.max()),
        "effective_sample_size": float(1.0 / weight.square().sum()),
        "count_cost_lt_5": int((cost < 5.0).sum()),
        "count_cost_lt_10": int((cost < 10.0).sum()),
        "count_cost_lt_20": int((cost < 20.0).sum()),
        "count_cost_lt_50": int((cost < 50.0).sum()),
        "best_cost_components": {
            name: float(snapshot[f"cost_{name}"][best_index])
            for name in component_names
        },
    }


@torch.no_grad()
def evaluate_weighted_output(
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    actions: torch.Tensor,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    actions = actions.reshape(1, controller.params.horizon, 2)
    trajectory = backend(
        history, initial_state, current_action, actions
    ).to(controller.device)
    components = controller.trajectory_cost_components(
        trajectory, actions, reference, current_action
    )
    total = sum(components.values())
    return {
        "cost": float(total[0]),
        "first_action": cpu_numpy(actions[0, 0]).tolist(),
        "cost_components": {
            name: float(value[0]) for name, value in components.items()
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(
        (args.snapshot.resolve().parent / "summary.json").read_text()
    )
    if metadata["model"]["backend"] != "dbm":
        raise ValueError("this fixed experiment requires a DBM snapshot")

    snapshot = np.load(args.snapshot)
    device = torch.device(args.device)
    params = TorchMPPIParams(**metadata["mppi_params"])
    cost_weights = TorchMPPICostWeights(**metadata["cost_weights"])
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**metadata["model"]["dbm_params"])
    )
    backend.set_initial_lateral_velocity(
        float(snapshot["initial_lateral_velocity"])
    )
    controller = TorchMPPIController(
        backend, params=params, cost_weights=cost_weights, device=device
    )
    history = torch.from_numpy(snapshot["history"]).to(device)
    initial_state = torch.from_numpy(snapshot["initial_state"]).to(device).reshape(
        1, 5
    )
    current_action = torch.from_numpy(snapshot["current_action"]).to(device).reshape(
        1, 2
    )
    reference = controller._prepare_reference(snapshot["reference"])
    warm_start = torch.from_numpy(snapshot["mean_knots_before"]).to(device)
    sigma = torch.tensor(params.noise_sigma, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    first_knots = make_antithetic_candidates(
        warm_start, sigma, args.round_samples, generator
    )
    synchronize(device)
    first_start = time.perf_counter()
    first = evaluate_knots(
        controller,
        backend,
        first_knots,
        history,
        initial_state,
        current_action,
        reference,
    )
    synchronize(device)
    first_seconds = time.perf_counter() - first_start

    fit_start = time.perf_counter()
    guided_center, standardized_step, response, fit_diagnostics = (
        fit_guided_center(
            first,
            warm_start,
            sigma,
            args.fit_ridge,
            args.step_damping,
            args.max_standardized_step,
        )
    )
    synchronize(device)
    fit_seconds = time.perf_counter() - fit_start

    first_best = first.knots[int(first.cost.argmin())]
    second_knots = make_antithetic_candidates(
        guided_center,
        sigma * args.local_noise_scale,
        args.round_samples,
        generator,
        second_anchor=first_best,
    )
    synchronize(device)
    second_start = time.perf_counter()
    second = evaluate_knots(
        controller,
        backend,
        second_knots,
        history,
        initial_state,
        current_action,
        reference,
    )
    synchronize(device)
    second_seconds = time.perf_counter() - second_start
    combined = concatenate_evaluations(first, second)
    combined_weight = torch.softmax(
        -(combined.cost - combined.cost.min()) / params.temperature, dim=0
    )
    guided_weighted_action_sequence = torch.sum(
        combined_weight[:, None, None] * combined.actions, dim=0
    )
    baseline_weighted_action_sequence = torch.from_numpy(
        snapshot["optimized_action_sequence"]
    ).to(device)
    baseline_weighted_output = evaluate_weighted_output(
        controller,
        backend,
        baseline_weighted_action_sequence,
        history,
        initial_state,
        current_action,
        reference,
    )
    guided_weighted_output = evaluate_weighted_output(
        controller,
        backend,
        guided_weighted_action_sequence,
        history,
        initial_state,
        current_action,
        reference,
    )

    exact_cost_error = float(
        torch.max(
            torch.abs(
                combined.cost - combined.residuals.square().sum(dim=1)
            )
        )
    )
    summary = {
        "snapshot": str(args.snapshot.resolve()),
        "backend": "dbm",
        "algorithm": "weighted empirical residual response + damped Gauss-Newton",
        "gradient_required": False,
        "seed": args.seed,
        "rollout_budget": {
            "baseline": int(len(snapshot["cost"])),
            "guided_first_round": args.round_samples,
            "guided_second_round": args.round_samples,
            "guided_total": 2 * args.round_samples,
        },
        "parameters": {
            "original_noise_sigma": list(params.noise_sigma),
            "local_noise_scale": args.local_noise_scale,
            "fit_ridge": args.fit_ridge,
            "step_damping": args.step_damping,
            "max_standardized_step": args.max_standardized_step,
        },
        "fit": fit_diagnostics,
        "timing_seconds": {
            "first_round_rollout": first_seconds,
            "response_fit_and_step": fit_seconds,
            "second_round_rollout": second_seconds,
            "total": first_seconds + fit_seconds + second_seconds,
        },
        "cost_residual_identity_max_abs_error": exact_cost_error,
        "guided_center_cost": float(second.cost[0]),
        "diagnostic_weighted_output_rollouts": {
            "note": (
                "These two verification rollouts are not counted in the "
                "256-candidate optimization budget."
            ),
            "baseline": baseline_weighted_output,
            "guided": guided_weighted_output,
        },
        "baseline": snapshot_baseline_summary(snapshot, params.temperature),
        "first_round": summarize(first, params.temperature),
        "second_round": summarize(second, params.temperature),
        "combined": summarize(combined, params.temperature),
    }
    summary["improvement"] = {
        "best_cost_absolute": (
            summary["baseline"]["best_cost"] - summary["combined"]["best_cost"]
        ),
        "best_cost_percent": 100.0
        * (
            summary["baseline"]["best_cost"] - summary["combined"]["best_cost"]
        )
        / summary["baseline"]["best_cost"],
        "median_cost_percent": 100.0
        * (
            summary["baseline"]["median_cost"]
            - summary["combined"]["median_cost"]
        )
        / summary["baseline"]["median_cost"],
        "additional_cost_lt_10": (
            summary["combined"]["count_cost_lt_10"]
            - summary["baseline"]["count_cost_lt_10"]
        ),
    }

    save_evaluation_arrays(
        args.output_dir / "guided_sampling.npz",
        first,
        second,
        combined,
        guided_center,
        standardized_step,
        response,
        guided_weighted_action_sequence,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    plot_comparison(
        args.output_dir,
        snapshot,
        first,
        second,
        combined,
        summary,
    )
    print(json.dumps(summary, indent=2))
    print((args.output_dir / "trajectory_error_guided_sampling.png").resolve())


if __name__ == "__main__":
    main()
