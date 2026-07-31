#!/usr/bin/env python3
"""Compare one to four sequential guided-MPPI stages at a fixed budget.

All strategies use the same 256 DBM rollouts and the current eight temporal
knots (16 scalar control variables).  Every non-final stage fits the empirical
trajectory-residual response and moves the center with the damped
Gauss-Newton update implemented by
``guide_mppi_sampling_from_trajectory_error.py``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
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
from guide_mppi_sampling_from_trajectory_error import (
    RolloutEvaluation,
    concatenate_evaluations,
    cpu_numpy,
    evaluate_knots,
    fit_guided_center,
    summarize,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/snapshot.npz"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/"
    "guided_stage_count_comparison_dbm_step0340"
)
DEFAULT_SEEDS = (3407, 1, 2, 3, 4, 5, 6, 7, 8, 9)


@dataclass
class StrategyRun:
    summary: dict[str, object]
    stages: list[RolloutEvaluation]
    combined: RolloutEvaluation
    weighted_action_sequence: torch.Tensor
    weighted_trajectory: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds; the first seed is used for detailed plots.",
    )
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument("--max-standardized-step", type=float, default=1.0)
    parser.add_argument("--final-noise-scale", type=float, default=0.10)
    return parser.parse_args()


def sample_allocation(stage_count: int) -> list[int]:
    allocations = {
        1: [256],
        2: [128, 128],
        3: [86, 86, 84],
        4: [64, 64, 64, 64],
    }
    return allocations[stage_count]


def noise_scale_schedule(stage_count: int, final_scale: float) -> np.ndarray:
    if stage_count == 1:
        return np.ones(1, dtype=np.float32)
    return np.geomspace(1.0, final_scale, stage_count).astype(np.float32)


def make_device_independent_antithetic_candidates(
    center: torch.Tensor,
    sigma: torch.Tensor,
    count: int,
    rng: np.random.Generator,
    second_anchor: torch.Tensor | None,
    action_min: tuple[float, float],
    action_max: tuple[float, float],
) -> torch.Tensor:
    """Generate identical float32 candidates regardless of rollout device."""
    if count < 4 or count % 2:
        raise ValueError("stage sample counts must be even integers >= 4")
    center_np = cpu_numpy(center).astype(np.float32, copy=False)
    sigma_np = cpu_numpy(sigma).astype(np.float32, copy=False)
    action_min_np = np.asarray(action_min, dtype=np.float32)
    action_max_np = np.asarray(action_max, dtype=np.float32)

    # Draw the extra anchor first so the broad-stage random prefix is shared
    # across the 1/2/3/4-stage comparisons as far as their sample counts allow.
    if second_anchor is None:
        extra = rng.standard_normal(center_np.shape).astype(np.float32)
        second_np = center_np + extra * sigma_np
    else:
        second_np = cpu_numpy(second_anchor).astype(np.float32, copy=False)

    pair_count = (count - 2) // 2
    direction = rng.standard_normal(
        (pair_count, *center_np.shape)
    ).astype(np.float32)
    perturbation = direction * sigma_np
    candidates = np.concatenate(
        (
            center_np[None],
            second_np[None],
            center_np[None] + perturbation,
            center_np[None] - perturbation,
        ),
        axis=0,
    )
    candidates = np.clip(candidates, action_min_np, action_max_np)
    return torch.from_numpy(candidates).to(center.device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_strategy(
    stage_count: int,
    seed: int,
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
    warm_start: torch.Tensor,
    base_sigma: torch.Tensor,
    fit_ridge: float,
    step_damping: float,
    max_standardized_step: float,
    final_noise_scale: float,
) -> StrategyRun:
    allocation = sample_allocation(stage_count)
    scales = noise_scale_schedule(stage_count, final_noise_scale)
    rng = np.random.default_rng(seed)
    center = warm_start.clone()
    global_best_knots: torch.Tensor | None = None
    global_best_cost = float("inf")
    stages: list[RolloutEvaluation] = []
    stage_summaries: list[dict[str, object]] = []
    cumulative_count = 0
    cumulative_best = float("inf")

    synchronize(controller.device)
    strategy_start = time.perf_counter()
    for stage_index, (sample_count, scale) in enumerate(
        zip(allocation, scales), start=1
    ):
        stage_sigma = base_sigma * float(scale)
        knots = make_device_independent_antithetic_candidates(
            center,
            stage_sigma,
            sample_count,
            rng,
            global_best_knots,
            controller.params.action_min,
            controller.params.action_max,
        )
        synchronize(controller.device)
        rollout_start = time.perf_counter()
        evaluation = evaluate_knots(
            controller,
            backend,
            knots,
            history,
            initial_state,
            current_action,
            reference,
        )
        synchronize(controller.device)
        rollout_seconds = time.perf_counter() - rollout_start
        stages.append(evaluation)

        best_index = int(evaluation.cost.argmin())
        stage_best = float(evaluation.cost[best_index])
        if stage_best < global_best_cost:
            global_best_cost = stage_best
            global_best_knots = evaluation.knots[best_index].clone()
        cumulative_count += sample_count
        cumulative_best = min(cumulative_best, stage_best)

        stage_summary = summarize(evaluation, controller.params.temperature)
        stage_summary.update(
            {
                "stage_index": stage_index,
                "sample_count": sample_count,
                "cumulative_sample_count": cumulative_count,
                "noise_scale": float(scale),
                "noise_sigma": cpu_numpy(stage_sigma).tolist(),
                "cumulative_best_cost": cumulative_best,
                "rollout_seconds": rollout_seconds,
            }
        )

        if stage_index < stage_count:
            synchronize(controller.device)
            fit_start = time.perf_counter()
            next_center, _, _, fit_diagnostics = fit_guided_center(
                evaluation,
                center,
                stage_sigma,
                fit_ridge,
                step_damping,
                max_standardized_step,
            )
            synchronize(controller.device)
            stage_summary["guidance_fit_seconds"] = (
                time.perf_counter() - fit_start
            )
            stage_summary["guidance"] = fit_diagnostics
            center = next_center
        stage_summaries.append(stage_summary)

    combined = stages[0]
    for evaluation in stages[1:]:
        combined = concatenate_evaluations(combined, evaluation)
    if len(combined.cost) != 256:
        raise AssertionError("every strategy must consume exactly 256 candidates")

    final = stages[-1]
    final_weight = torch.softmax(
        -(final.cost - final.cost.min()) / controller.params.temperature,
        dim=0,
    )
    weighted_action_sequence = torch.sum(
        final_weight[:, None, None] * final.actions, dim=0
    )
    weighted_trajectory = backend(
        history,
        initial_state,
        current_action,
        weighted_action_sequence[None],
    ).to(controller.device)[0]
    weighted_components = controller.trajectory_cost_components(
        weighted_trajectory[None],
        weighted_action_sequence[None],
        reference,
        current_action,
    )
    weighted_cost = sum(weighted_components.values())[0]
    synchronize(controller.device)
    elapsed_seconds = time.perf_counter() - strategy_start

    combined_summary = summarize(combined, controller.params.temperature)
    final_summary = summarize(final, controller.params.temperature)
    final_summary["normalized_effective_sample_size"] = (
        final_summary["effective_sample_size"] / len(final.cost)
    )
    summary = {
        "stage_count": stage_count,
        "seed": seed,
        "sample_allocation": allocation,
        "noise_scale_schedule": scales.tolist(),
        "total_candidate_count": len(combined.cost),
        "stages": stage_summaries,
        "combined": combined_summary,
        "final_stage": final_summary,
        "weighted_output": {
            "cost": float(weighted_cost),
            "first_action": cpu_numpy(weighted_action_sequence[0]).tolist(),
            "cost_components": {
                name: float(value[0])
                for name, value in weighted_components.items()
            },
        },
        "elapsed_seconds": elapsed_seconds,
    }
    return StrategyRun(
        summary=summary,
        stages=stages,
        combined=combined,
        weighted_action_sequence=weighted_action_sequence,
        weighted_trajectory=weighted_trajectory,
    )


def aggregate_seed_results(
    seed_results: dict[int, dict[int, StrategyRun]]
) -> dict[str, object]:
    stage_counts = sorted(next(iter(seed_results.values())))
    aggregate: dict[str, object] = {}
    best_by_seed = {
        seed: min(
            result.summary["combined"]["best_cost"]
            for result in results.values()
        )
        for seed, results in seed_results.items()
    }
    for stage_count in stage_counts:
        results = [
            seed_results[seed][stage_count].summary
            for seed in seed_results
        ]

        def statistics(values) -> dict[str, float]:
            values_np = np.asarray(list(values), dtype=np.float64)
            return {
                "mean": float(values_np.mean()),
                "std": float(values_np.std()),
                "min": float(values_np.min()),
                "max": float(values_np.max()),
            }

        aggregate[str(stage_count)] = {
            "trial_count": len(results),
            "best_cost": statistics(
                result["combined"]["best_cost"] for result in results
            ),
            "weighted_output_cost": statistics(
                result["weighted_output"]["cost"] for result in results
            ),
            "final_stage_median_cost": statistics(
                result["final_stage"]["median_cost"] for result in results
            ),
            "combined_count_cost_lt_10": statistics(
                result["combined"]["count_cost_lt_10"] for result in results
            ),
            "final_stage_normalized_ess": statistics(
                result["final_stage"]["normalized_effective_sample_size"]
                for result in results
            ),
            "best_cost_win_count": sum(
                abs(result["combined"]["best_cost"] - best_by_seed[result["seed"]])
                <= 1e-7
                for result in results
            ),
        }
    return aggregate


def save_primary_arrays(
    output_path: Path, primary_runs: dict[int, StrategyRun]
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for stage_count, result in primary_runs.items():
        prefix = f"stages_{stage_count}"
        arrays[f"{prefix}_combined_cost"] = cpu_numpy(result.combined.cost)
        arrays[f"{prefix}_combined_actions"] = cpu_numpy(result.combined.actions)
        arrays[f"{prefix}_combined_trajectories"] = cpu_numpy(
            result.combined.trajectories
        )
        arrays[f"{prefix}_weighted_action_sequence"] = cpu_numpy(
            result.weighted_action_sequence
        )
        arrays[f"{prefix}_weighted_trajectory"] = cpu_numpy(
            result.weighted_trajectory
        )
        arrays[f"{prefix}_candidate_stage"] = np.concatenate(
            [
                np.full(len(stage.cost), index, dtype=np.int16)
                for index, stage in enumerate(result.stages, start=1)
            ]
        )
        for index, stage in enumerate(result.stages, start=1):
            arrays[f"{prefix}_stage_{index}_knots"] = cpu_numpy(stage.knots)
            arrays[f"{prefix}_stage_{index}_cost"] = cpu_numpy(stage.cost)
    np.savez_compressed(output_path, **arrays)


def write_primary_csv(
    path: Path, primary_runs: dict[int, StrategyRun]
) -> None:
    fields = [
        "stage_count",
        "allocation",
        "best_cost",
        "weighted_output_cost",
        "combined_median_cost",
        "final_median_cost",
        "final_p95_cost",
        "combined_cost_lt_5",
        "combined_cost_lt_10",
        "combined_cost_lt_20",
        "final_ess",
        "final_normalized_ess",
        "elapsed_seconds",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for stage_count, run in primary_runs.items():
            summary = run.summary
            writer.writerow(
                {
                    "stage_count": stage_count,
                    "allocation": "+".join(
                        str(value) for value in summary["sample_allocation"]
                    ),
                    "best_cost": summary["combined"]["best_cost"],
                    "weighted_output_cost": summary["weighted_output"]["cost"],
                    "combined_median_cost": summary["combined"]["median_cost"],
                    "final_median_cost": summary["final_stage"]["median_cost"],
                    "final_p95_cost": summary["final_stage"]["p95_cost"],
                    "combined_cost_lt_5": summary["combined"]["count_cost_lt_5"],
                    "combined_cost_lt_10": summary["combined"]["count_cost_lt_10"],
                    "combined_cost_lt_20": summary["combined"]["count_cost_lt_20"],
                    "final_ess": summary["final_stage"]["effective_sample_size"],
                    "final_normalized_ess": summary["final_stage"][
                        "normalized_effective_sample_size"
                    ],
                    "elapsed_seconds": summary["elapsed_seconds"],
                }
            )


def style_axis(axis) -> None:
    axis.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_results(
    output_dir: Path,
    snapshot: np.lib.npyio.NpzFile,
    primary_runs: dict[int, StrategyRun],
    aggregate: dict[str, object],
    primary_seed: int,
) -> None:
    stage_counts = np.asarray(sorted(primary_runs), dtype=np.int64)
    colors = {
        1: "#777777",
        2: "#E69F00",
        3: "#009E73",
        4: "#0072B2",
    }
    baseline_cost = snapshot["cost"]
    baseline_best_index = int(np.argmin(baseline_cost))
    historical_best = float(baseline_cost[baseline_best_index])
    reference = snapshot["reference"]
    initial_state = snapshot["initial_state"]

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
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    axis = axes[0, 0]
    for stage_count, run in primary_runs.items():
        x = [stage["cumulative_sample_count"] for stage in run.summary["stages"]]
        y = [stage["cumulative_best_cost"] for stage in run.summary["stages"]]
        axis.plot(
            x,
            y,
            "o-",
            color=colors[stage_count],
            linewidth=2,
            label=f"{stage_count} stage" + ("s" if stage_count > 1 else ""),
        )
    axis.scatter(
        [256],
        [historical_best],
        marker="*",
        color="#CC79A7",
        s=80,
        label="saved Gaussian baseline",
        zorder=5,
    )
    axis.set_yscale("log")
    axis.set_title("Primary seed — cumulative best cost")
    axis.set_xlabel("cumulative DBM rollouts")
    axis.set_ylabel("best cost so far")
    axis.set_xticks([64, 86, 128, 172, 192, 256])
    axis.legend()
    style_axis(axis)

    axis = axes[0, 1]
    best = [primary_runs[value].summary["combined"]["best_cost"] for value in stage_counts]
    output = [primary_runs[value].summary["weighted_output"]["cost"] for value in stage_counts]
    position = np.arange(len(stage_counts))
    width = 0.36
    best_bars = axis.bar(
        position - width / 2,
        best,
        width,
        color="#56B4E9",
        label="best candidate",
    )
    output_bars = axis.bar(
        position + width / 2,
        output,
        width,
        color="#0072B2",
        label="weighted output",
    )
    axis.axhline(
        historical_best,
        color="#CC79A7",
        linestyle="--",
        linewidth=1.5,
        label=f"saved baseline best={historical_best:.3f}",
    )
    axis.bar_label(best_bars, fmt="%.3f", padding=2, fontsize=8)
    axis.bar_label(output_bars, fmt="%.3f", padding=2, fontsize=8)
    axis.set_xticks(position, [str(value) for value in stage_counts])
    axis.set_title("Primary seed — final solution quality")
    axis.set_xlabel("sequential stage count")
    axis.set_ylabel("cost")
    axis.legend()
    style_axis(axis)

    axis = axes[0, 2]
    aggregate_best_mean = np.asarray(
        [aggregate[str(value)]["best_cost"]["mean"] for value in stage_counts]
    )
    aggregate_best_std = np.asarray(
        [aggregate[str(value)]["best_cost"]["std"] for value in stage_counts]
    )
    aggregate_output_mean = np.asarray(
        [aggregate[str(value)]["weighted_output_cost"]["mean"] for value in stage_counts]
    )
    aggregate_output_std = np.asarray(
        [aggregate[str(value)]["weighted_output_cost"]["std"] for value in stage_counts]
    )
    axis.errorbar(
        stage_counts - 0.05,
        aggregate_best_mean,
        yerr=aggregate_best_std,
        fmt="o-",
        capsize=4,
        color="#009E73",
        linewidth=2,
        label="best candidate",
    )
    axis.errorbar(
        stage_counts + 0.05,
        aggregate_output_mean,
        yerr=aggregate_output_std,
        fmt="s-",
        capsize=4,
        color="#0072B2",
        linewidth=2,
        label="weighted output",
    )
    axis.set_xticks(stage_counts)
    axis.set_title("10-seed repeatability (mean ± std)")
    axis.set_xlabel("sequential stage count")
    axis.set_ylabel("cost")
    axis.legend()
    style_axis(axis)

    axis = axes[1, 0]
    final_median = np.asarray(
        [primary_runs[value].summary["final_stage"]["median_cost"] for value in stage_counts]
    )
    final_p95 = np.asarray(
        [primary_runs[value].summary["final_stage"]["p95_cost"] for value in stage_counts]
    )
    median_bars = axis.bar(
        position - width / 2,
        final_median,
        width,
        color="#56B4E9",
        label="final-stage median",
    )
    p95_bars = axis.bar(
        position + width / 2,
        final_p95,
        width,
        color="#E69F00",
        label="final-stage P95",
    )
    axis.set_yscale("log")
    axis.set_xticks(position, [str(value) for value in stage_counts])
    axis.set_title("Final-stage cost distribution")
    axis.set_xlabel("sequential stage count")
    axis.set_ylabel("cost (log scale)")
    axis.legend(loc="upper right")
    style_axis(axis)
    ess_axis = axis.twinx()
    normalized_ess = [
        primary_runs[value].summary["final_stage"]["normalized_effective_sample_size"]
        for value in stage_counts
    ]
    ess_axis.plot(
        position,
        normalized_ess,
        "D--",
        color="#7E57C2",
        label="normalized ESS",
    )
    ess_axis.set_ylabel("ESS / final-stage samples")
    ess_axis.set_ylim(0, max(normalized_ess) * 1.35)
    lines = axis.get_legend_handles_labels()
    lines_ess = ess_axis.get_legend_handles_labels()
    axis.legend(lines[0] + lines_ess[0], lines[1] + lines_ess[1], loc="upper right")

    axis = axes[1, 1]
    for threshold, marker in ((5, "o"), (10, "s"), (20, "D")):
        values = [
            primary_runs[count].summary["combined"][f"count_cost_lt_{threshold}"]
            for count in stage_counts
        ]
        axis.plot(
            stage_counts,
            values,
            marker=marker,
            linewidth=2,
            label=f"cost < {threshold}",
        )
        for x_value, y_value in zip(stage_counts, values):
            axis.annotate(
                str(y_value),
                (x_value, y_value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_xticks(stage_counts)
    axis.set_title("Useful candidates across all 256 rollouts")
    axis.set_xlabel("sequential stage count")
    axis.set_ylabel("candidate count")
    axis.legend()
    style_axis(axis)

    axis = axes[1, 2]
    axis.plot(
        reference[:, 0],
        reference[:, 1],
        color="#111111",
        linestyle="--",
        linewidth=2.2,
        label="reference",
    )
    historical_path = np.vstack(
        (
            initial_state[None, :2],
            snapshot["predicted_trajectories"][baseline_best_index, :, :2],
        )
    )
    axis.plot(
        historical_path[:, 0],
        historical_path[:, 1],
        color="#CC79A7",
        linewidth=1.6,
        label="saved Gaussian best",
    )
    for stage_count, run in primary_runs.items():
        path = np.vstack(
            (initial_state[None, :2], cpu_numpy(run.weighted_trajectory)[:, :2])
        )
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=colors[stage_count],
            linewidth=2,
            label=f"{stage_count}-stage weighted",
        )
    axis.scatter(
        initial_state[0],
        initial_state[1],
        marker="*",
        color="#111111",
        s=80,
        label="fixed state",
        zorder=5,
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title("Primary seed — weighted-output trajectories")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.legend(ncol=2)
    style_axis(axis)

    best_four = aggregate["4"]["best_cost"]
    figure.suptitle(
        "Fixed DBM snapshot — splitting 256 rollouts into sequential guidance stages\n"
        f"8 temporal knots / 16 scalar variables; primary seed={primary_seed}; "
        f"4-stage 10-seed best cost={best_four['mean']:.3f}±{best_four['std']:.3f}",
        fontsize=14,
    )
    figure.savefig(output_dir / "guided_stage_count_comparison.png", bbox_inches="tight")
    figure.savefig(output_dir / "guided_stage_count_comparison.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    if not 0 < args.final_noise_scale <= 1:
        raise ValueError("final-noise-scale must be within (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        (args.snapshot.resolve().parent / "summary.json").read_text()
    )
    if metadata["model"]["backend"] != "dbm":
        raise ValueError("stage-count comparison requires a fixed DBM snapshot")
    snapshot = np.load(args.snapshot)
    device = torch.device(args.device)
    params = TorchMPPIParams(**metadata["mppi_params"])
    if params.num_knots != 8:
        raise ValueError("this comparison fixes 8 temporal knots / 16 variables")
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
    initial_state = torch.from_numpy(snapshot["initial_state"]).to(device).reshape(1, 5)
    current_action = torch.from_numpy(snapshot["current_action"]).to(device).reshape(1, 2)
    reference = controller._prepare_reference(snapshot["reference"])
    warm_start = torch.from_numpy(snapshot["mean_knots_before"]).to(device)
    base_sigma = torch.tensor(params.noise_sigma, dtype=torch.float32, device=device)

    seed_results: dict[int, dict[int, StrategyRun]] = {}
    for seed in seeds:
        seed_results[seed] = {}
        for stage_count in (1, 2, 3, 4):
            seed_results[seed][stage_count] = run_strategy(
                stage_count,
                seed,
                controller,
                backend,
                history,
                initial_state,
                current_action,
                reference,
                warm_start,
                base_sigma,
                args.fit_ridge,
                args.step_damping,
                args.max_standardized_step,
                args.final_noise_scale,
            )

    primary_seed = seeds[0]
    primary_runs = seed_results[primary_seed]
    aggregate = aggregate_seed_results(seed_results)
    original_best_index = int(np.argmin(snapshot["cost"]))
    summary = {
        "snapshot": str(args.snapshot.resolve()),
        "backend": "dbm",
        "control_parameterization": {
            "temporal_knots": params.num_knots,
            "action_channels": params.action_dim,
            "scalar_variables": params.num_knots * params.action_dim,
            "horizon_steps": params.horizon,
        },
        "comparison": {
            "total_rollouts_per_strategy": 256,
            "allocations": {
                str(stage_count): sample_allocation(stage_count)
                for stage_count in (1, 2, 3, 4)
            },
            "noise_schedules": {
                str(stage_count): noise_scale_schedule(
                    stage_count, args.final_noise_scale
                ).tolist()
                for stage_count in (1, 2, 3, 4)
            },
            "sampler": "device-independent NumPy antithetic pairs",
            "next_stage_center": "weighted empirical response + damped Gauss-Newton",
            "final_output_weights": "last stage only",
        },
        "parameters": {
            "base_noise_sigma": list(params.noise_sigma),
            "final_noise_scale": args.final_noise_scale,
            "fit_ridge": args.fit_ridge,
            "step_damping": args.step_damping,
            "max_standardized_step": args.max_standardized_step,
        },
        "historical_saved_gaussian_baseline": {
            "candidate_count": len(snapshot["cost"]),
            "best_cost": float(snapshot["cost"][original_best_index]),
            "best_index": original_best_index,
            "note": (
                "Reference only: it uses the original CUDA Gaussian sample set, "
                "not the controlled antithetic samples used for the stage-count ablation."
            ),
        },
        "primary_seed": primary_seed,
        "primary": {
            str(stage_count): run.summary
            for stage_count, run in primary_runs.items()
        },
        "repeat_seeds": list(seeds),
        "aggregate": aggregate,
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    save_primary_arrays(
        args.output_dir / "primary_seed_results.npz", primary_runs
    )
    write_primary_csv(
        args.output_dir / "primary_seed_comparison.csv", primary_runs
    )
    plot_results(
        args.output_dir, snapshot, primary_runs, aggregate, primary_seed
    )
    print(json.dumps(summary, indent=2))
    print((args.output_dir / "guided_stage_count_comparison.png").resolve())


if __name__ == "__main__":
    main()
