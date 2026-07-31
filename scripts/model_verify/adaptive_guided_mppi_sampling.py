#!/usr/bin/env python3
"""Adaptive, gradient-free sequential guidance on a fixed DBM snapshot.

The total budget remains 256 rollouts.  After two stages, measured center
improvement and empirical-model agreement decide whether the remaining budget
is used in one final batch or split across another fit/update pair.  The DBM is
only called through forward rollouts, so the policy remains portable to Query
and ONNX backends.
"""

from __future__ import annotations

import argparse
import csv
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
from compare_guided_mppi_stage_counts import (
    StrategyRun,
    make_device_independent_antithetic_candidates,
    run_strategy,
    synchronize,
)
from guide_mppi_sampling_from_trajectory_error import (
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
    / "outputs/mppi_sampling_snapshot/adaptive_guided_dbm_step0340"
)
DEFAULT_SEEDS = (3407, 1, 2, 3, 4, 5, 6, 7, 8, 9)


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
    )
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument("--max-standardized-step", type=float, default=1.0)
    parser.add_argument("--minimum-noise-scale", type=float, default=0.06)
    parser.add_argument("--split-improvement-threshold", type=float, default=0.15)
    parser.add_argument("--split-agreement-threshold", type=float, default=0.25)
    parser.add_argument("--split-fit-error-threshold", type=float, default=0.45)
    return parser.parse_args()


def predicted_center_cost(
    evaluation, standardized_step: torch.Tensor, response: torch.Tensor
) -> float:
    baseline_residual = evaluation.residuals[0]
    predicted_residual = baseline_residual + standardized_step.reshape(-1) @ response
    return float(predicted_residual.square().sum())


def proposal_diagnostics(
    baseline_cost: float, predicted_cost: float, actual_cost: float
) -> dict[str, float]:
    predicted_reduction = baseline_cost - predicted_cost
    actual_reduction = baseline_cost - actual_cost
    agreement = actual_reduction / max(abs(predicted_reduction), 1e-6)
    relative_improvement = actual_reduction / max(abs(baseline_cost), 1e-6)
    return {
        "baseline_center_cost": baseline_cost,
        "predicted_center_cost": predicted_cost,
        "actual_center_cost": actual_cost,
        "predicted_reduction": predicted_reduction,
        "actual_reduction": actual_reduction,
        "actual_to_predicted_reduction": agreement,
        "relative_actual_improvement": relative_improvement,
    }


def first_adaptive_scale(relative_fit_error: float) -> float:
    """Use a broader second stage when the first response fit is uncertain."""
    return float(np.clip(0.22 + 0.25 * relative_fit_error, 0.28, 0.40))


def trust_region_scale(
    current_scale: float,
    agreement: float,
    relative_improvement: float,
    current_fit_error: float,
    next_predicted_relative_improvement: float,
    minimum_scale: float,
) -> tuple[float, str]:
    """Adapt the next perturbation radius using trust-region-style feedback."""
    strong_recovery = bool(
        current_fit_error <= 0.20 and next_predicted_relative_improvement >= 0.20
    )
    if strong_recovery:
        factor, reason = 0.35, "high-confidence fit predicts a strong recovery"
    elif agreement < 0.0:
        factor, reason = 0.90, "center worsened: retain exploration radius"
    elif agreement < 0.25:
        factor, reason = 0.80, "poor prediction agreement: shrink cautiously"
    elif agreement < 0.75:
        factor, reason = 0.65, "usable prediction agreement"
    else:
        factor, reason = 0.50, "good prediction agreement"
    if relative_improvement < 0.05 and not strong_recovery:
        factor = max(factor, 0.75)
        reason += "; small gain prevents premature collapse"
    next_scale = float(
        np.clip(current_scale * factor, minimum_scale, current_scale * 0.95)
    )
    return next_scale, reason


def run_adaptive_strategy(
    seed: int,
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
    warm_start: torch.Tensor,
    base_sigma: torch.Tensor,
    args: argparse.Namespace,
) -> StrategyRun:
    rng = np.random.default_rng(seed)
    center = warm_start.clone()
    scale = 1.0
    allocation = [96, 80]
    stages = []
    stage_summaries: list[dict[str, object]] = []
    global_best_knots: torch.Tensor | None = None
    global_best_cost = float("inf")
    pending_prediction: dict[str, float] | None = None
    cumulative_count = 0
    strategy_start = time.perf_counter()

    stage_index = 0
    while stage_index < len(allocation):
        stage_index += 1
        sample_count = allocation[stage_index - 1]
        stage_sigma = base_sigma * scale
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
        cumulative_count += sample_count

        best_index = int(evaluation.cost.argmin())
        stage_best_cost = float(evaluation.cost[best_index])
        if stage_best_cost < global_best_cost:
            global_best_cost = stage_best_cost
            global_best_knots = evaluation.knots[best_index].clone()

        stage_summary = summarize(evaluation, controller.params.temperature)
        stage_summary.update(
            {
                "stage_index": stage_index,
                "sample_count": sample_count,
                "cumulative_sample_count": cumulative_count,
                "noise_scale": scale,
                "noise_sigma": cpu_numpy(stage_sigma).tolist(),
                "cumulative_best_cost": global_best_cost,
                "rollout_seconds": rollout_seconds,
            }
        )

        if pending_prediction is not None:
            proposal = proposal_diagnostics(
                pending_prediction["baseline_center_cost"],
                pending_prediction["predicted_center_cost"],
                float(evaluation.cost[0]),
            )
            stage_summary["previous_center_proposal"] = proposal
        else:
            proposal = None

        # After observing stage 2, decide whether another fitted update is worth
        # reserving half of the remaining 80 candidates.
        if stage_index == 2:
            fit_error_used_for_split = float(
                stage_summaries[0]["guidance"]["relative_weighted_fit_error"]
            )
            should_split = bool(
                proposal is not None
                and proposal["relative_actual_improvement"]
                >= args.split_improvement_threshold
                and proposal["actual_to_predicted_reduction"]
                >= args.split_agreement_threshold
                and fit_error_used_for_split <= args.split_fit_error_threshold
            )
            allocation.extend([40, 40] if should_split else [80])
            stage_summary["remaining_budget_decision"] = {
                "split_remaining_budget": should_split,
                "remaining_allocation": allocation[2:],
                "relative_improvement_threshold": args.split_improvement_threshold,
                "agreement_threshold": args.split_agreement_threshold,
                "fit_error_threshold": args.split_fit_error_threshold,
                "fit_error_used": fit_error_used_for_split,
            }

        if cumulative_count < 256:
            synchronize(controller.device)
            fit_start = time.perf_counter()
            next_center, standardized_step, response, fit_diagnostics = (
                fit_guided_center(
                    evaluation,
                    center,
                    stage_sigma,
                    args.fit_ridge,
                    args.step_damping,
                    args.max_standardized_step,
                )
            )
            synchronize(controller.device)
            stage_summary["guidance_fit_seconds"] = time.perf_counter() - fit_start
            stage_summary["guidance"] = fit_diagnostics
            pending_prediction = {
                "baseline_center_cost": float(evaluation.cost[0]),
                "predicted_center_cost": predicted_center_cost(
                    evaluation, standardized_step, response
                ),
            }
            center = next_center

            if stage_index == 1:
                next_scale = first_adaptive_scale(
                    fit_diagnostics["relative_weighted_fit_error"]
                )
                scale_reason = "first-fit uncertainty rule"
            else:
                assert proposal is not None
                next_predicted_relative_improvement = (
                    pending_prediction["baseline_center_cost"]
                    - pending_prediction["predicted_center_cost"]
                ) / max(abs(pending_prediction["baseline_center_cost"]), 1e-6)
                next_scale, scale_reason = trust_region_scale(
                    scale,
                    proposal["actual_to_predicted_reduction"],
                    proposal["relative_actual_improvement"],
                    fit_diagnostics["relative_weighted_fit_error"],
                    next_predicted_relative_improvement,
                    args.minimum_noise_scale,
                )
                stage_summary["next_predicted_relative_improvement"] = (
                    next_predicted_relative_improvement
                )
            stage_summary["next_noise_scale"] = next_scale
            stage_summary["noise_scale_reason"] = scale_reason
            scale = next_scale

        stage_summaries.append(stage_summary)

    if cumulative_count != 256:
        raise AssertionError(f"adaptive strategy used {cumulative_count} rollouts")
    combined = stages[0]
    for evaluation in stages[1:]:
        combined = concatenate_evaluations(combined, evaluation)

    final = stages[-1]
    final_weight = torch.softmax(
        -(final.cost - final.cost.min()) / controller.params.temperature, dim=0
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
    final_summary = summarize(final, controller.params.temperature)
    final_summary["p10_cost"] = float(torch.quantile(final.cost, 0.10))
    combined_summary = summarize(combined, controller.params.temperature)
    combined_summary["p10_cost"] = float(torch.quantile(combined.cost, 0.10))
    summary = {
        "name": "adaptive",
        "seed": seed,
        "stage_count": len(stages),
        "sample_allocation": allocation,
        "total_candidate_count": len(combined.cost),
        "stages": stage_summaries,
        "combined": combined_summary,
        "final_stage": final_summary,
        "weighted_output": {
            "cost": float(weighted_cost),
            "first_action": cpu_numpy(weighted_action_sequence[0]).tolist(),
            "cost_components": {
                name: float(value[0]) for name, value in weighted_components.items()
            },
        },
        "elapsed_seconds": time.perf_counter() - strategy_start,
    }
    return StrategyRun(
        summary=summary,
        stages=stages,
        combined=combined,
        weighted_action_sequence=weighted_action_sequence,
        weighted_trajectory=weighted_trajectory,
    )


def progress_rows(run: StrategyRun, name: str) -> list[dict[str, object]]:
    rows = []
    collected = []
    cumulative_count = 0
    for index, stage in enumerate(run.stages, start=1):
        cost = cpu_numpy(stage.cost)
        collected.append(cost)
        cumulative = np.concatenate(collected)
        cumulative_count += len(cost)
        rows.append(
            {
                "strategy": name,
                "stage_index": index,
                "stage_samples": len(cost),
                "cumulative_rollouts": cumulative_count,
                "stage_best_cost": float(cost.min()),
                "stage_p10_cost": float(np.percentile(cost, 10)),
                "cumulative_best_cost": float(cumulative.min()),
                "cumulative_p10_cost": float(np.percentile(cumulative, 10)),
            }
        )
    return rows


def save_primary_adaptive_arrays(path: Path, run: StrategyRun) -> None:
    arrays = {
        "combined_cost": cpu_numpy(run.combined.cost),
        "combined_actions": cpu_numpy(run.combined.actions),
        "combined_trajectories": cpu_numpy(run.combined.trajectories),
        "candidate_stage": np.concatenate(
            [
                np.full(len(stage.cost), index, dtype=np.int16)
                for index, stage in enumerate(run.stages, start=1)
            ]
        ),
        "weighted_action_sequence": cpu_numpy(run.weighted_action_sequence),
        "weighted_trajectory": cpu_numpy(run.weighted_trajectory),
    }
    for index, stage in enumerate(run.stages, start=1):
        arrays[f"stage_{index}_knots"] = cpu_numpy(stage.knots)
        arrays[f"stage_{index}_cost"] = cpu_numpy(stage.cost)
    np.savez_compressed(path, **arrays)


def plot_comparison(
    output_dir: Path,
    primary_runs: dict[str, StrategyRun],
    repeated: dict[int, dict[str, StrategyRun]],
) -> None:
    colors = {"fixed-2": "#E69F00", "fixed-3": "#009E73", "fixed-4": "#0072B2", "adaptive": "#D55E00"}
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for name, run in primary_runs.items():
        rows = progress_rows(run, name)
        x = [row["cumulative_rollouts"] for row in rows]
        axes[0].plot(
            x,
            [row["cumulative_best_cost"] for row in rows],
            "o-",
            color=colors[name],
            linewidth=2.2,
            label=f"{name}: {rows[-1]['cumulative_best_cost']:.3f}",
        )
        axes[1].plot(
            x,
            [row["cumulative_p10_cost"] for row in rows],
            "o-",
            color=colors[name],
            linewidth=2.2,
            label=f"{name}: {rows[-1]['cumulative_p10_cost']:.3f}",
        )
    axes[0].set_title("Primary seed — cumulative best cost")
    axes[1].set_title("Primary seed — cumulative P10 cost")
    axes[1].set_yscale("log")
    for axis in axes[:2]:
        axis.set_xlabel("cumulative DBM rollouts")
        axis.set_ylabel("cost (lower is better)")
        axis.grid(True, color="#D8DEE9", alpha=0.8)
        axis.legend(fontsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    seed_values = sorted(repeated)
    fixed_best = [
        repeated[seed]["fixed-3"].summary["combined"]["best_cost"]
        for seed in seed_values
    ]
    adaptive_best = [
        repeated[seed]["adaptive"].summary["combined"]["best_cost"]
        for seed in seed_values
    ]
    for index, (fixed_value, adaptive_value) in enumerate(
        zip(fixed_best, adaptive_best)
    ):
        axes[2].plot(
            [0, 1],
            [fixed_value, adaptive_value],
            "o-",
            color="#999999" if adaptive_value <= fixed_value else "#CC6677",
            alpha=0.8,
        )
        axes[2].annotate(str(seed_values[index]), (1, adaptive_value), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7)
    axes[2].set_xticks([0, 1], ["fixed-3", "adaptive"])
    axes[2].set_ylabel("best cost (lower is better)")
    axes[2].set_title("Each seed — final best cost")
    axes[2].grid(True, axis="y", color="#D8DEE9", alpha=0.8)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    figure.suptitle(
        "Fixed DBM snapshot — adaptive budget and trust-region sigma\n"
        "16 scalar variables; exactly 256 rollouts per strategy",
        fontsize=15,
    )
    figure.savefig(output_dir / "adaptive_guided_comparison.png", bbox_inches="tight")
    figure.savefig(output_dir / "adaptive_guided_comparison.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.snapshot.resolve().parent / "summary.json").read_text())
    snapshot = np.load(args.snapshot)
    device = torch.device(args.device)
    params = TorchMPPIParams(**metadata["mppi_params"])
    weights = TorchMPPICostWeights(**metadata["cost_weights"])
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**metadata["model"]["dbm_params"])
    )
    backend.set_initial_lateral_velocity(float(snapshot["initial_lateral_velocity"]))
    controller = TorchMPPIController(
        backend, params=params, cost_weights=weights, device=device
    )
    history = torch.from_numpy(snapshot["history"]).to(device)
    initial_state = torch.from_numpy(snapshot["initial_state"]).to(device).reshape(1, 5)
    current_action = torch.from_numpy(snapshot["current_action"]).to(device).reshape(1, 2)
    reference = controller._prepare_reference(snapshot["reference"])
    warm_start = torch.from_numpy(snapshot["mean_knots_before"]).to(device)
    base_sigma = torch.tensor(params.noise_sigma, dtype=torch.float32, device=device)

    repeated: dict[int, dict[str, StrategyRun]] = {}
    for seed in seeds:
        repeated[seed] = {}
        for stage_count in (2, 3, 4):
            repeated[seed][f"fixed-{stage_count}"] = run_strategy(
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
                0.10,
            )
        repeated[seed]["adaptive"] = run_adaptive_strategy(
            seed,
            controller,
            backend,
            history,
            initial_state,
            current_action,
            reference,
            warm_start,
            base_sigma,
            args,
        )

    primary = repeated[seeds[0]]
    summary: dict[str, object] = {
        "snapshot": str(args.snapshot.resolve()),
        "backend": "dbm",
        "primary_seed": seeds[0],
        "repeat_seeds": list(seeds),
        "adaptive_rule": {
            "initial_allocation": [96, 80],
            "remaining_budget": 80,
            "split_allocation": [40, 40],
            "unsplit_allocation": [80],
            "split_improvement_threshold": args.split_improvement_threshold,
            "split_agreement_threshold": args.split_agreement_threshold,
            "split_fit_error_threshold": args.split_fit_error_threshold,
            "minimum_noise_scale": args.minimum_noise_scale,
        },
        "primary": {name: run.summary for name, run in primary.items()},
        "per_seed": {
            str(seed): {
                name: {
                    "allocation": run.summary["sample_allocation"],
                    "stage_count": len(run.stages),
                    "best_cost": run.summary["combined"]["best_cost"],
                    "combined_p10_cost": float(torch.quantile(run.combined.cost, 0.10)),
                    "final_stage_p10_cost": float(
                        torch.quantile(run.stages[-1].cost, 0.10)
                    ),
                    "weighted_output_cost": run.summary["weighted_output"]["cost"],
                }
                for name, run in runs.items()
            }
            for seed, runs in repeated.items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for name, run in primary.items():
        rows.extend(progress_rows(run, name))
    with (args.output_dir / "primary_cost_progression.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_primary_adaptive_arrays(
        args.output_dir / "primary_adaptive_results.npz", primary["adaptive"]
    )
    plot_comparison(args.output_dir, primary, repeated)
    for seed in seeds:
        values = summary["per_seed"][str(seed)]
        adaptive = values["adaptive"]
        fixed = values["fixed-3"]
        print(
            f"seed={seed}: adaptive allocation={adaptive['allocation']}, "
            f"best={adaptive['best_cost']:.6f}, "
            f"P10={adaptive['combined_p10_cost']:.6f}; "
            f"fixed-3 best={fixed['best_cost']:.6f}, "
            f"P10={fixed['combined_p10_cost']:.6f}"
        )
    print((args.output_dir / "adaptive_guided_comparison.png").resolve())


if __name__ == "__main__":
    main()
