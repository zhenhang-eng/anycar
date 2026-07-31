#!/usr/bin/env python3
"""Run fixed and adaptive sequential guidance with the Query rollout model.

The experiment reuses the clean DBM step-340 state, history, reference, warm
start, cost, and 256-rollout budget.  Only the rollout backend is replaced by
the small-car deterministic Query checkpoint.  No Query gradients are used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
from car_foundation.query_deployment import (
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)
from adaptive_guided_mppi_sampling import (
    progress_rows,
    run_adaptive_strategy,
)
from compare_guided_mppi_stage_counts import StrategyRun, run_strategy
from guide_mppi_sampling_from_trajectory_error import cpu_numpy, evaluate_knots


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/snapshot.npz"
)
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/query_guided_on_dbm_state_step0340"
)
DEFAULT_SEEDS = (3407, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument("--max-standardized-step", type=float, default=1.0)
    parser.add_argument("--minimum-noise-scale", type=float, default=0.06)
    parser.add_argument("--split-improvement-threshold", type=float, default=0.15)
    parser.add_argument("--split-agreement-threshold", type=float, default=0.25)
    parser.add_argument("--split-fit-error-threshold", type=float, default=0.55)
    return parser.parse_args()


def metric_summary(
    run: StrategyRun, dbm_cross_evaluation: dict[str, float]
) -> dict[str, object]:
    return {
        "allocation": run.summary["sample_allocation"],
        "stage_count": len(run.stages),
        "best_cost": run.summary["combined"]["best_cost"],
        "combined_p10_cost": float(torch.quantile(run.combined.cost, 0.10)),
        "final_stage_p10_cost": float(
            torch.quantile(run.stages[-1].cost, 0.10)
        ),
        "weighted_output_cost": run.summary["weighted_output"]["cost"],
        "dbm_best_action_cost": dbm_cross_evaluation["best_action_cost"],
        "dbm_weighted_action_cost": dbm_cross_evaluation[
            "weighted_action_cost"
        ],
        "elapsed_seconds": run.summary["elapsed_seconds"],
    }


@torch.no_grad()
def dbm_cross_evaluate(
    run: StrategyRun,
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    best_index = int(run.combined.cost.argmin())
    actions = torch.stack(
        (run.combined.actions[best_index], run.weighted_action_sequence), dim=0
    )
    trajectories = backend(
        history, initial_state, current_action, actions
    ).to(controller.device)
    components = controller.trajectory_cost_components(
        trajectories, actions, reference, current_action
    )
    cost = sum(components.values())
    return {
        "best_action_cost": float(cost[0]),
        "weighted_action_cost": float(cost[1]),
    }


def save_primary_arrays(
    path: Path,
    runs: dict[str, StrategyRun],
    baseline,
    dbm_weighted_trajectories: dict[str, np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "baseline_cost": cpu_numpy(baseline.cost),
        "baseline_actions": cpu_numpy(baseline.actions),
        "baseline_trajectories": cpu_numpy(baseline.trajectories),
    }
    for name, run in runs.items():
        prefix = name.replace("-", "_")
        arrays[f"{prefix}_combined_cost"] = cpu_numpy(run.combined.cost)
        arrays[f"{prefix}_combined_actions"] = cpu_numpy(run.combined.actions)
        arrays[f"{prefix}_combined_trajectories"] = cpu_numpy(
            run.combined.trajectories
        )
        arrays[f"{prefix}_candidate_stage"] = np.concatenate(
            [
                np.full(len(stage.cost), index, dtype=np.int16)
                for index, stage in enumerate(run.stages, start=1)
            ]
        )
        arrays[f"{prefix}_weighted_action_sequence"] = cpu_numpy(
            run.weighted_action_sequence
        )
        arrays[f"{prefix}_weighted_trajectory"] = cpu_numpy(
            run.weighted_trajectory
        )
        for index, stage in enumerate(run.stages, start=1):
            arrays[f"{prefix}_stage_{index}_knots"] = cpu_numpy(stage.knots)
            arrays[f"{prefix}_stage_{index}_cost"] = cpu_numpy(stage.cost)
        arrays[f"{prefix}_dbm_weighted_trajectory"] = (
            dbm_weighted_trajectories[name]
        )
    np.savez_compressed(path, **arrays)


def plot_results(
    output_dir: Path,
    snapshot: np.lib.npyio.NpzFile,
    primary: dict[str, StrategyRun],
    repeated: dict[int, dict[str, StrategyRun]],
    baseline,
    checkpoint: Path,
    adaptive_fit_threshold: float,
    primary_dbm_cross: dict[str, dict[str, float]],
    primary_dbm_weighted_trajectories: dict[str, np.ndarray],
) -> None:
    names = ("fixed-1", "fixed-2", "fixed-3", "fixed-4", "adaptive")
    colors = {
        "fixed-1": "#777777",
        "fixed-2": "#E69F00",
        "fixed-3": "#009E73",
        "fixed-4": "#0072B2",
        "adaptive": "#D55E00",
    }
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for panel, key, title, log_scale in (
        (axes[0, 0], "cumulative_best_cost", "Cumulative best cost", False),
        (axes[0, 1], "cumulative_p10_cost", "Cumulative P10 cost", True),
    ):
        for name in names:
            rows = progress_rows(primary[name], name)
            x = [row["cumulative_rollouts"] for row in rows]
            y = [row[key] for row in rows]
            panel.plot(
                x,
                y,
                "o-",
                linewidth=2.1,
                color=colors[name],
                label=f"{name}: {y[-1]:.3f}",
            )
        if log_scale:
            panel.set_yscale("log")
        panel.set_title(f"Primary seed — {title}")
        panel.set_xlabel("cumulative Query rollouts")
        panel.set_ylabel("cost (lower is better)")
        panel.legend(fontsize=8)

    axis = axes[0, 2]
    x = np.arange(len(names))
    width = 0.36
    best = [primary[name].summary["combined"]["best_cost"] for name in names]
    weighted = [primary[name].summary["weighted_output"]["cost"] for name in names]
    best_bars = axis.bar(x - width / 2, best, width, color="#56B4E9", label="best")
    weighted_bars = axis.bar(
        x + width / 2, weighted, width, color="#0072B2", label="weighted output"
    )
    axis.bar_label(best_bars, fmt="%.3f", fontsize=8, padding=2)
    axis.bar_label(weighted_bars, fmt="%.3f", fontsize=8, padding=2)
    axis.set_xticks(x, names, rotation=20)
    axis.set_ylabel("cost")
    axis.set_title("Primary seed — final solution")
    axis.legend()

    axis = axes[1, 0]
    seed_values = sorted(repeated)
    comparison_names = ("fixed-3", "fixed-4", "adaptive")
    for seed in seed_values:
        values = [
            repeated[seed][name].summary["combined"]["best_cost"]
            for name in comparison_names
        ]
        axis.plot(range(3), values, "o-", color="#999999", alpha=0.65)
        axis.annotate(
            str(seed),
            (2, values[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    axis.set_xticks(range(3), comparison_names)
    axis.set_ylabel("best cost")
    axis.set_title("Each seed — final best cost")

    axis = axes[1, 1]
    reference = snapshot["reference"]
    if len(reference) == 51:
        reference = reference[1:]
    initial = snapshot["initial_state"]
    axis.plot(
        reference[:, 0], reference[:, 1], "k--", linewidth=2.2, label="reference"
    )
    for name in ("fixed-3", "fixed-4", "adaptive"):
        query_path = np.vstack(
            (initial[None, :2], cpu_numpy(primary[name].weighted_trajectory)[:, :2])
        )
        dbm_path = np.vstack(
            (initial[None, :2], primary_dbm_weighted_trajectories[name][:, :2])
        )
        axis.plot(
            query_path[:, 0],
            query_path[:, 1],
            color=colors[name],
            linewidth=2,
            label=f"{name}: Query prediction",
        )
        axis.plot(
            dbm_path[:, 0],
            dbm_path[:, 1],
            color=colors[name],
            linewidth=1.8,
            linestyle=":",
            label=f"{name}: DBM replay",
        )
    axis.scatter(initial[0], initial[1], marker="*", color="k", s=80, zorder=5)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Primary seed — Query prediction vs DBM replay")
    axis.legend(fontsize=8, ncol=2)

    axis = axes[1, 2]
    best_dbm = [primary_dbm_cross[name]["best_action_cost"] for name in names]
    weighted_dbm = [
        primary_dbm_cross[name]["weighted_action_cost"] for name in names
    ]
    best_dbm_bars = axis.bar(
        x - width / 2,
        best_dbm,
        width,
        color="#F0A35E",
        label="Query-best action replayed in DBM",
    )
    weighted_dbm_bars = axis.bar(
        x + width / 2,
        weighted_dbm,
        width,
        color="#D55E00",
        label="weighted action replayed in DBM",
    )
    axis.axhline(
        float(snapshot["cost"].min()),
        color="#CC79A7",
        linestyle="--",
        linewidth=1.5,
        label=f"saved DBM candidate best={float(snapshot['cost'].min()):.3f}",
    )
    axis.bar_label(best_dbm_bars, fmt="%.2f", fontsize=7, padding=2)
    axis.bar_label(weighted_dbm_bars, fmt="%.2f", fontsize=7, padding=2)
    axis.set_xticks(x, names, rotation=20)
    axis.set_ylabel("DBM cost")
    axis.set_title("Primary seed — DBM replay of Query-optimized actions")
    axis.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Small-car Query model — sequential guided sampling on fixed DBM state\n"
        f"checkpoint={checkpoint.parent.name}; 16 variables; 256 rollouts; "
        f"adaptive fit threshold={adaptive_fit_threshold:.2f}",
        fontsize=15,
    )
    figure.savefig(output_dir / "query_guided_comparison.png", bbox_inches="tight")
    figure.savefig(output_dir / "query_guided_comparison.svg", bbox_inches="tight")
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
    model = QueryDeploymentModel.from_checkpoint(args.checkpoint, device)
    if abs(model.dt - params.dt) > 1e-9 or abs(model.wheelbase - 0.21) > 1e-6:
        raise ValueError(
            f"expected small-car Query dt=0.05/wheelbase=0.21, got "
            f"dt={model.dt}, wheelbase={model.wheelbase}"
        )
    backend = TorchQueryRolloutBackend(model)
    controller = TorchMPPIController(
        backend, params=params, cost_weights=weights, device=device
    )
    history = torch.from_numpy(snapshot["history"]).to(device)
    initial_state = torch.from_numpy(snapshot["initial_state"]).to(device).reshape(1, 5)
    current_action = torch.from_numpy(snapshot["current_action"]).to(device).reshape(1, 2)
    reference = controller._prepare_reference(snapshot["reference"])
    warm_start = torch.from_numpy(snapshot["mean_knots_before"]).to(device)
    base_sigma = torch.tensor(params.noise_sigma, dtype=torch.float32, device=device)

    baseline_knots = torch.from_numpy(snapshot["sampled_knots"]).to(device)
    baseline = evaluate_knots(
        controller,
        backend,
        baseline_knots,
        history,
        initial_state,
        current_action,
        reference,
    )

    repeated: dict[int, dict[str, StrategyRun]] = {}
    for seed in seeds:
        repeated[seed] = {}
        for stage_count in (1, 2, 3, 4):
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
    dbm_backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**metadata["model"]["dbm_params"])
    )
    dbm_backend.set_initial_lateral_velocity(
        float(snapshot["initial_lateral_velocity"])
    )
    dbm_controller = TorchMPPIController(
        dbm_backend, params=params, cost_weights=weights, device=device
    )
    dbm_cross = {
        str(seed): {
            name: dbm_cross_evaluate(
                run,
                dbm_controller,
                dbm_backend,
                history,
                initial_state,
                current_action,
                reference,
            )
            for name, run in runs.items()
        }
        for seed, runs in repeated.items()
    }
    primary_dbm_actions = torch.stack(
        [primary[name].weighted_action_sequence for name in primary], dim=0
    )
    primary_dbm_weighted_trajectories_tensor = dbm_backend(
        history,
        initial_state,
        current_action,
        primary_dbm_actions,
    ).to(device)
    primary_dbm_weighted_trajectories = {
        name: cpu_numpy(primary_dbm_weighted_trajectories_tensor[index])
        for index, name in enumerate(primary)
    }
    per_seed = {
        str(seed): {
            name: metric_summary(run, dbm_cross[str(seed)][name])
            for name, run in runs.items()
        }
        for seed, runs in repeated.items()
    }
    aggregate = {}
    for name in primary:
        aggregate[name] = {}
        for metric in (
            "best_cost",
            "weighted_output_cost",
            "combined_p10_cost",
            "final_stage_p10_cost",
            "dbm_best_action_cost",
            "dbm_weighted_action_cost",
        ):
            values = np.asarray(
                [per_seed[str(seed)][name][metric] for seed in seeds],
                dtype=np.float64,
            )
            aggregate[name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    for name in ("fixed-1", "fixed-2", "fixed-3", "fixed-4"):
        aggregate["adaptive"][f"wins_vs_{name}"] = {
            metric: sum(
                per_seed[str(seed)]["adaptive"][metric]
                < per_seed[str(seed)][name][metric]
                for seed in seeds
            )
            for metric in (
                "best_cost",
                "weighted_output_cost",
                "combined_p10_cost",
                "final_stage_p10_cost",
                "dbm_best_action_cost",
                "dbm_weighted_action_cost",
            )
        }

    summary = {
        "scenario_snapshot": str(args.snapshot.resolve()),
        "scenario_source_backend": metadata["model"]["backend"],
        "rollout_backend": "query-pytorch",
        "query_checkpoint": str(args.checkpoint.resolve()),
        "query_dt": model.dt,
        "query_wheelbase": model.wheelbase,
        "gradient_used": False,
        "dbm_cross_evaluation": (
            "Diagnostic only and excluded from each strategy's 256 Query rollout budget"
        ),
        "total_rollouts_per_strategy": 256,
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
        "primary_seed": seeds[0],
        "repeat_seeds": list(seeds),
        "saved_candidate_query_baseline": {
            "candidate_count": len(baseline.cost),
            "best_cost": float(baseline.cost.min()),
            "p10_cost": float(torch.quantile(baseline.cost, 0.10)),
        },
        "primary": {name: run.summary for name, run in primary.items()},
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_primary_arrays(
        args.output_dir / "primary_results.npz",
        primary,
        baseline,
        primary_dbm_weighted_trajectories,
    )

    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="") as file:
        fields = [
            "seed",
            "strategy",
            "allocation",
            "best_cost",
            "weighted_output_cost",
            "combined_p10_cost",
            "final_stage_p10_cost",
            "dbm_best_action_cost",
            "dbm_weighted_action_cost",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for seed in seeds:
            for name, metrics in per_seed[str(seed)].items():
                writer.writerow(
                    {
                        "seed": seed,
                        "strategy": name,
                        **{field: metrics[field] for field in fields[2:]},
                    }
                )

    plot_results(
        args.output_dir,
        snapshot,
        primary,
        repeated,
        baseline,
        args.checkpoint,
        args.split_fit_error_threshold,
        dbm_cross[str(seeds[0])],
        primary_dbm_weighted_trajectories,
    )
    print(
        f"saved candidates under Query: best={float(baseline.cost.min()):.6f}, "
        f"P10={float(torch.quantile(baseline.cost, 0.10)):.6f}"
    )
    for seed in seeds:
        values = per_seed[str(seed)]
        print(
            f"seed={seed}: fixed-3={values['fixed-3']['best_cost']:.6f}, "
            f"fixed-4={values['fixed-4']['best_cost']:.6f}, "
            f"adaptive={values['adaptive']['best_cost']:.6f} "
            f"allocation={values['adaptive']['allocation']}"
        )
    print((args.output_dir / "query_guided_comparison.png").resolve())


if __name__ == "__main__":
    main()
