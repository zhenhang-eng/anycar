#!/usr/bin/env python3
"""Visualize all four fixed Query-guidance stages on the saved primary seed."""

from __future__ import annotations

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/query_guided_on_dbm_state_step0340"
)
SNAPSHOT_PATH = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/snapshot.npz"
)
COLORS = ("#777777", "#E69F00", "#009E73", "#0072B2")


def wrapped_angle_difference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    difference = lhs - rhs
    return np.arctan2(np.sin(difference), np.cos(difference))


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    summary = json.loads((RESULT_DIR / "summary.json").read_text())
    snapshot_summary = json.loads(
        (SNAPSHOT_PATH.parent / "summary.json").read_text()
    )
    arrays = np.load(RESULT_DIR / "primary_results.npz")
    snapshot = np.load(SNAPSHOT_PATH)
    prefix = "fixed_4_"
    cost = arrays[prefix + "combined_cost"]
    actions = arrays[prefix + "combined_actions"]
    query_trajectories = arrays[prefix + "combined_trajectories"]
    stage_label = arrays[prefix + "candidate_stage"]
    reference = snapshot["reference"]
    if len(reference) == 51:
        reference = reference[1:]
    initial_state = snapshot["initial_state"]
    current_action = snapshot["current_action"]

    params = TorchMPPIParams(**snapshot_summary["mppi_params"])
    weights = TorchMPPICostWeights(**snapshot_summary["cost_weights"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dbm_backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**snapshot_summary["model"]["dbm_params"])
    )
    dbm_backend.set_initial_lateral_velocity(
        float(snapshot["initial_lateral_velocity"])
    )
    dbm_controller = TorchMPPIController(
        dbm_backend, params=params, cost_weights=weights, device=device
    )
    best_indices = []
    stage_cost = []
    rows = []
    collected = []
    stage_summaries = summary["primary"]["fixed-4"]["stages"]
    for stage in range(1, 5):
        indices = np.flatnonzero(stage_label == stage)
        current_cost = cost[indices]
        best_index = int(indices[np.argmin(current_cost)])
        best_indices.append(best_index)
        stage_cost.append(current_cost)
        collected.append(current_cost)
        cumulative = np.concatenate(collected)
        rows.append(
            {
                "stage": stage,
                "sample_count": len(current_cost),
                "cumulative_rollouts": len(cumulative),
                "sigma_scale": stage_summaries[stage - 1]["noise_scale"],
                "center_cost": float(current_cost[0]),
                "best_cost": float(current_cost.min()),
                "p10_cost": float(np.percentile(current_cost, 10)),
                "median_cost": float(np.median(current_cost)),
                "cumulative_best_cost": float(cumulative.min()),
                "cumulative_p10_cost": float(np.percentile(cumulative, 10)),
                "relative_fit_error": (
                    stage_summaries[stage - 1].get("guidance", {}).get(
                        "relative_weighted_fit_error", float("nan")
                    )
                ),
            }
        )

    best_actions = torch.from_numpy(actions[best_indices]).to(device)
    with torch.no_grad():
        dbm_trajectories_tensor = dbm_backend(
            torch.from_numpy(snapshot["history"]).to(device),
            torch.from_numpy(initial_state).to(device).reshape(1, 5),
            torch.from_numpy(current_action).to(device).reshape(1, 2),
            best_actions,
        ).to(device)
        dbm_components = dbm_controller.trajectory_cost_components(
            dbm_trajectories_tensor,
            best_actions,
            torch.from_numpy(reference).to(device),
            torch.from_numpy(current_action).to(device).reshape(1, 2),
        )
        dbm_cost = sum(dbm_components.values()).cpu().numpy()
    dbm_trajectories = dbm_trajectories_tensor.cpu().numpy()
    for row, value in zip(rows, dbm_cost):
        row["dbm_replay_cost"] = float(value)

    with (RESULT_DIR / "query_fixed4_stage_progression.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_query_trajectories = query_trajectories[best_indices]
    best_action_np = actions[best_indices]
    time = np.arange(1, params.horizon + 1) * params.dt
    stages = np.arange(1, 5)

    figure, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    axis = axes[0, 0]
    box = axis.boxplot(
        stage_cost,
        tick_labels=[f"S{stage}" for stage in stages],
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axis.set_yscale("log")
    axis.set_ylabel("Query cost (log scale)")
    axis.set_title("Each new batch — cost distribution")
    for stage, row in zip(stages, rows):
        axis.scatter(stage, row["best_cost"], marker="*", s=75, color=COLORS[stage - 1], zorder=5)
        axis.annotate(
            f"best={row['best_cost']:.2f}\nP10={row['p10_cost']:.2f}",
            (stage, row["best_cost"]),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    axis = axes[0, 1]
    for key, marker, label in (
        ("center_cost", "x", "current center"),
        ("best_cost", "o", "current batch best"),
        ("p10_cost", "s", "current batch P10"),
        ("median_cost", "D", "current batch median"),
    ):
        axis.plot(stages, [row[key] for row in rows], marker=marker, linewidth=2, label=label)
    axis.set_yscale("log")
    axis.set_xticks(stages)
    axis.set_xlabel("sequential stage")
    axis.set_ylabel("Query cost (log scale)")
    axis.set_title("New-batch quality after each center update")
    axis.legend(fontsize=8)

    axis = axes[0, 2]
    cumulative_rollouts = [row["cumulative_rollouts"] for row in rows]
    axis.plot(
        cumulative_rollouts,
        [row["cumulative_best_cost"] for row in rows],
        "o-",
        linewidth=2.2,
        label="cumulative best",
    )
    axis.plot(
        cumulative_rollouts,
        [row["cumulative_p10_cost"] for row in rows],
        "s-",
        linewidth=2.2,
        label="cumulative P10",
    )
    axis.set_yscale("log")
    axis.set_xticks(cumulative_rollouts)
    axis.set_xlabel("cumulative Query rollouts")
    axis.set_ylabel("Query cost (log scale)")
    axis.set_title("Best/P10 across all samples collected so far")
    axis.legend()

    axis = axes[1, 0]
    axis.plot(reference[:, 0], reference[:, 1], "k--", linewidth=2.3, label="reference")
    for index, stage in enumerate(stages):
        query_path = np.vstack((initial_state[None, :2], best_query_trajectories[index, :, :2]))
        dbm_path = np.vstack((initial_state[None, :2], dbm_trajectories[index, :, :2]))
        axis.plot(query_path[:, 0], query_path[:, 1], color=COLORS[index], linewidth=2, label=f"S{stage} Query")
        axis.plot(dbm_path[:, 0], dbm_path[:, 1], color=COLORS[index], linewidth=1.6, linestyle=":", label=f"S{stage} DBM")
    axis.scatter(initial_state[0], initial_state[1], marker="*", color="k", s=80, zorder=5)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Stage-best XY — Query prediction vs DBM replay")
    axis.legend(fontsize=7, ncol=2)

    axis = axes[1, 1]
    axis.plot(time, reference[:, 2], "k--", linewidth=2.2, label="reference")
    for index, stage in enumerate(stages):
        axis.plot(time, best_query_trajectories[index, :, 2], color=COLORS[index], linewidth=2, label=f"S{stage} Query")
        axis.plot(time, dbm_trajectories[index, :, 2], color=COLORS[index], linewidth=1.5, linestyle=":", label=f"S{stage} DBM")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("yaw [rad]")
    axis.set_title("Stage-best predicted yaw")

    axis = axes[1, 2]
    axis.plot(time, reference[:, 3], "k--", linewidth=2.2, label="reference")
    for index, stage in enumerate(stages):
        axis.plot(time, best_query_trajectories[index, :, 3], color=COLORS[index], linewidth=2, label=f"S{stage} Query")
        axis.plot(time, dbm_trajectories[index, :, 3], color=COLORS[index], linewidth=1.5, linestyle=":", label=f"S{stage} DBM")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("vx [m/s]")
    axis.set_title("Stage-best predicted longitudinal speed")

    axis = axes[2, 0]
    for index, stage in enumerate(stages):
        axis.plot(time, best_action_np[index, :, 0], color=COLORS[index], linewidth=2, label=f"S{stage}")
    axis.axhline(current_action[0], color="k", linestyle="--", linewidth=1.4, label="current action")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("acceleration command")
    axis.set_title("Stage-best acceleration sequence")
    axis.legend(fontsize=8)

    axis = axes[2, 1]
    for index, stage in enumerate(stages):
        axis.plot(time, best_action_np[index, :, 1], color=COLORS[index], linewidth=2, label=f"S{stage}")
    axis.axhline(current_action[1], color="k", linestyle="--", linewidth=1.4, label="current action")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("steering command")
    axis.set_title("Stage-best steering sequence")
    axis.legend(fontsize=8)

    axis = axes[2, 2]
    width = 0.36
    query_best = [row["best_cost"] for row in rows]
    query_bars = axis.bar(stages - width / 2, query_best, width, color=COLORS, alpha=0.65, label="Query predicted")
    dbm_bars = axis.bar(stages + width / 2, dbm_cost, width, color=COLORS, label="DBM replay")
    axis.axhline(float(snapshot["cost"].min()), color="#CC79A7", linestyle="--", linewidth=1.5, label=f"saved DBM best={float(snapshot['cost'].min()):.3f}")
    axis.bar_label(query_bars, fmt="%.2f", fontsize=8, padding=2)
    axis.bar_label(dbm_bars, fmt="%.2f", fontsize=8, padding=2)
    axis.set_xticks(stages, [f"S{stage}" for stage in stages])
    axis.set_ylabel("cost")
    axis.set_title("Stage-best action: Query cost vs DBM replay")
    axis.legend(fontsize=8)

    for axis in axes.flat:
        style_axis(axis)
    figure.suptitle(
        "Small-car Query — fixed four-stage guidance, primary seed 3407\n"
        "64+64+64+64 rollouts; sigma scale 1.0→0.464→0.215→0.1; solid=Query, dotted=DBM replay",
        fontsize=15,
    )
    figure.savefig(RESULT_DIR / "query_fixed4_stage_progression.png", bbox_inches="tight")
    figure.savefig(RESULT_DIR / "query_fixed4_stage_progression.svg", bbox_inches="tight")
    plt.close(figure)

    for row in rows:
        print(
            f"S{row['stage']}: Query best={row['best_cost']:.6f}, "
            f"P10={row['p10_cost']:.6f}, median={row['median_cost']:.6f}, "
            f"DBM replay={row['dbm_replay_cost']:.6f}"
        )
    print((RESULT_DIR / "query_fixed4_stage_progression.png").resolve())


if __name__ == "__main__":
    main()
