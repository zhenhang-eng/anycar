#!/usr/bin/env python3
"""Visualize cost distribution and representative MPPI snapshot candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/snapshot.npz"
)


COLORS = {
    "Best": "#0072B2",
    "Warm-start": "#009E73",
    "P25": "#56B4E9",
    "P50": "#E69F00",
    "P75": "#7E57C2",
    "P95": "#CC79A7",
    "Worst": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def representative_indices(cost: np.ndarray) -> dict[str, int]:
    order = np.argsort(cost)
    quantile_index = lambda quantile: int(
        order[round(quantile * (len(order) - 1))]
    )
    return {
        "Best": int(order[0]),
        "Warm-start": 0,
        "P25": quantile_index(0.25),
        "P50": quantile_index(0.50),
        "P75": quantile_index(0.75),
        "P95": quantile_index(0.95),
        "Worst": int(order[-1]),
    }


def style_axis(axis) -> None:
    axis.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def add_all_prediction_lines(axis, time, values, log_cost) -> None:
    collection = LineCollection(
        [np.column_stack((time, row)) for row in values],
        cmap="viridis",
        norm=Normalize(float(log_cost.min()), float(log_cost.max())),
        linewidths=0.45,
        alpha=0.10,
        zorder=1,
    )
    collection.set_array(log_cost)
    axis.add_collection(collection)
    axis.autoscale()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.snapshot.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = np.load(args.snapshot)
    metadata = json.loads((args.snapshot.resolve().parent / "summary.json").read_text())
    backend_name = metadata["model"]["backend"]
    model_label = "Torch DBM" if backend_name == "dbm" else "Query"
    state = snapshot["initial_state"]
    current_action = snapshot["current_action"]
    reference = snapshot["reference"]
    actions = snapshot["sampled_action_sequences"]
    trajectories = snapshot["predicted_trajectories"]
    cost = snapshot["cost"]
    weight = snapshot["weight"]
    selected = representative_indices(cost)
    horizon_time = (np.arange(actions.shape[1]) + 1) * 0.05

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

    order = np.argsort(cost)
    log_cost = np.log10(cost)

    # Sorted cost and cumulative weight expose how many samples actually
    # contribute to the MPPI update.
    axis = axes[0, 0]
    rank = np.arange(1, len(cost) + 1)
    axis.plot(rank, cost[order], color="#315B7D", linewidth=2, label="sorted cost")
    rank_by_index = np.empty(len(cost), dtype=np.int64)
    rank_by_index[order] = rank
    for name, index in selected.items():
        axis.scatter(
            rank_by_index[index], cost[index], color=COLORS[name], s=30,
            edgecolor="white", linewidth=0.5, zorder=4,
        )
    axis.set_yscale("log")
    axis.set_title("Cost rank and cumulative MPPI weight")
    axis.set_xlabel("candidate rank (low cost to high)")
    axis.set_ylabel("total cost (log scale)")
    style_axis(axis)
    weight_axis = axis.twinx()
    weight_axis.plot(rank, np.cumsum(weight[order]), color="#D55E00",
                     linestyle="--", linewidth=1.8, label="cumulative weight")
    weight_axis.set_ylabel("cumulative MPPI weight")
    weight_axis.set_ylim(-0.02, 1.02)
    lines = axis.get_lines() + weight_axis.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], loc="center right")

    # All rollouts are drawn faintly and colored by log-cost; representative
    # trajectories are overlaid with consistent colors used in other panels.
    axis = axes[0, 1]
    paths = [
        np.vstack((state[None, :2], trajectory[:, :2]))
        for trajectory in trajectories
    ]
    collection = LineCollection(
        paths,
        cmap="viridis",
        norm=Normalize(float(log_cost.min()), float(log_cost.max())),
        linewidths=0.55,
        alpha=0.20,
    )
    collection.set_array(log_cost)
    axis.add_collection(collection)
    axis.plot(reference[:, 0], reference[:, 1], color="#222222", linestyle="--",
              linewidth=2.2, label="reference")
    for name, index in selected.items():
        path = paths[index]
        axis.plot(path[:, 0], path[:, 1], color=COLORS[name], linewidth=2.0,
                  label=f"{name} #{index}")
        axis.scatter(path[-1, 0], path[-1, 1], color=COLORS[name], s=22, zorder=4)
    axis.scatter(state[0], state[1], marker="*", color="#111111", s=95,
                 label="fixed state", zorder=5)
    axis.autoscale()
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(f"{model_label}-predicted XY rollouts")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.legend(loc="best", ncol=2)
    style_axis(axis)
    colorbar = figure.colorbar(collection, ax=axis, pad=0.01, shrink=0.82)
    colorbar.set_label("log10(total cost)")

    axis = axes[0, 2]
    for name, index in selected.items():
        axis.plot(horizon_time, actions[index, :, 1], color=COLORS[name],
                  linewidth=1.9, label=f"{name} #{index}")
    axis.scatter([0], [current_action[1]], marker="*", color="#111111", s=55,
                 label="current action", zorder=4)
    axis.axhline(-1, color="#777777", linestyle=":", linewidth=1)
    axis.axhline(1, color="#777777", linestyle=":", linewidth=1)
    axis.set_title("Representative steering sequences")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("normalized steering")
    axis.set_ylim(-1.08, 1.08)
    axis.legend(loc="best", ncol=2)
    style_axis(axis)

    yaw_predictions = np.stack(
        [
            np.unwrap(np.concatenate(([state[2]], trajectory[:, 2])))[1:]
            for trajectory in trajectories
        ]
    )
    reference_yaw = np.unwrap(reference[:, 2])
    axis = axes[1, 0]
    add_all_prediction_lines(axis, horizon_time, yaw_predictions, log_cost)
    axis.plot(horizon_time, reference_yaw[1:], color="#222222", linestyle="--",
              linewidth=2.1, label="reference")
    for name, index in selected.items():
        axis.plot(horizon_time, yaw_predictions[index], color=COLORS[name],
                  linewidth=1.9, label=f"{name} #{index}")
    axis.scatter([0], [state[2]], marker="*", color="#111111", s=55,
                 label="current state", zorder=4)
    axis.set_title("Predicted yaw")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("unwrapped yaw [rad]")
    axis.legend(loc="best", ncol=2)
    style_axis(axis)

    axis = axes[1, 1]
    add_all_prediction_lines(axis, horizon_time, trajectories[:, :, 3], log_cost)
    axis.plot(horizon_time, reference[1:, 3], color="#222222", linestyle="--",
              linewidth=2.1, label="reference")
    for name, index in selected.items():
        axis.plot(horizon_time, trajectories[index, :, 3], color=COLORS[name],
                  linewidth=1.9, label=f"{name} #{index}")
    axis.scatter([0], [state[3]], marker="*", color="#111111", s=55,
                 label="current state", zorder=4)
    axis.set_title("Predicted longitudinal speed")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("vx [m/s]")
    axis.legend(loc="best", ncol=2)
    style_axis(axis)

    axis = axes[1, 2]
    add_all_prediction_lines(axis, horizon_time, trajectories[:, :, 4], log_cost)
    reference_yawrate = np.diff(reference_yaw) / 0.05
    axis.plot(horizon_time, reference_yawrate, color="#222222", linestyle="--",
              linewidth=2.1, label="reference from yaw")
    for name, index in selected.items():
        axis.plot(horizon_time, trajectories[index, :, 4], color=COLORS[name],
                  linewidth=1.9,
                  label=f"{name} #{index}")
    axis.scatter([0], [state[4]], marker="*", color="#111111", s=55,
                 label="current state", zorder=4)
    axis.set_title("Predicted yaw rate")
    axis.set_xlabel("prediction time [s]")
    axis.set_ylabel("yaw rate [rad/s]")
    axis.legend(loc="best", ncol=2)
    style_axis(axis)

    baseline = metadata["baseline"]
    figure.suptitle(
        f"Fixed clean {model_label}-MPPI snapshot — {len(cost)} candidates, "
        f"step {metadata['scenario']['control_step']} "
        f"(t={metadata['scenario']['simulated_time_s']:.1f} s)\n"
        f"best cost={baseline['best_cost']:.3f}, median={baseline['median_cost']:.1f}, "
        f"ESS={baseline['effective_sample_size']:.3f}, "
        f"best weight={weight[order[0]]:.3%}",
        fontsize=14,
    )

    png_path = output_dir / "sampling_cost_and_trajectories.png"
    svg_path = output_dir / "sampling_cost_and_trajectories.svg"
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    print(png_path.resolve())
    print(svg_path.resolve())
    print(
        json.dumps(
            {
                name: {"index": index, "cost": float(cost[index])}
                for name, index in selected.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
