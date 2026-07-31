#!/usr/bin/env python3
"""Plot stage-by-stage cost progress for the fixed guided-MPPI comparison.

The plot deliberately uses only the saved primary seed.  It separates the
quality of the newly sampled stage from the quality of the cumulative sample
pool, and reports both the minimum and lower-tail P10 cost.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/"
    "guided_stage_count_comparison_dbm_step0340/primary_seed_results.npz"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent
ALLOCATIONS = {
    1: [256],
    2: [128, 128],
    3: [86, 86, 84],
    4: [64, 64, 64, 64],
}
COLORS = {
    1: "#777777",
    2: "#E69F00",
    3: "#009E73",
    4: "#0072B2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--percentile", type=float, default=10.0)
    return parser.parse_args()


def summarize_progress(
    arrays: np.lib.npyio.NpzFile, percentile: float
) -> dict[int, list[dict[str, float | int]]]:
    progress: dict[int, list[dict[str, float | int]]] = {}
    for stage_count, allocation in ALLOCATIONS.items():
        prefix = f"stages_{stage_count}_"
        cost = arrays[prefix + "combined_cost"]
        stage_label = arrays[prefix + "candidate_stage"]
        if cost.shape != (256,) or stage_label.shape != (256,):
            raise ValueError(f"unexpected saved shape for {stage_count} stages")
        if list(np.bincount(stage_label)[1:]) != allocation:
            raise ValueError(f"saved allocation mismatch for {stage_count} stages")

        rows: list[dict[str, float | int]] = []
        for stage_index in range(1, stage_count + 1):
            current_cost = cost[stage_label == stage_index]
            cumulative_cost = cost[stage_label <= stage_index]
            rows.append(
                {
                    "strategy_stage_count": stage_count,
                    "stage_index": stage_index,
                    "stage_sample_count": len(current_cost),
                    "cumulative_rollouts": len(cumulative_cost),
                    "current_best_cost": float(np.min(current_cost)),
                    "current_p10_cost": float(
                        np.percentile(current_cost, percentile)
                    ),
                    "cumulative_best_cost": float(np.min(cumulative_cost)),
                    "cumulative_p10_cost": float(
                        np.percentile(cumulative_cost, percentile)
                    ),
                }
            )
        progress[stage_count] = rows
    return progress


def write_csv(
    path: Path, progress: dict[int, list[dict[str, float | int]]]
) -> None:
    fields = list(progress[1][0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for stage_count in sorted(progress):
            writer.writerows(progress[stage_count])


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xticks([64, 86, 128, 172, 192, 256])
    axis.set_xlim(48, 272)
    axis.set_xlabel("cumulative DBM rollouts")


def plot_metric(
    axis: plt.Axes,
    progress: dict[int, list[dict[str, float | int]]],
    key: str,
    title: str,
    log_scale: bool,
) -> None:
    for stage_count in sorted(progress):
        rows = progress[stage_count]
        x = np.asarray([row["cumulative_rollouts"] for row in rows])
        y = np.asarray([row[key] for row in rows])
        final_value = y[-1]
        axis.plot(
            x,
            y,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=COLORS[stage_count],
            label=f"{stage_count} stage{'s' if stage_count > 1 else ''}: "
            f"final {final_value:.3f}",
        )
        for row, x_value, y_value in zip(rows[:-1], x[:-1], y[:-1]):
            axis.annotate(
                f"S{row['stage_index']}",
                (x_value, y_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLORS[stage_count],
            )
    if log_scale:
        axis.set_yscale("log")
    axis.set_title(title)
    axis.set_ylabel("cost (lower is better)")
    axis.legend(fontsize=8, loc="upper right")
    style_axis(axis)


def plot_progress(
    output_dir: Path,
    progress: dict[int, list[dict[str, float | int]]],
    percentile: float,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 190,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.7), constrained_layout=True)
    plot_metric(
        axes[0],
        progress,
        "cumulative_best_cost",
        "Best cost found after each sequential update",
        log_scale=False,
    )
    plot_metric(
        axes[1],
        progress,
        "cumulative_p10_cost",
        f"P{percentile:g} cost of all samples collected so far",
        log_scale=True,
    )
    figure.suptitle(
        "Fixed DBM snapshot — cost improvement after each sequential update\n"
        "primary seed 3407; total budget 256; P10 = lower-cost 10% threshold",
        fontsize=15,
    )
    figure.savefig(output_dir / "guided_cost_progression.png", bbox_inches="tight")
    figure.savefig(output_dir / "guided_cost_progression.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 < args.percentile < 100:
        raise ValueError("percentile must be within (0, 100)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as arrays:
        progress = summarize_progress(arrays, args.percentile)
    write_csv(args.output_dir / "guided_cost_progression.csv", progress)
    plot_progress(args.output_dir, progress, args.percentile)

    for stage_count in sorted(progress):
        final = progress[stage_count][-1]
        print(
            f"{stage_count} stage(s): final-stage "
            f"best={final['current_best_cost']:.6f}, "
            f"P{args.percentile:g}={final['current_p10_cost']:.6f}; "
            f"cumulative best={final['cumulative_best_cost']:.6f}, "
            f"P{args.percentile:g}={final['cumulative_p10_cost']:.6f}"
        )


if __name__ == "__main__":
    main()
