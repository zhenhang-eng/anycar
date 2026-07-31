#!/usr/bin/env python3
"""Plot the Query 1/2/3/4-stage result with the DBM figure's exact panels.

This is intentionally a plotting-only script: it consumes the saved Query
experiment so regenerating the matched figure does not spend rollout budget or
change any optimization result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/query_guided_on_dbm_state_step0340"
)
DEFAULT_SNAPSHOT = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/snapshot.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    return parser.parse_args()


def style_axis(axis) -> None:
    axis.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    summary = json.loads((result_dir / "summary.json").read_text())
    arrays = np.load(result_dir / "primary_results.npz")
    snapshot = np.load(args.snapshot)

    stage_counts = np.arange(1, 5, dtype=np.int64)
    names = [f"fixed-{count}" for count in stage_counts]
    primary = summary["primary"]
    aggregate = summary["aggregate"]
    colors = {
        1: "#777777",
        2: "#E69F00",
        3: "#009E73",
        4: "#0072B2",
    }
    baseline_cost = arrays["baseline_cost"]
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
    for stage_count, name in zip(stage_counts, names):
        stages = primary[name]["stages"]
        x = [stage["cumulative_sample_count"] for stage in stages]
        y = [stage["cumulative_best_cost"] for stage in stages]
        axis.plot(
            x,
            y,
            "o-",
            color=colors[int(stage_count)],
            linewidth=2,
            label=f"{stage_count} stage" + ("s" if stage_count > 1 else ""),
        )
    axis.scatter(
        [256],
        [historical_best],
        marker="*",
        color="#CC79A7",
        s=80,
        label="saved Gaussian baseline (Query reevaluation)",
        zorder=5,
    )
    axis.set_yscale("log")
    axis.set_title("Primary seed — cumulative best cost")
    axis.set_xlabel("cumulative Query rollouts")
    axis.set_ylabel("best cost so far")
    axis.set_xticks([64, 86, 128, 172, 192, 256])
    axis.legend()
    style_axis(axis)

    axis = axes[0, 1]
    best = [primary[name]["combined"]["best_cost"] for name in names]
    output = [primary[name]["weighted_output"]["cost"] for name in names]
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
    axis.set_ylabel("Query cost")
    axis.legend()
    style_axis(axis)

    axis = axes[0, 2]
    aggregate_best_mean = np.asarray(
        [aggregate[name]["best_cost"]["mean"] for name in names]
    )
    aggregate_best_std = np.asarray(
        [aggregate[name]["best_cost"]["std"] for name in names]
    )
    aggregate_output_mean = np.asarray(
        [aggregate[name]["weighted_output_cost"]["mean"] for name in names]
    )
    aggregate_output_std = np.asarray(
        [aggregate[name]["weighted_output_cost"]["std"] for name in names]
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
    axis.set_ylabel("Query cost")
    axis.legend()
    style_axis(axis)

    axis = axes[1, 0]
    final_median = np.asarray(
        [primary[name]["final_stage"]["median_cost"] for name in names]
    )
    final_p95 = np.asarray(
        [primary[name]["final_stage"]["p95_cost"] for name in names]
    )
    axis.bar(
        position - width / 2,
        final_median,
        width,
        color="#56B4E9",
        label="final-stage median",
    )
    axis.bar(
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
    axis.set_ylabel("Query cost (log scale)")
    style_axis(axis)
    ess_axis = axis.twinx()
    normalized_ess = [
        primary[name]["final_stage"]["normalized_effective_sample_size"]
        for name in names
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
    axis.legend(
        lines[0] + lines_ess[0],
        lines[1] + lines_ess[1],
        loc="upper right",
    )

    axis = axes[1, 1]
    for threshold, marker in ((5, "o"), (10, "s"), (20, "D")):
        values = [
            primary[name]["combined"][f"count_cost_lt_{threshold}"]
            for name in names
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
            arrays["baseline_trajectories"][baseline_best_index, :, :2],
        )
    )
    axis.plot(
        historical_path[:, 0],
        historical_path[:, 1],
        color="#CC79A7",
        linewidth=1.6,
        label="saved Gaussian best (Query)",
    )
    for stage_count, name in zip(stage_counts, names):
        path = np.vstack(
            (
                initial_state[None, :2],
                arrays[f"fixed_{stage_count}_weighted_trajectory"][:, :2],
            )
        )
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=colors[int(stage_count)],
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

    best_four = aggregate["fixed-4"]["best_cost"]
    figure.suptitle(
        "Fixed DBM snapshot — Query rollout model, 256 rollouts split into "
        "sequential guidance stages\n"
        f"8 temporal knots / 16 scalar variables; primary seed="
        f"{summary['primary_seed']}; 4-stage 10-seed best Query cost="
        f"{best_four['mean']:.3f}±{best_four['std']:.3f}",
        fontsize=14,
    )
    output_png = result_dir / "query_stage_count_comparison_matched.png"
    output_svg = result_dir / "query_stage_count_comparison_matched.svg"
    figure.savefig(output_png, bbox_inches="tight")
    figure.savefig(output_svg, bbox_inches="tight")
    plt.close(figure)
    print(output_png)


if __name__ == "__main__":
    main()
