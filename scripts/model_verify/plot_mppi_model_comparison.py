#!/usr/bin/env python3
"""Plot aligned old-Query, small-car-Query, and DBM Quick Start bags."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_mppi_rosbags import arrays_from_bag, read_bag


COLORS = {
    "Old Query (3.9 m)": "#D55E00",
    "Small-car Query (0.21 m)": "#0072B2",
    "Torch DBM": "#009E73",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-query-bag", type=Path, required=True)
    parser.add_argument("--small-query-bag", type=Path, required=True)
    parser.add_argument("--dbm-bag", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument("--target-speed", type=float, default=2.0)
    return parser.parse_args()


def aligned_steps(series):
    return min(
        min(
            len(data["state"]) - 1,
            len(data["action"]),
            len(data["lateral_error"]),
            len(data["mppi_time"]),
        )
        for data in series.values()
    )


def style_axis(axis):
    axis.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def main():
    args = parse_args()
    series = {
        "Old Query (3.9 m)": arrays_from_bag(read_bag(args.old_query_bag)),
        "Small-car Query (0.21 m)": arrays_from_bag(
            read_bag(args.small_query_bag)
        ),
        "Torch DBM": arrays_from_bag(read_bag(args.dbm_bag)),
    }
    steps = aligned_steps(series)
    if steps < 1:
        raise RuntimeError("bags have no common control transitions")
    time = np.arange(steps) * args.sim_dt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 180,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    trajectory_axis = axes[0, 0]
    for name, data in series.items():
        state = data["state"][: steps + 1]
        trajectory_axis.plot(
            state[:, 0], state[:, 1], label=name, color=COLORS[name], linewidth=2
        )
        trajectory_axis.scatter(
            state[-1, 0], state[-1, 1], color=COLORS[name], s=28, zorder=3
        )
    trajectory_axis.scatter(
        0.0, 0.0, marker="*", color="#222222", s=90, label="Start", zorder=4
    )
    trajectory_axis.set_title("Closed-loop trajectory")
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.axis("equal")
    trajectory_axis.legend(loc="upper left", frameon=True)
    style_axis(trajectory_axis)

    lateral_axis = axes[0, 1]
    for name, data in series.items():
        values = np.abs(data["lateral_error"][:steps])
        lateral_axis.plot(time, values, color=COLORS[name], linewidth=1.8)
        lateral_axis.text(
            0.98,
            {"Old Query (3.9 m)": 0.92, "Small-car Query (0.21 m)": 0.83,
             "Torch DBM": 0.74}[name],
            f"{name}: MAE {values.mean():.3f} m",
            color=COLORS[name],
            ha="right",
            transform=lateral_axis.transAxes,
        )
    lateral_axis.set_title("Absolute lateral tracking error")
    lateral_axis.set_xlabel("simulated time [s]")
    lateral_axis.set_ylabel("absolute error [m]")
    style_axis(lateral_axis)

    speed_axis = axes[1, 0]
    for name, data in series.items():
        state = data["state"][:steps]
        speed = np.linalg.norm(state[:, 3:5], axis=1)
        speed_axis.plot(time, speed, color=COLORS[name], linewidth=1.8)
    speed_axis.axhline(
        args.target_speed,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
        label=f"Target {args.target_speed:g} m/s",
    )
    speed_axis.set_title("Vehicle speed")
    speed_axis.set_xlabel("simulated time [s]")
    speed_axis.set_ylabel("speed [m/s]")
    speed_axis.legend(loc="lower right")
    style_axis(speed_axis)

    timing_axis = axes[1, 1]
    for name, data in series.items():
        timing = data["mppi_time"][:steps] * 1000.0
        timing_axis.plot(time, timing, color=COLORS[name], linewidth=1.3)
        timing_axis.text(
            0.98,
            {"Old Query (3.9 m)": 0.92, "Small-car Query (0.21 m)": 0.83,
             "Torch DBM": 0.74}[name],
            f"{name}: mean {timing.mean():.1f} ms",
            color=COLORS[name],
            ha="right",
            transform=timing_axis.transAxes,
        )
    timing_axis.axhline(
        args.sim_dt * 1000.0,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
        label=f"{args.sim_dt * 1000:g} ms deadline",
    )
    timing_axis.set_title("MPPI computation time")
    timing_axis.set_xlabel("simulated time [s]")
    timing_axis.set_ylabel("time [ms]")
    timing_axis.legend(loc="lower right")
    style_axis(timing_axis)

    figure.suptitle(
        f"AnyCar Quick Start — aligned {steps} steps / "
        f"{steps * args.sim_dt:.2f} s",
        fontsize=15,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "closed_loop_comparison.png"
    svg_path = args.output_dir / "closed_loop_comparison.svg"
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    print(f"aligned_steps: {steps}")
    print(f"png: {png_path}")
    print(f"svg: {svg_path}")


if __name__ == "__main__":
    main()
