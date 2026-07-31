#!/usr/bin/env python3
"""Compare the first complete Quick Start lap for small-car Query and DBM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from compare_mppi_rosbags import arrays_from_bag, read_bag


COLORS = {
    "Small-car Query (0.21 m)": "#0072B2",
    "Torch DBM": "#009E73",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-query-bag", type=Path, required=True)
    parser.add_argument("--dbm-bag", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument("--target-speed", type=float, default=2.0)
    parser.add_argument(
        "--figure-title",
        default="AnyCar Quick Start — first complete lap",
    )
    parser.add_argument("--noise-position-std", type=float, default=None)
    parser.add_argument("--noise-yaw-std", type=float, default=None)
    parser.add_argument("--noise-velocity-std", type=float, default=None)
    parser.add_argument("--noise-yawrate-std", type=float, default=None)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--state-topic", default="/odometry")
    parser.add_argument(
        "--use-track-distance",
        action="store_true",
        help="Evaluate lateral error from the selected state and reference track.",
    )
    return parser.parse_args()


def first_lap(data, track_xy, dt):
    tree = KDTree(track_xy)
    distance_to_track, nearest_index = tree.query(data["state"][:, :2])
    track_points = len(track_xy)
    index_delta = (
        (np.diff(nearest_index) + track_points // 2) % track_points
        - track_points // 2
    )
    progress = np.concatenate(([0], np.cumsum(index_delta)))
    completed = np.flatnonzero(progress >= track_points)
    if not len(completed):
        raise RuntimeError(
            f"bag only completed {progress[-1] / track_points:.3f} laps"
        )
    steps = min(
        int(completed[0]),
        len(data["state"]) - 1,
        len(data["action"]),
        len(data["lateral_error"]),
        len(data["mppi_time"]),
    )
    if steps < 1:
        raise RuntimeError("invalid first-lap transition count")
    return {
        "steps": steps,
        "state": data["state"][: steps + 1],
        "action": data["action"][:steps],
        "lateral_error": data["lateral_error"][:steps],
        "mppi_time": data["mppi_time"][:steps],
        "progress": progress[: steps + 1] / track_points,
        "nearest_distance": distance_to_track[: steps + 1],
        "time": np.arange(steps) * dt,
    }


def metrics(lap, dt, target_speed):
    lateral = lap["lateral_error"]
    abs_lateral = np.abs(lateral)
    state = lap["state"]
    speed = np.linalg.norm(state[:-1, 3:5], axis=1)
    speed_error = speed - target_speed
    timing_ms = lap["mppi_time"] * 1000.0
    distance = np.linalg.norm(np.diff(state[:, :2], axis=0), axis=1).sum()
    return {
        "steps": lap["steps"],
        "simulated_lap_time_s": lap["steps"] * dt,
        "measured_distance_m": float(distance),
        "final_track_progress": float(lap["progress"][-1]),
        "lateral_error": {
            "mae_m": float(abs_lateral.mean()),
            "rmse_m": float(np.sqrt(np.mean(lateral**2))),
            "p95_abs_m": float(np.percentile(abs_lateral, 95)),
            "max_abs_m": float(abs_lateral.max()),
        },
        "scalar_speed_error": {
            "mae_mps": float(np.abs(speed_error).mean()),
            "rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
            "final_speed_mps": float(np.linalg.norm(state[-1, 3:5])),
        },
        "mppi_time": {
            "mean_ms": float(timing_ms.mean()),
            "mean_after_warmup_ms": float(timing_ms[1:].mean()),
            "p95_ms": float(np.percentile(timing_ms, 95)),
            "max_ms": float(timing_ms.max()),
            "deadline_miss_rate": float(np.mean(timing_ms > dt * 1000.0)),
        },
        "maximum_nearest_track_distance_m": float(
            lap["nearest_distance"].max()
        ),
        "all_finite": bool(
            np.isfinite(state).all()
            and np.isfinite(lateral).all()
            and np.isfinite(timing_ms).all()
        ),
    }


def style_axis(axis):
    axis.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def distance_to_closed_polyline(points, track_xy):
    segment_begin = track_xy
    segment_end = np.roll(track_xy, -1, axis=0)
    segment = segment_end - segment_begin
    denominator = np.maximum(np.sum(segment * segment, axis=1), 1e-12)
    offset = points[:, None, :] - segment_begin[None, :, :]
    ratio = np.clip(
        np.sum(offset * segment[None, :, :], axis=2) / denominator[None, :],
        0.0,
        1.0,
    )
    projection = segment_begin[None, :, :] + ratio[..., None] * segment[None, :, :]
    return np.linalg.norm(points[:, None, :] - projection, axis=2).min(axis=1)


def main():
    args = parse_args()
    track = np.loadtxt(args.track, delimiter=",", skiprows=1)
    track_xy = track[:, :2]
    raw = {
        "Small-car Query (0.21 m)": arrays_from_bag(
            read_bag(args.small_query_bag), state_topic=args.state_topic
        ),
        "Torch DBM": arrays_from_bag(
            read_bag(args.dbm_bag), state_topic=args.state_topic
        ),
    }
    laps = {
        name: first_lap(data, track_xy, args.sim_dt)
        for name, data in raw.items()
    }
    if args.use_track_distance:
        for lap in laps.values():
            lap["lateral_error"] = distance_to_closed_polyline(
                lap["state"][:-1, :2], track_xy
            )
    summary = {
        "protocol": {
            "track": str(args.track.resolve()),
            "track_points": len(track_xy),
            "simulator_dt_s": args.sim_dt,
            "target_speed_mps": args.target_speed,
            "state_topic": args.state_topic,
            "lateral_error_source": (
                "ground-truth distance to reference track polyline"
                if args.use_track_distance
                else "/lateral_error"
            ),
            "observation_noise": {
                "position_std_m": args.noise_position_std,
                "yaw_std_rad": args.noise_yaw_std,
                "velocity_std_mps": args.noise_velocity_std,
                "yawrate_std_radps": args.noise_yawrate_std,
                "seed": args.noise_seed,
            },
            "lap_definition": "first nearest-track cyclic-index progress >= 1.0",
        },
        **{
            name: metrics(lap, args.sim_dt, args.target_speed)
            for name, lap in laps.items()
        },
    }

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
    closed_track = np.vstack((track_xy, track_xy[0]))
    trajectory_axis.plot(
        closed_track[:, 0], closed_track[:, 1], "--", color="#555555",
        linewidth=1.3, label="Reference track"
    )
    for name, lap in laps.items():
        state = lap["state"]
        trajectory_axis.plot(
            state[:, 0], state[:, 1], color=COLORS[name], linewidth=2,
            label=name
        )
        trajectory_axis.scatter(
            state[-1, 0], state[-1, 1], color=COLORS[name], s=28, zorder=3
        )
    trajectory_axis.scatter(
        0.0, 0.0, marker="*", color="#222222", s=90, label="Start", zorder=4
    )
    trajectory_axis.set_title("First complete lap trajectory")
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.axis("equal")
    trajectory_axis.legend(loc="upper left")
    style_axis(trajectory_axis)

    lateral_axis = axes[0, 1]
    for name, lap in laps.items():
        progress = lap["progress"][:-1] * 100.0
        values = np.abs(lap["lateral_error"])
        lateral_axis.plot(progress, values, color=COLORS[name], linewidth=1.8)
        result = summary[name]
        y = 0.91 if "Query" in name else 0.82
        lateral_axis.text(
            0.98, y,
            f"{name}: MAE {result['lateral_error']['mae_m']:.3f} m",
            color=COLORS[name], ha="right", transform=lateral_axis.transAxes
        )
    lateral_axis.set_title("Absolute lateral tracking error")
    lateral_axis.set_xlabel("lap progress [%]")
    lateral_axis.set_ylabel("absolute error [m]")
    lateral_axis.set_xlim(0, 100.5)
    style_axis(lateral_axis)

    speed_axis = axes[1, 0]
    for name, lap in laps.items():
        progress = lap["progress"][:-1] * 100.0
        speed = np.linalg.norm(lap["state"][:-1, 3:5], axis=1)
        speed_axis.plot(progress, speed, color=COLORS[name], linewidth=1.8,
                        label=name)
    speed_axis.axhline(
        args.target_speed, color="#333333", linestyle="--", linewidth=1.4,
        label=f"Target {args.target_speed:g} m/s"
    )
    speed_axis.set_title("Vehicle speed")
    speed_axis.set_xlabel("lap progress [%]")
    speed_axis.set_ylabel("speed [m/s]")
    speed_axis.set_xlim(0, 100.5)
    speed_axis.legend(loc="lower right")
    style_axis(speed_axis)

    timing_axis = axes[1, 1]
    for name, lap in laps.items():
        progress = lap["progress"][:-1] * 100.0
        timing = lap["mppi_time"] * 1000.0
        timing_axis.plot(progress, timing, color=COLORS[name], linewidth=1.3)
        result = summary[name]
        y = 0.91 if "Query" in name else 0.82
        timing_axis.text(
            0.98, y,
            f"{name}: mean {result['mppi_time']['mean_ms']:.1f} ms",
            color=COLORS[name], ha="right", transform=timing_axis.transAxes
        )
    timing_axis.axhline(
        args.sim_dt * 1000.0, color="#333333", linestyle="--", linewidth=1.4,
        label=f"{args.sim_dt * 1000:g} ms deadline"
    )
    timing_axis.set_title("MPPI computation time")
    timing_axis.set_xlabel("lap progress [%]")
    timing_axis.set_ylabel("time [ms]")
    timing_axis.set_xlim(0, 100.5)
    timing_axis.legend(loc="lower right")
    style_axis(timing_axis)

    figure.suptitle(args.figure_title, fontsize=15)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    png_path = args.output_dir / "full_lap_comparison.png"
    svg_path = args.output_dir / "full_lap_comparison.svg"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"png: {png_path}")
    print(f"svg: {svg_path}")


if __name__ == "__main__":
    main()
