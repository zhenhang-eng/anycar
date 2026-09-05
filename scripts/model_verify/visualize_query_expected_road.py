#!/usr/bin/env python3
"""Visualize the Query expected-road sampling contract (4 variants x 5 speeds).

Reconstructs the exact PeriodicArcLengthRoad geometry and reference sampling
from collect_query_expected_road_closed_loop.py (same ROAD_VARIANTS spec,
radius rule, waypoint construction, and arc-length reference generation),
then renders one figure per speed bin: full closed centerline plus a zoomed
window showing the 51-point [x,y,yaw,v_ref] reference from a projected
vehicle position with a lateral offset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "scripts/model_verify")
from collect_query_expected_road_closed_loop import (
    PeriodicArcLengthRoad,
    ROAD_VARIANTS,
    make_waypoints,
)

SPEEDS_KPH = (40, 55, 70, 85, 100)
DT = 0.05
COLORS = {
    "mild_left_nominal": "#1f77b4",
    "moderate_right_nominal": "#d62728",
    "varying_left_recovery": "#2ca02c",
    "varying_right_recovery": "#ff7f0e",
}


def reference_window(road, state, speed_mps, count=51):
    s0 = road.project_s(state[:2])
    s = s0 + np.arange(count, dtype=np.float64) * speed_mps * DT
    pose = road.sample(s)
    ref = np.concatenate(
        (pose, np.full((count, 1), speed_mps)), axis=1
    ).astype(np.float32)
    return ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/figures/query_expected_road_sampling.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(
        len(SPEEDS_KPH), len(ROAD_VARIANTS),
        figsize=(4 * len(ROAD_VARIANTS), 3.4 * len(SPEEDS_KPH)),
    )
    # Column titles
    for col, spec in enumerate(ROAD_VARIANTS):
        name = spec["name"].replace("_", " ")
        axes[0, col].set_title(
            f"{name}\n{spec['kind']} | {'left' if spec['direction']>0 else 'right'}"
            f" | ω={spec['target_yawrate_abs_rps']:.3f} rad/s",
            fontsize=11,
        )

    for row, kph in enumerate(SPEEDS_KPH):
        speed = kph / 3.6
        for col, spec in enumerate(ROAD_VARIANTS):
            ax = axes[row, col]
            waypoints, geometry = make_waypoints(dict(spec), speed)
            road = PeriodicArcLengthRoad(waypoints)

            # Full closed centerline (subsampled for plotting)
            s_dense = np.linspace(0, road.total_length, 4000, endpoint=False)
            center = road.sample(s_dense)
            ax.plot(
                center[:, 0], center[:, 1],
                color=COLORS[spec["name"]], linewidth=1.2, alpha=0.85,
                label=f"centerline R={geometry['base_radius_m']:.0f} m",
            )

            # Vehicle position with the variant's recovery offset
            s_start = 0.25 * road.total_length
            pose0 = road.sample(s_start)
            lateral = spec["lateral_offset_m"]
            heading = spec["heading_error_rad"]
            car = np.array([
                pose0[0] - np.sin(pose0[2]) * lateral,
                pose0[1] + np.cos(pose0[2]) * lateral,
                pose0[2] + heading,
                speed, 0.0, 0.0,
            ])
            ref = reference_window(road, car, speed)

            # Reference window (51 points, 2.5 s)
            ax.plot(
                ref[:, 0], ref[:, 1],
                color="black", linewidth=2.0, marker="o", markersize=2.2,
                linestyle="-", alpha=0.9, label="reference 51pts (2.5 s)",
            )
            ax.plot(
                car[0], car[1], marker="^", color="crimson",
                markersize=9, markeredgewidth=1.2, label="vehicle (projected)",
            )

            # Zoom window around the reference segment
            pad = 0.35 * max(
                ref[:, 0].ptp(), ref[:, 1].ptp(), 12.0
            ) + 6.0
            x_mid, y_mid = ref[:, 0].mean(), ref[:, 1].mean()
            ax.set_xlim(x_mid - pad, x_mid + pad)
            ax.set_ylim(y_mid - pad, y_mid + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25, linewidth=0.5)

            if col == 0:
                ax.set_ylabel(
                    f"{kph} km/h\n({speed:.1f} m/s)\ny [m]", fontsize=10
                )
            if row == len(SPEEDS_KPH) - 1:
                ax.set_xlabel("x [m]", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Query expected-road sampling: PeriodicArcLengthRoad (4 variants × 5 speeds)\n"
        "radius = max(150, v/ω); reference = arc-length projection + 51 points at v·dt "
        "(dt=0.05 s, horizon 2.5 s); recovery variants add ±0.75 m lateral / ±0.03 rad heading",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
