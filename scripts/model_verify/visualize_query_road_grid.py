#!/usr/bin/env python3
"""One figure, four road variants in a 2x2 grid (A top-left, B top-right,
C bottom-left, D bottom-right)."""
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

DT = 0.05
COLORS = {
    "mild_left_nominal": "#1f77b4",
    "moderate_right_nominal": "#d62728",
    "varying_left_recovery": "#2ca02c",
    "varying_right_recovery": "#ff7f0e",
}
TITLES = {
    "mild_left_nominal": "A · Mild Left — nominal circle",
    "moderate_right_nominal": "B · Moderate Right — nominal circle",
    "varying_left_recovery": "C · Left Recovery — oval, +0.75 m / +0.03 rad",
    "varying_right_recovery": "D · Right Recovery — oval, −0.75 m / −0.03 rad",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/figures/query_roads_2x2.png"))
    parser.add_argument("--speed-kph", type=int, default=70)
    args = parser.parse_args()

    speed = args.speed_kph / 3.6
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    for index, spec in enumerate(ROAD_VARIANTS):
        ax = axes[index // 2][index % 2]
        waypoints, geo = make_waypoints(dict(spec), speed)
        road = PeriodicArcLengthRoad(waypoints)

        s = np.linspace(0, road.total_length, 4000, endpoint=False)
        center = road.sample(s)
        ax.plot(center[:, 0], center[:, 1],
                color=COLORS[spec["name"]], linewidth=2.5)

        # Direction arrows
        for frac in (0.0, 0.25, 0.5, 0.75):
            pose = road.sample(frac * road.total_length)
            nxt = road.sample((frac + 0.012) * road.total_length)
            ax.annotate(
                "", xy=(nxt[0], nxt[1]), xytext=(pose[0], pose[1]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8,
                                mutation_scale=18),
            )

        # Vehicle start + reference window
        s_frac = 0.22
        pose0 = road.sample(s_frac * road.total_length)
        lateral = spec["lateral_offset_m"]
        heading = spec["heading_error_rad"]
        car = np.array([
            pose0[0] - np.sin(pose0[2]) * lateral,
            pose0[1] + np.cos(pose0[2]) * lateral,
            pose0[2] + heading, speed, 0.0, 0.0,
        ])
        s0 = road.project_s(car[:2])
        ref = road.sample(s0 + np.arange(51) * speed * DT)

        ax.plot(car[0], car[1], marker="^", color="crimson", markersize=13,
                markeredgecolor="black", label="vehicle start")
        ax.plot(ref[:, 0], ref[:, 1], color="black", linewidth=2.2,
                marker="o", markersize=3,
                label=f"reference {speed*DT*50:.0f} m @ {args.speed_kph} km/h")

        ax.set_title(
            f"{TITLES[spec['name']]}\n"
            f"ω={spec['target_yawrate_abs_rps']:.3f} rad/s · R={geo['base_radius_m']:.0f} m",
            fontsize=13, fontweight="bold", loc="left",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel("y [m]", fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(
        f"Query expected-road variants @ {args.speed_kph} km/h\n"
        "A/B nominal tracking (constant curvature) · C/D recovery "
        "(varying curvature + 0.75 m lateral / 0.03 rad heading offset)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
