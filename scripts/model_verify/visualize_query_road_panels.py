#!/usr/bin/env python3
"""Four separate road-variant figures (A/B/C/D), one variant per image.

Each figure shows one closed periodic centerline with its reference window,
so the geometry itself is the message — no panel crowding.
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

DT = 0.05
PANEL = {"A": 0, "B": 1, "C": 2, "D": 3}
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
SUBTITLES = {
    "mild_left_nominal": "gentlest curvature: ω = 0.012 rad/s, constant radius",
    "moderate_right_nominal": "tighter steady bend: ω = 0.030 rad/s, constant radius",
    "varying_left_recovery": "varying curvature (oval 1.10×/0.90×), vehicle starts 0.75 m outside with heading error",
    "varying_right_recovery": "varying curvature (oval 1.10×/0.90×), vehicle starts 0.75 m outside with heading error",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/figures/query_roads"))
    parser.add_argument("--speed-kph", type=int, default=70)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    speed = args.speed_kph / 3.6
    for label, index in PANEL.items():
        spec = ROAD_VARIANTS[index]
        waypoints, geo = make_waypoints(dict(spec), speed)
        road = PeriodicArcLengthRoad(waypoints)

        fig, ax = plt.subplots(figsize=(9, 9))
        s = np.linspace(0, road.total_length, 4000, endpoint=False)
        center = road.sample(s)
        ax.plot(center[:, 0], center[:, 1],
                color=COLORS[spec["name"]], linewidth=3.0)

        # Direction arrows along the road
        for frac in (0.0, 0.25, 0.5, 0.75):
            pose = road.sample(frac * road.total_length)
            ax.annotate(
                "", xy=road.sample((frac + 0.012) * road.total_length)[:2],
                xytext=pose[:2],
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0,
                                mutation_scale=22),
            )

        # Vehicle + reference window at the recovery offset (if any)
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

        ax.plot(car[0], car[1], marker="^", color="crimson", markersize=16,
                markeredgecolor="black", label="vehicle start")
        ax.plot(ref[:, 0], ref[:, 1], color="black", linewidth=2.5,
                marker="o", markersize=3.5,
                label=f"reference window · {speed*DT*50:.0f} m @ {args.speed_kph} km/h")

        radius = geo["base_radius_m"]
        ax.set_title(
            f"{TITLES[spec['name']]}\n{SUBTITLES[spec['name']]}  ·  R = {radius:.0f} m",
            fontsize=14, fontweight="bold", loc="left",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
        ax.set_xlabel("x [m]", fontsize=11)
        ax.set_ylabel("y [m]", fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()

        path = args.output_dir / f"road_{label.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
