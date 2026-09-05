#!/usr/bin/env python3
"""Presentation figure for the Query expected-road sampling contract.

Panels:
  A  four road variants (legend outside)
  B  sampling mechanism in Frenet frame, cross-track axis compressed to
     +/-1.6 m so the 0.75 m offset and the 0.61 m bow are visually dominant
  C  speed -> reference window length, horizontal bar chart (no more tiny fan)
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
COLORS = {
    "mild_left_nominal": "#1f77b4",
    "moderate_right_nominal": "#d62728",
    "varying_left_recovery": "#2ca02c",
    "varying_right_recovery": "#ff7f0e",
}
SHORT = {
    "mild_left_nominal": "mild left · circle · ω=0.012",
    "moderate_right_nominal": "moderate right · circle · ω=0.030",
    "varying_left_recovery": "left recovery · oval · ω=0.040",
    "varying_right_recovery": "right recovery · oval · ω=0.040",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/figures/query_expected_road_overview.png"))
    args = parser.parse_args()

    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.25, 0.9), hspace=0.34,
                          wspace=0.26, top=0.88, bottom=0.06)
    speed70 = 70 / 3.6

    # ---- Panel A ----
    axA = fig.add_subplot(gs[0, 0])
    for spec in ROAD_VARIANTS:
        waypoints, geo = make_waypoints(dict(spec), speed70)
        road = PeriodicArcLengthRoad(waypoints)
        s = np.linspace(0, road.total_length, 3000, endpoint=False)
        c = road.sample(s)
        label = f"{SHORT[spec['name']]}  (R={geo['base_radius_m']:.0f} m)"
        axA.plot(c[:, 0], c[:, 1], color=COLORS[spec["name"]],
                 linewidth=1.8, label=label)
    axA.set_aspect("equal", adjustable="box")
    axA.set_title("A · 4 road variants @ 70 km/h\nclosed periodic centerlines",
                  fontsize=13, fontweight="bold", loc="left")
    axA.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0),
               framealpha=0.95, borderpad=0.6)
    axA.set_xlabel("x [m]", fontsize=9)
    axA.set_ylabel("y [m]", fontsize=9)
    axA.grid(alpha=0.2, linewidth=0.5)

    # ---- Panel B: Frenet zoom with exaggerated cross scale ----
    axB = fig.add_subplot(gs[0, 1:])
    spec = ROAD_VARIANTS[2]  # left recovery: +0.75 m, +0.03 rad
    waypoints, geo = make_waypoints(dict(spec), speed70)
    road = PeriodicArcLengthRoad(waypoints)
    s_frac = 0.25
    pose0 = road.sample(s_frac * road.total_length)
    car = np.array([
        pose0[0] - np.sin(pose0[2]) * spec["lateral_offset_m"],
        pose0[1] + np.cos(pose0[2]) * spec["lateral_offset_m"],
    ])
    s0 = road.project_s(car)
    ref = road.sample(s0 + np.arange(51) * speed70 * DT)

    tangent = pose0[2]
    cos_t, sin_t = np.cos(tangent), np.sin(tangent)

    def to_frenet(points):
        delta = np.asarray(points)[:, :2] - pose0[:2]
        along = delta[:, 0] * cos_t + delta[:, 1] * sin_t
        cross = -delta[:, 0] * sin_t + delta[:, 1] * cos_t
        return along, cross

    s_local = np.linspace(s0 - 55, s0 + speed70 * DT * 51 + 55, 900)
    local = road.sample(s_local)
    la, lc = to_frenet(local)
    ra, rc = to_frenet(ref[:, :2])
    (ca,), (cc,) = to_frenet(car[None])

    axB.plot(la, lc, color=COLORS[spec["name"]], linewidth=3.5,
             label="road centerline (left recovery oval)")
    axB.plot(ra, rc, color="black", linewidth=2.8, marker="o",
             markersize=4.5, label="reference · 51 pts · dt=0.05 s (2.5 s)")
    axB.annotate(
        "", xy=(0.0, 0.0), xytext=(ca, cc),
        arrowprops=dict(arrowstyle="-|>", color="crimson", lw=3.0,
                        mutation_scale=24),
    )
    axB.plot(ca, cc, marker="^", color="crimson", markersize=18,
             markeredgecolor="black", zorder=6,
             label="vehicle · lateral offset +0.75 m")
    axB.plot(0.0, 0.0, marker="x", color="crimson", markersize=14,
             markeredgewidth=3.5, zorder=6, label="arc-length projection s₀")
    axB.axhline(0.0, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)

    axB.set_xlim(-55, 55)
    axB.set_ylim(-1.6, 1.6)
    axB.set_xticks(np.arange(-50, 51, 10))
    axB.set_yticks(np.arange(-1.5, 1.6, 0.5))
    axB.set_title(
        "B · How the reference is sampled (Frenet frame, cross-axis ×12, 70 km/h)\n"
        "project onto centerline → arc length s₀ + k·v_ref·dt → 51 points [x, y, yaw, v_ref]",
        fontsize=13, fontweight="bold", loc="left")
    axB.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    axB.set_xlabel("along-track [m]", fontsize=10)
    axB.set_ylabel("cross-track [m]", fontsize=10)
    axB.grid(alpha=0.3, linewidth=0.5)

    # ---- Panel C: window length vs speed, bar chart ----
    axC = fig.add_subplot(gs[1, :])
    speeds_kph = (40, 55, 70, 85, 100)
    lengths = [kph / 3.6 * 2.5 for kph in speeds_kph]
    viridis = plt.cm.viridis(np.linspace(0.15, 0.9, len(speeds_kph)))
    bars = axC.barh(
        [f"{k} km/h" for k in speeds_kph], lengths,
        color=viridis, edgecolor="black", linewidth=0.8, height=0.62,
    )
    for bar, length in zip(bars, lengths):
        axC.text(
            length + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{length:.0f} m", va="center", fontsize=12, fontweight="bold",
        )
    axC.set_xlim(0, 82)
    axC.invert_yaxis()
    axC.set_xlabel("reference window arc length [m]  =  v_ref × 2.5 s", fontsize=11)
    axC.set_title(
        "C · Speed sets the reference horizon length",
        fontsize=13, fontweight="bold", loc="left")
    axC.grid(axis="x", alpha=0.3, linewidth=0.5)
    axC.tick_params(labelsize=11)

    fig.suptitle(
        "Query expected-road sampling · PeriodicArcLengthRoad · "
        "5 speeds × 4 variants",
        fontsize=16, fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
