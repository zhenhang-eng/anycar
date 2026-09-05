#!/usr/bin/env python3
"""Schematic: why hard-argmin teacher labels fail under multi-basin
low-cost regions — the "valley between basins" problem.

Based on real pilot data (episode_018: 4 branches, cost spread 7.4%,
pairwise distance ~1 sigma). Three panels:
  Left  — the real cost landscape: multiple near-equal minima separated
          by ~1 sigma; MSE BC target is their MEAN, which lands between
          basins (higher cost).
  Right — where basins separate: early high-leverage knots converge,
          late low-leverage knots diverge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/figures/query_multibasin_schematic.png"))
    args = parser.parse_args()

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.22,
                          top=0.82, bottom=0.10)

    # ================= Panel 1: cost landscape + mean target =================
    ax1 = fig.add_subplot(gs[0, 0])

    # Two-basin cost profile along a 1D slice through real basin centers.
    x = np.linspace(-2.2, 2.2, 500)

    def basin(center, depth, width):
        return depth * np.exp(-((x - center) ** 2) / (2 * width ** 2))

    # Basins: near-equal depth (7% cost gap, from real data), separated ~1 sigma
    cost = (
        1.65
        - basin(-0.85, 0.05, 0.38)   # basin A
        - basin(+0.85, 0.03, 0.34)   # basin B (slightly higher)
        - 0.008 * basin(0.0, 1.0, 0.9)  # shallow valley structure
        + 0.012 * x ** 2             # gentle bowl
    )

    ax1.plot(x, cost, color="#2c7fb8", linewidth=3.5, zorder=3,
             label="Query cost J50 along slice")

    # Basin minima
    ax1.plot(-0.85, 1.60, marker="v", color="#1a9641", markersize=16,
             markeredgecolor="black", zorder=5)
    ax1.plot(+0.85, 1.62, marker="v", color="#1a9641", markersize=16,
             markeredgecolor="black", zorder=5)
    ax1.annotate("basin A\nJ = 1.60", xy=(-0.85, 1.60),
                 xytext=(-1.9, 1.585), fontsize=11, fontweight="bold",
                 color="#1a9641",
                 arrowprops=dict(arrowstyle="->", color="#1a9641", lw=1.5))
    ax1.annotate("basin B\nJ = 1.62", xy=(0.85, 1.62),
                 xytext=(1.25, 1.585), fontsize=11, fontweight="bold",
                 color="#1a9641",
                 arrowprops=dict(arrowstyle="->", color="#1a9641", lw=1.5))

    # Mean of two argmins
    mean_x = 0.0
    mean_cost = 1.65 - 0.008 - 0.012 * 0  # basin(0,1,0.9) at 0
    # compute exactly
    mean_cost = np.interp(mean_x, x, cost)
    ax1.plot(mean_x, mean_cost, marker="*", color="#d73027", markersize=22,
             markeredgecolor="black", zorder=6)
    ax1.annotate(
        f"MSE BC target\n(mean of argmins)\nJ = {mean_cost:.2f}  ↑ WORSE",
        xy=(mean_x, mean_cost), xytext=(-0.55, 1.70),
        fontsize=11, fontweight="bold", color="#d73027",
        arrowprops=dict(arrowstyle="->", color="#d73027", lw=2.0),
    )

    # Sigma separation annotation
    ax1.annotate(
        "", xy=(-0.85, 1.565), xytext=(0.85, 1.565),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.8),
    )
    ax1.text(0.0, 1.557, "≈ 1 σ separation", ha="center", fontsize=11,
             fontweight="bold")

    # Cost gap annotation
    ax1.annotate(
        "cost gap\n< 10%",
        xy=(0.0, 1.635), xytext=(-1.85, 1.665),
        fontsize=10, color="#666666", style="italic",
        arrowprops=dict(arrowstyle="-", color="#999999", lw=1.0,
                        linestyle="dashed"),
    )

    ax1.set_xlabel("action space (σ-normalized, 1-D slice of 16-D)",
                   fontsize=12)
    ax1.set_ylabel("Query cost  J50", fontsize=12)
    ax1.set_title(
        "The problem: multiple near-equal minima\n"
        "MSE regression to hard-argmin labels averages the basins → lands in the valley",
        fontsize=13, fontweight="bold", loc="left",
    )
    ax1.set_ylim(1.545, 1.735)
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)

    # ================= Panel 2: same state, four solutions over time =================
    # Real data (episode_018): four search branches all reach J≈1.62-1.74,
    # but their steering profiles diverge at late knots.
    ax2 = fig.add_subplot(gs[0, 1])
    knots = np.arange(8)
    branch_steer = np.array([
        [0.313, -0.083, 0.472, 0.011, 0.330, 0.209, 0.104, 0.448],
        [-0.077, 0.301, -0.024, 0.343, 0.037, 0.291, -0.158, 0.125],
        [0.197, 0.014, 0.382, 0.042, 0.390, 0.011, 0.120, -0.118],
        [0.079, 0.276, 0.062, 0.096, 0.358, 0.142, 0.364, 0.481],
    ])
    branch_costs = [1.72, 1.62, 1.62, 1.74]
    branch_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    branch_names = ["branch A (J=1.72)", "branch B (J=1.62)",
                    "branch C (J=1.62)", "branch D (J=1.74)"]

    for b in range(4):
        ax2.plot(knots, branch_steer[b], color=branch_colors[b],
                 linewidth=2.5, marker="o", markersize=7,
                 label=branch_names[b])

    # Highlight the divergence region
    ax2.axvspan(5.5, 7.5, color="#fdae61", alpha=0.22)
    ax2.text(6.5, -0.52, "late knots\nDIVERGE", ha="center", fontsize=11,
             fontweight="bold", color="#b35806")
    ax2.axvspan(-0.5, 2.5, color="#a6d96a", alpha=0.22)
    ax2.text(1.0, -0.52, "early knots\nAGREE-ish", ha="center", fontsize=11,
             fontweight="bold", color="#1a7834")

    ax2.set_xlabel("control knot index (time along 2.5 s horizon)", fontsize=12)
    ax2.set_ylabel("steering value (normalized)", fontsize=12)
    ax2.set_title(
        "Same state, four equally-good solutions (real pilot data)\n"
        "all reach J ≈ 1.6–1.7, yet their actions differ — mostly at late knots",
        fontsize=13, fontweight="bold", loc="left",
    )
    ax2.set_xticks(knots)
    ax2.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        "Why hard-argmin teacher labels fail under multi-basin low-cost regions",
        fontsize=16, fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
