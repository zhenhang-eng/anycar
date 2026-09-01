#!/usr/bin/env python3
"""Plot multi-elite label coverage and fresh-seed DBM oracle diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_multi_elite_expansion_20260804_v1"
)
DEFAULT_ORACLE = Path("outputs/mppi_proposal/dbm_multi_elite_oracle_20260804_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_summary = json.loads((args.multi_labels / "summary.json").read_text())
    oracle = json.loads((args.oracle_dir / "summary.json").read_text())
    with (args.oracle_dir / "per_snapshot.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = args.output or args.oracle_dir / "multicenter_oracle_result.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "warm": "#7f8c8d",
        "teacher": "#d95319",
        "mean": "#8c6bb1",
        "oracle": "#2878b5",
        "mixture": "#2ca02c",
    }
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)

    axis = axes[0, 0]
    split_names = ("train", "validation", "test")
    split_rows = {
        split: [row for row in rows if row["split"] == split] for split in split_names
    }
    # The oracle run is test-only, so use the immutable label summary for train/validation.
    distributions = {}
    for split in split_names:
        paths = sorted((args.multi_labels).glob(f"episode_*/*.npz"))
        split_episodes = set(json.loads((args.multi_labels / "splits.json").read_text())[split])
        counts = []
        for path in paths:
            if path.parent.name not in split_episodes:
                continue
            with np.load(path, allow_pickle=False) as label:
                counts.append(int(label["elite_count"]))
        distributions[split] = counts
    x = np.arange(3)
    bottom = np.zeros(3)
    palette = ("#d9e2ec", "#9fbad0", "#5d91bb", "#26699c")
    for count, color in zip((1, 2, 3, 4), palette):
        values = np.asarray(
            [sum(value == count for value in distributions[split]) for split in split_names]
        )
        axis.bar(x, values, bottom=bottom, color=color, label=f"{count} elite")
        bottom += values
    axis.set_xticks(x, ("Train (360)", "Validation (120)", "Test (120)"))
    axis.set_ylabel("Snapshots")
    axis.set_title("A. Diverse stable elite labels")
    axis.legend(frameon=False, ncol=4, fontsize=8)

    axis = axes[0, 1]
    metric = oracle["overall"]["methods"]
    names = (
        "warm",
        "teacher",
        "elite_mean_k2",
        "full_budget_state_oracle_k2",
        "elite_mean_k3",
        "full_budget_state_oracle_k3",
    )
    labels = ("Warm", "Teacher", "Mean K2", "Oracle K2", "Mean K3", "Oracle K3")
    values = [metric[name]["mean"] for name in names]
    bar_colors = (
        colors["warm"],
        colors["teacher"],
        colors["mean"],
        colors["oracle"],
        colors["mean"],
        colors["oracle"],
    )
    bars = axis.bar(np.arange(len(names)), values, color=bar_colors)
    axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axis.set_xticks(np.arange(len(names)), labels, rotation=18, ha="right")
    axis.set_ylabel("Mean weighted-output DBM cost")
    axis.set_title("B. Full test set, 256 candidates per evaluated center")

    axis = axes[1, 0]
    x = np.arange(3)
    width = 0.24
    eligible_counts = []
    mean_gain = []
    oracle_gain = []
    mixture_gain = []
    for count in (2, 3, 4):
        eligible = [row for row in rows if int(row["elite_count"]) >= count]
        eligible_counts.append(len(eligible))
        teacher = np.asarray([float(row["teacher_cost"]) for row in eligible])
        mean_gain.append(
            float(
                np.mean(
                    teacher
                    - np.asarray([float(row[f"elite_mean_k{count}_cost"]) for row in eligible])
                )
            )
        )
        oracle_gain.append(
            float(
                np.mean(
                    teacher
                    - np.asarray(
                        [
                            float(row[f"full_budget_state_oracle_k{count}_cost"])
                            for row in eligible
                        ]
                    )
                )
            )
        )
        mixture_gain.append(
            float(
                np.mean(
                    teacher
                    - np.asarray(
                        [float(row[f"fixed_budget_mixture_k{count}_cost"]) for row in eligible]
                    )
                )
            )
        )
    bars_a = axis.bar(x - width, mean_gain, width, color=colors["mean"], label="Center mean")
    bars_b = axis.bar(x, oracle_gain, width, color=colors["oracle"], label="Perfect selector")
    bars_c = axis.bar(x + width, mixture_gain, width, color=colors["mixture"], label="Fixed-budget mixture")
    for bars in (bars_a, bars_b, bars_c):
        axis.bar_label(bars, fmt="%+.3f", padding=2, fontsize=8)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, [f"K={count}\n(n={n})" for count, n in zip((2, 3, 4), eligible_counts)])
    axis.set_ylabel("Teacher cost - method cost (positive is better)")
    axis.set_title("C. Only snapshots with at least K valid elites")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    scale = np.asarray((0.0, 0.25, 0.50, 0.75, 1.0))
    scale_cost = np.asarray(
        (
            metric["warm"]["mean"],
            metric["teacher_scale_25"]["mean"],
            metric["teacher_scale_50"]["mean"],
            metric["teacher_scale_75"]["mean"],
            metric["teacher"]["mean"],
        )
    )
    axis.plot(scale, scale_cost, marker="o", linewidth=2, color=colors["teacher"])
    for x_value, y_value in zip(scale, scale_cost):
        axis.annotate(f"{y_value:.3f}", (x_value, y_value), xytext=(0, 7), textcoords="offset points", ha="center")
    axis.set_xticks(scale)
    axis.set_xlabel("Fraction of T1 teacher residual applied to warm")
    axis.set_ylabel("Mean weighted-output DBM cost")
    axis.set_title("D. Residual shrinkage is expensive")

    figure.suptitle(
        "DBM multi-elite oracle: mode averaging hurts, extra centers add little beyond T1 teacher",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output.resolve())


if __name__ == "__main__":
    main()
