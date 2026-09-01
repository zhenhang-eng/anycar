#!/usr/bin/env python3
"""Plot the T2 BC label fit and fresh-seed DBM proposal evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TRAINING = Path("outputs/mppi_proposal/bc_t1_conv_v1/training_summary.json")
DEFAULT_EVALUATION = Path("outputs/mppi_proposal/bc_t1_conv_v1/offline_dbm_eval_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = json.loads(args.training_summary.read_text())
    evaluation = json.loads((args.evaluation_dir / "summary.json").read_text())
    with (args.evaluation_dir / "per_snapshot.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = args.output or args.evaluation_dir / "bc_t1_result.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    selected_path = Path(training["selected_checkpoint"])
    selected = next(
        run for run in training["runs"] if Path(run["checkpoint"]) == selected_path
    )
    split_order = ("train", "validation", "test")
    splits = tuple(split for split in split_order if split in evaluation["by_split"])
    if not splits:
        raise ValueError("evaluation summary does not contain any episode split")
    split_labels = tuple(
        f"{split.capitalize()} ({evaluation['by_split'][split]['snapshot_count']})"
        for split in splits
    )
    colors = {"warm": "#7f8c8d", "network": "#2878b5", "teacher": "#d95319"}
    split_colors = {
        "train": "#4c78a8",
        "validation": "#f2a541",
        "test": "#c44e52",
    }

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)

    axis = axes[0, 0]
    x = np.arange(len(splits))
    width = 0.24
    for offset, method in zip((-1, 0, 1), ("warm", "network", "teacher")):
        values = [
            evaluation["by_split"][split]["methods"][method][
                "weighted_output_cost"
            ]["mean"]
            for split in splits
        ]
        bars = axis.bar(
            x + offset * width,
            values,
            width,
            label=method.capitalize(),
            color=colors[method],
        )
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axis.set_xticks(x, split_labels)
    axis.set_ylabel("Mean weighted-output DBM cost (lower is better)")
    axis.set_title("A. Fresh-seed proposal quality")
    axis.legend(frameon=False, ncol=3)

    axis = axes[0, 1]
    offset = 0
    boundaries = []
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        gains = np.sort(
            [
                float(row["warm_weighted_output_cost"])
                - float(row["network_weighted_output_cost"])
                for row in split_rows
            ]
        )
        indices = np.arange(len(gains)) + offset
        axis.scatter(
            indices,
            gains,
            s=22,
            color=split_colors[split],
            label=split.capitalize(),
            zorder=3,
        )
        offset += len(gains)
        boundaries.append(offset)
    axis.axhline(0.0, color="black", linewidth=1)
    for boundary in boundaries[:-1]:
        axis.axvline(boundary - 0.5, color="#aaaaaa", linewidth=0.8)
    axis.set_xlabel("Snapshots, sorted within each episode split")
    axis.set_ylabel("Warm cost - network cost (positive is better)")
    axis.set_title("B. Per-snapshot network gain")
    axis.legend(frameon=False, ncol=3)

    axis = axes[1, 0]
    all_values = []
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        warm = np.asarray(
            [float(row["warm_weighted_output_cost"]) for row in split_rows]
        )
        network = np.asarray(
            [float(row["network_weighted_output_cost"]) for row in split_rows]
        )
        all_values.extend(warm.tolist())
        all_values.extend(network.tolist())
        axis.scatter(
            warm,
            network,
            s=28,
            alpha=0.78,
            color=split_colors[split],
            label=split.capitalize(),
        )
    limits = (min(all_values) - 0.5, max(all_values) + 0.5)
    axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Warm weighted-output cost")
    axis.set_ylabel("Network weighted-output cost")
    axis.set_title("C. Paired comparison (below diagonal is better)")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    prediction = [
        selected["label_fit"][split]["prediction_delta_rms"] for split in splits
    ]
    teacher = [selected["label_fit"][split]["teacher_delta_rms"] for split in splits]
    bars_a = axis.bar(
        x - width / 2,
        prediction,
        width,
        label="Network residual",
        color=colors["network"],
    )
    bars_b = axis.bar(
        x + width / 2,
        teacher,
        width,
        label="Teacher residual",
        color=colors["teacher"],
    )
    axis.bar_label(bars_a, fmt="%.3f", padding=2, fontsize=8)
    axis.bar_label(bars_b, fmt="%.3f", padding=2, fontsize=8)
    axis.set_xticks(x, split_labels)
    axis.set_ylabel("RMS center residual")
    axis.set_title("D. BC prediction shrinks toward warm start")
    axis.legend(frameon=False)

    figure.suptitle(
        "T2 lightweight BC proposal: "
        f"{selected['parameter_count']:,} parameters, "
        f"selected {selected['trust_multiplier']:g}-sigma seed {selected['seed']}",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output.resolve())


if __name__ == "__main__":
    main()
