#!/usr/bin/env python3
"""Combine the formal S0-S3 and S4-S5 uncertainty-capacity results."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ("dx_body", "dy_body", "dvx", "dyawrate")
ORDER = (
    "s0_conditional_mlp",
    "s1_film",
    "s4_concat_transformer",
    "s5_history_cross_attention",
)
LABELS = {
    "s0_conditional_mlp": "S0 conditional MLP",
    "s1_film": "S1 layer fusion + FiLM",
    "s4_concat_transformer": "S4 concat + Transformer",
    "s5_history_cross_attention": "S5 history cross-attention",
}
COLORS = {
    "s0_conditional_mlp": "#4C78A8",
    "s1_film": "#F58518",
    "s4_concat_transformer": "#B279A2",
    "s5_history_cross_attention": "#E45756",
}


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-summary",
        type=Path,
        default=root
        / "outputs/formal_real_query_uncertainty_s0_s3/20260728T162845/summary.json",
    )
    parser.add_argument(
        "--capacity-summary",
        type=Path,
        default=root
        / "outputs/formal_real_query_uncertainty_capacity/20260728T164000/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "outputs/formal_real_query_uncertainty_capacity/20260728T164000/comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base = json.loads(args.base_summary.read_text())
    capacity = json.loads(args.capacity_summary.read_text())
    metrics = {**base["metrics"], **capacity["metrics"]}
    fits = {**base["fits"], **capacity["fits"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = (
        ("spearman_sigma_abs_error", "Mean Spearman", True),
        ("top10_error_auc", "Mean top-10% AUC", True),
        ("top10_error_recall_by_top10_sigma", "Mean top-10% recall", True),
        ("ence", "Mean ENCE", False),
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for axis, (metric, title, higher) in zip(axes, specs):
        values = [metrics[name]["aggregate"][metric] for name in ORDER]
        axis.bar(range(len(ORDER)), values, color=[COLORS[name] for name in ORDER])
        axis.set_xticks(range(len(ORDER)), [name.split("_")[0].upper() for name in ORDER])
        axis.set_title(f"{title}\n({'higher' if higher else 'lower'} is better)")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "01_capacity_aggregate.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(CHANNELS))
    width = 0.2
    for index, name in enumerate(ORDER):
        offset = (index - 1.5) * width
        axes[0].bar(
            x + offset,
            [metrics[name]["channels"][channel]["spearman_sigma_abs_error"] for channel in CHANNELS],
            width,
            color=COLORS[name],
            label=LABELS[name],
        )
        axes[1].bar(
            x + offset,
            [metrics[name]["channels"][channel]["top10_error_auc"] for channel in CHANNELS],
            width,
            color=COLORS[name],
            label=LABELS[name],
        )
    axes[0].set_title("Sigma-error Spearman")
    axes[1].set_title("Top-10% error AUC")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xticks(x, CHANNELS)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "02_capacity_channel_metrics.png", dpi=180)
    plt.close(fig)

    mean_parameters = 2_478_970
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name in ORDER:
        parameters = fits[name]["parameter_count"] / mean_parameters * 100.0
        latency = fits[name]["benchmark"]["1"]["milliseconds_per_batch"]
        axes[0].scatter(
            parameters,
            metrics[name]["aggregate"]["spearman_sigma_abs_error"],
            s=90,
            color=COLORS[name],
            label=LABELS[name],
        )
        axes[1].scatter(
            latency,
            metrics[name]["aggregate"]["top10_error_auc"],
            s=90,
            color=COLORS[name],
            label=LABELS[name],
        )
        axes[0].annotate(name.split("_")[0].upper(), (parameters, metrics[name]["aggregate"]["spearman_sigma_abs_error"]), xytext=(5, 5), textcoords="offset points")
        axes[1].annotate(name.split("_")[0].upper(), (latency, metrics[name]["aggregate"]["top10_error_auc"]), xytext=(5, 5), textcoords="offset points")
    axes[0].set(xlabel="Head parameters / mean-model parameters (%)", ylabel="Mean Spearman", title="Capacity versus error ranking")
    axes[1].set(xlabel="Head-only batch-1 latency (ms)", ylabel="Mean top-10% AUC", title="Latency versus high-error detection")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "03_accuracy_compute_frontier.png", dpi=180)
    plt.close(fig)

    combined = {
        "sources": {
            "base": str(args.base_summary.resolve()),
            "capacity": str(args.capacity_summary.resolve()),
        },
        "metrics": {name: metrics[name] for name in ORDER},
        "fits": {name: fits[name] for name in ORDER},
    }
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
