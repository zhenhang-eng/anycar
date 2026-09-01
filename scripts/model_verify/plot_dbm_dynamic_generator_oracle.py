#!/usr/bin/env python3
"""Plot fixed-state dynamic-generator oracle comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.result_dir / "summary.json").read_text())
    with (args.result_dir / "per_snapshot.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    names = list(summary["banks"])
    short = [name.split("_", 1)[0] for name in names]
    selected = np.asarray([summary["banks"][name]["selected_audit_mean"] for name in names])
    clairvoyant = np.asarray([summary["banks"][name]["clairvoyant_audit_mean"] for name in names])
    unrestricted = float(summary["unrestricted_audit_mean"])
    target = float(summary["gate"]["target_clairvoyant_mean"])
    episodes = [row["episode"].replace("episode_", "e") for row in rows]
    focus = [name for name in names if name in (
        "B0_current", "B2_current_adaptive", "B5_hybrid_current_elites",
        "B6_residual_response", "B7_hybrid_current_response",
    )]
    recovery = np.asarray(
        [[float(row[f"{name}_coverage_recovery"]) for row in rows] for name in focus]
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, selected, 0.36, label="selection -> audit")
    axes[0].bar(x + 0.18, clairvoyant, 0.36, label="audit clairvoyant")
    axes[0].axhline(unrestricted, color="black", linestyle="--", label="unrestricted")
    axes[0].axhline(target, color="tab:red", linestyle=":", label="50% recovery gate")
    axes[0].set_xticks(x, short)
    axes[0].set_ylabel("audit mean cost (lower is better)")
    axes[0].set_title("Same 33 slots and 64-candidate budget")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    width = 0.8 / len(focus)
    for index, name in enumerate(focus):
        axes[1].bar(
            np.arange(len(rows)) - 0.4 + width / 2 + index * width,
            recovery[index], width, label=name.split("_", 1)[0],
        )
    axes[1].axhline(0.5, color="tab:red", linestyle=":", label="gate")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(rows)), episodes)
    axes[1].set_ylabel("coverage gap recovery fraction")
    axes[1].set_title("Per-snapshot oracle recovery")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Forward-only dynamic center-generator pilot")
    fig.tight_layout()
    output = args.result_dir / "dynamic_generator_oracle.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
