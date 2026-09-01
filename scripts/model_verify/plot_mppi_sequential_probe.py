#!/usr/bin/env python3
"""Plot budget efficiency and fresh DBM results for sequential MPPI probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TRAINING = Path(
    "outputs/mppi_proposal/sequential_probe_sac_20260805_v2/training_summary.json"
)
DEFAULT_FRESH = Path(
    "outputs/mppi_proposal/sequential_probe_teacher_eval_20260805_v5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = json.loads(args.training_summary.read_text())
    fresh = json.loads((args.fresh_dir / "summary.json").read_text())
    with np.load(args.fresh_dir / "fresh_eval.npz", allow_pickle=False) as data:
        method_names = data["method_names"].astype(str).tolist()
        cost = np.asarray(data["costs"], np.float32)
    output = args.output or args.fresh_dir / "sequential_probe_result.png"
    audit = training["audit_test"]
    budget_keys = [key for key in ("1", "2", "3", "4", "8") if key in audit]
    rollout_budget = [audit[key]["candidate_rollout_budget"] for key in budget_keys]
    mean_gain = [audit[key]["mean_independent_seed_advantage"] for key in budget_keys]
    rollout_budget.append(audit["full_33_probe"]["candidate_rollout_budget"])
    mean_gain.append(audit["full_33_probe"]["mean_independent_seed_advantage"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(rollout_budget, mean_gain, "o-", linewidth=2, color="#0068b5")
    axis.axvline(256, color="#e87722", linestyle="--", linewidth=1.5)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("DBM candidate rollouts used for probes")
    axis.set_ylabel("Independent-seed cost reduction vs guided")
    axis.set_title("A. Probe-budget efficiency (audit replay)")
    axis.grid(alpha=0.25)
    ratio = audit["4"]["mean_independent_seed_advantage"] / audit[
        "full_33_probe"
    ]["mean_independent_seed_advantage"]
    axis.annotate(
        f"4 probes retain {100 * ratio:.1f}% of 33-probe mean gain",
        xy=(256, audit["4"]["mean_independent_seed_advantage"]),
        xytext=(320, max(mean_gain) * 0.64),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )

    axis = axes[0, 1]
    methods = [
        "guided", "critic_pos_0p10", "fixed_priority_probe",
        "sequential_probe", "t1_teacher",
    ]
    labels = [
        "Guided", "Critic +0.10", "Fixed probes (4)",
        "Sequential (4)", "T1 teacher",
    ]
    means = [fresh["method_cost"][name]["mean"] for name in methods]
    p95 = [fresh["method_cost"][name]["p95_context"] for name in methods]
    x = np.arange(len(methods))
    width = 0.36
    axis.bar(x - width / 2, means, width, label="Mean cost", color="#4c78a8")
    axis.bar(x + width / 2, p95, width, label="P95 context cost", color="#f58518")
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.set_ylabel("Weighted-output DBM cost (lower is better)")
    axis.set_title("B. Fully fresh DBM gate")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)

    sequential_index = method_names.index("sequential_probe")
    guided_index = method_names.index("guided")
    teacher_index = method_names.index("t1_teacher")
    context_cost = cost.mean(axis=3).reshape(-1, cost.shape[2])
    guided_gain = context_cost[:, guided_index] - context_cost[:, sequential_index]
    teacher_gain = context_cost[:, teacher_index] - context_cost[:, sequential_index]
    axis = axes[1, 0]
    for values, label, color in (
        (guided_gain, "Sequential vs guided", "#54a24b"),
        (teacher_gain, "Sequential vs T1 teacher", "#b279a2"),
    ):
        ordered = np.sort(values)
        probability = np.arange(1, len(ordered) + 1) / len(ordered)
        axis.plot(ordered, probability, linewidth=2, label=label, color=color)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlim(-20, 20)
    axis.set_xlabel("Context cost reduction (positive is better)")
    axis.set_ylabel("Empirical CDF")
    axis.set_title("C. Paired context-gain distribution (clipped x view)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    histogram = np.asarray(fresh["actor_action_histogram"], np.int64)
    top = np.argsort(histogram)[::-1][:7]
    checkpoint = torch_load_action_names(args.training_summary.parent)
    axis = axes[1, 1]
    axis.barh(
        np.arange(len(top)), histogram[top][::-1], color="#72b7b2"
    )
    axis.set_yticks(
        np.arange(len(top)), [checkpoint[index] for index in top[::-1]]
    )
    axis.set_xlabel("Final selected contexts (of 600)")
    axis.set_title("D. Final best-so-far center")
    axis.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Same-state sequential MPPI probing: repeated Actor inference with frozen networks",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


def torch_load_action_names(training_dir: Path) -> list[str]:
    import torch

    checkpoint = torch.load(
        training_dir / "sequential_probe_sac.pt", map_location="cpu"
    )
    return list(checkpoint["action_names"])


if __name__ == "__main__":
    main()
