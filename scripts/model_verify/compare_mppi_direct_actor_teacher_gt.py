#!/usr/bin/env python3
"""Compare the current Direct Actor, T1 teacher, and DBM numerical oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor
from train_mppi_direct_center_actor_critic import (
    DEFAULT_PARENT,
    DEFAULT_RISK,
    DEFAULT_SOURCE,
    build_inputs,
    evaluate_actor_direct_costs,
    load_direct_partition,
)
from train_mppi_two_pass_feedback_critic import load_partition as load_state_partition


DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_center_replay_diverse_20260806_v2"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/direct_center_actor_trust_step_20260806_v2/"
    "direct_center_actor_trust_selected.pt"
)
DEFAULT_GT = Path(
    "outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v2"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_actor_teacher_gt_validation_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def distribution(cost: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(cost)),
        "median": float(np.median(cost)),
        "p95": float(np.quantile(cost, 0.95)),
        "maximum": float(np.max(cost)),
    }


def comparison(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    gap = lhs - rhs
    return {
        "mean_cost_gap": float(np.mean(gap)),
        "median_cost_gap": float(np.median(gap)),
        "p95_cost_gap": float(np.quantile(gap, 0.95)),
        "lhs_win_fraction": float(np.mean(lhs < rhs)),
        "tie_fraction": float(np.mean(np.isclose(lhs, rhs, atol=1e-5))),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    maximum = float(checkpoint["maximum_delta_sigma"])
    splits = json.loads((args.labels / "splits.json").read_text())
    episodes = splits["validation"]
    state = load_state_partition(
        args.source, args.parent_labels, episodes, "selection"
    )
    replay = load_direct_partition(
        args.labels, args.parent_labels, args.risk_labels, episodes, maximum
    )
    inputs = build_inputs(state, replay, checkpoint)
    actor = TorchMPPIDeterministicCenterActor(maximum, dropout=0.0).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor_cost = evaluate_actor_direct_costs(
        actor, inputs, replay, args.source, args.batch_size, device
    )

    gt_summary = json.loads((args.gt / "summary.json").read_text())
    if gt_summary["split"] != "validation" or gt_summary["snapshot_count"] != 300:
        raise AssertionError("GT is not the complete validation split")
    gt_lookup = {
        (row["episode"], row["snapshot"]): row for row in gt_summary["rows"]
    }
    warm_cost = np.asarray([
        gt_lookup[(path.parent.name, path.name)]["warm_cost"]
        for path in replay.label_paths
    ], np.float32)
    j16_cost = np.asarray([
        gt_lookup[(path.parent.name, path.name)]["j16_best_found"]
        for path in replay.label_paths
    ], np.float32)
    j100_cost = np.asarray([
        gt_lookup[(path.parent.name, path.name)]["j100_best_found"]
        for path in replay.label_paths
    ], np.float32)
    teacher_index = replay.center_names.index("t1_teacher")
    teacher_cost = replay.direct_cost[:, teacher_index]
    stored_teacher_from_gt = np.asarray([
        gt_lookup[(path.parent.name, path.name)]["teacher_cost"]
        for path in replay.label_paths
    ], np.float32)
    teacher_replay_error = float(np.max(np.abs(teacher_cost - stored_teacher_from_gt)))
    initial_index = replay.center_names.index("bootstrap_actor")
    prior_actor_cost = replay.direct_cost[:, initial_index]
    guard_cost = np.minimum(prior_actor_cost, actor_cost)

    speed = []
    for path in replay.label_paths:
        source_path = args.source / path.parent.name / "snapshots" / path.name
        with np.load(source_path, allow_pickle=False) as source:
            speed.append(float(source["reference_speed_override_mps"]))
    speed = np.asarray(speed)
    methods = {
        "warm": warm_cost,
        "prior_actor": prior_actor_cost,
        "current_actor": actor_cost,
        "two_center_guard": guard_cost,
        "t1_teacher": teacher_cost,
        "j16_best_found": j16_cost,
        "j100_best_found": j100_cost,
    }
    recoverable = float(np.mean(warm_cost - j16_cost))
    aggregate = {
        name: {
            **distribution(cost),
            "gap_to_j16_mean": float(np.mean(cost - j16_cost)),
            "gap_to_j100_mean": float(np.mean(cost - j100_cost)),
            "fraction_warm_to_j16_gap_recovered": float(
                np.mean(warm_cost - cost) / recoverable
            ),
        }
        for name, cost in methods.items()
    }
    by_speed = {}
    for value in sorted(set(speed.tolist())):
        mask = speed == value
        by_speed[f"{value:.1f}"] = {
            name: {
                "count": int(mask.sum()),
                "mean": float(np.mean(cost[mask])),
                "gap_to_j16_mean": float(np.mean(cost[mask] - j16_cost[mask])),
            }
            for name, cost in methods.items()
        }
    summary = {
        "format_version": 1,
        "method": "paired Direct Actor/T1 teacher/J16/J100 DBM comparison",
        "qualification": "VALIDATION_ONLY_TEST_SEALED",
        "validation_snapshots": gt_summary["snapshot_count"],
        "validation_feedback_contexts": len(actor_cost),
        "teacher_replay_max_abs_error": teacher_replay_error,
        "aggregate": aggregate,
        "paired": {
            "actor_vs_teacher": comparison(actor_cost, teacher_cost),
            "guard_vs_teacher": comparison(guard_cost, teacher_cost),
            "teacher_vs_j16": comparison(teacher_cost, j16_cost),
            "actor_vs_j16": comparison(actor_cost, j16_cost),
        },
        "by_reference_speed_mps": by_speed,
        "gt_semantics": gt_summary["semantics"],
        "test_policy": "test split not loaded or evaluated",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
