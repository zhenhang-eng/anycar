#!/usr/bin/env python3
"""Replay interpolation from a frozen distilled Actor to validation J16."""

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
    actor_outputs,
    build_inputs,
    load_direct_partition,
)
from train_mppi_j16_oracle_distillation import (
    evaluate_center_methods,
    load_gt_knots,
    sha256_file,
)
from train_mppi_two_pass_feedback_critic import load_partition as load_state_partition


DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_center_replay_diverse_20260806_v2"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/direct_center_j16_distillation_20260807_v1/"
    "actor_scale6_seed1.pt"
)
DEFAULT_GT = Path("outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v2")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_center_j16_distillation_20260807_v1/"
    "validation_interpolation_analysis.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=(0, 0.25, 0.5, 0.75, 0.9, 0.95, 1)
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def distribution(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p95": float(np.quantile(value, 0.95)),
        "maximum": float(np.max(value)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["qualification"] != "VALIDATION_ONLY_TEST_SEALED":
        raise AssertionError("checkpoint does not seal test")
    splits = json.loads((args.labels / "splits.json").read_text())
    episodes = splits["validation"]
    state = load_state_partition(
        args.source, args.parent_labels, episodes, "selection"
    )
    maximum = float(checkpoint["maximum_delta_sigma"])
    replay = load_direct_partition(
        args.labels, args.parent_labels, args.risk_labels, episodes, maximum
    )
    inputs = build_inputs(state, replay, checkpoint)
    actor = TorchMPPIDeterministicCenterActor(maximum, dropout=0.0).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    _, actor_center = actor_outputs(actor, inputs, 128, device)
    oracle_knots, j16_cost, _ = load_gt_knots(
        args.gt, "validation", replay.label_paths
    )
    methods = {
        f"alpha_{alpha:g}": (
            (1.0 - alpha) * actor_center + alpha * oracle_knots
        ).astype(np.float32)
        for alpha in args.alphas
    }
    cost = evaluate_center_methods(methods, replay.label_paths, args.source, device)
    speed = []
    for path in replay.label_paths:
        source_path = args.source / path.parent.name / "snapshots" / path.name
        with np.load(source_path, allow_pickle=False) as source:
            speed.append(float(source["reference_speed_override_mps"]))
    speed = np.asarray(speed)
    summary = {
        "format_version": 1,
        "method": "validation interpolation from frozen Actor to J16 oracle",
        "qualification": "VALIDATION_ONLY_TEST_SEALED",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "validation_contexts": len(speed),
        "j16": distribution(j16_cost),
        "alphas": {},
        "test_policy": "test split not loaded or evaluated",
    }
    for name, value in cost.items():
        summary["alphas"][name] = {
            **distribution(value),
            "by_reference_speed_mps": {
                f"{one_speed:.1f}": float(np.mean(value[speed == one_speed]))
                for one_speed in sorted(set(speed.tolist()))
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
