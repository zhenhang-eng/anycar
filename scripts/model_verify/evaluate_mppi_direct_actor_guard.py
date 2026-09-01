#!/usr/bin/env python3
"""Evaluate a deterministic two-center DBM guard on held-out validation episodes."""

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
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_center_actor_two_center_guard_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tail-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    maximum = float(checkpoint["maximum_delta_sigma"])
    splits = json.loads((args.labels / "splits.json").read_text())
    validation_episodes = splits["validation"]
    if set(validation_episodes) & set(splits["test_sealed_not_generated"]):
        raise AssertionError("validation/test episode overlap")
    state = load_state_partition(
        args.source, args.parent_labels, validation_episodes, "selection"
    )
    replay = load_direct_partition(
        args.labels, args.parent_labels, args.risk_labels,
        validation_episodes, maximum,
    )
    inputs = build_inputs(state, replay, checkpoint)
    actor = TorchMPPIDeterministicCenterActor(maximum, dropout=0.0).to(device)
    if checkpoint.get("actor_class") == "TorchMPPIDeterministicCenterActor":
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    else:
        actor.load_stochastic_actor_state_dict(checkpoint["actor_state_dict"])
    actor_cost = evaluate_actor_direct_costs(
        actor, inputs, replay, args.source, args.batch_size, device
    )
    initial_index = replay.center_names.index("bootstrap_actor")
    initial_cost = replay.direct_cost[:, initial_index]
    actor_gain = initial_cost - actor_cost
    guard_cost = np.minimum(initial_cost, actor_cost)
    guard_gain = initial_cost - guard_cost

    tail = []
    for rank, index in enumerate(np.argsort(actor_gain)[:args.tail_count], start=1):
        label_path = replay.label_paths[index]
        source_path = (
            args.source / label_path.parent.name / "snapshots" / label_path.name
        )
        metadata = json.loads(source_path.with_suffix(".json").read_text())["scenario"]
        with np.load(source_path, allow_pickle=False) as source:
            initial_state = np.asarray(source["initial_state"], np.float32)
        tail.append({
            "rank": rank,
            "episode": label_path.parent.name,
            "snapshot": label_path.name,
            "context": int(replay.context_in_file[index]),
            "actor_gain": float(actor_gain[index]),
            "initial_cost": float(initial_cost[index]),
            "actor_cost": float(actor_cost[index]),
            "reference_speed_mps": float(metadata["reference_speed_override_mps"]),
            "simulated_time_s": float(metadata["simulated_time_s"]),
            "frenet_lateral_m": float(metadata["frenet_lateral_m"]),
            "frenet_heading_error_rad": float(metadata["frenet_heading_error_rad"]),
            "vx_mps": float(initial_state[3]),
            "yawrate_rad_s": float(initial_state[4]),
        })

    summary = {
        "format_version": 1,
        "method": "deterministic two-center model-cost guard",
        "qualification": "VALIDATION_ONLY_TEST_SEALED",
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_episodes": validation_episodes,
        "validation_contexts": len(actor_cost),
        "initial_direct_cost_mean": float(initial_cost.mean()),
        "actor_direct_cost_mean": float(actor_cost.mean()),
        "actor_gain_mean": float(actor_gain.mean()),
        "actor_gain_median": float(np.median(actor_gain)),
        "actor_gain_p05": float(np.quantile(actor_gain, 0.05)),
        "actor_worst_gain": float(actor_gain.min()),
        "guard_direct_cost_mean": float(guard_cost.mean()),
        "guard_gain_mean": float(guard_gain.mean()),
        "guard_gain_median": float(np.median(guard_gain)),
        "guard_gain_p05": float(np.quantile(guard_gain, 0.05)),
        "guard_worst_gain": float(guard_gain.min()),
        "guard_uses_actor_fraction": float(np.mean(actor_cost < initial_cost)),
        "positive_actor_gain_sum": float(actor_gain[actor_gain > 0].sum()),
        "negative_actor_gain_sum": float(actor_gain[actor_gain < 0].sum()),
        "tail_contexts": tail,
        "interpretation": (
            "DBM perfect-model upper bound only; Query must separately qualify "
            "pairwise cost ranking before using this guard."
        ),
        "test_policy": "episode_105..119 not loaded or evaluated",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
