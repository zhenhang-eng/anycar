#!/usr/bin/env python3
"""Independently replay the selected train-only Alpha Actor--Critic checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_alpha_actor_critic import (
    DEFAULT_OUTPUT,
    critic_metrics,
    gate_metrics,
    grouped_metrics,
    make_critic,
    policy_eval_args,
)
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)
from train_mppi_direct_trust_alpha_policy import (
    evaluate,
    extra_tensors,
    make_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def numeric_max_error(left, right) -> float:
    errors = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() & right.keys():
            errors.append(numeric_max_error(left[key], right[key]))
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        errors.append(abs(float(left) - float(right)))
    return max(errors, default=0.0)


def main() -> None:
    args = parse_args()
    summary = json.loads((args.run / "training_summary.json").read_text())
    checkpoint = Path(summary["selected_checkpoint"])
    if sha256_file(checkpoint) != summary["selected_checkpoint_sha256"]:
        raise AssertionError("selected Alpha AC checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu")
    if sha256_file(Path(payload["initial_policy"])) != payload["initial_policy_sha256"]:
        raise AssertionError("TR2-B initialization hash mismatch")
    labels = Path(payload["labels"])
    for name, expected in payload["labels_hashes"].items():
        if sha256_file(labels / name) != expected:
            raise AssertionError(f"TR1 label hash mismatch: {name}")
    old_path = Path(payload["old_actor"])
    if sha256_file(old_path) != payload["old_actor_sha256"]:
        raise AssertionError("old Actor hash mismatch")
    old_payload = load_actor_payload(old_path)
    data, _, splits = load_dataset(labels, old_payload)
    device = torch.device(args.device)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    selection_index = np.flatnonzero(
        np.isin(data.episodes, splits["internal_selection"])
    )
    initial_payload = torch.load(payload["initial_policy"], map_location="cpu")
    initial_policy = make_policy(old_payload, device, dropout=0.0)
    initial_policy.load_state_dict(initial_payload["policy_state_dict"], strict=True)
    policy = make_policy(old_payload, device, dropout=0.0)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    q1 = make_critic(initial_policy, device, dropout=0.0)
    q2 = make_critic(initial_policy, device, dropout=0.0)
    q1.load_state_dict(payload["critic1_state_dict"], strict=True)
    q2.load_state_dict(payload["critic2_state_dict"], strict=True)
    train_args = argparse.Namespace(**payload["training_arguments"])
    train_args.device = args.device
    train_args.evaluation_batch_size = 128
    eval_args = policy_eval_args(train_args, initial_payload)
    initial_eval_args = argparse.Namespace(**vars(eval_args))
    initial_eval_args.move_threshold = float(initial_payload["move_threshold"])
    eval_args.move_threshold = float(payload["move_threshold"])
    initial_metrics = evaluate(
        initial_policy, data, tensors, extra, selection_index, initial_eval_args, device
    )
    selected_metrics = evaluate(
        policy, data, tensors, extra, selection_index, eval_args, device
    )
    critic = critic_metrics(
        q1, q2, data, tensors, extra, selection_index, train_args, device
    )
    speed = grouped_metrics(
        policy, data, tensors, extra, selection_index, eval_args, device
    )
    gates = gate_metrics(selected_metrics)
    speed_gate = all(row["gain_mean"] >= 0.0 for row in speed.values())
    improved = (
        selected_metrics["direct_cost"]["mean"]
        < initial_metrics["direct_cost"]["mean"] - 1e-6
    )
    saved_error = numeric_max_error(
        payload["internal_selection_metrics"], selected_metrics
    )
    qualification = (
        "ALPHA_AC_VALIDATED_READY_FOR_FORMAL_GATE"
        if all(gates.values()) and speed_gate and improved and saved_error <= 1e-5
        else "ALPHA_AC_VALIDATION_FAIL_RETAIN_TR2B"
    )
    report = {
        "format_version": 1,
        "run": str(args.run.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_seed": int(payload["seed"]),
        "selected_critic_epoch": int(payload["selected_critic_epoch"]),
        "selected_actor_epoch": int(payload["selected_actor_epoch"]),
        "internal_selection_context_count": len(selection_index),
        "internal_selection_transition_count": int(
            len(selection_index) * data.alpha_grid.shape[1]
        ),
        "initial_tr2b_metrics": initial_metrics,
        "selected_actor_metrics": selected_metrics,
        "critic_metrics": critic,
        "by_reference_speed_mps": speed,
        "tail_gates": gates,
        "all_speed_mean_nonregression": speed_gate,
        "improves_calibrated_tr2b": improved,
        "saved_metric_max_error": saved_error,
        "qualification": qualification,
        "test_policy": "formal validation and test not loaded or evaluated",
    }
    (args.run / "alpha_ac_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
