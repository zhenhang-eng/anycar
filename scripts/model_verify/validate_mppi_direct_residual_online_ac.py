#!/usr/bin/env python3
"""Independently replay a 16-D residual direct Actor--Critic run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPIContinuousCenterCritic,
    TorchMPPIDeterministicCenterActor,
)
from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import (
    actor_inputs,
    center_from_residual,
    evaluate,
    load_j16,
    make_base_policy,
    module_batch,
    residual_outputs,
)
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--reward-absolute-tolerance", type=float, default=5e-4,
        help="Float32 CUDA DBM replay tolerance; policy metrics remain at --tolerance.",
    )
    parser.add_argument("--reward-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.run / "direct_residual_online_ac_selected.pt"
    replay_path = args.run / "direct_residual_replay.npz"
    summary = json.loads((args.run / "summary.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    frozen_batch_size = int(
        checkpoint["training_arguments"].get("evaluation_batch_size", 256)
    )
    if sha256_file(Path(checkpoint["base_alpha_checkpoint"])) != checkpoint["base_alpha_sha256"]:
        raise AssertionError("base Alpha checkpoint hash mismatch")
    labels = Path(checkpoint["labels"])
    for name, digest in checkpoint["labels_hashes"].items():
        if sha256_file(labels / name) != digest:
            raise AssertionError(f"label hash mismatch: {name}")
    for path, digest in zip(checkpoint["j16_summaries"], checkpoint["j16_summary_sha256"]):
        if sha256_file(Path(path)) != digest:
            raise AssertionError(f"J16 summary hash mismatch: {path}")

    device = torch.device(args.device)
    base_payload = torch.load(checkpoint["base_alpha_checkpoint"], map_location="cpu")
    old_payload = load_actor_payload(Path(base_payload["old_actor"]))
    data, _, splits = load_dataset(labels, old_payload)
    fit_index = np.flatnonzero(np.isin(data.episodes, checkpoint["fit_episodes"]))
    selection_index = np.flatnonzero(np.isin(data.episodes, checkpoint["selection_episodes"]))
    if set(data.episodes[fit_index]) & set(data.episodes[selection_index]):
        raise AssertionError("fit/selection episode leakage")
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    base_policy = make_base_policy(base_payload, device)
    all_index = np.arange(len(data.episodes))
    _, _, _, base_center = deterministic_outputs(
        base_policy, tensors, extra, all_index,
        float(checkpoint["base_move_threshold"]), frozen_batch_size, device,
    )
    base_cost = direct_cost(
        base_center, data, tensors, all_index, frozen_batch_size, device
    )
    j16_center, j16_cost, _ = load_j16(
        labels, [Path(path) for path in checkpoint["j16_summaries"]], len(data.episodes)
    )
    del j16_center
    inputs = actor_inputs(tensors, base_center, device)
    actor = TorchMPPIDeterministicCenterActor(
        float(checkpoint["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    validation_args = argparse.Namespace(evaluation_batch_size=frozen_batch_size)
    metrics, actor_action, actor_cost = evaluate(
        actor, inputs, data, tensors, selection_index, base_cost, j16_cost,
        validation_args, device,
    )
    saved_metrics = checkpoint["internal_selection_metrics"]
    metric_errors = {
        "direct_cost_mean": abs(
            metrics["direct_cost"]["mean"] - saved_metrics["direct_cost"]["mean"]
        ),
        "gain_vs_base_mean": abs(
            metrics["gain_vs_base_alpha"]["mean"]
            - saved_metrics["gain_vs_base_alpha"]["mean"]
        ),
        "gap_vs_j16_mean": abs(
            metrics["gap_vs_j16_best_found"]["mean"]
            - saved_metrics["gap_vs_j16_best_found"]["mean"]
        ),
    }
    _, center1 = residual_outputs(
        actor, inputs, selection_index, frozen_batch_size, device
    )
    _, center2 = residual_outputs(
        actor, inputs, selection_index, frozen_batch_size, device
    )
    deterministic_error = float(np.max(np.abs(center1 - center2)))

    with np.load(replay_path, allow_pickle=False) as replay:
        context = np.asarray(replay["context_index"], np.int64)
        action = np.asarray(replay["action"], np.float32)
        reward = np.asarray(replay["reward"], np.float32)
        source = np.asarray(replay["source"]).astype(str)
    if not (len(context) == len(action) == len(reward) == len(source)):
        raise AssertionError("replay arrays have different lengths")
    if not np.all(np.isin(context, fit_index)):
        raise AssertionError("replay contains non-fit context")
    reward_error = 0.0
    reward_relative_error = 0.0
    reward_tolerance_pass = True
    # Preserve collection-call boundaries. Broad unstable trajectories can
    # amplify tiny CUDA kernel-order differences if independent probe/trust
    # calls are concatenated into new batches during validation.
    start = 0
    while start < len(context):
        end = start + 1
        while end < len(context) and source[end] == source[start]:
            end += 1
        one_context = context[start:end]
        center, effective = center_from_residual(
            base_center[one_context], data.sigma[one_context],
            action[start:end],
            float(checkpoint["maximum_residual_sigma"]),
        )
        if np.max(np.abs(effective - action[start:end])) > args.tolerance:
            raise AssertionError("replay action is not clipping-effective action")
        cost = direct_cost(
            center, data, tensors, one_context, frozen_batch_size, device
        )
        expected = base_cost[one_context] - cost
        stored = reward[start:end]
        difference = np.abs(expected - stored)
        scale = np.maximum(1.0, np.maximum(np.abs(expected), np.abs(stored)))
        reward_error = max(reward_error, float(np.max(difference)))
        reward_relative_error = max(
            reward_relative_error, float(np.max(difference / scale))
        )
        reward_tolerance_pass &= bool(np.all(
            difference <= (
                args.reward_absolute_tolerance
                + args.reward_relative_tolerance * scale
            )
        ))
        start = end

    q1 = TorchMPPIContinuousCenterCritic(dropout=0.0).to(device)
    q2 = TorchMPPIContinuousCenterCritic(dropout=0.0).to(device)
    q1.load_state_dict(checkpoint["critic1_state_dict"], strict=True)
    q2.load_state_dict(checkpoint["critic2_state_dict"], strict=True)
    predicted = []
    with torch.no_grad():
        for start in range(0, len(selection_index), args.batch_size):
            absolute = torch.from_numpy(
                selection_index[start:start + args.batch_size]
            ).to(device)
            action_tensor = torch.from_numpy(
                actor_action[start:start + args.batch_size]
            ).to(device)
            zero = torch.zeros_like(action_tensor)
            actor_q = torch.minimum(
                module_batch(q1, inputs, absolute, action_tensor),
                module_batch(q2, inputs, absolute, action_tensor),
            )
            base_q = torch.minimum(
                module_batch(q1, inputs, absolute, zero),
                module_batch(q2, inputs, absolute, zero),
            )
            predicted.append((actor_q - base_q).cpu().numpy())
    predicted = np.concatenate(predicted)
    actual_gain = base_cost[selection_index] - actor_cost
    critic = {
        "actor_delta_correlation": float(np.corrcoef(predicted, actual_gain)[0, 1]),
        "actor_delta_sign_accuracy": float(np.mean(
            (predicted > 0.0) == (actual_gain > 0.0)
        )),
        "predicted_move_fraction": float(np.mean(predicted > 0.0)),
    }
    maximum_metric_error = max(metric_errors.values())
    passed = (
        reward_tolerance_pass
        and deterministic_error == 0.0
        and maximum_metric_error <= args.tolerance
    )
    result = {
        "qualification": "PASS" if passed else "FAIL",
        "checkpoint": str(checkpoint_path.resolve()),
        "replay_count": int(len(context)),
        "fit_selection_overlap": [],
        "maximum_replay_reward_abs_error": reward_error,
        "maximum_replay_reward_scaled_relative_error": reward_relative_error,
        "replay_reward_tolerance": {
            "absolute": args.reward_absolute_tolerance,
            "relative": args.reward_relative_tolerance,
        },
        "deterministic_repeat_max_abs_error": deterministic_error,
        "metric_abs_error": metric_errors,
        "maximum_metric_abs_error": maximum_metric_error,
        "critic_internal_selection": critic,
        "formal_validation_test_loaded": False,
    }
    (args.run / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
