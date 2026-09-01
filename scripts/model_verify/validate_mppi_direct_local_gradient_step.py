#!/usr/bin/env python3
"""Independently validate the frozen local-gradient signed-step experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_mppi_direct_local_gradient_step import policy_metrics
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import make_base_policy
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_local_gradient_step_eval_20260812_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--replay-count", type=int, default=1024)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def max_numeric_error(reference: dict, actual: dict) -> float:
    errors = []
    for section in ("cost", "gain"):
        for key, value in reference[section].items():
            errors.append(abs(float(value) - float(actual[section][key])))
    for key in ("wins", "losses", "ties", "win_fraction", "regression_fraction"):
        errors.append(abs(float(reference[key]) - float(actual[key])))
    return max(errors, default=0.0)


def main() -> None:
    args = parse_args()
    summary = json.loads((args.output_dir / "summary.json").read_text())
    with np.load(args.output_dir / "step_eval.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    split = arrays["split"].astype(str)
    validation = split == "internal_validation"
    heldout = split == "internal_selection"
    if int(validation.sum()) != int(summary["validation_context_count"]):
        raise AssertionError("validation context count changed")
    if int(heldout.sum()) != int(summary["heldout_context_count"]):
        raise AssertionError("heldout context count changed")
    signed = arrays["signed_radii_sigma"].astype(np.float32)
    zero_index = int(np.argmin(np.abs(signed)))
    positive_indices = np.flatnonzero(signed >= 0.0)
    validation_mean = arrays["cost"][validation][:, positive_indices].mean(axis=0)
    chosen_index = int(positive_indices[np.argmin(validation_mean)])
    chosen_radius = float(signed[chosen_index])
    selected_radius_error = abs(chosen_radius - float(summary["selected_radius_sigma"]))
    negative_index = int(np.argmin(np.abs(signed + chosen_radius)))
    base = arrays["cost"][heldout, zero_index]
    positive = arrays["cost"][heldout, chosen_index]
    negative = arrays["cost"][heldout, negative_index]
    positive_metric = policy_metrics(base, positive)
    negative_metric = policy_metrics(base, negative)
    positive_metric_error = max_numeric_error(
        summary["heldout"]["fixed_positive_step"], positive_metric
    )
    negative_metric_error = max_numeric_error(
        summary["heldout"]["same_radius_negative_control"], negative_metric
    )

    initial_actor_path = Path(summary["initial_actor"])
    initial_payload = torch.load(initial_actor_path, map_location="cpu")
    alpha_payload = torch.load(initial_payload["base_alpha_checkpoint"], map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial_payload["labels"]), old_payload)
    device = torch.device(args.device)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.evaluation_batch_size, device,
    )
    del alpha_center
    rng = np.random.default_rng(260812)
    total = arrays["cost"].size
    count = min(args.replay_count, total)
    flat = rng.choice(total, size=count, replace=False)
    row, column = np.unravel_index(flat, arrays["cost"].shape)
    context_index = arrays["context_index"][row].astype(np.int64)
    center = arrays["center"][row, column].astype(np.float32)
    replay = direct_cost(
        center, data, tensors, context_index, args.evaluation_batch_size, device
    )
    stored = arrays["cost"][row, column]
    reward_replay_error = float(np.max(np.abs(replay - stored)))
    chosen_effective_radius_error = float(np.max(np.abs(
        arrays["effective_radius_sigma"][:, chosen_index] - chosen_radius
    )))
    episode = arrays["episode"].astype(str)
    overlap = len(set(episode[validation]) & set(episode[heldout]))

    qualification = "PASS"
    if selected_radius_error > 1e-8:
        qualification = "FAIL_RADIUS_SELECTION_REPLAY"
    elif max(positive_metric_error, negative_metric_error) > 1e-6:
        qualification = "FAIL_METRIC_REPLAY"
    elif reward_replay_error > 5e-4:
        qualification = "FAIL_DBM_COST_REPLAY"
    elif overlap:
        qualification = "FAIL_SPLIT_LEAKAGE"
    result = {
        "format_version": 1,
        "qualification": qualification,
        "selected_radius_sigma": chosen_radius,
        "selected_radius_max_abs_error": selected_radius_error,
        "positive_metric_max_abs_error": positive_metric_error,
        "negative_metric_max_abs_error": negative_metric_error,
        "reward_replay_count": int(count),
        "reward_replay_max_abs_error": reward_replay_error,
        "chosen_effective_radius_max_abs_error": chosen_effective_radius_error,
        "validation_heldout_episode_overlap": int(overlap),
        "positive_metric": positive_metric,
        "negative_metric": negative_metric,
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if qualification != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
