#!/usr/bin/env python3
"""Independently replay the full-16D local-Critic labels and heldout metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPIActorCenteredLocalCritic,
    TorchMPPIDeterministicCenterActor,
)
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy, residual_outputs
import train_mppi_direct_local_gradient_critic as local_train
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_local_gradient_critic_20260812_v1"
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


def load_local_labels(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def numeric_metric_error(reference: dict[str, Any], actual: dict[str, Any]) -> float:
    errors = []
    for key, value in reference.items():
        if isinstance(value, (int, float)) and key in actual:
            errors.append(abs(float(value) - float(actual[key])))
    return max(errors, default=0.0)


def main() -> None:
    args = parse_args()
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    device = torch.device(args.device)
    initial_actor_path = Path(summary["initial_actor"])
    initial_payload = torch.load(initial_actor_path, map_location="cpu")
    alpha_path = Path(initial_payload["base_alpha_checkpoint"])
    alpha_payload = torch.load(alpha_path, map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    labels_path = Path(initial_payload["labels"])
    data, _, _ = load_dataset(labels_path, old_payload)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.evaluation_batch_size, device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    maximum_residual_sigma = float(initial_payload["maximum_residual_sigma"])
    actor = TorchMPPIDeterministicCenterActor(maximum_residual_sigma, dropout=0.0).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()

    local = load_local_labels(args.output_dir / "local_forward_labels.npz")
    radii = np.asarray(local.pop("probe_radii_sigma"), np.float32)
    episode = np.asarray(local.pop("episode")).astype(str)
    context_index = np.asarray(local["context_index"], np.int64)
    if not np.array_equal(episode, data.episodes[context_index].astype(str)):
        raise AssertionError("saved label episode/context mapping changed")
    fit_count = int(summary["fit_context_count"])
    heldout_positions = np.arange(fit_count, len(context_index), dtype=np.int64)
    heldout_episode_set = set(episode[heldout_positions])
    if heldout_episode_set != set(summary["split"]["heldout_episodes"]):
        raise AssertionError("heldout episode set changed")
    if heldout_episode_set & set(episode[:fit_count]):
        raise AssertionError("fit/heldout episode leakage")

    actor_action, actor_center = residual_outputs(
        actor, inputs, context_index, args.evaluation_batch_size, device
    )
    actor_action_error = float(np.max(np.abs(actor_action - local["actor_action"])))
    actor_center_error = float(np.max(np.abs(actor_center - local["actor_center"])))

    models = []
    for checkpoint_path in summary["checkpoints"]:
        payload = torch.load(checkpoint_path, map_location="cpu")
        model = TorchMPPIActorCenteredLocalCritic(dropout=0.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
    local_train.RADII_FOR_METRICS = radii
    heldout_metrics = local_train.metrics(
        models, inputs, local, heldout_positions, device,
        args.evaluation_batch_size,
        float(summary["training_arguments"]["meaningful_delta"]),
    )
    heldout_metric_error = numeric_metric_error(
        summary["heldout_metrics"], heldout_metrics
    )

    rng = np.random.default_rng(260812)
    count = min(args.replay_count, len(context_index) * len(radii) * 33)
    flat = rng.choice(len(context_index) * len(radii) * 33, size=count, replace=False)
    position, remainder = np.divmod(flat, len(radii) * 33)
    radius_index, bank_index = np.divmod(remainder, 33)
    global_index = context_index[position]
    action = local["actions"][position, radius_index, bank_index]
    sigma = data.sigma[global_index]
    center = np.clip(
        alpha_center[global_index]
        + action * maximum_residual_sigma * sigma[:, None, :],
        -1.0, 1.0,
    ).astype(np.float32)
    replay_cost = direct_cost(
        center, data, tensors, global_index, args.evaluation_batch_size, device
    )
    stored_cost = local["cost"][position, radius_index, bank_index]
    reward_replay_error = float(np.max(np.abs(replay_cost - stored_cost)))

    grouped: dict[str, dict[str, Any]] = {"reference_speed": {}, "scenario": {}}
    heldout_global = context_index[heldout_positions]
    for value in sorted(np.unique(data.reference_speed[heldout_global])):
        positions = heldout_positions[np.isclose(data.reference_speed[heldout_global], value)]
        grouped["reference_speed"][f"{float(value):.1f}"] = local_train.metrics(
            models, inputs, local, positions, device, args.evaluation_batch_size,
            float(summary["training_arguments"]["meaningful_delta"]),
        )
    for value in sorted(np.unique(data.scenario[heldout_global])):
        positions = heldout_positions[data.scenario[heldout_global] == value]
        grouped["scenario"][str(value)] = local_train.metrics(
            models, inputs, local, positions, device, args.evaluation_batch_size,
            float(summary["training_arguments"]["meaningful_delta"]),
        )

    qualification = "PASS"
    if actor_action_error > 1e-6 or actor_center_error > 1e-6:
        qualification = "FAIL_ACTOR_RECONSTRUCTION"
    elif heldout_metric_error > 1e-6:
        qualification = "FAIL_CHECKPOINT_METRIC_REPLAY"
    elif reward_replay_error > 5e-4:
        qualification = "FAIL_FORWARD_REWARD_REPLAY"
    result = {
        "format_version": 1,
        "qualification": qualification,
        "output_dir": str(args.output_dir.resolve()),
        "actor_action_max_abs_error": actor_action_error,
        "actor_center_max_abs_error": actor_center_error,
        "heldout_metric_max_abs_error": heldout_metric_error,
        "reward_replay_count": int(count),
        "reward_replay_max_abs_error": reward_replay_error,
        "fit_heldout_episode_overlap": 0,
        "heldout_metrics": heldout_metrics,
        "grouped_heldout_metrics": grouped,
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if qualification != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
