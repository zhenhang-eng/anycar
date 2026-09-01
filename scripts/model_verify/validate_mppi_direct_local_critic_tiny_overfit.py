#!/usr/bin/env python3
"""Independently validate the local-Critic tiny-set overfit artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPIActorCenteredLocalCritic,
    TorchMPPIDeterministicCenterActor,
)
from generate_dbm_proposal_teacher import sha256_file
from overfit_mppi_direct_local_gradient_critic import (
    gradient_metrics,
    memorization_gate,
)
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_ARTIFACT = Path(
    "outputs/mppi_proposal/direct_local_critic_tiny_overfit_20260813_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?", default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def close(left: float, right: float, tolerance: float = 1e-7) -> None:
    if not np.isclose(left, right, rtol=tolerance, atol=tolerance):
        raise AssertionError(f"metric mismatch: {left} != {right}")


def main() -> None:
    args = parse_args()
    analysis_path = args.artifact / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis["qualification"] != "TINY_OVERFIT_PASS_BASIC_AND_MULTITASK":
        raise AssertionError("tiny-set artifact did not pass both arms")
    if analysis["contract"] != {
        "new_dbm_rollouts": 0,
        "actor_frozen": True,
        "critic_updated": True,
        "formal_validation_loaded": False,
        "test_loaded": False,
        "selected_from": "internal-fit train episodes only",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "scheduler": False,
        "early_stopping": False,
    }:
        raise AssertionError("tiny-set contract changed")

    source_summary_path = Path(analysis["source_summary"])
    source_labels_path = Path(analysis["source_labels"])
    initial_actor_path = Path(analysis["initial_actor"])
    for path, expected in (
        (source_summary_path, analysis["source_summary_sha256"]),
        (source_labels_path, analysis["source_labels_sha256"]),
        (initial_actor_path, analysis["initial_actor_sha256"]),
        (
            args.artifact / "tiny_set_manifest.json",
            analysis["tiny_set_manifest_sha256"],
        ),
        (args.artifact / "tiny_set.npz", analysis["tiny_set_npz_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise AssertionError(f"hash mismatch: {path}")

    source_summary = json.loads(source_summary_path.read_text())
    with np.load(source_labels_path, allow_pickle=False) as archive:
        labels = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.artifact / "tiny_set.npz", allow_pickle=False) as tiny:
        selected = np.asarray(tiny["label_position"], np.int64)
        context_index = np.asarray(tiny["context_index"], np.int64)
        target = np.asarray(tiny["target_gradient"], np.float32)
        episode = np.asarray(tiny["episode"])
    if len(selected) != analysis["context_count"]:
        raise AssertionError("tiny-set context count changed")
    if len(np.unique(episode)) != len(episode):
        raise AssertionError("tiny-set repeats an episode")
    if not np.array_equal(context_index, labels["context_index"][selected]):
        raise AssertionError("tiny-set context mapping changed")
    if not np.array_equal(episode, labels["episode"][selected]):
        raise AssertionError("tiny-set episode mapping changed")
    if not np.array_equal(target, labels["gradient_by_radius"][selected, 0]):
        raise AssertionError("tiny-set target is not the stored 0.05-sigma label")
    train_episodes = set(source_summary["split"]["train_episodes"])
    if not set(episode).issubset(train_episodes):
        raise AssertionError("tiny-set contains a non-training episode")

    device = torch.device(args.device)
    initial_payload = torch.load(initial_actor_path, map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial_payload["labels"]), old_payload)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy,
        tensors,
        extra,
        np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        256,
        device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    actor = TorchMPPIDeterministicCenterActor(
        float(initial_payload["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()
    index_tensor = torch.from_numpy(context_index).to(device)
    with torch.no_grad():
        actor_action, actor_center = actor(*(value[index_tensor] for value in inputs))
    actor_action_error = float(np.max(np.abs(
        actor_action.cpu().numpy() - labels["actor_action"][selected]
    )))
    actor_center_error = float(np.max(np.abs(
        actor_center.cpu().numpy() - labels["actor_center"][selected]
    )))
    if actor_action_error > 5e-5 or actor_center_error > 5e-5:
        raise AssertionError("frozen Actor reconstruction failed")

    with np.load(args.artifact / "predictions.npz", allow_pickle=False) as archive:
        stored_prediction = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(stored_prediction["target_gradient"], target):
        raise AssertionError("stored prediction target changed")

    replay_rows = []
    for record in analysis["runs"]:
        checkpoint_path = Path(record["checkpoint"])
        if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu")
        if payload["arm"] != record["arm"] or payload["seed"] != record["seed"]:
            raise AssertionError("checkpoint identity changed")
        if not np.array_equal(payload["selected_label_positions"], selected):
            raise AssertionError("checkpoint selection changed")
        model = TorchMPPIActorCenteredLocalCritic(dropout=0.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            _, gradient, _ = model.local_parameters(
                *(value[index_tensor] for value in inputs), actor_action
            )
        prediction = gradient.flatten(1).cpu().numpy()
        key = f"{record['arm']}_seed{record['seed']}"
        maximum_prediction_error = float(np.max(np.abs(
            prediction - stored_prediction[key]
        )))
        if maximum_prediction_error > 1e-6:
            raise AssertionError(
                f"prediction replay error {maximum_prediction_error}: {key}"
            )
        metrics = gradient_metrics(prediction, target)
        if not memorization_gate(metrics):
            raise AssertionError(f"replayed checkpoint fails gate: {key}")
        for section, field in (
            ("cosine", "median"),
            ("cosine", "p10"),
            ("norm_ratio", "median"),
            ("norm_ratio", "p10"),
            ("norm_ratio", "p90"),
        ):
            close(
                metrics[section][field],
                record["best_metrics"][section][field],
            )
        replay_rows.append({
            "arm": record["arm"],
            "seed": record["seed"],
            "maximum_prediction_error": maximum_prediction_error,
            "metrics": metrics,
        })

    validation = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PASS",
        "analysis": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "context_count": len(selected),
        "run_count": len(replay_rows),
        "frozen_actor_maximum_action_error": actor_action_error,
        "frozen_actor_maximum_center_error": actor_center_error,
        "maximum_prediction_replay_error": max(
            row["maximum_prediction_error"] for row in replay_rows
        ),
        "runs": replay_rows,
        "contract": {
            "new_dbm_rollouts": 0,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "actor_updated": False,
            "critic_checkpoint_replayed": True,
        },
    }
    path = args.artifact / "validation_summary.json"
    path.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({
        "qualification": validation["qualification"],
        "context_count": validation["context_count"],
        "run_count": validation["run_count"],
        "maximum_prediction_replay_error": validation[
            "maximum_prediction_replay_error"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
