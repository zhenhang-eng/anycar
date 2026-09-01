#!/usr/bin/env python3
"""Reload and independently replay high-speed Actor/twin-Critic pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization, ego_reference_features
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from mppi_a2_actors import DirectNoAnchorGTXActor
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic


DEFAULT_ROOT = Path(
    "outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_20260830_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).ravel()
    right = np.asarray(right, np.float64).ravel()
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def load_source(summary: dict) -> dict[str, np.ndarray]:
    replay_dir = Path(summary["source"]["replay_dir"])
    teacher_dir = Path(summary["source"]["teacher_dir"])
    replay_path = replay_dir / "replay.npz"
    teacher_path = teacher_dir / "labels.npz"
    if sha256(replay_path) != summary["source"]["replay_sha256"]:
        raise AssertionError("replay hash mismatch")
    if sha256(teacher_path) != summary["source"]["teacher_sha256"]:
        raise AssertionError("teacher hash mismatch")
    with np.load(replay_path, allow_pickle=False) as loaded:
        replay = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(teacher_path, allow_pickle=False) as loaded:
        teacher = {name: np.asarray(loaded[name]) for name in loaded.files}
    reference = np.stack([
        ego_reference_features(value, float(state[3]))
        for value, state in zip(replay["reference_ego"], replay["state_six"])
    ]).astype(np.float32)
    current = np.stack([
        np.asarray((state[3], state[5], *action), np.float32)
        for state, action in zip(replay["state_six"], replay["current_action"])
    ])
    return {
        "history": replay["history"].astype(np.float32),
        "reference": reference,
        "current": current,
        "state_six": replay["state_six"].astype(np.float32),
        "current_action": replay["current_action"].astype(np.float32),
        "rollout_reference": replay["reference"][:, 1:].astype(np.float32),
        "anchor": teacher["anchor_knots"].astype(np.float32),
        "teacher": teacher["teacher_knots"].astype(np.float32),
        "anchor_cost": teacher["anchor_cost"].astype(np.float32),
        "teacher_cost": teacher["teacher_cost"].astype(np.float32),
        "actions": teacher["search_centers"].astype(np.float32),
        "costs": teacher["search_costs"].astype(np.float32),
        "episode": teacher["episode_id"].astype(str),
    }


def inputs(data: dict[str, np.ndarray], payload: dict) -> tuple[np.ndarray, ...]:
    normalizer = MPPIProposalNormalization.from_dict(payload["normalization"])
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["reference"], data["current"]
    )
    count = len(history)
    return (
        history.astype(np.float32), reference.astype(np.float32), current.astype(np.float32),
        np.zeros((count, 8, 2), np.float32), np.zeros((count, 74), np.float32),
        np.zeros((count, 32), np.float32),
    )


def actor_predict(model, values, rows, device):
    model.eval()
    with torch.no_grad():
        tensors = tuple(torch.from_numpy(value[rows]).to(device) for value in values)
        return model(*tensors)[1].cpu().numpy().astype(np.float32)


def critic_predict(model, values, actions, rows, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            local = rows[start : start + 16]
            output.append(model(
                torch.from_numpy(values[0][local]).to(device),
                torch.from_numpy(values[1][local]).to(device),
                torch.from_numpy(values[2][local]).to(device),
                torch.from_numpy(actions[local]).to(device),
            ).cpu().numpy())
    return np.concatenate(output)


def replay_cost(data, knots, rows, backend, weights, params, device, batch_size):
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            local = rows[start : start + batch_size]
            action = interpolate_knots(
                torch.from_numpy(knots[start : start + len(local)]).to(device),
                params.horizon,
            ).unsqueeze(1)
            output.append(batched_cost(
                backend, weights, action,
                torch.from_numpy(data["state_six"][local]).to(device),
                torch.from_numpy(data["current_action"][local]).to(device),
                torch.from_numpy(data["rollout_reference"][local]).to(device),
            )[:, 0].cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["contract"]["formal_validation_or_test_created"]:
        raise AssertionError("formal split unexpectedly used")
    data = load_source(summary)
    source_indices = np.asarray(summary["contract"].get(
        "source_indices", np.arange(len(data["episode"]), dtype=np.int64)
    ), np.int64)
    full_count = len(data["episode"])
    data = {
        key: value[source_indices]
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == full_count
        else value
        for key, value in data.items()
    }
    device = torch.device(args.device)
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)

    all_rows = np.arange(len(data["anchor"]), dtype=np.int64)
    anchor_replay = replay_cost(
        data, data["anchor"], all_rows, backend, weights, params, device, args.batch_size
    )
    teacher_replay = replay_cost(
        data, data["teacher"], all_rows, backend, weights, params, device, args.batch_size
    )
    anchor_error = float(np.max(np.abs(anchor_replay - data["anchor_cost"])))
    teacher_error = float(np.max(np.abs(teacher_replay - data["teacher_cost"])))
    replay_tolerance = max(
        1e-3, 1e-6 * float(max(data["anchor_cost"].max(), data["teacher_cost"].max()))
    )
    if anchor_error > replay_tolerance or teacher_error > replay_tolerance:
        raise AssertionError("source DBM replay mismatch")

    records = []
    actor_recovery_errors = []
    critic_correlation_errors = []
    for expected in summary["records"]:
        checkpoint = Path(expected["checkpoint"])
        if sha256(checkpoint) != expected["checkpoint_sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {checkpoint}")
        payload = torch.load(checkpoint, map_location=device)
        rows = np.asarray(payload["oof_indices"], np.int64)
        fit = np.asarray(payload["fit_indices"], np.int64)
        selection = np.asarray(payload["selection_indices"], np.int64)
        if set(data["episode"][rows]) & set(data["episode"][fit]):
            raise AssertionError("OOF/fit episode leakage")
        if set(data["episode"][rows]) & set(data["episode"][selection]):
            raise AssertionError("OOF/selection episode leakage")
        value_inputs = inputs(data, payload)

        actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
        predicted_center = actor_predict(actor, value_inputs, rows, device)
        predicted_cost = replay_cost(
            data, predicted_center, rows, backend, weights, params, device, args.batch_size
        )
        gain = data["anchor_cost"][rows] - predicted_cost
        teacher_gain = data["anchor_cost"][rows] - data["teacher_cost"][rows]
        recovery = float(np.sum(gain) / np.sum(teacher_gain))
        expected_recovery = expected["actor"]["oof"]["aggregate_teacher_gain_recovery"]
        actor_recovery_errors.append(abs(recovery - expected_recovery))

        physical_log = []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
            prediction = critic_predict(critic, value_inputs, data["actions"], rows, device)
            training = payload[f"critic{twin}_training"]
            physical_log.append(
                prediction * float(training["target_std"]) + float(training["target_mean"])
            )
        conservative = np.maximum(physical_log[0], physical_log[1])
        corr = correlation(conservative, np.log1p(data["costs"][rows]))
        expected_corr = expected["critic_twin_conservative"]["oof"]["pearson_log_cost"]
        critic_correlation_errors.append(abs(corr - expected_corr))
        records.append({
            "fold": int(payload["fold"]), "seed": int(payload["seed"]),
            "actor_oof_recovery": recovery,
            "critic_twin_oof_pearson_log_cost": corr,
            "episode_disjoint": True,
        })
    max_actor_error = float(max(actor_recovery_errors))
    max_critic_error = float(max(critic_correlation_errors))
    if max_actor_error > 1e-6 or max_critic_error > 1e-6:
        raise AssertionError("summary replay mismatch")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_TWIN_CRITIC_PRETRAIN_INDEPENDENT_REPLAY_PASS",
        "summary_sha256": sha256(summary_path),
        "checkpoint_count": len(records),
        "source_anchor_dbm_max_abs_error": anchor_error,
        "source_teacher_dbm_max_abs_error": teacher_error,
        "source_dbm_absolute_tolerance": replay_tolerance,
        "actor_recovery_max_abs_error_vs_summary": max_actor_error,
        "critic_correlation_max_abs_error_vs_summary": max_critic_error,
        "episode_group_leakage_count": 0,
        "formal_validation_or_test_created": False,
        "records": records,
    }
    report_path = root / "validator_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
