#!/usr/bin/env python3
"""Independently replay and validate the high-speed along/cross decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import interpolate_knots


DEFAULT_INPUT = Path(
    "outputs/mppi_proposal/highspeed_along_cross_cost_20260830_v1"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = args.input_dir.resolve()
    summary_path = root / "summary.json"
    artifact_path = root / "decomposition.npz"
    summary = json.loads(summary_path.read_text())
    if sha256(artifact_path) != summary["artifact_sha256"]:
        raise AssertionError("decomposition artifact hash mismatch")
    replay_path = Path(summary["inputs"]["replay"])
    if sha256(replay_path) != summary["inputs"]["replay_sha256"]:
        raise AssertionError("source replay hash mismatch")
    with np.load(artifact_path, allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(replay_path, allow_pickle=False) as loaded:
        replay = {name: np.asarray(loaded[name]) for name in loaded.files}
    rows = data["source_indices"].astype(np.int64)
    centers = data["centers"].astype(np.float32)
    batch, candidates = centers.shape[:2]
    params = TorchMPPIParams(num_samples=64)
    weights = TorchMPPICostWeights()
    backend = TorchDynamicBicycleRolloutBackend()
    device = torch.device(args.device)
    actions = interpolate_knots(torch.from_numpy(centers).to(device), params.horizon)
    flat_actions = actions.reshape(batch * candidates, params.horizon, 2)
    initial = torch.from_numpy(replay["state_six"][rows]).to(device)
    flat_initial = initial[:, None].expand(-1, candidates, -1).reshape(batch * candidates, 6)
    with torch.no_grad():
        full = backend.rollout_full_state_differentiable(flat_initial, flat_actions)
    trajectory = full[..., [0, 1, 2, 3, 5]].reshape(batch, candidates, params.horizon, 5)
    reference = torch.from_numpy(replay["reference"][rows, 1:]).to(device)
    current = torch.from_numpy(replay["current_action"][rows]).to(device)
    error = trajectory[..., :2] - reference[:, None, :, :2]
    yaw = reference[:, None, :, 2]
    along = error[..., 0] * torch.cos(yaw) + error[..., 1] * torch.sin(yaw)
    cross = -error[..., 0] * torch.sin(yaw) + error[..., 1] * torch.cos(yaw)
    yaw_delta = trajectory[..., 2] - yaw
    wrapped_yaw = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
    previous = torch.cat((
        current[:, None, None].expand(-1, candidates, 1, -1), actions[:, :, :-1]
    ), dim=2)
    rate = actions - previous
    replayed = {
        "position_along": weights.position * along.square().sum(-1),
        "position_cross": weights.position * cross.square().sum(-1),
        "position": weights.position * error.square().sum(-1).sum(-1),
        "yaw": weights.yaw * wrapped_yaw.square().sum(-1),
        "vx": weights.vx * (
            trajectory[..., 3] - reference[:, None, :, 3]
        ).square().sum(-1),
        "acceleration_rate": weights.acceleration_rate * rate[..., 0].square().sum(-1),
        "steering_rate": weights.steering_rate * rate[..., 1].square().sum(-1),
    }
    replayed = {key: value.cpu().numpy() for key, value in replayed.items()}
    errors = {
        key: float(np.max(np.abs(replayed[key] - data[key]))) for key in replayed
    }
    total = (
        replayed["position"] + replayed["yaw"] + replayed["vx"]
        + replayed["acceleration_rate"] + replayed["steering_rate"]
    )
    errors["total_cost"] = float(np.max(np.abs(total - data["total_cost"])))
    identity_error = float(np.max(np.abs(
        replayed["position"] - replayed["position_along"] - replayed["position_cross"]
    )))
    relative = float(np.max(
        np.abs(total - data["expected_cost"])
        / np.maximum(np.abs(data["expected_cost"]), 1.0)
    ))
    passed = max(errors.values()) <= 1e-5 and identity_error <= 0.25 and relative <= 2e-6
    qualification = (
        "HIGHSPEED_ALONG_CROSS_DECOMPOSITION_INDEPENDENT_REPLAY_PASS"
        if passed else "HIGHSPEED_ALONG_CROSS_DECOMPOSITION_INDEPENDENT_REPLAY_FAIL"
    )
    report = {
        "qualification": qualification,
        "independent_dbm_replay_max_abs_errors": errors,
        "position_identity_max_abs_error": identity_error,
        "stored_total_cost_max_relative_error": relative,
        "state_count": int(batch),
        "proposal_count": int(candidates),
        "formal_validation_or_test_opened": False,
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
