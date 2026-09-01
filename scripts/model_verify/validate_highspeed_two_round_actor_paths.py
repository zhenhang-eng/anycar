#!/usr/bin/env python3
"""Independently replay all candidates in the 600-state two-round path bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots


DEFAULT_INPUT = Path(
    "outputs/mppi_proposal/highspeed_two_round_actor_paths_20260830_v1"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state-batch", type=int, default=8)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    summary = json.loads((root / "summary.json").read_text())
    artifact_path = Path(summary["artifact"])
    if sha256(artifact_path) != summary["artifact_sha256"]:
        raise AssertionError("path artifact hash mismatch")
    replay_path = Path(summary["sources"]["replay"])
    if sha256(replay_path) != summary["sources"]["replay_sha256"]:
        raise AssertionError("replay source hash mismatch")
    if summary["contract"]["formal_validation_or_test_created"]:
        raise AssertionError("path source is not train-only")
    with np.load(artifact_path, allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(replay_path, allow_pickle=False) as loaded:
        replay = {name: np.asarray(loaded[name]) for name in loaded.files}
    if not np.array_equal(data["source_indices"], np.arange(600)):
        raise AssertionError("expected complete replay order")
    if not np.array_equal(data["episode_id"], replay["episode_id"]):
        raise AssertionError("episode alignment mismatch")
    if not np.all(data["evaluation_counts"] == 195):
        raise AssertionError("unexpected candidate count")
    centers = data["evaluated_centers"].astype(np.float32)
    stored = data["evaluated_costs"].astype(np.float32)
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    device = torch.device(args.device)
    replayed = np.empty_like(stored)
    for start in range(0, 600, args.state_batch):
        stop = min(start + args.state_batch, 600)
        actions = interpolate_knots(
            torch.from_numpy(centers[start:stop]).to(device), params.horizon
        )
        with torch.no_grad():
            replayed[start:stop] = batched_cost(
                backend, weights, actions,
                torch.from_numpy(replay["state_six"][start:stop]).to(device),
                torch.from_numpy(replay["current_action"][start:stop]).to(device),
                torch.from_numpy(replay["reference"][start:stop, 1:]).to(device),
            ).cpu().numpy()
    max_cost_error = float(np.max(np.abs(replayed - stored)))
    relative_cost_error = float(np.max(
        np.abs(replayed - stored) / np.maximum(np.abs(stored), 1.0)
    ))
    path = data["path_centers"]
    path_cost = data["path_costs"]
    path_lookup_error = 0.0
    missing_path_centers = 0
    for row in range(600):
        keys = {np.round(center, 7).tobytes(): index for index, center in enumerate(centers[row])}
        for actor in range(3):
            for step in range(3):
                key = np.round(path[row, actor, step], 7).tobytes()
                if key not in keys:
                    missing_path_centers += 1
                else:
                    path_lookup_error = max(
                        path_lookup_error,
                        abs(float(stored[row, keys[key]]) - float(path_cost[row, actor, step])),
                    )
    monotonic_violations = int(np.sum(np.diff(path_cost, axis=2) > 1e-5))
    baseline_violations = int(np.sum(path_cost[..., 2] > path_cost[..., 0] + 1e-5))
    passed = (
        max_cost_error == 0.0 and relative_cost_error == 0.0
        and path_lookup_error == 0.0 and missing_path_centers == 0
        and monotonic_violations == 0 and baseline_violations == 0
    )
    qualification = (
        "HIGHSPEED_TWO_ROUND_ACTOR_PATHS_INDEPENDENT_REPLAY_PASS"
        if passed else "HIGHSPEED_TWO_ROUND_ACTOR_PATHS_INDEPENDENT_REPLAY_FAIL"
    )
    report = {
        "qualification": qualification,
        "candidate_replay_max_abs_error": max_cost_error,
        "candidate_replay_max_relative_error": relative_cost_error,
        "path_lookup_max_abs_error": path_lookup_error,
        "missing_path_centers": missing_path_centers,
        "monotonic_violations": monotonic_violations,
        "baseline_violations": baseline_violations,
        "contexts": 600, "candidates_per_context": 195,
        "total_candidate_replays": int(600 * 195),
        "formal_validation_or_test_opened": False,
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
