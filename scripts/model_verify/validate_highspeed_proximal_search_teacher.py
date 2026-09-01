#!/usr/bin/env python3
"""Independently replay high-speed proximal teacher anchor/label costs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots


DEFAULT_INPUT = Path(
    "outputs/mppi_proposal/highspeed_proximal_teacher_20260830_v1"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = args.input_dir.resolve()
    summary = json.loads((root / "summary.json").read_text())
    labels_path = root / "labels.npz"
    if sha256(labels_path) != summary["labels_sha256"]:
        raise AssertionError("labels SHA256 mismatch")
    replay_path = Path(summary["source_replay"])
    if sha256(replay_path) != summary["source_replay_sha256"]:
        raise AssertionError("source replay SHA256 mismatch")
    labels = np.load(labels_path, allow_pickle=False)
    source = np.load(replay_path, allow_pickle=False)
    count = len(labels["anchor_cost"])
    if count != summary["results"]["contexts"] or count != len(source["state_six"]):
        raise AssertionError("context count mismatch")
    if labels["search_centers"].shape != (count, 129, 8, 2):
        raise AssertionError("search center shape mismatch")
    if labels["search_costs"].shape != (count, 129):
        raise AssertionError("search cost shape mismatch")
    if not np.isfinite(labels["search_costs"]).all():
        raise AssertionError("search costs contain NaN/Inf")
    if not np.allclose(labels["search_centers"][:, 0], labels["anchor_knots"]):
        raise AssertionError("candidate zero is not the warm anchor")
    best = labels["search_costs"].min(axis=1)
    if not np.allclose(best, labels["teacher_cost"], rtol=2e-5, atol=2e-3):
        raise AssertionError("teacher is not the stored bank argmin")
    if np.any(labels["teacher_cost"] > labels["anchor_cost"] + 1e-5):
        raise AssertionError("warm-relative baseline violation")

    device = torch.device(args.device)
    centers = np.stack((labels["anchor_knots"], labels["teacher_knots"]), axis=1)
    replay_cost = []
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    with torch.no_grad():
        for start in range(0, count, 32):
            stop = min(start + 32, count)
            knots = torch.as_tensor(centers[start:stop], dtype=torch.float32, device=device)
            actions = interpolate_knots(knots, 50)
            reference = source["reference"][start:stop, 1:]
            replay_cost.append(
                batched_cost(
                    backend,
                    weights,
                    actions,
                    torch.as_tensor(source["state_six"][start:stop], dtype=torch.float32, device=device),
                    torch.as_tensor(source["current_action"][start:stop], dtype=torch.float32, device=device),
                    torch.as_tensor(reference, dtype=torch.float32, device=device),
                ).cpu().numpy()
            )
    replay_cost = np.concatenate(replay_cost)
    expected = np.stack((labels["anchor_cost"], labels["teacher_cost"]), axis=1)
    maximum_error = float(np.max(np.abs(replay_cost - expected)))
    if not np.allclose(replay_cost, expected, rtol=2e-5, atol=2e-3):
        raise AssertionError(f"DBM cost replay mismatch: {maximum_error}")
    target = (source["state_six"][:, 3] >= 40.0 / 3.6) & (
        source["state_six"][:, 3] <= 100.0 / 3.6
    )
    if not np.array_equal(target, labels["target_speed_mask"]):
        raise AssertionError("target speed mask mismatch")
    report = {
        "status": "ok",
        "qualification": summary["qualification"],
        "contexts": count,
        "candidate_center_rollouts": count * 129,
        "baseline_violations": 0,
        "maximum_anchor_teacher_dbm_replay_error": maximum_error,
        "target_speed_contexts": int(target.sum()),
        "formal_validation_or_test_created": False,
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
