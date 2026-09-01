#!/usr/bin/env python3
"""Independently validate the compact high-speed initial-state DBM replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_20260828_v1"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    summary = json.loads((root / "summary.json").read_text())
    archive = root / "replay.npz"
    if digest(archive) != summary["archive_sha256"]:
        raise AssertionError("archive SHA256 mismatch")
    with np.load(archive, allow_pickle=False) as data:
        count = int(summary["context_count"])
        samples = int(summary["candidate_rollout_count"] // count)
        expected = {
            "state_six": (count, 6),
            "current_action": (count, 2),
            "history": (count, 250, 7),
            "reference": (count, 51, 4),
            "reference_ego": (count, 51, 4),
            "mean_knots_before": (count, 8, 2),
            "sampled_knots": (count, samples, 8, 2),
            "sampled_action_sequences": (count, samples, 50, 2),
            "predicted_trajectories_full": (count, samples, 50, 6),
            "cost": (count, samples),
            "weight": (count, samples),
            "optimized_action_sequence": (count, 50, 2),
        }
        for name, shape in expected.items():
            if data[name].shape != shape:
                raise AssertionError(f"{name} shape {data[name].shape} != {shape}")
            if not np.isfinite(data[name]).all():
                raise AssertionError(f"{name} contains NaN/Inf")
        if not np.allclose(data["weight"].sum(axis=1), 1.0, atol=2e-5):
            raise AssertionError("candidate weights do not sum to one")
        replay_best = data["cost"].min(axis=1)
        if not np.allclose(replay_best, data["replay_best_cost"], atol=2e-5):
            raise AssertionError("stored replay best cost mismatch")
        vx = data["state_six"][:, 3]
        target = (vx >= 40.0 / 3.6) & (vx <= 100.0 / 3.6)
        if not np.isclose(target.mean(), summary["target_40_100_kph_fraction"]):
            raise AssertionError("target speed fraction mismatch")
        if len(np.unique(data["episode_id"])) != summary["episode_count"]:
            raise AssertionError("episode count mismatch")
        if not np.array_equal(
            np.unique(data["control_step"]),
            np.arange(summary["steps_per_episode"]),
        ):
            raise AssertionError("control-step coverage mismatch")
    report = {
        "status": "ok",
        "qualification": summary["qualification"],
        "context_count": summary["context_count"],
        "candidate_rollout_count": summary["candidate_rollout_count"],
        "episode_count": summary["episode_count"],
        "target_40_100_kph_fraction": summary["target_40_100_kph_fraction"],
        "archive_sha256": summary["archive_sha256"],
        "formal_validation_or_test_created": False,
    }
    (root / "validator_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
