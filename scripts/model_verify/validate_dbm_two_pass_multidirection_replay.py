#!/usr/bin/env python3
"""Validate multidirection replay hashes, reconstruction, seeds, and rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import sha256_file
from generate_dbm_two_pass_multidirection_replay import (
    DIRECTION_NAMES,
    direction_bank,
    make_centers,
)


DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="?", type=Path, default=DEFAULT_LABELS)
    return parser.parse_args()


def main() -> None:
    root = parse_args().labels.resolve()
    summary = json.loads((root / "summary.json").read_text())
    source_root = Path(summary["source_collection"])
    parent_root = Path(summary["parent_labels"])
    risk_root = Path(summary["risk_labels"])
    paths = sorted(root.glob("episode_*/*.npz"))
    if len(paths) != summary["snapshot_count"]:
        raise AssertionError("snapshot count mismatch")
    radii = [float(value) for value in summary["radii_sigma"]]
    center_count = 1 + len(DIRECTION_NAMES) * len(radii)
    repeat_count = int(summary["repeats_per_partition"])
    maximum_error = 0.0
    for index, path in enumerate(paths, 1):
        episode = path.parent.name
        source_path = source_root / episode / "snapshots" / path.name
        parent_path = parent_root / episode / path.name
        risk_path = risk_root / episode / path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(risk_path, allow_pickle=False) as risk, np.load(
            path, allow_pickle=False
        ) as data:
            expected_hashes = {
                "source_snapshot_sha256": sha256_file(source_path),
                "parent_label_sha256": sha256_file(parent_path),
                "risk_label_sha256": sha256_file(risk_path),
            }
            for name, expected in expected_hashes.items():
                if str(data[name]) != expected:
                    raise AssertionError(f"{path}: {name} mismatch")
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            action_min = np.asarray(params["action_min"], np.float32)
            action_max = np.asarray(params["action_max"], np.float32)
            base = np.asarray(parent["base_center_knots"], np.float32)
            for prefix, expected_seeds in (
                ("", summary["selection_seeds"]),
                ("audit_", summary["audit_seeds"]),
            ):
                parent_prefix = "" if not prefix else "audit_"
                anchors = np.asarray(data[f"{prefix}guided_center_knots"], np.float32)
                feedback = np.asarray(parent[f"{parent_prefix}first_pass_feedback"], np.float32)
                expected_feedback_hash = np.asarray([
                    hashlib.sha256(value.tobytes()).hexdigest() for value in feedback
                ])
                if not np.array_equal(
                    data[f"{prefix}feedback_sha256"], expected_feedback_hash
                ):
                    raise AssertionError(f"{path}: feedback hash mismatch")
                if not np.array_equal(
                    data[f"{prefix}first_pass_seed"],
                    parent[f"{parent_prefix}first_pass_seed"],
                ):
                    raise AssertionError(f"{path}: first-pass seed mismatch")
                if not np.array_equal(
                    data[f"{prefix}evaluation_seeds"],
                    np.asarray(expected_seeds, np.int64),
                ):
                    raise AssertionError(f"{path}: evaluation seeds mismatch")
                expected_directions = direction_bank(
                    feedback,
                    np.asarray(parent[f"{parent_prefix}first_pass_knots"], np.float32),
                    np.asarray(parent[f"{parent_prefix}first_pass_cost"], np.float32),
                    base,
                    sigma,
                    np.asarray(risk[f"{prefix}normalized_critic_direction"], np.float32),
                    float(summary["preconditioner_damping"]),
                )
                directions = np.asarray(data[f"{prefix}directions"], np.float32)
                if directions.shape != (repeat_count, len(DIRECTION_NAMES), 8, 2):
                    raise AssertionError(f"{path}: direction shape mismatch")
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(directions - expected_directions))),
                )
                raw, centers = make_centers(
                    anchors, directions, radii, sigma, action_min, action_max
                )
                actual_centers = np.asarray(data[f"{prefix}centers"], np.float32)
                if actual_centers.shape != (repeat_count, center_count, 8, 2):
                    raise AssertionError(f"{path}: center shape mismatch")
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(raw - data[f"{prefix}raw_centers"]))),
                    float(np.max(np.abs(centers - actual_centers))),
                )
                cost = np.asarray(data[f"{prefix}proposal_output_cost_by_seed"], np.float32)
                expected_shape = (repeat_count, center_count, len(expected_seeds))
                if cost.shape != expected_shape:
                    raise AssertionError(f"{path}: reward shape mismatch")
                advantage = cost[:, :1] - cost
                checks = {
                    "paired_advantage_by_seed": advantage,
                    "paired_advantage_mean": advantage.mean(axis=2),
                    "paired_advantage_std": advantage.std(axis=2),
                    "paired_advantage_p10": np.quantile(advantage, 0.10, axis=2),
                    "paired_win_probability": np.mean(advantage > 0.0, axis=2),
                }
                for name, expected in checks.items():
                    if not np.allclose(
                        data[f"{prefix}{name}"], expected, rtol=1e-5, atol=1e-5
                    ):
                        raise AssertionError(f"{path}: {prefix}{name} mismatch")
                numeric = (directions, raw, centers, cost)
                if not all(np.isfinite(value).all() for value in numeric):
                    raise AssertionError(f"{path}: non-finite value")
        if index % 400 == 0 or index == len(paths):
            print(f"[{index}/{len(paths)}] validated", flush=True)
    if maximum_error > 1e-6:
        raise AssertionError(f"maximum reconstruction error {maximum_error}")
    print(json.dumps({
        "status": "ok", "labels": str(root), "snapshot_count": len(paths),
        "center_count": center_count,
        "reward_seeds_per_partition": len(summary["selection_seeds"]),
        "maximum_reconstruction_error": maximum_error,
    }, indent=2))


if __name__ == "__main__":
    main()
