#!/usr/bin/env python3
"""Validate repeated second-pass risk-replay labels without DBM rerollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        nargs="?",
        type=Path,
        default=Path(
            "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
            "dbm_two_pass_risk_replay_diverse_20260805_v1"
        ),
    )
    return parser.parse_args()


def main() -> None:
    root = parse_args().labels.resolve()
    summary = json.loads((root / "summary.json").read_text())
    for checkpoint, expected_hash in zip(
        summary["critic_checkpoints"], summary["critic_checkpoint_sha256"]
    ):
        if sha256_file(Path(checkpoint)) != expected_hash:
            raise AssertionError(f"critic checkpoint hash mismatch: {checkpoint}")
    source_root = Path(summary["source_collection"])
    parent_root = Path(summary["parent_labels"])
    paths = sorted(root.glob("episode_*/*.npz"))
    if len(paths) != int(summary["snapshot_count"]):
        raise AssertionError("snapshot count mismatch")
    radii = np.asarray(summary["radii_sigma"], np.float32)
    center_count = 1 + 2 * len(radii)
    repeat_count = int(summary["repeats_per_partition"])
    seed_count = len(summary["selection_seeds"])
    if seed_count != len(summary["audit_seeds"]):
        raise AssertionError("selection/audit seed counts differ")
    maximum_reconstruction_error = 0.0
    for index, path in enumerate(paths, 1):
        source_path = source_root / path.parent.name / "snapshots" / path.name
        parent_path = parent_root / path.parent.name / path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(path, allow_pickle=False) as data:
            if str(data["source_snapshot_sha256"]) != sha256_file(source_path):
                raise AssertionError(f"{path}: source hash mismatch")
            if str(data["parent_label_sha256"]) != sha256_file(parent_path):
                raise AssertionError(f"{path}: parent hash mismatch")
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            action_min = np.asarray(params["action_min"], np.float32)
            action_max = np.asarray(params["action_max"], np.float32)
            for prefix, expected_seeds in (
                ("", summary["selection_seeds"]),
                ("audit_", summary["audit_seeds"]),
            ):
                parent_prefix = "" if not prefix else "audit_"
                anchors = np.asarray(data[f"{prefix}guided_center_knots"], np.float32)
                direction = np.asarray(data[f"{prefix}normalized_critic_direction"], np.float32)
                raw = np.asarray(data[f"{prefix}raw_centers"], np.float32)
                centers = np.asarray(data[f"{prefix}centers"], np.float32)
                costs = np.asarray(data[f"{prefix}proposal_output_cost_by_seed"], np.float32)
                if anchors.shape != (repeat_count, 8, 2):
                    raise AssertionError(f"{path}: anchor shape mismatch")
                if centers.shape != (repeat_count, center_count, 8, 2):
                    raise AssertionError(f"{path}: center shape mismatch")
                if costs.shape != (repeat_count, center_count, seed_count):
                    raise AssertionError(f"{path}: cost shape mismatch")
                if not np.array_equal(
                    data[f"{prefix}evaluation_seeds"], np.asarray(expected_seeds, np.int64)
                ):
                    raise AssertionError(f"{path}: evaluation seeds mismatch")
                if not np.array_equal(
                    data[f"{prefix}first_pass_seed"], parent[f"{parent_prefix}first_pass_seed"]
                ):
                    raise AssertionError(f"{path}: first-pass seed mismatch")
                if not np.array_equal(anchors, parent[f"{parent_prefix}guided_center_knots"]):
                    raise AssertionError(f"{path}: parent anchor mismatch")
                feedback = np.asarray(parent[f"{parent_prefix}first_pass_feedback"], np.float32)
                feedback_hash = np.asarray(
                    [hashlib.sha256(value.tobytes()).hexdigest() for value in feedback]
                )
                if not np.array_equal(data[f"{prefix}feedback_sha256"], feedback_hash):
                    raise AssertionError(f"{path}: feedback hash mismatch")
                reconstructed = [anchors]
                # Reconstruct [anchor, +r, -r, ...] explicitly.
                bank = []
                for anchor, one_direction in zip(anchors, direction):
                    one = [anchor]
                    for radius in radii:
                        delta = radius * one_direction * sigma.reshape(1, 2)
                        one.extend((anchor + delta, anchor - delta))
                    bank.append(one)
                reconstructed = np.asarray(bank, np.float32)
                maximum_reconstruction_error = max(
                    maximum_reconstruction_error,
                    float(np.max(np.abs(raw - reconstructed))),
                )
                if not np.array_equal(centers, np.clip(raw, action_min, action_max)):
                    raise AssertionError(f"{path}: clipped centers mismatch")
                advantage = costs[:, :1] - costs
                checks = {
                    "paired_advantage_by_seed": advantage,
                    "paired_advantage_mean": advantage.mean(axis=2),
                    "paired_advantage_std": advantage.std(axis=2),
                    "paired_advantage_p10": np.quantile(advantage, 0.10, axis=2),
                    "paired_win_probability": np.mean(advantage > 0, axis=2),
                }
                for name, expected in checks.items():
                    actual = np.asarray(data[f"{prefix}{name}"])
                    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-5):
                        raise AssertionError(f"{path}: {prefix}{name} mismatch")
                numeric = [
                    anchors, direction, raw, centers, costs,
                    data[f"{prefix}critic_gradient_mean"],
                    data[f"{prefix}critic_gradient_std"],
                ]
                if not all(np.isfinite(value).all() for value in numeric):
                    raise AssertionError(f"{path}: non-finite value")
        if index % 400 == 0:
            print(f"[{index}/{len(paths)}] validated", flush=True)
    if maximum_reconstruction_error > 1e-6:
        raise AssertionError(
            f"maximum center reconstruction error {maximum_reconstruction_error}"
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "labels": str(root),
                "snapshot_count": len(paths),
                "center_count": center_count,
                "reward_seeds_per_partition": seed_count,
                "maximum_center_reconstruction_error": maximum_reconstruction_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
