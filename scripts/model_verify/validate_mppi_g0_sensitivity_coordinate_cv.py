#!/usr/bin/env python3
"""Validate the final Critic sensitivity-coordinate experiment artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/g0_sensitivity_coordinate_cv_20260818_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary_path = args.output_dir / "summary.json"
    transform_path = args.output_dir / "coordinate_transforms.npz"
    summary = json.loads(summary_path.read_text())

    assert summary["qualification"] == summary["final_decision"]["qualification"]
    assert summary["unit_tests"] == {
        "batch_equals_per_sample": True,
        "zero_delta_at_reference": True,
        "permutation_consistent": True,
    }
    contract = summary["coordinate_contract"]
    assert contract["physical_action_order"] == ["acceleration", "steering"]
    assert contract["early_steering_flat_indices"] == [1, 3, 5]
    assert not contract["formal_validation_loaded"]
    assert not contract["test_loaded"]
    assert contract["actor_frozen"]

    records = summary["records"]
    assert len(records) == 18
    assert {
        (row["arm"], int(row["fold"]), int(row["seed"])) for row in records
    } == {
        (arm, fold, seed)
        for arm in ("PA_G0", "PA_G0_COORD")
        for fold in range(3)
        for seed in range(3)
    }
    for row in records:
        checkpoint = Path(row["checkpoint"])
        assert checkpoint.is_file()
        assert sha256_file(checkpoint) == row["checkpoint_sha256"]

    for label, path_key, hash_key in (
        ("initial_actor", "initial_actor", "initial_actor_sha256"),
        ("fresh_fd", "fresh_fd", "fresh_fd_sha256"),
        ("labels", "labels", "labels_sha256"),
        ("manifest", "manifest", "manifest_sha256"),
        ("gradient_baseline", "gradient_baseline", "gradient_baseline_sha256"),
    ):
        path = Path(summary["sources"][path_key])
        assert path.is_file(), label
        assert sha256_file(path) == summary["sources"][hash_key], label

    with np.load(transform_path, allow_pickle=False) as payload:
        for fold in range(3):
            matrix = payload[f"B_fold{fold}"].astype(np.float64)
            inverse = payload[f"B_inverse_fold{fold}"].astype(np.float64)
            assert matrix.shape == inverse.shape == (16, 16)
            assert np.all(np.isfinite(matrix)) and np.all(np.isfinite(inverse))
            assert np.allclose(inverse @ matrix, np.eye(16), atol=2e-6)
            assert np.linalg.matrix_rank(matrix) == 16

    final = summary["final_decision"]
    numeric = [value for value in final.values() if isinstance(value, float)]
    assert np.all(np.isfinite(numeric))
    assert set(summary["strict_pooled_final"]) == {"0", "1", "2"}
    print(json.dumps({
        "status": "PASS",
        "qualification": summary["qualification"],
        "records": len(records),
        "summary_sha256": sha256_file(summary_path),
        "coordinate_transforms_sha256": sha256_file(transform_path),
    }, indent=2))


if __name__ == "__main__":
    main()
