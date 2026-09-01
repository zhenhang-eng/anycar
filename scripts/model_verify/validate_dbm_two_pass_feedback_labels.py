#!/usr/bin/env python3
"""Validate hashes, shapes, seed isolation, and local rank of two-pass labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels", type=Path,
        nargs="?",
        default=Path(
            "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
            "dbm_two_pass_feedback_diverse_20260805_v1"
        ),
    )
    return parser.parse_args()


def main() -> None:
    root = parse_args().labels.resolve()
    summary = json.loads((root / "summary.json").read_text())
    source_root = Path(summary["source_collection"])
    parent_root = Path(summary["parent_labels"])
    repeat_count = int(summary["replicates_per_partition"])
    paths = sorted(root.glob("episode_*/*.npz"))
    if len(paths) != int(summary["snapshot_count"]):
        raise AssertionError("snapshot count differs from summary")
    expected_selection = np.asarray(summary["selection_seed_pairs"], np.int64)
    expected_audit = np.asarray(summary["audit_seed_pairs"], np.int64)
    if set(expected_selection.reshape(-1)) & set(expected_audit.reshape(-1)):
        raise AssertionError("selection and audit seeds overlap")
    minimum_rank = 16
    maximum_abs_anchor_error = 0.0
    for index, path in enumerate(paths, 1):
        source_path = source_root / path.parent.name / "snapshots" / path.name
        parent_path = parent_root / path.parent.name / path.name
        with np.load(path, allow_pickle=False) as data:
            if str(data["source_snapshot_sha256"]) != sha256_file(source_path):
                raise AssertionError(f"{path}: source hash mismatch")
            if str(data["parent_label_sha256"]) != sha256_file(parent_path):
                raise AssertionError(f"{path}: parent hash mismatch")
            for prefix, seeds in (("", expected_selection), ("audit_", expected_audit)):
                if not np.array_equal(data[f"{prefix}first_pass_seed"], seeds[:, 0]):
                    raise AssertionError(f"{path}: first-pass seeds mismatch")
                if not np.array_equal(data[f"{prefix}second_pass_seed"], seeds[:, 1]):
                    raise AssertionError(f"{path}: second-pass seeds mismatch")
                feedback = data[f"{prefix}first_pass_feedback"]
                centers = data[f"{prefix}centers"]
                anchor = data[f"{prefix}guided_center_knots"]
                costs = data[f"{prefix}proposal_weighted_output_cost"]
                if feedback.shape != (repeat_count, int(summary["feedback_dimension"])):
                    raise AssertionError(f"{path}: feedback shape mismatch")
                if centers.shape != (repeat_count, 33, 8, 2):
                    raise AssertionError(f"{path}: center shape mismatch")
                if costs.shape != (repeat_count, 33):
                    raise AssertionError(f"{path}: cost shape mismatch")
                if not all(np.isfinite(value).all() for value in (feedback, centers, costs)):
                    raise AssertionError(f"{path}: non-finite value")
                maximum_abs_anchor_error = max(
                    maximum_abs_anchor_error,
                    float(np.max(np.abs(centers[:, 0] - anchor))),
                )
                for repeat in range(repeat_count):
                    design = (centers[repeat] - anchor[repeat]).reshape(33, -1)
                    minimum_rank = min(minimum_rank, int(np.linalg.matrix_rank(design)))
        if index % 400 == 0:
            print(f"[{index}/{len(paths)}] validated", flush=True)
    if maximum_abs_anchor_error != 0.0:
        raise AssertionError("center zero is not the guided center")
    if minimum_rank != 16:
        raise AssertionError(f"minimum local rank is {minimum_rank}, expected 16")
    print(
        json.dumps(
            {
                "status": "ok", "labels": str(root), "snapshot_count": len(paths),
                "replicates_per_partition": repeat_count,
                "minimum_local_rank": minimum_rank,
                "maximum_abs_anchor_error": maximum_abs_anchor_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
