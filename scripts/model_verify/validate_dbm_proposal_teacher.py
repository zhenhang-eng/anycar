#!/usr/bin/env python3
"""Validate a T0 DBM proposal-teacher sidecar against its immutable source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import (
    FORMAT_VERSION,
    candidate_cost,
    configs_match_collection,
    load_cost_configs,
    sha256_file,
    stable_mppi_weight,
)


REQUIRED_ARRAYS = {
    "format_version",
    "config_ids",
    "candidate_cost",
    "candidate_weight",
    "best_candidate_index",
    "best_candidate_knots",
    "best_teacher_delta_knots",
    "soft_teacher_center_knots",
    "soft_teacher_delta_knots",
    "warm_candidate_cost",
    "best_candidate_cost",
    "warm_best_regret",
    "soft_weighted_candidate_cost",
    "soft_expected_best_gap",
    "effective_sample_size",
    "best_candidate_clipped",
    "sampling_mean_knots",
    "source_snapshot_sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_root", type=Path)
    return parser.parse_args()


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
        maximum = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name}: maximum absolute error={maximum}")


def main() -> None:
    args = parse_args()
    root = args.label_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise AssertionError("unsupported teacher manifest format")
    config_path = root / "cost_configs.json"
    if sha256_file(config_path) != manifest["embedded_cost_config_sha256"]:
        raise AssertionError("embedded cost config SHA256 mismatch")
    _, configs = load_cost_configs(config_path)
    config_ids = [config["id"] for config in configs]
    source = Path(manifest["source_collection"])
    splits = manifest["splits"]
    split_members = [episode for members in splits.values() for episode in members]
    if len(split_members) != len(set(split_members)):
        raise AssertionError("episode splits overlap")

    index = manifest["source_index"]
    recomputed_index_lines = []
    collection_cost_error = 0.0
    collection_weight_error = 0.0
    for item in index:
        source_path = source / item["source_relative_path"]
        label_path = root / item["label_relative_path"]
        source_hash = sha256_file(source_path)
        if source_hash != item["source_snapshot_sha256"]:
            raise AssertionError(f"{source_path}: source SHA256 mismatch")
        recomputed_index_lines.append(f"{item['source_relative_path']} {source_hash}")
        with np.load(source_path, allow_pickle=False) as source_data, np.load(
            label_path, allow_pickle=False
        ) as label:
            missing = REQUIRED_ARRAYS.difference(label.files)
            if missing:
                raise AssertionError(f"{label_path}: missing arrays {sorted(missing)}")
            if int(label["format_version"]) != FORMAT_VERSION:
                raise AssertionError(f"{label_path}: unsupported format")
            if list(label["config_ids"].astype(str)) != config_ids:
                raise AssertionError(f"{label_path}: config IDs differ")
            if str(label["source_snapshot_sha256"]) != source_hash:
                raise AssertionError(f"{label_path}: embedded source hash differs")
            sampled = np.asarray(source_data["sampled_knots"], dtype=np.float64)
            center = np.asarray(source_data["sampling_mean_knots"], dtype=np.float64)
            expected_costs = []
            expected_weights = []
            expected_best_indices = []
            expected_best_knots = []
            expected_soft_centers = []
            expected_ess = []
            source_params = json.loads(str(source_data["mppi_params_json"]))
            source_weights = json.loads(str(source_data["cost_weights_json"]))
            for config in configs:
                cost = candidate_cost(source_data, config["cost_weights"])
                weight = stable_mppi_weight(cost, float(config["temperature"]))
                best_index = int(np.argmin(cost))
                soft_center = np.sum(weight[:, None, None] * sampled, axis=0)
                expected_costs.append(cost)
                expected_weights.append(weight)
                expected_best_indices.append(best_index)
                expected_best_knots.append(sampled[best_index])
                expected_soft_centers.append(soft_center)
                expected_ess.append(1.0 / np.sum(np.square(weight)))
                if configs_match_collection(
                    config, source_weights, float(source_params["temperature"])
                ):
                    collection_cost_error = max(
                        collection_cost_error,
                        float(np.max(np.abs(cost - source_data["cost"]))),
                    )
                    collection_weight_error = max(
                        collection_weight_error,
                        float(np.max(np.abs(weight - source_data["weight"]))),
                    )
            expected_costs_array = np.asarray(expected_costs)
            expected_weights_array = np.asarray(expected_weights)
            expected_best_knots_array = np.asarray(expected_best_knots)
            expected_soft_centers_array = np.asarray(expected_soft_centers)
            assert_close("candidate_cost", label["candidate_cost"], expected_costs_array)
            assert_close(
                "candidate_weight", label["candidate_weight"], expected_weights_array
            )
            if not np.array_equal(
                label["best_candidate_index"], np.asarray(expected_best_indices)
            ):
                raise AssertionError(f"{label_path}: best candidate indices differ")
            assert_close(
                "best_candidate_knots", label["best_candidate_knots"], expected_best_knots_array
            )
            assert_close(
                "best_teacher_delta_knots",
                label["best_teacher_delta_knots"],
                expected_best_knots_array - center,
            )
            assert_close(
                "soft_teacher_center_knots",
                label["soft_teacher_center_knots"],
                expected_soft_centers_array,
            )
            assert_close(
                "soft_teacher_delta_knots",
                label["soft_teacher_delta_knots"],
                expected_soft_centers_array - center,
            )
            assert_close("effective_sample_size", label["effective_sample_size"], expected_ess)
            if not np.allclose(np.sum(label["candidate_weight"], axis=1), 1.0, atol=2e-5):
                raise AssertionError(f"{label_path}: candidate weights do not sum to one")
            for name in label.files:
                value = label[name]
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    # NaN is intentionally used only when a config does not match collection.
                    if name not in {
                        "collection_cost_max_abs_error",
                        "collection_weight_max_abs_error",
                    }:
                        raise AssertionError(f"{label_path}: {name} contains NaN/Inf")

    fingerprint = hashlib.sha256("\n".join(recomputed_index_lines).encode()).hexdigest()
    if fingerprint != manifest["source_collection_fingerprint_sha256"]:
        raise AssertionError("source collection fingerprint mismatch")
    with (root / "labels.csv").open(newline="") as stream:
        row_count = sum(1 for _ in csv.DictReader(stream))
    expected_rows = len(index) * len(configs)
    if row_count != expected_rows:
        raise AssertionError(f"labels.csv has {row_count} rows, expected {expected_rows}")
    print(
        json.dumps(
            {
                "status": "ok",
                "label_root": str(root),
                "snapshots": len(index),
                "configs": config_ids,
                "label_rows": row_count,
                "collection_cost_max_abs_error": collection_cost_error,
                "collection_weight_max_abs_error": collection_weight_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
