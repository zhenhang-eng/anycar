#!/usr/bin/env python3
"""Validate a DBM multi-elite sidecar against source snapshots and T1 labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_dbm_multi_elite_labels import (
    FORMAT_VERSION,
    load_config,
    select_elite_indices,
    standardized_distance,
)
from generate_dbm_proposal_teacher import sha256_file


REQUIRED_ARRAYS = {
    "format_version",
    "source_snapshot_sha256",
    "source_t1_label_sha256",
    "source_t1_teacher_shortlist_index",
    "noise_sigma",
    "elite_count",
    "elite_valid_mask",
    "elite_shortlist_indices",
    "elite_center_names",
    "elite_centers",
    "elite_delta_knots",
    "elite_selection_score",
    "elite_weighted_output_cost_mean",
    "elite_weighted_output_cost_std",
    "elite_p10_cost_mean",
    "elite_softmin_cost_mean",
    "elite_seed_wins_vs_warm",
    "elite_pairwise_standardized_distance",
    "warm_selection_score",
    "warm_weighted_output_cost_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_root", type=Path)
    return parser.parse_args()


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
        maximum = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name}: maximum absolute error={maximum}")


def validate_one(
    label_path: Path,
    source_path: Path,
    t1_path: Path,
    config: dict,
    expected_source_hash: str,
    expected_t1_hash: str,
) -> int:
    with np.load(source_path, allow_pickle=False) as source, np.load(
        t1_path, allow_pickle=False
    ) as t1, np.load(label_path, allow_pickle=False) as label:
        missing = REQUIRED_ARRAYS.difference(label.files)
        if missing:
            raise AssertionError(f"{label_path}: missing arrays {sorted(missing)}")
        if int(label["format_version"]) != FORMAT_VERSION:
            raise AssertionError(f"{label_path}: unsupported format")
        if str(label["source_snapshot_sha256"]) != expected_source_hash:
            raise AssertionError(f"{label_path}: source hash mismatch")
        if str(label["source_t1_label_sha256"]) != expected_t1_hash:
            raise AssertionError(f"{label_path}: T1 hash mismatch")
        for name in label.files:
            value = label[name]
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise AssertionError(f"{label_path}: {name} contains NaN/Inf")
        params = json.loads(str(source["mppi_params_json"]))
        sigma = np.asarray(params["noise_sigma"], dtype=np.float32)
        assert_close("noise sigma", label["noise_sigma"], sigma)
        selected, cost_mean, cost_std, seed_wins = select_elite_indices(
            t1, sigma, config
        )
        count = len(selected)
        maximum = int(config["max_elites"])
        if int(label["elite_count"]) != count:
            raise AssertionError(f"{label_path}: elite count differs")
        expected_mask = np.arange(maximum) < count
        if not np.array_equal(label["elite_valid_mask"], expected_mask):
            raise AssertionError(f"{label_path}: invalid elite mask")
        indices = np.asarray(label["elite_shortlist_indices"], dtype=np.int32)
        if indices.shape != (maximum,) or not np.array_equal(
            indices[:count], np.asarray(selected, dtype=np.int32)
        ):
            raise AssertionError(f"{label_path}: elite indices differ")
        if not np.all(indices[count:] == -1):
            raise AssertionError(f"{label_path}: invalid padded indices")
        centers = np.asarray(t1["shortlist_centers"], dtype=np.float32)
        names = np.asarray(t1["shortlist_center_names"]).astype(str)
        warm_index = int(t1["warm_shortlist_index"])
        warm = np.asarray(source["sampling_mean_knots"], dtype=np.float32)
        expected_centers = np.broadcast_to(warm, (maximum, 8, 2)).copy()
        expected_centers[:count] = centers[selected]
        assert_close("elite centers", label["elite_centers"], expected_centers)
        assert_close(
            "elite delta", label["elite_delta_knots"], expected_centers - warm[None]
        )
        expected_names = np.full(maximum, "<invalid>", dtype="<U96")
        expected_names[:count] = names[selected]
        if not np.array_equal(label["elite_center_names"].astype(str), expected_names):
            raise AssertionError(f"{label_path}: elite names differ")
        if int(label["source_t1_teacher_shortlist_index"]) != int(
            t1["teacher_shortlist_index"]
        ):
            raise AssertionError(f"{label_path}: teacher index differs")
        if selected[0] != int(t1["teacher_shortlist_index"]):
            raise AssertionError(f"{label_path}: elite zero is not T1 teacher")
        expected_score = np.full(maximum, float(t1["selection_score"][warm_index]))
        expected_score[:count] = np.asarray(t1["selection_score"])[selected]
        assert_close("elite score", label["elite_selection_score"], expected_score)
        expected_mean = np.full(maximum, cost_mean[warm_index])
        expected_mean[:count] = cost_mean[selected]
        expected_std = np.full(maximum, cost_std[warm_index])
        expected_std[:count] = cost_std[selected]
        expected_wins = np.zeros(maximum, dtype=np.int32)
        expected_wins[:count] = seed_wins[selected]
        assert_close(
            "elite weighted cost mean",
            label["elite_weighted_output_cost_mean"],
            expected_mean,
        )
        assert_close(
            "elite weighted cost std",
            label["elite_weighted_output_cost_std"],
            expected_std,
        )
        if not np.array_equal(label["elite_seed_wins_vs_warm"], expected_wins):
            raise AssertionError(f"{label_path}: seed-win count differs")
        expected_p10 = np.full(
            maximum, float(np.asarray(t1["proposal_p10_cost"])[warm_index].mean())
        )
        expected_p10[:count] = np.asarray(t1["proposal_p10_cost"])[selected].mean(
            axis=1
        )
        expected_softmin = np.full(
            maximum, float(np.asarray(t1["proposal_softmin_cost"])[warm_index].mean())
        )
        expected_softmin[:count] = np.asarray(t1["proposal_softmin_cost"])[
            selected
        ].mean(axis=1)
        assert_close("elite P10", label["elite_p10_cost_mean"], expected_p10)
        assert_close(
            "elite softmin", label["elite_softmin_cost_mean"], expected_softmin
        )
        expected_pairwise = np.zeros((maximum, maximum), dtype=np.float32)
        expected_pairwise[:count, :count] = standardized_distance(
            expected_centers[:count], sigma
        )
        assert_close(
            "elite pairwise distance",
            label["elite_pairwise_standardized_distance"],
            expected_pairwise,
        )
        assert_close(
            "warm score", label["warm_selection_score"], t1["selection_score"][warm_index]
        )
        assert_close(
            "warm weighted cost",
            label["warm_weighted_output_cost_mean"],
            cost_mean[warm_index],
        )
        if count > 1:
            valid_distances = expected_pairwise[:count, :count][
                np.triu_indices(count, 1)
            ]
            if np.min(valid_distances) + 1e-6 < float(
                config["minimum_standardized_center_distance"]
            ):
                raise AssertionError(f"{label_path}: elites violate diversity threshold")
        return count


def main() -> None:
    args = parse_args()
    root = args.label_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if int(manifest.get("format_version", 0)) != FORMAT_VERSION:
        raise AssertionError("unsupported multi-elite manifest format")
    config_path = root / "multi_elite_config.json"
    if sha256_file(config_path) != manifest["embedded_config_sha256"]:
        raise AssertionError("embedded config hash mismatch")
    config = load_config(config_path)
    source_root = Path(manifest["source_collection"])
    t1_root = Path(manifest["source_t1_labels"])
    if sha256_file(t1_root / "manifest.json") != manifest["source_t1_manifest_sha256"]:
        raise AssertionError("source T1 manifest hash mismatch")
    fingerprint_lines = []
    counts = []
    for item in manifest["source_index"]:
        source_path = source_root / item["source_relative_path"]
        t1_path = t1_root / item["t1_relative_path"]
        label_path = root / item["label_relative_path"]
        source_hash = sha256_file(source_path)
        t1_hash = sha256_file(t1_path)
        if source_hash != item["source_snapshot_sha256"]:
            raise AssertionError(f"{source_path}: manifest source hash differs")
        if t1_hash != item["source_t1_label_sha256"]:
            raise AssertionError(f"{t1_path}: manifest T1 hash differs")
        label_hash = sha256_file(label_path)
        if label_hash != item["label_sha256"]:
            raise AssertionError(f"{label_path}: label hash differs")
        counts.append(
            validate_one(
                label_path, source_path, t1_path, config, source_hash, t1_hash
            )
        )
        fingerprint_lines.append(
            f"{item['source_relative_path']}\t{source_hash}\t"
            f"{item['t1_relative_path']}\t{t1_hash}\t"
            f"{item['label_relative_path']}\t{label_hash}"
        )
    fingerprint = hashlib.sha256("\n".join(fingerprint_lines).encode()).hexdigest()
    if fingerprint != manifest["source_index_fingerprint_sha256"]:
        raise AssertionError("source index fingerprint differs")
    values = np.asarray(counts)
    print(
        json.dumps(
            {
                "status": "ok",
                "label_root": str(root),
                "snapshots": len(values),
                "elite_count_mean": float(values.mean()),
                "snapshots_with_at_least_two_elites": int(np.sum(values >= 2)),
                "snapshots_with_four_elites": int(np.sum(values == 4)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
