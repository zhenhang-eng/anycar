#!/usr/bin/env python3
"""Validate T1 multi-center DBM teacher labels and selection arithmetic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_dbm_multicenter_teacher import (
    FORMAT_VERSION,
    load_config,
    sha256_file,
    softmin,
    stable_weight,
)


REQUIRED_ARRAYS = {
    "format_version",
    "source_snapshot_sha256",
    "t0_label_sha256",
    "pool_center_names",
    "pool_centers",
    "pool_direct_cost",
    "shortlist_pool_indices",
    "shortlist_center_names",
    "shortlist_centers",
    "shortlist_direct_action_sequences",
    "shortlist_direct_trajectories",
    "shortlist_direct_cost",
    "proposal_evaluation_seeds",
    "proposal_candidate_cost",
    "proposal_candidate_weight",
    "proposal_weighted_action_sequences",
    "proposal_weighted_trajectories",
    "proposal_weighted_output_cost",
    "proposal_best_cost",
    "proposal_p10_cost",
    "proposal_median_cost",
    "proposal_softmin_cost",
    "proposal_effective_sample_size",
    "proposal_clip_fraction",
    "center_shift_standardized_rms",
    "center_boundary_fraction",
    "selection_score",
    "warm_shortlist_index",
    "teacher_shortlist_index",
    "teacher_center_source",
    "teacher_center_knots",
    "teacher_delta_knots",
    "teacher_direct_cost",
    "teacher_weighted_action_sequences",
    "teacher_weighted_trajectories",
    "audit_center_names",
    "audit_centers",
    "audit_evaluation_seeds",
    "audit_proposal_candidate_cost",
    "audit_proposal_candidate_weight",
    "audit_proposal_weighted_action_sequences",
    "audit_proposal_weighted_trajectories",
    "audit_proposal_weighted_output_cost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_root", type=Path)
    return parser.parse_args()


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(actual, expected, rtol=3e-4, atol=3e-4):
        maximum = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name}: maximum absolute error={maximum}")


def trajectory_cost(
    trajectories: np.ndarray,
    actions: np.ndarray,
    reference: np.ndarray,
    current_action: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    horizon = actions.shape[-2]
    if len(reference) == horizon + 1:
        reference = reference[1:]
    position = np.square(trajectories[..., :2] - reference[..., :2]).sum(axis=-1)
    yaw_delta = trajectories[..., 2] - reference[..., 2]
    yaw = np.square(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))
    vx = np.square(trajectories[..., 3] - reference[..., 3])
    previous = np.concatenate(
        (
            np.broadcast_to(current_action, (*actions.shape[:-2], 1, 2)),
            actions[..., :-1, :],
        ),
        axis=-2,
    )
    action_rate = np.square(actions - previous)
    return (
        float(weights["position"]) * position.sum(axis=-1)
        + float(weights["yaw"]) * yaw.sum(axis=-1)
        + float(weights["vx"]) * vx.sum(axis=-1)
        + float(weights["acceleration_rate"]) * action_rate[..., 0].sum(axis=-1)
        + float(weights["steering_rate"]) * action_rate[..., 1].sum(axis=-1)
    )


def validate_proposal_block(
    label: np.lib.npyio.NpzFile,
    prefix: str,
    temperature: float,
    reference: np.ndarray,
    current_action: np.ndarray,
    weights: dict[str, float],
) -> None:
    key = lambda name: f"{prefix}{name}"
    cost = np.asarray(label[key("proposal_candidate_cost")], dtype=np.float64)
    stored_weight = np.asarray(
        label[key("proposal_candidate_weight")], dtype=np.float64
    )
    expected_weight = np.empty_like(stored_weight)
    expected_softmin = np.empty(cost.shape[:2], dtype=np.float64)
    for center_index in range(cost.shape[0]):
        for seed_index in range(cost.shape[1]):
            expected_weight[center_index, seed_index] = stable_weight(
                cost[center_index, seed_index], temperature
            )
            expected_softmin[center_index, seed_index] = softmin(
                cost[center_index, seed_index], temperature
            )
    assert_close(f"{prefix}candidate_weight", stored_weight, expected_weight)
    assert_close(
        f"{prefix}best_cost", label[key("proposal_best_cost")], cost.min(axis=2)
    )
    assert_close(
        f"{prefix}p10_cost",
        label[key("proposal_p10_cost")],
        np.quantile(cost, 0.10, axis=2),
    )
    assert_close(
        f"{prefix}median_cost",
        label[key("proposal_median_cost")],
        np.median(cost, axis=2),
    )
    assert_close(
        f"{prefix}softmin_cost",
        label[key("proposal_softmin_cost")],
        expected_softmin,
    )
    assert_close(
        f"{prefix}ESS",
        label[key("proposal_effective_sample_size")],
        1.0 / np.square(expected_weight).sum(axis=2),
    )
    weighted_actions = np.asarray(
        label[key("proposal_weighted_action_sequences")], dtype=np.float64
    )
    weighted_trajectories = np.asarray(
        label[key("proposal_weighted_trajectories")], dtype=np.float64
    )
    expected_output_cost = trajectory_cost(
        weighted_trajectories,
        weighted_actions,
        reference,
        current_action,
        weights,
    )
    assert_close(
        f"{prefix}weighted_output_cost",
        label[key("proposal_weighted_output_cost")],
        expected_output_cost,
    )


def validate_label(
    label_path: Path,
    source_path: Path,
    t0_path: Path,
    config: dict,
    source_hash: str,
    t0_hash: str,
) -> dict[str, float | bool | str]:
    with np.load(source_path, allow_pickle=False) as source, np.load(
        label_path, allow_pickle=False
    ) as label:
        missing = REQUIRED_ARRAYS.difference(label.files)
        if missing:
            raise AssertionError(f"{label_path}: missing arrays {sorted(missing)}")
        if int(label["format_version"]) != FORMAT_VERSION:
            raise AssertionError(f"{label_path}: unsupported format")
        if str(label["source_snapshot_sha256"]) != source_hash:
            raise AssertionError(f"{label_path}: source hash differs")
        if str(label["t0_label_sha256"]) != t0_hash:
            raise AssertionError(f"{label_path}: T0 hash differs")
        for name in label.files:
            value = label[name]
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise AssertionError(f"{label_path}: {name} contains NaN/Inf")
        weights = config["objective"]["cost_weights"]
        temperature = float(config["objective"]["temperature"])
        reference = np.asarray(source["reference"], dtype=np.float64)
        current_action = np.asarray(source["current_action"], dtype=np.float64)
        warm = np.asarray(source["sampling_mean_knots"], dtype=np.float64)
        shortlist = np.asarray(label["shortlist_centers"], dtype=np.float64)
        names = list(label["shortlist_center_names"].astype(str))
        shortlist_count = int(config["search"]["shortlist_count"])
        seed_count = len(config["proposal_evaluation"]["seeds"])
        sample_count = int(config["proposal_evaluation"]["num_samples"])
        if shortlist.shape != (shortlist_count, 8, 2):
            raise AssertionError(f"{label_path}: invalid shortlist shape")
        if label["proposal_candidate_cost"].shape != (
            shortlist_count,
            seed_count,
            sample_count,
        ):
            raise AssertionError(f"{label_path}: invalid proposal cost shape")
        if not np.array_equal(
            label["proposal_evaluation_seeds"],
            np.asarray(config["proposal_evaluation"]["seeds"]),
        ):
            raise AssertionError(f"{label_path}: proposal seeds differ")
        warm_index = int(label["warm_shortlist_index"])
        teacher_index = int(label["teacher_shortlist_index"])
        if "warm" not in names[warm_index].split("|"):
            raise AssertionError(f"{label_path}: warm index/name mismatch")
        assert_close("warm center", shortlist[warm_index], warm)
        direct_cost = trajectory_cost(
            np.asarray(label["shortlist_direct_trajectories"], dtype=np.float64),
            np.asarray(label["shortlist_direct_action_sequences"], dtype=np.float64),
            reference,
            current_action,
            weights,
        )
        assert_close("shortlist direct cost", label["shortlist_direct_cost"], direct_cost)
        validate_proposal_block(
            label, "", temperature, reference, current_action, weights
        )
        score_components = {
            "weighted_output_cost_mean": np.asarray(
                label["proposal_weighted_output_cost"]
            ).mean(axis=1),
            "weighted_output_cost_std": np.asarray(
                label["proposal_weighted_output_cost"]
            ).std(axis=1),
            "p10_cost_mean": np.asarray(label["proposal_p10_cost"]).mean(axis=1),
            "softmin_cost_mean": np.asarray(label["proposal_softmin_cost"]).mean(
                axis=1
            ),
            "center_shift_standardized_rms": np.asarray(
                label["center_shift_standardized_rms"]
            ),
            "boundary_fraction": np.asarray(label["center_boundary_fraction"]),
        }
        expected_score = np.zeros(shortlist_count, dtype=np.float64)
        for name, values in score_components.items():
            assert_close(f"score component {name}", label[f"score_component_{name}"], values)
            expected_score += float(config["selection_score"][name]) * values
        assert_close("selection score", label["selection_score"], expected_score)
        if teacher_index != int(np.argmin(expected_score)):
            raise AssertionError(f"{label_path}: teacher does not minimize selection score")
        assert_close("teacher center", label["teacher_center_knots"], shortlist[teacher_index])
        assert_close(
            "teacher delta", label["teacher_delta_knots"], shortlist[teacher_index] - warm
        )
        if str(label["teacher_center_source"]) != names[teacher_index]:
            raise AssertionError(f"{label_path}: teacher source name differs")
        assert_close(
            "teacher direct cost",
            np.asarray(label["teacher_direct_cost"]),
            np.asarray(label["shortlist_direct_cost"][teacher_index]),
        )
        assert_close(
            "teacher weighted actions",
            label["teacher_weighted_action_sequences"],
            label["proposal_weighted_action_sequences"][teacher_index],
        )
        assert_close(
            "teacher weighted trajectories",
            label["teacher_weighted_trajectories"],
            label["proposal_weighted_trajectories"][teacher_index],
        )
        audit_seeds = np.asarray(config["proposal_evaluation"]["audit_seeds"])
        if not np.array_equal(label["audit_evaluation_seeds"], audit_seeds):
            raise AssertionError(f"{label_path}: audit seeds differ")
        assert_close("audit warm center", label["audit_centers"][0], warm)
        assert_close(
            "audit teacher center", label["audit_centers"][1], shortlist[teacher_index]
        )
        validate_proposal_block(
            label, "audit_", temperature, reference, current_action, weights
        )
        selection_gain = float(expected_score[warm_index] - expected_score[teacher_index])
        audit_out = np.asarray(label["audit_proposal_weighted_output_cost"])
        return {
            "teacher_is_warm": teacher_index == warm_index,
            "teacher_source": names[teacher_index],
            "selection_score_gain": selection_gain,
            "audit_weighted_output_cost_gain": float(
                audit_out[0].mean() - audit_out[1].mean()
            ),
        }


def main() -> None:
    args = parse_args()
    root = args.label_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise AssertionError("unsupported T1 manifest format")
    config_path = root / "teacher_config.json"
    if sha256_file(config_path) != manifest["embedded_teacher_config_sha256"]:
        raise AssertionError("embedded teacher config hash mismatch")
    config = load_config(config_path)
    source_root = Path(manifest["source_collection"])
    t0_root = Path(manifest["t0_labels"])
    fingerprint_lines: list[str] = []
    results = []
    for item in manifest["source_index"]:
        source_path = source_root / item["source_relative_path"]
        label_path = root / item["label_relative_path"]
        episode_id = Path(item["source_relative_path"]).parts[0]
        t0_path = t0_root / episode_id / label_path.name
        source_hash = sha256_file(source_path)
        t0_hash = sha256_file(t0_path)
        if source_hash != item["source_snapshot_sha256"]:
            raise AssertionError(f"{source_path}: source hash mismatch")
        if t0_hash != item["t0_label_sha256"]:
            raise AssertionError(f"{t0_path}: T0 hash mismatch")
        fingerprint_lines.append(
            f"{item['source_relative_path']} {source_hash} {t0_hash}"
        )
        results.append(
            validate_label(
                label_path, source_path, t0_path, config, source_hash, t0_hash
            )
        )
    fingerprint = hashlib.sha256("\n".join(fingerprint_lines).encode()).hexdigest()
    if fingerprint != manifest["source_fingerprint_sha256"]:
        raise AssertionError("source fingerprint mismatch")
    with (root / "labels.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(results):
        raise AssertionError("labels.csv row count differs from manifest")
    audit_gain = np.asarray(
        [result["audit_weighted_output_cost_gain"] for result in results]
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "label_root": str(root),
                "snapshots": len(results),
                "teacher_is_warm_count": sum(
                    bool(result["teacher_is_warm"]) for result in results
                ),
                "heldout_audit_weighted_output_improved_count": int(
                    np.sum(audit_gain > 0)
                ),
                "heldout_audit_weighted_output_gain_mean": float(audit_gain.mean()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
