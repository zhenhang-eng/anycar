#!/usr/bin/env python3
"""Independently reconstruct the target-coverage absolute Query replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "car_foundation"))

from car_foundation.mppi_proposal_policy import ego_reference_features  # noqa: E402


DEFAULT_REPLAY = Path("/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_target_coverage_absolute_replay_20260903_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, nargs="?", default=DEFAULT_REPLAY)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float("inf")
    if actual.dtype.kind in "US" or expected.dtype.kind in "US":
        return 0.0 if np.array_equal(actual, expected) else 1.0
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.0


def maximum(values: list[float]) -> float:
    return float(max(values, default=0.0))


def main() -> None:
    args = parse_args()
    root = args.replay.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(Path(manifest["config"]).read_text())
    base = Path(manifest["base_absolute_replay"])
    coverage = Path(manifest["coverage_fullrank"])
    base_manifest = json.loads((base / "manifest.json").read_text())
    base_validation = json.loads((base / "validation.json").read_text())
    coverage_manifest = json.loads((coverage / "manifest.json").read_text())
    coverage_validation = json.loads((coverage / "validation.json").read_text())
    replay_path = root / "replay.npz"
    hash_checks = {
        "config": sha256(Path(manifest["config"])) == manifest["config_sha256"],
        "base_manifest": sha256(base / "manifest.json") == manifest["base_manifest_sha256"],
        "base_validation": sha256(base / "validation.json") == manifest["base_validation_sha256"],
        "base_replay": sha256(base / "replay.npz") == manifest["base_replay_sha256"],
        "coverage_manifest": sha256(coverage / "manifest.json") == manifest["coverage_manifest_sha256"],
        "coverage_validation": sha256(coverage / "validation.json") == manifest["coverage_validation_sha256"],
        "coverage_bank": sha256(coverage / "bank.npz") == manifest["coverage_bank_sha256"],
        "query_checkpoint": sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "replay": sha256(replay_path) == manifest["replay_sha256"],
        "summary": sha256(root / "summary.json") == manifest["summary_sha256"],
    }
    if base_validation["qualification"] not in {
        "QUERY_SINGLE_CENTER_ABSOLUTE_REPLAY_INDEPENDENT_PASS",
        "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS",
    }:
        raise AssertionError("base qualification changed")
    if coverage_validation["qualification"] != "QUERY_EXPECTED_ROAD_FULLRANK_PASS":
        raise AssertionError("coverage qualification changed")

    with np.load(base / "replay.npz", allow_pickle=False) as archive:
        old = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(coverage / "bank.npz", allow_pickle=False) as archive:
        bank = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(replay_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    old_count = len(old["state"])
    new_count = len(bank["state"])
    maximum_count = old["candidate_knots"].shape[1]
    bank_count = bank["candidate_knots"].shape[1]

    prefix_errors = [error(data[name][:old_count], old[name]) for name in old]
    context_errors = []
    for name in (
        "episode_id", "episode_index", "row_in_episode", "control_step", "speed_kph",
        "speed_index", "variant_index", "road_name", "fold_id", "source_snapshot_sha256",
        "state", "current_action", "history", "reference", "reference_ego",
    ):
        context_errors.append(error(data[name][old_count:], bank[name]))
    context_errors.extend(
        (
            error(data["row_index"][old_count:], np.arange(old_count, old_count + new_count)),
            error(
                data["critic_reference"][old_count:],
                np.stack([ego_reference_features(value, float(state[3])) for value, state in zip(bank["reference_ego"], bank["state"])]),
            ),
            error(data["critic_current"][old_count:], np.concatenate((bank["state"][:, 3:5], bank["current_action"]), axis=1)),
            error(data["warm_knots"][old_count:], bank["mean_knots_before"]),
            error(data["warm_cost"][old_count:], bank["warm_direct_cost_replayed"]),
            error(data["fullrank_teacher_knots"][old_count:], bank["fullrank_teacher_knots"]),
            error(data["fullrank_teacher_cost"][old_count:], bank["fullrank_teacher_direct_cost"]),
        )
    )

    new_slice = slice(old_count, old_count + new_count)
    candidate_errors = [
        error(data["candidate_knots"][new_slice, :bank_count], bank["candidate_knots"]),
        error(data["candidate_cost"][new_slice, :bank_count], bank["candidate_cost"]),
        error(data["candidate_valid_mask"][new_slice, :bank_count], np.ones((new_count, bank_count), bool)),
        error(data["candidate_clipped_mask"][new_slice, :bank_count], np.any(bank["candidate_clipped_mask"], axis=(2, 3))),
        error(data["candidate_source"][new_slice, :bank_count], np.zeros((new_count, bank_count), np.int8)),
        error(data["candidate_source_index"][new_slice, :bank_count], np.broadcast_to(np.arange(bank_count, dtype=np.int16), (new_count, bank_count))),
        error(data["candidate_canonical_eligible_mask"][new_slice, :bank_count], np.ones((new_count, bank_count), bool)),
        error(data["candidate_valid_mask"][new_slice, bank_count:], np.zeros((new_count, maximum_count - bank_count), bool)),
        error(data["candidate_source"][new_slice, bank_count:], np.full((new_count, maximum_count - bank_count), -1, np.int8)),
        error(data["candidate_count"][new_slice], np.full(new_count, bank_count, np.int16)),
    ]
    coverage_folds = data["fold_id"][new_slice]
    split_checks = {
        "coverage_only_fit_folds": bool(np.all(np.isin(coverage_folds, config["fit_folds"]))),
        "coverage_absent_inner": bool(np.all(coverage_folds != config["inner_selection_fold"])),
        "coverage_absent_outer": bool(np.all(coverage_folds != config["outer_fold"])),
        "old_inner_exact_count": int(np.sum(data["fold_id"][:old_count] == config["inner_selection_fold"])) == int(np.sum(old["fold_id"] == config["inner_selection_fold"])),
        "old_outer_exact_count": int(np.sum(data["fold_id"][:old_count] == config["outer_fold"])) == int(np.sum(old["fold_id"] == config["outer_fold"])),
        "coverage_fold_balance": len({int(np.sum(coverage_folds == fold)) for fold in config["fit_folds"]}) == 1,
    }
    valid = data["candidate_valid_mask"]
    finite = bool(np.isfinite(data["candidate_cost"][valid]).all() and np.isfinite(data["candidate_knots"][valid]).all())
    sealed = (
        not manifest.get("formal_validation_or_test_consumed", True)
        and not manifest.get("dbm_fields_or_labels_consumed")
        and not manifest.get("query_analytic_gradient_consumed", True)
        and not base_manifest.get("formal_validation_or_test_consumed", True)
        and not coverage_manifest.get("formal_validation_or_test_consumed", True)
    )
    checks = {
        "artifact_hashes": all(hash_checks.values()),
        "base_prefix_exact": maximum(prefix_errors) == 0.0,
        "coverage_context_exact": maximum(context_errors) <= 1e-7,
        "coverage_candidate_bank_exact": maximum(candidate_errors) <= 1e-7,
        "row_count": len(data["state"]) == manifest["state_count"] == old_count + new_count,
        "row_index_contiguous": bool(np.array_equal(data["row_index"], np.arange(old_count + new_count))),
        "split_contract": all(split_checks.values()),
        "valid_candidate_count": int(valid.sum()) == int(manifest["valid_candidate_count"]),
        "finite_valid_candidates": finite,
        "sealed_boundary": sealed,
    }
    passed = all(checks.values())
    output = {
        "qualification": "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS" if passed else "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replay": str(root),
        "manifest_sha256": sha256(manifest_path),
        "checks": checks,
        "hash_checks": hash_checks,
        "split_checks": split_checks,
        "maximum_errors": {
            "base_prefix": maximum(prefix_errors),
            "coverage_context": maximum(context_errors),
            "coverage_candidate": maximum(candidate_errors),
        },
        "counts": {
            "base_rows": old_count,
            "coverage_rows": new_count,
            "total_rows": len(data["state"]),
            "valid_candidates": int(valid.sum()),
            "coverage_fold_rows": {str(fold): int(np.sum(coverage_folds == fold)) for fold in range(5)},
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }
    (root / "validation.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
