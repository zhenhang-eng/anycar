#!/usr/bin/env python3
"""Append validated target-road rows to the absolute Query replay as fit-only data."""

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


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_target_coverage_absolute_replay_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"] or config["query_analytic_gradient_consumed"]:
        raise AssertionError("sealed-boundary contract violated")
    base = Path(config["base_absolute_replay"]).resolve()
    coverage = Path(config["coverage_fullrank"]).resolve()
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")

    base_manifest = json.loads((base / "manifest.json").read_text())
    base_validation = json.loads((base / "validation.json").read_text())
    coverage_manifest = json.loads((coverage / "manifest.json").read_text())
    coverage_validation = json.loads((coverage / "validation.json").read_text())
    if base_validation["qualification"] not in {
        "QUERY_SINGLE_CENTER_ABSOLUTE_REPLAY_INDEPENDENT_PASS",
        "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS",
    }:
        raise AssertionError("base absolute replay is not independently qualified")
    if coverage_validation["qualification"] != "QUERY_EXPECTED_ROAD_FULLRANK_PASS":
        raise AssertionError("coverage full-rank bank is not independently qualified")
    for value in (base_manifest, coverage_manifest):
        if value.get("formal_validation_or_test_consumed", True) or value.get("dbm_fields_or_labels_consumed") or value.get("query_analytic_gradient_consumed", False):
            raise AssertionError("source violates sealed-boundary contract")
    if base_manifest["query_checkpoint_sha256"] != coverage_manifest["query_checkpoint_sha256"]:
        raise AssertionError("base and coverage use different Query checkpoints")

    with np.load(base / "replay.npz", allow_pickle=False) as archive:
        old = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(coverage / "bank.npz", allow_pickle=False) as archive:
        bank = {name: np.asarray(archive[name]) for name in archive.files}
    old_count = len(old["state"])
    new_count = len(bank["state"])
    maximum = old["candidate_knots"].shape[1]
    bank_count = bank["candidate_knots"].shape[1]
    if old_count != base_manifest["state_count"] or new_count != coverage_manifest["row_count"]:
        raise AssertionError("source row-count mismatch")
    if bank_count > maximum:
        raise AssertionError("coverage candidate bank does not fit base padded width")

    new: dict[str, np.ndarray] = {}
    direct_context = (
        "episode_id", "episode_index", "row_in_episode", "control_step", "speed_kph",
        "speed_index", "variant_index", "road_name", "fold_id", "source_snapshot_sha256",
        "state", "current_action", "history", "reference", "reference_ego",
    )
    for name in direct_context:
        new[name] = bank[name]
    new["row_index"] = np.arange(old_count, old_count + new_count, dtype=np.int64)
    new["critic_reference"] = np.stack(
        [ego_reference_features(value, float(state[3])) for value, state in zip(bank["reference_ego"], bank["state"])]
    ).astype(np.float32)
    new["critic_current"] = np.concatenate((bank["state"][:, 3:5], bank["current_action"]), axis=1).astype(np.float32)
    new["warm_knots"] = bank["mean_knots_before"].astype(np.float32)
    new["warm_cost"] = bank["warm_direct_cost_replayed"].astype(np.float32)
    new["fullrank_teacher_knots"] = bank["fullrank_teacher_knots"].astype(np.float32)
    new["fullrank_teacher_cost"] = bank["fullrank_teacher_direct_cost"].astype(np.float32)
    new["landscape_context_mask"] = np.zeros(new_count, bool)
    new["landscape_source_audit_row"] = np.full(new_count, -1, np.int16)

    candidate_shapes = {
        "candidate_knots": ((new_count, maximum, 8, 2), np.float32, 0),
        "candidate_cost": ((new_count, maximum), np.float32, 0),
        "candidate_valid_mask": ((new_count, maximum), bool, False),
        "candidate_clipped_mask": ((new_count, maximum), bool, False),
        "candidate_source": ((new_count, maximum), np.int8, -1),
        "candidate_source_index": ((new_count, maximum), np.int16, -1),
        "candidate_branch": ((new_count, maximum), np.int8, -1),
        "candidate_round": ((new_count, maximum), np.int8, -1),
        "candidate_local_index": ((new_count, maximum), np.int16, -1),
        "candidate_shadow_mask": ((new_count, maximum), bool, False),
        "candidate_canonical_eligible_mask": ((new_count, maximum), bool, False),
    }
    for name, (shape, dtype, fill) in candidate_shapes.items():
        new[name] = np.full(shape, fill, dtype=dtype)
    new["candidate_knots"][:, :bank_count] = bank["candidate_knots"]
    new["candidate_cost"][:, :bank_count] = bank["candidate_cost"]
    new["candidate_valid_mask"][:, :bank_count] = True
    new["candidate_clipped_mask"][:, :bank_count] = np.any(bank["candidate_clipped_mask"], axis=(2, 3))
    new["candidate_source"][:, :bank_count] = 0
    new["candidate_source_index"][:, :bank_count] = np.arange(bank_count, dtype=np.int16)
    new["candidate_canonical_eligible_mask"][:, :bank_count] = True
    new["candidate_count"] = np.full(new_count, bank_count, np.int16)

    if set(new) != set(old):
        raise AssertionError(f"field mismatch: missing={sorted(set(old)-set(new))}, extra={sorted(set(new)-set(old))}")
    data = {name: np.concatenate((old[name], new[name]), axis=0) for name in old}
    if not np.array_equal(data["row_index"], np.arange(old_count + new_count)):
        raise AssertionError("row indices are not contiguous")
    fit_folds = np.asarray(config["fit_folds"], np.int64)
    if not np.all(np.isin(new["fold_id"], fit_folds)):
        raise AssertionError("coverage rows escaped fit folds")
    if np.any(new["fold_id"] == int(config["inner_selection_fold"])) or np.any(new["fold_id"] == int(config["outer_fold"])):
        raise AssertionError("coverage rows leaked into inner/outer folds")
    valid = data["candidate_valid_mask"]
    if not np.isfinite(data["candidate_cost"][valid]).all() or not np.isfinite(data["candidate_knots"][valid]).all():
        raise AssertionError("non-finite valid candidate")

    output.mkdir(parents=True)
    replay_path = output / "replay.npz"
    np.savez_compressed(replay_path, **data)
    summary = {
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "state_count": old_count + new_count,
        "base_state_count": old_count,
        "coverage_state_count": new_count,
        "episode_count": int(len(np.unique(data["episode_id"]))),
        "maximum_candidates_per_state": maximum,
        "valid_candidate_count": int(valid.sum()),
        "coverage_candidates_per_state": bank_count,
        "candidate_cost": stats(data["candidate_cost"][valid]),
        "states_per_fold": {str(fold): int(np.sum(data["fold_id"] == fold)) for fold in range(5)},
        "coverage_states_per_fold": {str(fold): int(np.sum(new["fold_id"] == fold)) for fold in range(5)},
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-target-coverage-absolute-replay-v1",
        "dataset_type": "train-only-pure-query-absolute-action-cost-replay-with-targeted-fit-expansion",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "base_absolute_replay": str(base),
        "base_manifest_sha256": sha256(base / "manifest.json"),
        "base_validation_sha256": sha256(base / "validation.json"),
        "base_replay_sha256": sha256(base / "replay.npz"),
        "coverage_fullrank": str(coverage),
        "coverage_manifest_sha256": sha256(coverage / "manifest.json"),
        "coverage_validation_sha256": sha256(coverage / "validation.json"),
        "coverage_bank_sha256": sha256(coverage / "bank.npz"),
        "fullrank_source": str(coverage),
        "landscape_source": base_manifest["landscape_source"],
        "query_checkpoint": base_manifest["query_checkpoint"],
        "query_checkpoint_sha256": base_manifest["query_checkpoint_sha256"],
        "replay_sha256": sha256(replay_path),
        "summary_sha256": sha256(summary_path),
        "state_count": old_count + new_count,
        "base_state_count": old_count,
        "coverage_state_count": new_count,
        "episode_count": summary["episode_count"],
        "valid_candidate_count": summary["valid_candidate_count"],
        "split_contract": {
            "fit_folds": config["fit_folds"],
            "inner_selection_fold": config["inner_selection_fold"],
            "outer_fold": config["outer_fold"],
            "coverage_role": "fit-only",
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "limitations": [
            "Only target cells 55:2 and 100:3 receive new fit rows.",
            "Coverage rows have the 132-candidate full-rank bank and no forward-response landscape bank.",
            "The original 600-row prefix, including inner fold 1 and outer fold 0, is immutable.",
        ],
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
