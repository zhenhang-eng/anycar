#!/usr/bin/env python3
"""Independently validate the single-center absolute Query Replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_query_single_center_absolute_replay import CONTEXT_FIELDS, sha256


DEFAULT_REPLAY = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "query_single_center_absolute_replay_20260902_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, nargs="?", default=DEFAULT_REPLAY)
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def main() -> None:
    args = parse_args()
    output = args.replay.resolve()
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    replay_path = output / "replay.npz"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    checks = {}
    metrics = {}
    checks["artifact_hashes"] = (
        sha256(replay_path) == manifest["replay_sha256"]
        and sha256(summary_path) == manifest["summary_sha256"]
        and sha256(Path(manifest["config"])) == manifest["config_sha256"]
    )
    fullrank = Path(manifest["fullrank_source"])
    landscape = Path(manifest["landscape_source"])
    checks["source_hashes"] = (
        sha256(fullrank / "manifest.json") == manifest["fullrank_manifest_sha256"]
        and sha256(fullrank / "validation.json") == manifest["fullrank_validation_sha256"]
        and sha256(fullrank / "bank.npz") == manifest["fullrank_bank_sha256"]
        and sha256(landscape / "manifest.json") == manifest["landscape_manifest_sha256"]
        and sha256(landscape / "validation.json") == manifest["landscape_validation_sha256"]
        and sha256(landscape / "landscape.npz") == manifest["landscape_bank_sha256"]
    )
    full_validation = json.loads((fullrank / "validation.json").read_text())
    land_validation = json.loads((landscape / "validation.json").read_text())
    checks["source_qualifications"] = (
        full_validation["qualification"] == "QUERY_EXPECTED_ROAD_FULLRANK_PASS"
        and land_validation["qualification"]
        == "QUERY_FORWARD_RESPONSE_FULL100_INDEPENDENT_PASS"
    )
    checks["sealed_boundaries"] = (
        not manifest["formal_validation_or_test_consumed"]
        and not manifest["dbm_fields_or_labels_consumed"]
        and not summary["formal_validation_or_test_consumed"]
        and not summary["dbm_fields_or_labels_consumed"]
    )
    with np.load(replay_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(fullrank / "bank.npz", allow_pickle=False) as archive:
        full = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(landscape / "landscape.npz", allow_pickle=False) as archive:
        land = {name: np.asarray(archive[name]) for name in archive.files}

    context_error = max(
        max_error(data[name], full[name])
        for name in ("state", "current_action", "history", "reference", "reference_ego")
    )
    context_meta = all(
        np.array_equal(data[name], full[name])
        for name in (
            "episode_id",
            "row_in_episode",
            "control_step",
            "speed_kph",
            "variant_index",
            "fold_id",
        )
    )
    metrics["context_max_error"] = context_error
    checks["context_consolidation"] = context_error == 0.0 and context_meta
    base_knot_error = max_error(data["candidate_knots"][:, :132], full["candidate_knots"])
    base_cost_error = max_error(data["candidate_cost"][:, :132], full["candidate_cost"])
    metrics["fullrank_knots_max_error"] = base_knot_error
    metrics["fullrank_cost_max_error"] = base_cost_error
    checks["fullrank_candidates"] = (
        base_knot_error == 0.0
        and base_cost_error == 0.0
        and np.all(data["candidate_valid_mask"][:, :132])
        and np.all(data["candidate_source"][:, :132] == 0)
    )

    lookup = {int(value): index for index, value in enumerate(full["row_index"])}
    mapped = np.asarray([lookup[int(value)] for value in land["row_index"]])
    landscape_error = 0.0
    metadata_exact = True
    for local_row, replay_row in enumerate(mapped):
        knots = [land["center_knots"][local_row, :, 0]]
        costs = [land["center_cost"][local_row, :, 0]]
        sources = [np.ones(5, np.int8)]
        branches = [np.arange(5, dtype=np.int8)]
        rounds = [np.full(5, -1, np.int8)]
        local_indices = [np.arange(5, dtype=np.int16)]
        shadows = [np.arange(5) == 4]
        eligible = [np.arange(5) < 4]
        source_indices = [np.arange(5, dtype=np.int16)]
        running = 5
        for round_index in range(4):
            for branch in range(5):
                knots.append(land["probe_knots"][local_row, branch, round_index])
                costs.append(land["probe_cost"][local_row, branch, round_index])
                sources.append(np.full(32, 2, np.int8))
                branches.append(np.full(32, branch, np.int8))
                rounds.append(np.full(32, round_index, np.int8))
                local_indices.append(np.arange(32, dtype=np.int16))
                shadows.append(np.full(32, branch == 4, bool))
                eligible.append(np.full(32, branch < 4, bool))
                source_indices.append(np.arange(running, running + 32, dtype=np.int16))
                running += 32
                knots.append(land["proposal_knots"][local_row, branch, round_index])
                costs.append(land["proposal_cost"][local_row, branch, round_index])
                sources.append(np.full(6, 3, np.int8))
                branches.append(np.full(6, branch, np.int8))
                rounds.append(np.full(6, round_index, np.int8))
                local_indices.append(np.arange(6, dtype=np.int16))
                shadows.append(np.full(6, branch == 4, bool))
                eligible.append(np.full(6, branch < 4, bool))
                source_indices.append(np.arange(running, running + 6, dtype=np.int16))
                running += 6
        start = 132
        landscape_error = max(
            landscape_error,
            max_error(data["candidate_knots"][replay_row, start:], np.concatenate(knots)),
            max_error(data["candidate_cost"][replay_row, start:], np.concatenate(costs)),
        )
        metadata_exact &= all(
            np.array_equal(data[name][replay_row, start:], expected)
            for name, expected in (
                ("candidate_source", np.concatenate(sources)),
                ("candidate_source_index", np.concatenate(source_indices)),
                ("candidate_branch", np.concatenate(branches)),
                ("candidate_round", np.concatenate(rounds)),
                ("candidate_local_index", np.concatenate(local_indices)),
                ("candidate_shadow_mask", np.concatenate(shadows)),
                ("candidate_canonical_eligible_mask", np.concatenate(eligible)),
            )
        )
    metrics["landscape_candidate_max_error"] = landscape_error
    checks["landscape_candidates"] = landscape_error == 0.0 and metadata_exact

    landscape_mask = np.zeros(600, bool)
    landscape_mask[mapped] = True
    checks["density_and_masks"] = (
        np.array_equal(data["landscape_context_mask"], landscape_mask)
        and np.all(data["candidate_count"][landscape_mask] == 897)
        and np.all(data["candidate_count"][~landscape_mask] == 132)
        and np.all(~data["candidate_valid_mask"][~landscape_mask, 132:])
        and int(np.sum(data["candidate_valid_mask"])) == 155700
    )
    valid = data["candidate_valid_mask"]
    checks["finite_and_bounds"] = (
        np.all(np.isfinite(data["candidate_cost"][valid]))
        and np.all(np.isfinite(data["candidate_knots"][valid]))
        and np.all(data["candidate_knots"][valid] >= -1.0)
        and np.all(data["candidate_knots"][valid] <= 1.0)
    )
    checks["fold_balance"] = (
        [int(np.sum(data["fold_id"] == fold)) for fold in range(5)] == [120] * 5
        and len(np.unique(data["episode_id"])) == 100
    )
    checks["summary_counts"] = (
        summary["state_count"] == 600
        and summary["episode_count"] == 100
        and summary["valid_candidate_count"] == 155700
        and summary["landscape_context_count"] == 100
    )
    qualification = (
        "QUERY_SINGLE_CENTER_ABSOLUTE_REPLAY_INDEPENDENT_PASS"
        if all(checks.values())
        else "QUERY_SINGLE_CENTER_ABSOLUTE_REPLAY_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification,
        "checks": checks,
        "metrics": metrics,
        "manifest_sha256": sha256(manifest_path),
        "replay_sha256": sha256(replay_path),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
    }
    dump_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2, default=json_default))
    if qualification.endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
