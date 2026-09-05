#!/usr/bin/env python3
"""Validate targeted Query expected-road coverage plus every closed-loop chain."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLLECTION = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "query_expected_road_target_coverage_expansion_20260903_v1"
)
BASE_VALIDATOR = REPO_ROOT / "scripts/model_verify/validate_query_expected_road_closed_loop.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path, nargs="?", default=DEFAULT_COLLECTION)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.collection.resolve()
    manifest_path, summary_path = root / "manifest.json", root / "summary.json"
    manifest, summary = json.loads(manifest_path.read_text()), json.loads(summary_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    source = Path(manifest["source_collection"])
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_seed_rows = {int(value["seed_row"]) for value in source_manifest["episodes"]}
    disjoint_hash_checks = []
    for collection_name, expected_hash in manifest.get("disjoint_seed_collection_manifest_sha256", {}).items():
        collection = Path(collection_name)
        collection_manifest_path = collection / "manifest.json"
        collection_manifest = json.loads(collection_manifest_path.read_text())
        source_seed_rows.update(int(value["seed_row"]) for value in collection_manifest["episodes"])
        disjoint_hash_checks.append(sha256(collection_manifest_path) == expected_hash)
    episodes = manifest["episodes"]
    cells = sorted({(int(value["speed_kph"]), int(value["variant_index"])) for value in episodes})
    checks = {
        "dataset_type": manifest["dataset_type"] == "anycar-query-expected-road-closed-loop",
        "role_targeted_train_only": manifest["role"] == "targeted train-only Query expected-road coverage expansion",
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": sha256(summary_path) == manifest["summary_sha256"],
        "checkpoint_hash": sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "seed_replay_hash": sha256(Path(manifest["seed_replay"])) == manifest["seed_replay_sha256"],
        "source_manifest_hash": sha256(source / "manifest.json") == manifest["source_collection_manifest_sha256"],
        "source_validation_hash": sha256(source / "validation.json") == manifest["source_collection_validation_sha256"],
        "diagnostic_validation_hash": sha256(Path(manifest["source_diagnostic"]) / "validation.json") == manifest["source_diagnostic_validation_sha256"],
        "episodes_match_config": episodes == config["episodes"],
        "episode_count": len(episodes) == int(summary["episode_count"]),
        "snapshot_count": int(summary["snapshot_count"]) == len(episodes) * int(manifest["collection"]["snapshots_per_episode"]),
        "unique_episode_ids": len({value["episode_id"] for value in episodes}) == len(episodes),
        "unique_episode_indices": len({int(value["episode_index"]) for value in episodes}) == len(episodes),
        "unique_seed_rows": len({int(value["seed_row"]) for value in episodes}) == len(episodes),
        "seed_rows_new": not bool(source_seed_rows & {int(value["seed_row"]) for value in episodes}),
        "target_cells": cells == sorted(tuple(map(int, value.split(":"))) for value in config["target_cells"]),
        "disjoint_collection_hashes": all(disjoint_hash_checks),
        "fit_folds_only": {int(value["fold_id"]) for value in episodes} == {2, 3, 4},
        "fold_balance": all(set(value.values()) == {2} for value in manifest["fold_cell_counts"].values()),
        "formal_test_sealed": not bool(manifest["formal_validation_or_test_consumed"]) and not bool(summary["formal_validation_or_test_consumed"]),
        "dbm_absent": not bool(manifest["dbm_fields_or_labels_consumed"]) and not bool(summary["dbm_fields_or_labels_consumed"]),
        "analytic_query_gradient_absent": not bool(manifest["query_analytic_gradient_consumed"]) and not bool(summary["query_analytic_gradient_consumed"]),
    }
    phase_errors, variation_errors, episode_hash_checks = [], [], []
    summary_by_id = {value["episode"]["episode_id"]: value for value in summary["episode_summaries"]}
    for episode in episodes:
        item = summary_by_id[episode["episode_id"]]
        actual_phase = float(item["road"]["initial_frenet_s_m"]) / float(item["road"]["total_length_m"])
        planned_phase = float(episode["planned_start_fraction"])
        phase_errors.append(abs((actual_phase - planned_phase + 0.5) % 1.0 - 0.5))
        variation_errors.append(abs(float(item["road"]["road_variation_normalized"]) - float(episode["planned_variation_normalized"])))
        episode_dir = root / episode["episode_id"]
        episode_hash_checks.extend(
            sha256(episode_dir / name) == item["artifacts"][name]["sha256"]
            for name in ("road.npz", "trace.npz", "snapshots.npz")
        )
    checks["phase_plan"] = max(phase_errors) <= 1e-10
    checks["variation_plan"] = max(variation_errors) <= 1e-12
    checks["episode_hashes"] = all(episode_hash_checks)
    cell_validations = {}
    with tempfile.TemporaryDirectory(prefix="query_target_coverage_validate_") as temporary:
        temporary_root = Path(temporary)
        for speed, variant in cells:
            local_root = temporary_root / f"cell_{speed}_{variant}"
            local_root.mkdir()
            local_episodes = [dict(value) for value in episodes if int(value["speed_kph"]) == speed and int(value["variant_index"]) == variant]
            for repeat, episode in enumerate(local_episodes):
                os.symlink(root / episode["episode_id"], local_root / episode["episode_id"], target_is_directory=True)
                episode["repeat_index"] = repeat
            local_manifest = dict(manifest)
            local_manifest["episodes"] = local_episodes
            local_manifest["collection"] = dict(manifest["collection"])
            local_manifest["collection"]["speed_bins_kph"] = [speed]
            local_manifest["collection"]["selected_variant_indices"] = [variant]
            local_manifest["collection"]["repeats_per_cell"] = len(local_episodes)
            (local_root / "manifest.json").write_text(json.dumps(local_manifest, indent=2, sort_keys=True) + "\n")
            local_validation = local_root / "validation.json"
            command = [
                sys.executable, str(BASE_VALIDATOR), str(local_root),
                "--device", args.device, "--replay-snapshots", str(len(local_episodes)),
                "--output", str(local_validation),
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if not local_validation.exists():
                raise RuntimeError(f"base validator failed before writing {speed}:{variant}: {result.stderr}")
            report = json.loads(local_validation.read_text())
            report["process_returncode"] = result.returncode
            cell_validations[f"{speed}:{variant}"] = report
            checks[f"cell_{speed}_{variant}_closed_loop"] = result.returncode == 0 and report["qualification"] == "QUERY_EXPECTED_ROAD_CLOSED_LOOP_PASS"
    passed = all(checks.values())
    report = {
        "qualification": "QUERY_EXPECTED_ROAD_TARGET_COVERAGE_PASS" if passed else "QUERY_EXPECTED_ROAD_TARGET_COVERAGE_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "collection": str(root),
        "manifest_sha256": sha256(manifest_path),
        "summary_sha256": sha256(summary_path),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "base_validator": str(BASE_VALIDATOR),
        "base_validator_sha256": sha256(BASE_VALIDATOR),
        "checks": checks,
        "maximum_plan_errors": {"start_fraction": max(phase_errors), "variation_normalized": max(variation_errors)},
        "cell_validations": cell_validations,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": report["qualification"],
        "checks": checks,
        "maximum_plan_errors": report["maximum_plan_errors"],
        "cell_qualifications": {name: value["qualification"] for name, value in cell_validations.items()},
        "cell_domains": {name: value["domain"] for name, value in cell_validations.items()},
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
