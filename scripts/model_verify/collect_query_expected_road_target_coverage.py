#!/usr/bin/env python3
"""Collect targeted train-only Query expected-road coverage episodes."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import collect_query_expected_road_closed_loop as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_expected_road_target_coverage_expansion_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def deterministic_contract() -> dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "mode": "warn_only",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
    }


def phase_fraction(episode_index: int) -> float:
    return float((0.071 + 0.113 * int(episode_index)) % 1.0)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"] or config["query_analytic_gradient_consumed"]:
        raise AssertionError("sealed boundary violation")
    deterministic = deterministic_contract()
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    checkpoint = Path(config["checkpoint"]).resolve()
    seed_replay = Path(config["seed_replay"]).resolve()
    source = Path(config["source_collection"]).resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_validation = json.loads((source / "validation.json").read_text())
    if source_validation["qualification"] != "QUERY_EXPECTED_ROAD_CLOSED_LOOP_PASS":
        raise AssertionError("source collection is not independently qualified")
    diagnostic = Path(config["source_diagnostic"]).resolve()
    diagnostic_validation = json.loads((diagnostic / "validation.json").read_text())
    if diagnostic_validation["qualification"] != "QUERY_OAC_FIXED_LR1E5_HARD_SLICE_DIAGNOSTIC_INDEPENDENT_PASS":
        raise AssertionError("source coverage diagnostic is not independently qualified")
    episodes = [dict(value) for value in config["episodes"]]
    if len({value["episode_id"] for value in episodes}) != len(episodes):
        raise AssertionError("episode ids must be unique")
    if len({int(value["episode_index"]) for value in episodes}) != len(episodes):
        raise AssertionError("episode indices must be unique because they seed MPPI")
    if len({int(value["seed_row"]) for value in episodes}) != len(episodes):
        raise AssertionError("seed rows must be unique")
    old_seed_rows = {int(value["seed_row"]) for value in source_manifest["episodes"]}
    disjoint_collections = [Path(value).resolve() for value in config.get("disjoint_seed_collections", [])]
    disjoint_seed_rows = set(old_seed_rows)
    disjoint_collection_hashes = {}
    for collection in disjoint_collections:
        collection_manifest_path = collection / "manifest.json"
        collection_manifest = json.loads(collection_manifest_path.read_text())
        disjoint_seed_rows.update(int(value["seed_row"]) for value in collection_manifest["episodes"])
        disjoint_collection_hashes[str(collection)] = base.sha256(collection_manifest_path)
    if disjoint_seed_rows & {int(value["seed_row"]) for value in episodes}:
        raise AssertionError("target expansion reuses a source-collection seed row")
    expected_cells = {tuple(map(int, value.split(":"))) for value in config["target_cells"]}
    if {(int(value["speed_kph"]), int(value["variant_index"])) for value in episodes} != expected_cells:
        raise AssertionError("episode cells differ from target cells")
    fold_cell_counts = {
        str(fold): {
            f"{speed}:{variant}": sum(
                int(value["fold_id"]) == fold
                and int(value["speed_kph"]) == speed
                and int(value["variant_index"]) == variant
                for value in episodes
            )
            for speed, variant in sorted(expected_cells)
        }
        for fold in (2, 3, 4)
    }
    if any(set(value.values()) != {2} for value in fold_cell_counts.values()):
        raise AssertionError("each fit fold must receive two episodes per target cell")
    for episode in episodes:
        actual_phase = phase_fraction(int(episode["episode_index"]))
        circular_error = abs((actual_phase - float(episode["planned_start_fraction"]) + 0.5) % 1.0 - 0.5)
        if circular_error > 1e-10:
            raise AssertionError(f"phase mismatch for {episode['episode_id']}")
        variant = base.effective_variant(
            int(episode["variant_index"]), int(episode["speed_kph"]),
            int(episode["repeat_index"]), int(config["variation_slot_denominator"]),
            float(config["road_variation_fraction"]),
        )
        if abs(float(variant["road_variation_normalized"]) - float(episode["planned_variation_normalized"])) > 1e-12:
            raise AssertionError(f"variation mismatch for {episode['episode_id']}")
    with np.load(seed_replay, allow_pickle=False) as seed_data:
        for episode in episodes:
            row = int(episode["seed_row"])
            if int(seed_data["speed_bin_kph"][row]) != int(episode["speed_kph"]):
                raise AssertionError(f"seed speed mismatch for {episode['episode_id']}")
    output.mkdir(parents=True)
    runtime_args = SimpleNamespace(
        device=args.device,
        repeats_per_cell=int(config["variation_slot_denominator"]),
        road_variation_fraction=float(config["road_variation_fraction"]),
        num_samples=int(config["mppi_num_samples"]),
        num_iterations=int(config["mppi_num_iterations"]),
        sampling_mode=str(config["mppi_sampling_mode"]),
        seed=int(config["collection_seed"]),
        burn_in_steps=int(config["burn_in_steps"]),
        snapshots_per_episode=int(config["snapshots_per_episode"]),
        snapshot_stride=int(config["snapshot_stride"]),
    )
    params = base.TorchMPPIParams(
        num_samples=runtime_args.num_samples,
        num_iterations=runtime_args.num_iterations,
        sampling_mode=runtime_args.sampling_mode,
        seed=runtime_args.seed,
    )
    manifest = {
        "format_version": 1,
        "dataset_type": "anycar-query-expected-road-closed-loop",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "targeted train-only Query expected-road coverage expansion",
        "config": str(config_path),
        "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "query_checkpoint": str(checkpoint),
        "query_checkpoint_sha256": base.sha256(checkpoint),
        "seed_replay": str(seed_replay),
        "seed_replay_sha256": base.sha256(seed_replay),
        "seed_replay_fields_used": ["history", "state_six[x,y,yaw,vx,yawrate]", "current_action", "source provenance", "speed bin"],
        "seed_replay_fields_explicitly_not_used": ["reference", "behavior_future_state", "behavior_future_action", "mean_knots_before", "cost"],
        "causal_contract": "step 0 repeats current_action; later steps use the preceding Query-MPPI shifted output knots",
        "reference_contract": "procedural periodic arc-length desired road; never recorded future trajectory",
        "plant_contract": "first state of a fresh frozen-Query optimized-sequence rollout is the deterministic next state",
        "source_collection": str(source),
        "source_collection_manifest_sha256": base.sha256(source / "manifest.json"),
        "source_collection_validation_sha256": base.sha256(source / "validation.json"),
        "disjoint_seed_collection_manifest_sha256": disjoint_collection_hashes,
        "source_diagnostic": str(diagnostic),
        "source_diagnostic_validation_sha256": base.sha256(diagnostic / "validation.json"),
        "targeted_noncartesian_cells": sorted(config["target_cells"]),
        "fold_cell_counts": fold_cell_counts,
        "collection": {
            "burn_in_steps": runtime_args.burn_in_steps,
            "snapshots_per_episode": runtime_args.snapshots_per_episode,
            "snapshot_stride": runtime_args.snapshot_stride,
            "repeats_per_cell": runtime_args.repeats_per_cell,
            "road_variation_fraction": runtime_args.road_variation_fraction,
            "speed_bins_kph": sorted({int(value["speed_kph"]) for value in episodes}),
            "base_road_variants": base.ROAD_VARIANTS,
            "selected_variant_indices": sorted({int(value["variant_index"]) for value in episodes}),
            "mppi": asdict(params),
            "sampling_bank_version": None,
        },
        "deterministic_runtime_contract": deterministic,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "episodes": episodes,
    }
    base.json_dump(output / "manifest.json", manifest)
    model = base.QueryDeploymentModel.from_checkpoint(checkpoint, args.device)
    episode_summaries = []
    seed_data = np.load(seed_replay, allow_pickle=False)
    try:
        for episode in episodes:
            print(f"collecting {episode['episode_id']} cell={episode['speed_kph']}:{episode['variant_index']} fold={episode['fold_id']}", flush=True)
            episode_summaries.append(base.collect_episode(episode, seed_data, model, runtime_args, output))
    finally:
        seed_data.close()
    all_speed = np.concatenate([np.load(output / value["episode"]["episode_id"] / "trace.npz")["state"][:, 3] for value in episode_summaries])
    all_yawrate = np.concatenate([np.load(output / value["episode"]["episode_id"] / "trace.npz")["state"][:, 4] for value in episode_summaries])
    summary = {
        "qualification": "QUERY_EXPECTED_ROAD_TARGET_COVERAGE_PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(episode_summaries),
        "snapshot_count": int(sum(value["snapshot_count"] for value in episode_summaries)),
        "actual_speed_mps": base.stats(all_speed),
        "actual_speed_kph": base.stats(all_speed * 3.6),
        "actual_yawrate_rps": base.stats(all_yawrate),
        "fold_cell_counts": fold_cell_counts,
        "episode_summaries": episode_summaries,
        "deterministic_runtime_contract": deterministic,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    base.json_dump(output / "summary.json", summary)
    manifest["summary_sha256"] = base.sha256(output / "summary.json")
    base.json_dump(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
