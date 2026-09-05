#!/usr/bin/env python3
"""Diagnose Query Actor update geometry over speed by road-variant strata."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
from analyze_query_actor_update_geometry import (  # noqa: E402
    critic_update_direction,
    empirical_bank_response,
    empirical_update_direction,
)
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402
from run_query_actor_aggregation_microstep import vector_cosine  # noqa: E402
from run_query_single_center_oac20to1 import actor_from_payload, load_inputs, sha256  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_actor_stratum_geometry_config_20260904_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    return np.asarray([
        [vector_cosine(vectors[left], vectors[right]) for right in range(len(vectors))]
        for left in range(len(vectors))
    ], np.float64)


def direction_report(
    vectors: np.ndarray, matrix: np.ndarray, labels: list[str], native: np.ndarray
) -> dict[str, Any]:
    equal_cell = vectors.mean(axis=0)
    off = matrix[np.triu_indices(len(labels), 1)]
    return {
        "norm_by_stratum": {
            label: float(np.linalg.norm(vectors[index].astype(np.float64)))
            for index, label in enumerate(labels)
        },
        "pair_cosine_minimum_all": float(off.min()),
        "pair_cosine_median_all": float(np.median(off)),
        "pair_negative_fraction_all": float(np.mean(off < 0.0)),
        "equal_cell_update_norm": float(np.linalg.norm(equal_cell.astype(np.float64))),
        "equal_cell_vs_native_cosine": vector_cosine(equal_cell, native),
        "matrix": matrix.tolist(),
    }


def compute_seed(
    seed: int, source_record: dict, geometry_record: dict, data: dict[str, np.ndarray],
    geometry_config: dict, config: dict, device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    checkpoint_path = Path(source_record["source_checkpoint"])
    online_path = Path(source_record["source_arrays"])
    geometry_arrays_path = Path(geometry_record["output_arrays"])
    if sha256(checkpoint_path) != source_record["source_checkpoint_sha256"]:
        raise AssertionError("source checkpoint hash mismatch")
    if sha256(online_path) != source_record["source_arrays_sha256"]:
        raise AssertionError("source online arrays hash mismatch")
    if sha256(geometry_arrays_path) != geometry_record["output_arrays_sha256"]:
        raise AssertionError("source geometry arrays hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    with np.load(online_path, allow_pickle=False) as archive:
        online = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(geometry_arrays_path, allow_pickle=False) as archive:
        prior = {name: np.asarray(archive[name]) for name in archive.files}
    fit = checkpoint["fit_indices"].astype(np.int64)
    selection = checkpoint["selection_indices"].astype(np.int64)
    outer = checkpoint["outer_indices_unevaluated"].astype(np.int64)
    empirical = empirical_bank_response(online, fit, geometry_config)
    visited = empirical["rows"]
    if not np.array_equal(visited, prior["visited_indices"]):
        raise AssertionError("visited state set changed")
    actor_inputs = load_inputs(data, checkpoint["actor_normalization"])
    critic_inputs = load_inputs(data, checkpoint["critic_normalization"])
    actor = actor_from_payload(checkpoint, "selected_actor_state_dict", device)
    critics, trainings = [], []
    for twin in (1, 2):
        critic = ConfigurableAbsoluteActionValueCritic().to(device)
        critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
        critic.eval()
        critics.append(critic)
        trainings.append(checkpoint[f"critic{twin}_training"])
    sigma = np.asarray(geometry_config["gradient_contract"]["noise_sigma"], np.float32).reshape(1, 2)
    batch_size = int(geometry_config["gradient_contract"]["batch_size"])
    strata = [
        (int(speed), int(variant))
        for speed in config["population"]["speed_values_kph"]
        for variant in config["population"]["variant_indices"]
    ]
    labels = [f"{speed}:{variant}" for speed, variant in strata]
    critic_updates, empirical_updates, counts = [], [], []
    parameter_names, parameter_offsets = None, None
    for speed, variant in strata:
        full_mask = (data["speed_kph"][fit] == speed) & (data["variant_index"][fit] == variant)
        visited_mask = (data["speed_kph"][visited] == speed) & (data["variant_index"][visited] == variant)
        full_rows, local_rows = fit[full_mask], visited[visited_mask]
        if len(full_rows) < 18 or len(local_rows) < 11:
            raise AssertionError(f"insufficient stratum coverage {speed}:{variant}")
        critic_update, names, offsets, _ = critic_update_direction(
            actor, critics, trainings, actor_inputs, critic_inputs, full_rows,
            geometry_config, device,
        )
        empirical_update, empirical_names, empirical_offsets = empirical_update_direction(
            actor, actor_inputs, local_rows, empirical["secant"][visited_mask],
            sigma, device, batch_size,
        )
        if names != empirical_names or not np.array_equal(offsets, empirical_offsets):
            raise AssertionError("parameter layout mismatch")
        if parameter_names is None:
            parameter_names, parameter_offsets = names, offsets
        elif names != parameter_names or not np.array_equal(offsets, parameter_offsets):
            raise AssertionError("parameter layout changed across strata")
        critic_updates.append(critic_update)
        empirical_updates.append(empirical_update)
        counts.append((len(full_rows), len(local_rows)))
    critic_updates = np.stack(critic_updates)
    empirical_updates = np.stack(empirical_updates)
    critic_matrix = cosine_matrix(critic_updates)
    empirical_matrix = cosine_matrix(empirical_updates)
    alignment = np.asarray([
        vector_cosine(critic_updates[index], empirical_updates[index])
        for index in range(len(strata))
    ], np.float64)
    report = {
        "seed": seed,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": source_record["source_checkpoint_sha256"],
        "source_online_arrays": str(online_path),
        "source_online_arrays_sha256": source_record["source_arrays_sha256"],
        "source_geometry_arrays": str(geometry_arrays_path),
        "source_geometry_arrays_sha256": geometry_record["output_arrays_sha256"],
        "stratum_counts": {
            label: {"fit": int(counts[index][0]), "visited": int(counts[index][1])}
            for index, label in enumerate(labels)
        },
        "critic": direction_report(critic_updates, critic_matrix, labels, prior["critic_full_update"]),
        "empirical_secant": direction_report(
            empirical_updates, empirical_matrix, labels, prior["empirical_secant_update"]
        ),
        "critic_vs_empirical_cosine_by_stratum": {
            label: float(alignment[index]) for index, label in enumerate(labels)
        },
        "critic_vs_empirical_positive_fraction": float(np.mean(alignment > 0.0)),
        "new_query_rollouts": 0,
        "oof_evaluated": False,
    }
    arrays = {
        "fit_indices": fit,
        "selection_indices_unevaluated": selection,
        "outer_indices_unevaluated": outer,
        "visited_indices": visited,
        "stratum_labels": np.asarray(labels),
        "stratum_speed_kph": np.asarray([value[0] for value in strata], np.int64),
        "stratum_variant_index": np.asarray([value[1] for value in strata], np.int64),
        "stratum_fit_count": np.asarray([value[0] for value in counts], np.int64),
        "stratum_visited_count": np.asarray([value[1] for value in counts], np.int64),
        "parameter_names": np.asarray(parameter_names),
        "parameter_offsets": parameter_offsets,
        "critic_stratum_update": critic_updates,
        "empirical_secant_stratum_update": empirical_updates,
        "critic_pair_cosine": critic_matrix,
        "empirical_secant_pair_cosine": empirical_matrix,
        "critic_vs_empirical_stratum_cosine": alignment,
    }
    return report, arrays


def route(records: list[dict[str, Any]], config: dict) -> dict[str, Any]:
    speeds = [int(value) for value in config["population"]["speed_values_kph"]]
    variants = [int(value) for value in config["population"]["variant_indices"]]
    labels = [f"{speed}:{variant}" for speed in speeds for variant in variants]
    label_index = {label: index for index, label in enumerate(labels)}
    threshold = float(config["decision_gate"]["material_negative_cosine"])
    required = int(config["decision_gate"]["stable_pair_minimum_seed_count"])
    pair_reports, critic_stable, empirical_stable = {}, [], []
    for speed in speeds:
        for left_index, left_variant in enumerate(variants):
            for right_variant in variants[left_index + 1 :]:
                left, right = f"{speed}:{left_variant}", f"{speed}:{right_variant}"
                i, j = label_index[left], label_index[right]
                critic = [record["critic"]["matrix"][i][j] for record in records]
                empirical = [record["empirical_secant"]["matrix"][i][j] for record in records]
                critic_count = int(np.sum(np.asarray(critic) <= threshold))
                empirical_count = int(np.sum(np.asarray(empirical) <= threshold))
                name = f"{left}|{right}"
                pair_reports[name] = {
                    "speed_kph": speed,
                    "critic_cosine_by_seed": critic,
                    "empirical_cosine_by_seed": empirical,
                    "critic_material_negative_seed_count": critic_count,
                    "empirical_material_negative_seed_count": empirical_count,
                }
                if critic_count >= required:
                    critic_stable.append(name)
                if empirical_count >= required:
                    empirical_stable.append(name)
    supported = sorted(set(critic_stable) & set(empirical_stable))
    critic_speeds = sorted({int(pair_reports[name]["speed_kph"]) for name in critic_stable})
    supported_speeds = sorted({int(pair_reports[name]["speed_kph"]) for name in supported})
    pcgrad_gate = (
        len(supported) >= int(config["decision_gate"]["pcgrad_supported_pair_minimum"])
        and len(supported_speeds) >= int(config["decision_gate"]["pcgrad_supported_speed_minimum"])
    )
    critic_only_gate = (
        len(critic_stable) >= int(config["decision_gate"]["critic_only_pair_minimum"])
        and len(critic_speeds) >= int(config["decision_gate"]["critic_only_speed_minimum"])
    )
    if pcgrad_gate:
        decision = "ADVANCE_STRATUM_PCGRAD_CAGRAD_PILOT"
    elif critic_only_gate:
        decision = "CRITIC_ONLY_STRATUM_CONFLICT_DIAGNOSE_LOCALITY"
    else:
        decision = "NO_SYSTEMATIC_ROAD_STRATUM_CONFLICT_DO_NOT_PCGRAD"
    return {
        "decision": decision,
        "critic_stable_pairs": critic_stable,
        "critic_stable_pair_count": len(critic_stable),
        "critic_stable_speeds": critic_speeds,
        "empirical_stable_pairs": empirical_stable,
        "empirical_stable_pair_count": len(empirical_stable),
        "supported_pairs": supported,
        "supported_pair_count": len(supported),
        "supported_speeds": supported_speeds,
        "pair_reports": pair_reports,
        "checks": {
            "pcgrad_supported_pair_and_speed_gate": pcgrad_gate,
            "critic_only_pair_and_speed_gate": critic_only_gate,
        },
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"] or config["query_analytic_gradient_consumed"]:
        raise AssertionError("sealed boundary violation")
    if int(config["new_query_rollouts"]) != 0:
        raise AssertionError("this is a zero-rollout diagnostic")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    geometry_root = Path(config["sources"]["geometry"])
    geometry_manifest_path = geometry_root / "manifest.json"
    geometry_summary_path = geometry_root / "summary.json"
    geometry_validation_path = geometry_root / "validation.json"
    expected_hashes = {
        geometry_manifest_path: config["sources"]["geometry_manifest_sha256"],
        geometry_summary_path: config["sources"]["geometry_summary_sha256"],
        geometry_validation_path: config["sources"]["geometry_validation_sha256"],
    }
    if any(sha256(path) != expected for path, expected in expected_hashes.items()):
        raise AssertionError("source geometry hash mismatch")
    if json.loads(geometry_validation_path.read_text())["qualification"] != config["sources"]["geometry_qualification"]:
        raise AssertionError("source geometry qualification changed")
    geometry_manifest = json.loads(geometry_manifest_path.read_text())
    geometry_summary = json.loads(geometry_summary_path.read_text())
    geometry_config = geometry_summary["contract"]
    coarse_summary = json.loads(Path(geometry_manifest["source_summary"]).read_text())
    loader = {"outputs": {"absolute_replay": geometry_config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    road_names = [value["name"] for value in collection_manifest["collection"]["base_road_variants"]]
    if road_names != config["population"]["variant_names"]:
        raise AssertionError("road variant semantics changed")
    device = torch.device(args.device)
    output.mkdir(parents=True)
    records, array_hashes = [], {}
    for position, seed in enumerate(config["population"]["seeds"]):
        source_record = geometry_summary["records"][position]
        if int(source_record["seed"]) != int(seed):
            raise AssertionError("source seed order changed")
        geometry_record = geometry_summary["records"][position]
        report, arrays = compute_seed(
            int(seed), source_record, geometry_record, data, geometry_config, config, device
        )
        arrays_path = output / f"seed_{seed}.npz"
        np.savez_compressed(arrays_path, **arrays)
        report["output_arrays"] = str(arrays_path)
        report["output_arrays_sha256"] = sha256(arrays_path)
        array_hashes[str(seed)] = report["output_arrays_sha256"]
        records.append(report)
        print(
            f"seed={seed} critic_neg={report['critic']['pair_negative_fraction_all']:.3f} "
            f"empirical_neg={report['empirical_secant']['pair_negative_fraction_all']:.3f} "
            f"alignment_positive={report['critic_vs_empirical_positive_fraction']:.3f}",
            flush=True,
        )
        del arrays
    routing = route(records, config)
    summary = {
        "qualification": "QUERY_ACTOR_STRATUM_GEOMETRY_COMPLETE_PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config,
        "records": records, "routing": routing, "decision": routing["decision"],
        "outer_fold_evaluated": False, "inner_newly_evaluated": False,
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False, "new_query_rollouts": 0,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-actor-stratum-geometry-v1", "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "geometry_manifest": str(geometry_manifest_path), "geometry_manifest_sha256": sha256(geometry_manifest_path),
        "geometry_summary": str(geometry_summary_path), "geometry_summary_sha256": sha256(geometry_summary_path),
        "geometry_validation": str(geometry_validation_path), "geometry_validation_sha256": sha256(geometry_validation_path),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "seed_arrays_sha256": array_hashes, "summary_sha256": sha256(output / "summary.json"),
        "outer_fold_evaluated": False, "inner_newly_evaluated": False,
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False, "new_query_rollouts": 0,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps(routing, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
