#!/usr/bin/env python3
"""Generate immutable T0 DBM proposal-teacher labels from saved MPPI candidates.

T0 performs no new dynamics rollout.  It recomputes candidate costs and MPPI
weights from the cost-independent features stored in the schema-v2 closed-loop
snapshots, then emits both the best sampled knots and a soft weighted center.
The latter is a proposal proxy: its own trajectory cost is intentionally not
claimed until a later DBM rollout stage evaluates that new center.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-proposal-teacher-t0-v1"
REQUIRED_COST_KEYS = {
    "position",
    "yaw",
    "vx",
    "yawrate",
    "acceleration_rate",
    "steering_rate",
}
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_train_seed_20260802_v2"
)
DEFAULT_CONFIG = Path(__file__).with_name(
    "dbm_teacher_cost_configs_20260803_v1.json"
)
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t0_20260803_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cost-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=1,
        help="Number of final sorted episodes assigned to validation.",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=1,
        help="Number of final sorted episodes assigned to test after validation.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def repository_state(root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = git("status", "--short").splitlines()
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(status),
            "dirty_files": status,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None, "dirty_files": []}


def load_cost_configs(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if payload.get("format_version") != 1:
        raise ValueError("cost config format_version must be 1")
    configs = payload.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("cost config must contain a non-empty configs list")
    ids: set[str] = set()
    for config in configs:
        config_id = config.get("id")
        if not isinstance(config_id, str) or not re.fullmatch(r"[a-z0-9_.-]+", config_id):
            raise ValueError(f"invalid cost config id: {config_id!r}")
        if config_id in ids:
            raise ValueError(f"duplicate cost config id: {config_id}")
        ids.add(config_id)
        temperature = float(config.get("temperature", 0.0))
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"{config_id}: temperature must be finite and positive")
        weights = config.get("cost_weights", {})
        if set(weights) != REQUIRED_COST_KEYS:
            raise ValueError(
                f"{config_id}: cost_weights keys must be {sorted(REQUIRED_COST_KEYS)}"
            )
        if any(not np.isfinite(float(value)) for value in weights.values()):
            raise ValueError(f"{config_id}: cost weights must be finite")
        if float(weights["yawrate"]) != 0.0:
            raise ValueError(
                f"{config_id}: yawrate must be zero because schema-v2 snapshots "
                "do not store an unweighted yaw-rate error feature"
            )
    return payload, configs


def discover_snapshots(source: Path) -> tuple[list[str], list[dict[str, Any]]]:
    episodes = sorted(path.name for path in source.glob("episode_*") if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episode directories under {source}")
    records: list[dict[str, Any]] = []
    for episode_id in episodes:
        episode_dir = source / episode_id
        manifest_path = episode_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format_version") != 2:
            raise ValueError(f"{manifest_path}: only source format_version 2 is supported")
        if manifest.get("controller_backend") != "dbm":
            raise ValueError(f"{manifest_path}: controller backend is not DBM")
        listed = manifest.get("snapshots", [])
        if not listed:
            raise ValueError(f"{manifest_path}: no snapshots listed")
        for item in listed:
            snapshot = episode_dir / item["snapshot"]
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)
            records.append(
                {
                    "episode_id": episode_id,
                    "control_step": int(item["control_step"]),
                    "path": snapshot,
                    "source_relative_path": str(snapshot.relative_to(source)),
                }
            )
    records.sort(key=lambda item: (item["episode_id"], item["control_step"]))
    return episodes, records


def make_splits(
    episodes: list[str], validation_count: int, test_count: int
) -> dict[str, list[str]]:
    if validation_count < 0 or test_count < 0:
        raise ValueError("split episode counts cannot be negative")
    if validation_count + test_count >= len(episodes):
        raise ValueError("at least one episode must remain in the training split")
    test_start = len(episodes) - test_count
    validation_start = test_start - validation_count
    return {
        "train": episodes[:validation_start],
        "validation": episodes[validation_start:test_start],
        "test": episodes[test_start:] if test_count else [],
    }


def split_for_episode(splits: dict[str, list[str]], episode_id: str) -> str:
    matches = [name for name, members in splits.items() if episode_id in members]
    if len(matches) != 1:
        raise AssertionError(f"{episode_id}: expected exactly one split, got {matches}")
    return matches[0]


def candidate_cost(data: np.lib.npyio.NpzFile, weights: dict[str, float]) -> np.ndarray:
    action_rate = np.asarray(data["feature_action_rate_sq"], dtype=np.float64)
    return (
        float(weights["position"])
        * np.asarray(data["feature_position_error_sq"], dtype=np.float64).sum(axis=1)
        + float(weights["yaw"])
        * np.asarray(data["feature_yaw_error_sq"], dtype=np.float64).sum(axis=1)
        + float(weights["vx"])
        * np.asarray(data["feature_vx_error_sq"], dtype=np.float64).sum(axis=1)
        + float(weights["acceleration_rate"]) * action_rate[..., 0].sum(axis=1)
        + float(weights["steering_rate"]) * action_rate[..., 1].sum(axis=1)
    )


def stable_mppi_weight(cost: np.ndarray, temperature: float) -> np.ndarray:
    unnormalized = np.exp(-(cost - np.min(cost)) / temperature)
    return unnormalized / np.sum(unnormalized)


def configs_match_collection(
    config: dict[str, Any], source_weights: dict[str, Any], source_temperature: float
) -> bool:
    return bool(
        np.isclose(float(config["temperature"]), source_temperature, rtol=0, atol=1e-12)
        and all(
            np.isclose(
                float(config["cost_weights"][name]),
                float(source_weights[name]),
                rtol=0,
                atol=1e-12,
            )
            for name in REQUIRED_COST_KEYS
        )
    )


def process_snapshot(
    record: dict[str, Any],
    configs: list[dict[str, Any]],
    split: str,
    source_hash: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    path = record["path"]
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != 2:
            raise ValueError(f"{path}: only source snapshot format 2 is supported")
        sampled = np.asarray(data["sampled_knots"], dtype=np.float64)
        center = np.asarray(data["sampling_mean_knots"], dtype=np.float64)
        clip_mask = np.asarray(data["sampled_knots_clipped"], dtype=bool)
        params = json.loads(str(data["mppi_params_json"]))
        source_weights = json.loads(str(data["cost_weights_json"]))
        source_temperature = float(params["temperature"])
        sigma = np.asarray(params["noise_sigma"], dtype=np.float64)
        action_min = np.asarray(params["action_min"], dtype=np.float64)
        action_max = np.asarray(params["action_max"], dtype=np.float64)
        count, knots, action_dim = sampled.shape
        if action_dim != 2 or center.shape != (knots, action_dim):
            raise ValueError(f"{path}: unsupported knot shapes")
        if not np.allclose(center, data["mean_knots_before"], rtol=0, atol=1e-7):
            raise AssertionError(f"{path}: sampling center differs from mean_knots_before")
        if not np.allclose(data["sampling_noise_knots"][0], 0.0, rtol=0, atol=1e-7):
            raise AssertionError(f"{path}: candidate zero is not the deterministic warm center")
        if not np.allclose(
            sampled[0], np.clip(center, action_min, action_max), rtol=0, atol=1e-7
        ):
            raise AssertionError(f"{path}: candidate zero does not reproduce the warm center")

        all_cost = []
        all_weight = []
        best_indices = []
        best_knots = []
        best_delta = []
        soft_centers = []
        soft_delta = []
        warm_costs = []
        best_costs = []
        regrets = []
        soft_expected_costs = []
        soft_expected_gaps = []
        ess_values = []
        best_clipped = []
        best_clip_knot_fraction = []
        best_distance = []
        soft_distance = []
        best_boundary_fraction = []
        soft_boundary_fraction = []
        collection_cost_error = []
        collection_weight_error = []
        matches_collection = []
        csv_rows: list[dict[str, Any]] = []

        for config in configs:
            cost = candidate_cost(data, config["cost_weights"])
            weight = stable_mppi_weight(cost, float(config["temperature"]))
            best_index = int(np.argmin(cost))
            selected = sampled[best_index]
            weighted_center = np.sum(weight[:, None, None] * sampled, axis=0)
            selected_delta = selected - center
            weighted_delta = weighted_center - center
            warm_cost = float(cost[0])
            best_cost = float(cost[best_index])
            soft_expected = float(np.sum(weight * cost))
            ess = float(1.0 / np.sum(np.square(weight)))
            exact_collection_match = configs_match_collection(
                config, source_weights, source_temperature
            )
            if exact_collection_match:
                cost_error = float(
                    np.max(np.abs(cost - np.asarray(data["cost"], dtype=np.float64)))
                )
                weight_error = float(
                    np.max(np.abs(weight - np.asarray(data["weight"], dtype=np.float64)))
                )
                if cost_error > 2e-3 or weight_error > 2e-5:
                    raise AssertionError(
                        f"{path}: collection objective reproduction failed "
                        f"(cost={cost_error}, weight={weight_error})"
                    )
            else:
                cost_error = float("nan")
                weight_error = float("nan")

            scale = sigma.reshape(1, action_dim)
            best_std_rms = float(np.sqrt(np.mean(np.square(selected_delta / scale))))
            soft_std_rms = float(np.sqrt(np.mean(np.square(weighted_delta / scale))))
            bound_epsilon = 1e-6
            best_bound = float(
                np.mean(
                    (selected <= action_min + bound_epsilon)
                    | (selected >= action_max - bound_epsilon)
                )
            )
            soft_bound = float(
                np.mean(
                    (weighted_center <= action_min + bound_epsilon)
                    | (weighted_center >= action_max - bound_epsilon)
                )
            )

            all_cost.append(cost)
            all_weight.append(weight)
            best_indices.append(best_index)
            best_knots.append(selected)
            best_delta.append(selected_delta)
            soft_centers.append(weighted_center)
            soft_delta.append(weighted_delta)
            warm_costs.append(warm_cost)
            best_costs.append(best_cost)
            regrets.append(warm_cost - best_cost)
            soft_expected_costs.append(soft_expected)
            soft_expected_gaps.append(soft_expected - best_cost)
            ess_values.append(ess)
            best_clipped.append(bool(np.any(clip_mask[best_index])))
            best_clip_knot_fraction.append(float(np.mean(clip_mask[best_index])))
            best_distance.append(best_std_rms)
            soft_distance.append(soft_std_rms)
            best_boundary_fraction.append(best_bound)
            soft_boundary_fraction.append(soft_bound)
            collection_cost_error.append(cost_error)
            collection_weight_error.append(weight_error)
            matches_collection.append(exact_collection_match)
            csv_rows.append(
                {
                    "episode_id": record["episode_id"],
                    "split": split,
                    "control_step": record["control_step"],
                    "config_id": config["id"],
                    "best_candidate_index": best_index,
                    "warm_candidate_is_best": best_index == 0,
                    "warm_candidate_cost": warm_cost,
                    "best_candidate_cost": best_cost,
                    "warm_best_regret": warm_cost - best_cost,
                    "soft_weighted_candidate_cost": soft_expected,
                    "soft_expected_best_gap": soft_expected - best_cost,
                    "effective_sample_size": ess,
                    "best_candidate_clipped": bool(np.any(clip_mask[best_index])),
                    "best_candidate_clip_knot_fraction": float(
                        np.mean(clip_mask[best_index])
                    ),
                    "all_candidate_clip_knot_fraction": float(np.mean(clip_mask)),
                    "best_delta_standardized_rms": best_std_rms,
                    "soft_delta_standardized_rms": soft_std_rms,
                    "matches_collection_objective": exact_collection_match,
                    "collection_cost_max_abs_error": cost_error,
                    "collection_weight_max_abs_error": weight_error,
                    "source_snapshot_sha256": source_hash,
                }
            )

        arrays = {
            "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
            "config_ids": np.asarray([config["id"] for config in configs]),
            "candidate_cost": np.asarray(all_cost, dtype=np.float32),
            "candidate_weight": np.asarray(all_weight, dtype=np.float32),
            "best_candidate_index": np.asarray(best_indices, dtype=np.int32),
            "best_candidate_knots": np.asarray(best_knots, dtype=np.float32),
            "best_teacher_delta_knots": np.asarray(best_delta, dtype=np.float32),
            "soft_teacher_center_knots": np.asarray(soft_centers, dtype=np.float32),
            "soft_teacher_delta_knots": np.asarray(soft_delta, dtype=np.float32),
            "warm_candidate_cost": np.asarray(warm_costs, dtype=np.float32),
            "best_candidate_cost": np.asarray(best_costs, dtype=np.float32),
            "warm_best_regret": np.asarray(regrets, dtype=np.float32),
            "soft_weighted_candidate_cost": np.asarray(
                soft_expected_costs, dtype=np.float32
            ),
            "soft_expected_best_gap": np.asarray(soft_expected_gaps, dtype=np.float32),
            "effective_sample_size": np.asarray(ess_values, dtype=np.float32),
            "best_candidate_clipped": np.asarray(best_clipped, dtype=bool),
            "best_candidate_clip_knot_fraction": np.asarray(
                best_clip_knot_fraction, dtype=np.float32
            ),
            "all_candidate_clip_knot_fraction": np.asarray(
                [float(np.mean(clip_mask))] * len(configs), dtype=np.float32
            ),
            "best_delta_standardized_rms": np.asarray(best_distance, dtype=np.float32),
            "soft_delta_standardized_rms": np.asarray(soft_distance, dtype=np.float32),
            "best_boundary_fraction": np.asarray(
                best_boundary_fraction, dtype=np.float32
            ),
            "soft_boundary_fraction": np.asarray(
                soft_boundary_fraction, dtype=np.float32
            ),
            "matches_collection_objective": np.asarray(matches_collection, dtype=bool),
            "collection_cost_max_abs_error": np.asarray(
                collection_cost_error, dtype=np.float32
            ),
            "collection_weight_max_abs_error": np.asarray(
                collection_weight_error, dtype=np.float32
            ),
            "sampling_mean_knots": center.astype(np.float32),
            "source_snapshot_sha256": np.asarray(source_hash),
        }
        metadata = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "source_relative_path": record["source_relative_path"],
            "source_snapshot_sha256": source_hash,
            "episode_id": record["episode_id"],
            "split": split,
            "control_step": record["control_step"],
            "candidate_count": count,
            "num_knots": knots,
            "config_ids": [config["id"] for config in configs],
            "primary_supervision_target": "soft_teacher_delta_knots",
            "target_semantics": {
                "best_teacher_delta_knots": "exact delta to the lowest-cost saved candidate",
                "soft_teacher_delta_knots": (
                    "MPPI-weighted saved-candidate center delta; proxy label whose own "
                    "rollout cost is not evaluated in T0"
                ),
                "soft_weighted_candidate_cost": (
                    "weighted expectation of saved candidate costs, not the rollout cost "
                    "of soft_teacher_center_knots"
                ),
            },
        }
        return arrays, metadata, csv_rows


def aggregate_summary(rows: list[dict[str, Any]], configs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for config in configs:
        selected = [row for row in rows if row["config_id"] == config["id"]]

        def values(name: str) -> np.ndarray:
            return np.asarray([row[name] for row in selected], dtype=np.float64)

        regret = values("warm_best_regret")
        ess = values("effective_sample_size")
        soft_gap = values("soft_expected_best_gap")
        summary[config["id"]] = {
            "snapshot_count": len(selected),
            "warm_candidate_best_count": int(
                sum(bool(row["warm_candidate_is_best"]) for row in selected)
            ),
            "warm_candidate_best_fraction": float(
                np.mean([bool(row["warm_candidate_is_best"]) for row in selected])
            ),
            "warm_best_regret": {
                "mean": float(np.mean(regret)),
                "median": float(np.median(regret)),
                "p90": float(np.quantile(regret, 0.9)),
                "maximum": float(np.max(regret)),
            },
            "effective_sample_size": {
                "mean": float(np.mean(ess)),
                "median": float(np.median(ess)),
                "minimum": float(np.min(ess)),
                "maximum": float(np.max(ess)),
            },
            "soft_expected_best_gap": {
                "mean": float(np.mean(soft_gap)),
                "median": float(np.median(soft_gap)),
                "p90": float(np.quantile(soft_gap, 0.9)),
            },
            "best_candidate_clipped_fraction": float(
                np.mean([bool(row["best_candidate_clipped"]) for row in selected])
            ),
            "best_delta_standardized_rms_mean": float(
                np.mean(values("best_delta_standardized_rms"))
            ),
            "soft_delta_standardized_rms_mean": float(
                np.mean(values("soft_delta_standardized_rms"))
            ),
            "collection_cost_max_abs_error": float(
                np.nanmax(values("collection_cost_max_abs_error"))
            )
            if any(row["matches_collection_objective"] for row in selected)
            else None,
            "collection_weight_max_abs_error": float(
                np.nanmax(values("collection_weight_max_abs_error"))
            )
            if any(row["matches_collection_objective"] for row in selected)
            else None,
        }
    return summary


def write_markdown_summary(
    path: Path,
    source: Path,
    splits: dict[str, list[str]],
    config_summary: dict[str, Any],
    candidate_counts: list[int],
) -> None:
    candidate_count_text = ", ".join(str(value) for value in candidate_counts)
    lines = [
        "# DBM proposal teacher T0 summary",
        "",
        f"- Source: `{source}`",
        f"- Generator: `{GENERATOR_ID}`",
        f"- T0 reuses saved DBM rollouts ({candidate_count_text} candidates per snapshot); "
        "no new dynamics rollout was run.",
        "- Primary BC target: `soft_teacher_delta_knots`.",
        "- The soft-center trajectory cost is not evaluated at T0; "
        "`soft_weighted_candidate_cost` is only an expectation over saved candidates.",
        "",
        "## Episode split",
        "",
    ]
    for name, episodes in splits.items():
        lines.append(f"- {name}: {', '.join(episodes) if episodes else '(empty)'}")
    lines.extend(["", "This 6/1/1 split is a pipeline split, not a final statistical claim.", ""])
    for config_id, metrics in config_summary.items():
        lines.extend(
            [
                f"## {config_id}",
                "",
                f"- Snapshots: {metrics['snapshot_count']}",
                "- Warm candidate already best: "
                f"{metrics['warm_candidate_best_count']}/{metrics['snapshot_count']} "
                f"({metrics['warm_candidate_best_fraction']:.3f})",
                "- Warm-to-best regret mean/median/P90: "
                f"{metrics['warm_best_regret']['mean']:.6f} / "
                f"{metrics['warm_best_regret']['median']:.6f} / "
                f"{metrics['warm_best_regret']['p90']:.6f}",
                "- ESS mean/median/range: "
                f"{metrics['effective_sample_size']['mean']:.3f} / "
                f"{metrics['effective_sample_size']['median']:.3f} / "
                f"[{metrics['effective_sample_size']['minimum']:.3f}, "
                f"{metrics['effective_sample_size']['maximum']:.3f}]",
                "- Best candidate clipped fraction: "
                f"{metrics['best_candidate_clipped_fraction']:.3f}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    config_path = args.cost_config.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable teacher sidecar: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    config_payload, configs = load_cost_configs(config_path)
    episodes, records = discover_snapshots(source)
    splits = make_splits(episodes, args.validation_episodes, args.test_episodes)
    repo_root = Path(__file__).resolve().parents[2]
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        write_json(
            staging / "splits.json",
            {
                "format_version": FORMAT_VERSION,
                "policy": "episode-level final-sorted holdout",
                "warning": (
                    f"Episode-level split over {len(episodes)} source episodes; "
                    "preserve this split for all controlled comparisons and do not "
                    "randomly split adjacent snapshots."
                ),
                **splits,
            },
        )
        shutil.copy2(config_path, staging / "cost_configs.json")
        rows: list[dict[str, Any]] = []
        source_index: list[dict[str, Any]] = []
        candidate_count_set: set[int] = set()
        for index, record in enumerate(records, start=1):
            split = split_for_episode(splits, record["episode_id"])
            source_hash = sha256_file(record["path"])
            arrays, metadata, snapshot_rows = process_snapshot(
                record, configs, split, source_hash
            )
            candidate_count_set.add(int(metadata["candidate_count"]))
            episode_output = staging / record["episode_id"]
            episode_output.mkdir(exist_ok=True)
            stem = f"step_{record['control_step']:06d}"
            np.savez_compressed(episode_output / f"{stem}.npz", **arrays)
            write_json(episode_output / f"{stem}.json", metadata)
            rows.extend(snapshot_rows)
            source_index.append(
                {
                    "source_relative_path": record["source_relative_path"],
                    "source_snapshot_sha256": source_hash,
                    "label_relative_path": f"{record['episode_id']}/{stem}.npz",
                }
            )
            print(f"[{index:03d}/{len(records):03d}] {record['source_relative_path']}")

        fieldnames = list(rows[0])
        with (staging / "labels.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        config_summary = aggregate_summary(rows, configs)
        candidate_counts = sorted(candidate_count_set)
        write_json(
            staging / "summary.json",
            {
                "format_version": FORMAT_VERSION,
                "generator_id": GENERATOR_ID,
                "source_collection": str(source),
                "snapshot_count": len(records),
                "label_row_count": len(rows),
                "candidate_counts_per_snapshot": candidate_counts,
                "config_summary": config_summary,
            },
        )
        write_markdown_summary(
            staging / "COLLECTION_SUMMARY.md",
            source,
            splits,
            config_summary,
            candidate_counts,
        )
        source_fingerprint = hashlib.sha256(
            "\n".join(
                f"{item['source_relative_path']} {item['source_snapshot_sha256']}"
                for item in source_index
            ).encode()
        ).hexdigest()
        write_json(
            staging / "manifest.json",
            {
                "format_version": FORMAT_VERSION,
                "dataset_type": "anycar-dbm-proposal-teacher-sidecar",
                "generator_id": GENERATOR_ID,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_collection": str(source),
                "source_snapshot_count": len(records),
                "source_collection_fingerprint_sha256": source_fingerprint,
                "source_index": source_index,
                "cost_config_source": str(config_path),
                "cost_config_sha256": sha256_file(config_path),
                "embedded_cost_config_sha256": sha256_file(
                    staging / "cost_configs.json"
                ),
                "cost_config": config_payload,
                "repository": repository_state(repo_root),
                "splits": splits,
                "primary_supervision_target": "soft_teacher_delta_knots",
                "alternative_supervision_target": "best_teacher_delta_knots",
                "limitations": [
                    "T0 searches only the "
                    + ", ".join(str(value) for value in candidate_counts)
                    + " candidates already stored per snapshot.",
                    "The soft weighted center is not rolled out; its exact DBM cost is unavailable at T0.",
                    "The episode split is suitable for pipeline development, not a final generalization claim.",
                ],
            },
        )
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "ok", "output": str(output), "snapshots": len(records)}))


if __name__ == "__main__":
    main()
