#!/usr/bin/env python3
"""Derive fixed-size multi-elite labels from a validated T1 DBM sidecar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from generate_dbm_proposal_teacher import repository_state, sha256_file, write_json


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-multi-elite-label-v1"
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_expansion_20260804_v3"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_expansion_20260804_v1"
)
DEFAULT_CONFIG = Path(__file__).with_name("dbm_multi_elite_config_20260804_v1.json")
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_multi_elite_expansion_20260804_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if int(config.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError("multi-elite config format_version must be 1")
    if int(config.get("max_elites", 0)) < 2:
        raise ValueError("max_elites must be at least two")
    if int(config.get("minimum_seed_wins_vs_warm", 0)) < 1:
        raise ValueError("minimum_seed_wins_vs_warm must be positive")
    if float(config.get("minimum_standardized_center_distance", 0.0)) <= 0:
        raise ValueError("minimum_standardized_center_distance must be positive")
    if float(config.get("maximum_weighted_output_cost_gap_from_teacher", -1.0)) < 0:
        raise ValueError("maximum cost gap must be non-negative")
    if config.get("ordering") != "teacher_first_then_selection_score":
        raise ValueError("unsupported elite ordering")
    if config.get("padding") != "warm_center_with_invalid_mask":
        raise ValueError("unsupported padding convention")
    return config


def standardized_distance(centers: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    delta = (centers[:, None] - centers[None, :]) / sigma.reshape(1, 1, 1, 2)
    return np.sqrt(np.mean(np.square(delta), axis=(2, 3)))


def select_elite_indices(
    label: np.lib.npyio.NpzFile,
    sigma: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(label["shortlist_centers"], dtype=np.float64)
    weighted_cost = np.asarray(
        label["proposal_weighted_output_cost"], dtype=np.float64
    )
    selection_score = np.asarray(label["selection_score"], dtype=np.float64)
    warm_index = int(label["warm_shortlist_index"])
    teacher_index = int(label["teacher_shortlist_index"])
    cost_mean = weighted_cost.mean(axis=1)
    cost_std = weighted_cost.std(axis=1)
    seed_wins = np.sum(
        weighted_cost < weighted_cost[warm_index][None] - float(
            config["weighted_output_cost_epsilon"]
        ),
        axis=1,
    )
    distance = standardized_distance(centers, sigma)
    selected = [teacher_index]
    for index in np.argsort(selection_score):
        index = int(index)
        if index in selected or index == warm_index:
            continue
        if seed_wins[index] < int(config["minimum_seed_wins_vs_warm"]):
            continue
        if cost_mean[index] >= cost_mean[warm_index] - float(
            config["weighted_output_cost_epsilon"]
        ):
            continue
        if cost_mean[index] > cost_mean[teacher_index] + float(
            config["maximum_weighted_output_cost_gap_from_teacher"]
        ):
            continue
        if any(
            distance[index, other]
            < float(config["minimum_standardized_center_distance"])
            for other in selected
        ):
            continue
        selected.append(index)
        if len(selected) == int(config["max_elites"]):
            break
    return selected, cost_mean, cost_std, seed_wins


def process_label(
    source_path: Path,
    t1_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_hash = sha256_file(source_path)
    t1_hash = sha256_file(t1_path)
    with np.load(source_path, allow_pickle=False) as source, np.load(
        t1_path, allow_pickle=False
    ) as label:
        if str(label["source_snapshot_sha256"]) != source_hash:
            raise AssertionError(f"{t1_path}: source snapshot hash mismatch")
        params = json.loads(str(source["mppi_params_json"]))
        sigma = np.asarray(params["noise_sigma"], dtype=np.float32)
        warm = np.asarray(source["sampling_mean_knots"], dtype=np.float32)
        centers = np.asarray(label["shortlist_centers"], dtype=np.float32)
        names = np.asarray(label["shortlist_center_names"]).astype(str)
        selected, cost_mean, cost_std, seed_wins = select_elite_indices(
            label, sigma, config
        )
        max_elites = int(config["max_elites"])
        count = len(selected)
        valid = np.zeros(max_elites, dtype=bool)
        valid[:count] = True
        shortlist_indices = np.full(max_elites, -1, dtype=np.int32)
        shortlist_indices[:count] = selected
        elite_names = np.full(max_elites, "<invalid>", dtype="<U96")
        elite_names[:count] = names[selected]
        elite_centers = np.broadcast_to(warm, (max_elites, 8, 2)).copy()
        elite_centers[:count] = centers[selected]
        elite_delta = elite_centers - warm[None]
        elite_selection_score = np.full(
            max_elites, float(label["selection_score"][int(label["warm_shortlist_index"])]),
            dtype=np.float32,
        )
        elite_selection_score[:count] = np.asarray(label["selection_score"])[selected]
        elite_cost_mean = np.full(
            max_elites,
            cost_mean[int(label["warm_shortlist_index"])],
            dtype=np.float32,
        )
        elite_cost_mean[:count] = cost_mean[selected]
        elite_cost_std = np.full(
            max_elites,
            cost_std[int(label["warm_shortlist_index"])],
            dtype=np.float32,
        )
        elite_cost_std[:count] = cost_std[selected]
        elite_seed_wins = np.zeros(max_elites, dtype=np.int32)
        elite_seed_wins[:count] = seed_wins[selected]
        elite_p10_mean = np.full(
            max_elites,
            float(np.asarray(label["proposal_p10_cost"])[int(label["warm_shortlist_index"])].mean()),
            dtype=np.float32,
        )
        elite_p10_mean[:count] = np.asarray(label["proposal_p10_cost"])[selected].mean(
            axis=1
        )
        elite_softmin_mean = np.full(
            max_elites,
            float(np.asarray(label["proposal_softmin_cost"])[int(label["warm_shortlist_index"])].mean()),
            dtype=np.float32,
        )
        elite_softmin_mean[:count] = np.asarray(label["proposal_softmin_cost"])[
            selected
        ].mean(axis=1)
        pairwise = np.zeros((max_elites, max_elites), dtype=np.float32)
        pairwise[:count, :count] = standardized_distance(
            elite_centers[:count], sigma
        ).astype(np.float32)
        arrays = {
            "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
            "source_snapshot_sha256": np.asarray(source_hash),
            "source_t1_label_sha256": np.asarray(t1_hash),
            "source_t1_teacher_shortlist_index": np.asarray(
                int(label["teacher_shortlist_index"]), dtype=np.int32
            ),
            "noise_sigma": sigma,
            "elite_count": np.asarray(count, dtype=np.int32),
            "elite_valid_mask": valid,
            "elite_shortlist_indices": shortlist_indices,
            "elite_center_names": elite_names,
            "elite_centers": elite_centers.astype(np.float32),
            "elite_delta_knots": elite_delta.astype(np.float32),
            "elite_selection_score": elite_selection_score,
            "elite_weighted_output_cost_mean": elite_cost_mean,
            "elite_weighted_output_cost_std": elite_cost_std,
            "elite_p10_cost_mean": elite_p10_mean,
            "elite_softmin_cost_mean": elite_softmin_mean,
            "elite_seed_wins_vs_warm": elite_seed_wins,
            "elite_pairwise_standardized_distance": pairwise,
            "warm_selection_score": np.asarray(
                label["selection_score"][int(label["warm_shortlist_index"])],
                dtype=np.float32,
            ),
            "warm_weighted_output_cost_mean": np.asarray(
                cost_mean[int(label["warm_shortlist_index"])], dtype=np.float32
            ),
        }
        metadata = {
            "elite_count": count,
            "teacher_source": str(elite_names[0]),
            "elite_sources": elite_names[:count].tolist(),
            "minimum_pairwise_standardized_distance": (
                float(np.min(pairwise[:count, :count][np.triu_indices(count, 1)]))
                if count > 1
                else None
            ),
            "teacher_weighted_output_cost_mean": float(elite_cost_mean[0]),
            "warm_weighted_output_cost_mean": float(
                arrays["warm_weighted_output_cost_mean"]
            ),
        }
        return arrays, metadata


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    t1_root = args.t1_labels.resolve()
    config = load_config(args.config)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    t1_manifest = json.loads((t1_root / "manifest.json").read_text())
    if Path(t1_manifest["source_collection"]).resolve() != source_root:
        raise ValueError("T1 source collection differs from --source")
    splits = json.loads((t1_root / "splits.json").read_text())
    split_by_episode = {
        episode: split
        for split, episodes in splits.items()
        if split in ("train", "validation", "test")
        for episode in episodes
    }
    label_paths = sorted(t1_root.glob("episode_*/*.npz"))
    if not label_paths:
        raise ValueError("no T1 labels found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    )
    records: list[dict[str, Any]] = []
    fingerprint_lines: list[str] = []
    try:
        shutil.copy2(args.config, temporary / "multi_elite_config.json")
        shutil.copy2(t1_root / "splits.json", temporary / "splits.json")
        for index, t1_path in enumerate(label_paths, start=1):
            episode_id = t1_path.parent.name
            source_path = source_root / episode_id / "snapshots" / t1_path.name
            arrays, metadata = process_label(source_path, t1_path, config)
            output_dir = temporary / episode_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / t1_path.name
            np.savez_compressed(output_path, **arrays)
            output_hash = sha256_file(output_path)
            source_rel = str(source_path.relative_to(source_root))
            t1_rel = str(t1_path.relative_to(t1_root))
            label_rel = str(output_path.relative_to(temporary))
            record = {
                "episode_id": episode_id,
                "split": split_by_episode[episode_id],
                "control_step": int(t1_path.stem.split("_")[-1]),
                "source_relative_path": source_rel,
                "t1_relative_path": t1_rel,
                "label_relative_path": label_rel,
                "source_snapshot_sha256": str(arrays["source_snapshot_sha256"]),
                "source_t1_label_sha256": str(arrays["source_t1_label_sha256"]),
                "label_sha256": output_hash,
                **metadata,
            }
            records.append(record)
            fingerprint_lines.append(
                f"{source_rel}\t{record['source_snapshot_sha256']}\t"
                f"{t1_rel}\t{record['source_t1_label_sha256']}\t"
                f"{label_rel}\t{output_hash}"
            )
            if index % 100 == 0:
                print(f"[{index:03d}/{len(label_paths):03d}] labeled")
        counts = np.asarray([record["elite_count"] for record in records])
        by_split = {}
        for split in ("train", "validation", "test"):
            values = np.asarray(
                [record["elite_count"] for record in records if record["split"] == split]
            )
            by_split[split] = {
                "snapshot_count": int(len(values)),
                "elite_count_mean": float(values.mean()),
                "snapshots_with_at_least_two_elites": int(np.sum(values >= 2)),
                "snapshots_with_four_elites": int(np.sum(values == 4)),
            }
        summary = {
            "format_version": FORMAT_VERSION,
            "snapshot_count": len(records),
            "elite_count_mean": float(counts.mean()),
            "elite_count_distribution": {
                str(value): int(np.sum(counts == value)) for value in np.unique(counts)
            },
            "snapshots_with_at_least_two_elites": int(np.sum(counts >= 2)),
            "snapshots_with_at_least_three_elites": int(np.sum(counts >= 3)),
            "snapshots_with_four_elites": int(np.sum(counts == 4)),
            "by_split": by_split,
        }
        manifest = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_collection": str(source_root),
            "source_t1_labels": str(t1_root),
            "source_t1_manifest_sha256": sha256_file(t1_root / "manifest.json"),
            "embedded_config_sha256": sha256_file(
                temporary / "multi_elite_config.json"
            ),
            "source_index_fingerprint_sha256": hashlib.sha256(
                "\n".join(fingerprint_lines).encode()
            ).hexdigest(),
            "repository": repository_state(Path(__file__).resolve().parents[2]),
            "source_index": records,
            "semantics": {
                "elite_zero": "The original T1 teacher center is always elite 0.",
                "additional_elites": (
                    "Additional centers beat warm on the configured seed count, remain "
                    "within the configured teacher cost gap, and satisfy greedy diversity."
                ),
                "invalid_padding": (
                    "Invalid slots copy warm center and must be ignored using elite_valid_mask."
                ),
            },
        }
        write_json(temporary / "manifest.json", manifest)
        write_json(temporary / "summary.json", summary)
        with (temporary / "labels.csv").open("w", newline="") as stream:
            fieldnames = [
                "episode_id",
                "split",
                "control_step",
                "elite_count",
                "teacher_source",
                "minimum_pairwise_standardized_distance",
                "teacher_weighted_output_cost_mean",
                "warm_weighted_output_cost_mean",
                "label_relative_path",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({name: record[name] for name in fieldnames})
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": "ok", "output": str(args.output), **summary}, indent=2))


if __name__ == "__main__":
    main()
