#!/usr/bin/env python3
"""Build a consolidated pure-Query replay and non-regressing T0 teacher.

The source collection is immutable.  This script consolidates its saved Query
rollouts, freshly evaluates the exact control-call entry warm center, and then
chooses a direct-cost teacher from warm, the lowest-cost saved Gaussian
candidate, and the MPPI weighted output.  No DBM value, gradient, trajectory,
or label is read, and no formal validation/test split is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import (  # noqa: E402
    TorchMPPIController,
    TorchMPPIParams,
)
from car_foundation.query_deployment import (  # noqa: E402
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "query_expected_road_train_20260901_v1"
)
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "query_expected_road_t0_20260901_v1"
)
TEACHER_SOURCE_NAMES = np.asarray(
    ("warm", "best_sampled", "weighted_output"), dtype="<U32"
)
KNOT_INDICES = np.asarray((0, 7, 14, 21, 28, 35, 42, 49), dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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


def grouped_metrics(replay: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    result = {
        "rows": int(mask.sum()),
        "warm_direct_cost": stats(replay["warm_direct_cost"][mask]),
        "best_sampled_direct_cost": stats(
            replay["best_sampled_direct_cost"][mask]
        ),
        "weighted_output_direct_cost": stats(
            replay["weighted_output_direct_cost"][mask]
        ),
        "teacher_direct_cost": stats(replay["teacher_direct_cost"][mask]),
        "teacher_gain_vs_warm": stats(replay["teacher_gain_vs_warm"][mask]),
        "best_sampled_gain_vs_warm": stats(
            replay["best_sampled_gain_vs_warm"][mask]
        ),
        "weighted_output_gain_vs_warm": stats(
            replay["weighted_output_gain_vs_warm"][mask]
        ),
        "effective_sample_size": stats(replay["effective_sample_size"][mask]),
        "candidate_clip_fraction": stats(
            replay["candidate_clip_fraction"][mask]
        ),
        "teacher_delta_sigma_rms": stats(
            replay["teacher_delta_sigma_rms"][mask]
        ),
        "teacher_strict_win_fraction": float(
            np.mean(replay["teacher_gain_vs_warm"][mask] > 0.0)
        ),
        "teacher_warm_tie_fraction": float(
            np.mean(replay["teacher_gain_vs_warm"][mask] == 0.0)
        ),
    }
    source = replay["teacher_source_index"][mask]
    result["teacher_source_counts"] = {
        str(name): int(np.sum(source == index))
        for index, name in enumerate(TEACHER_SOURCE_NAMES.tolist())
    }
    return result


def interpolate_knots(knots: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(knots, dtype=torch.float32)
    flat = tensor.reshape(-1, 8, 2)
    sequence = F.interpolate(
        flat.transpose(1, 2), size=50, mode="linear", align_corners=True
    ).transpose(1, 2)
    return sequence.reshape(*tensor.shape[:-2], 50, 2).numpy()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")

    source_manifest_path = source / "manifest.json"
    source_validation_path = source / "validation.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_validation = json.loads(source_validation_path.read_text())
    if source_manifest["dataset_type"] != "anycar-query-expected-road-closed-loop":
        raise ValueError("source is not a Query expected-road closed-loop collection")
    if source_validation["qualification"] not in {
        "QUERY_EXPECTED_ROAD_CLOSED_LOOP_PASS",
        "QUERY_EXPECTED_ROAD_TARGET_COVERAGE_PASS",
    }:
        raise AssertionError("source collection did not pass independent validation")
    if source_manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("source must keep formal validation/test sealed")
    if source_validation.get("formal_validation_or_test_consumed", True):
        raise AssertionError("source validation reports sealed data were consumed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    episode_chunks: list[dict[str, np.ndarray]] = []
    row_records: list[dict] = []
    source_snapshot_hashes: dict[str, str] = {}
    source_keys: tuple[str, ...] | None = None
    for episode in source_manifest["episodes"]:
        episode_id = str(episode["episode_id"])
        episode_dir = source / episode_id
        episode_summary = json.loads((episode_dir / "summary.json").read_text())
        snapshot_path = episode_dir / "snapshots.npz"
        snapshot_hash = sha256(snapshot_path)
        expected_hash = episode_summary["artifacts"]["snapshots.npz"]["sha256"]
        if snapshot_hash != expected_hash:
            raise AssertionError(f"source snapshot hash mismatch: {episode_id}")
        source_snapshot_hashes[episode_id] = snapshot_hash
        with np.load(snapshot_path, allow_pickle=False) as archive:
            if source_keys is None:
                source_keys = tuple(archive.files)
            elif tuple(archive.files) != source_keys:
                raise AssertionError(f"source key mismatch: {episode_id}")
            chunk = {name: np.asarray(archive[name]) for name in archive.files}
        episode_chunks.append(chunk)

        speed_index = list(source_manifest["collection"]["speed_bins_kph"]).index(
            int(episode["speed_kph"])
        )
        variant_index = int(episode["variant_index"])
        repeat_index = episode.get("repeat_index")
        explicit_fold_id = episode.get("fold_id")
        if explicit_fold_id is not None:
            fold_id = int(explicit_fold_id)
            split_strategy = "targeted-train-only-fit-folds-v1"
        elif repeat_index is None:
            fold_id = (speed_index + variant_index) % 5
            split_strategy = "episode-grouped-5fold-v1"
        else:
            fold_id = int(repeat_index)
            split_strategy = "cell-repeat-grouped-5fold-v2"
        road_name = str(episode_summary["road"]["name"])
        for row_in_episode, control_step in enumerate(chunk["control_step"]):
            row_records.append(
                {
                    "row_index": len(row_records),
                    "episode_id": episode_id,
                    "episode_index": int(episode["episode_index"]),
                    "row_in_episode": row_in_episode,
                    "control_step": int(control_step),
                    "speed_kph": int(episode["speed_kph"]),
                    "speed_index": speed_index,
                    "variant_index": variant_index,
                    "repeat_index": (
                        -1 if repeat_index is None else int(repeat_index)
                    ),
                    "road_name": road_name,
                    "fold_id": fold_id,
                    "seed_row": int(episode["seed_row"]),
                    "seed_source": str(episode_summary["seed"]["source"]),
                    "seed_source_file": str(episode_summary["seed"]["source_file"]),
                    "seed_source_window_index": int(
                        episode_summary["seed"]["source_window_index"]
                    ),
                    "source_snapshot_sha256": snapshot_hash,
                }
            )

    assert source_keys is not None
    replay: dict[str, np.ndarray] = {
        name: np.concatenate([chunk[name] for chunk in episode_chunks], axis=0)
        for name in source_keys
    }
    for key in row_records[0]:
        values = [row[key] for row in row_records]
        if isinstance(values[0], str):
            replay[key] = np.asarray(values, dtype=str)
        else:
            replay[key] = np.asarray(values)

    row_count = len(row_records)
    expected_row_count = int(
        json.loads((source / "summary.json").read_text())["snapshot_count"]
    )
    if row_count != expected_row_count:
        raise AssertionError(
            f"expected {expected_row_count} source snapshots, found {row_count}"
        )
    if replay["cost"].shape != (row_count, 256):
        raise AssertionError("expected 256 saved candidates per row")

    cost = replay["cost"].astype(np.float64)
    best_index = np.argmin(cost, axis=1).astype(np.int64)
    row_index = np.arange(row_count)
    warm_action = interpolate_knots(replay["mean_knots_before"])
    warm_trajectory = np.empty((row_count, 50, 5), dtype=np.float32)
    warm_cost = np.empty(row_count, dtype=np.float64)
    params = TorchMPPIParams(**source_manifest["collection"]["mppi"])
    model = QueryDeploymentModel.from_checkpoint(
        Path(source_manifest["query_checkpoint"]), args.device
    )
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(model), params, device=args.device
    )
    for index in range(row_count):
        result = controller.evaluate_action_sequences(
            replay["state"][index],
            replay["current_action"][index],
            replay["history"][index : index + 1],
            replay["reference"][index],
            warm_action[index : index + 1],
        )
        warm_trajectory[index] = result["trajectories"][0].cpu().numpy()
        warm_cost[index] = float(result["cost"][0].cpu())
    best_cost = cost[row_index, best_index]
    weighted_cost = replay["optimized_cost"].astype(np.float64)
    weighted_knots = replay["optimized_action_sequence"][:, KNOT_INDICES]
    center_cost = np.stack((warm_cost, best_cost, weighted_cost), axis=1)
    teacher_source_index = np.argmin(center_cost, axis=1).astype(np.int8)

    center_knots = np.stack(
        (
            replay["mean_knots_before"],
            replay["sampled_knots"][row_index, best_index],
            weighted_knots,
        ),
        axis=1,
    )
    center_action = np.stack(
        (
            warm_action,
            replay["sampled_action_sequences"][row_index, best_index],
            replay["optimized_action_sequence"],
        ),
        axis=1,
    )
    center_trajectory = np.stack(
        (
            warm_trajectory,
            replay["predicted_trajectories"][row_index, best_index],
            replay["optimized_trajectory"],
        ),
        axis=1,
    )
    teacher_knots = center_knots[row_index, teacher_source_index]
    teacher_action = center_action[row_index, teacher_source_index]
    teacher_trajectory = center_trajectory[row_index, teacher_source_index]
    teacher_cost = center_cost[row_index, teacher_source_index]
    teacher_delta = teacher_knots - replay["mean_knots_before"]
    sigma = np.asarray(source_manifest["collection"]["mppi"]["noise_sigma"])

    replay.update(
        {
            "knot_indices": np.broadcast_to(KNOT_INDICES, (row_count, 8)).copy(),
            "best_candidate_index": best_index,
            "warm_action_sequence": warm_action.astype(np.float32),
            "warm_trajectory": warm_trajectory,
            "last_iteration_candidate_zero_direct_cost": cost[:, 0],
            "candidate_center_knots": center_knots.astype(np.float32),
            "candidate_center_action_sequences": center_action.astype(np.float32),
            "candidate_center_trajectories": center_trajectory.astype(np.float32),
            "candidate_center_direct_cost": center_cost,
            "warm_direct_cost": warm_cost,
            "best_sampled_direct_cost": best_cost,
            "weighted_output_direct_cost": weighted_cost,
            "best_sampled_gain_vs_warm": warm_cost - best_cost,
            "weighted_output_gain_vs_warm": warm_cost - weighted_cost,
            "teacher_source_index": teacher_source_index,
            "teacher_source_name": TEACHER_SOURCE_NAMES[teacher_source_index],
            "teacher_knots": teacher_knots.astype(np.float32),
            "teacher_action_sequence": teacher_action.astype(np.float32),
            "teacher_trajectory": teacher_trajectory.astype(np.float32),
            "teacher_direct_cost": teacher_cost,
            "teacher_delta_knots": teacher_delta.astype(np.float32),
            "teacher_gain_vs_warm": warm_cost - teacher_cost,
            "teacher_delta_sigma_rms": np.sqrt(
                np.mean(np.square(teacher_delta / sigma), axis=(1, 2))
            ),
            "effective_sample_size": 1.0
            / np.sum(np.square(replay["weight"].astype(np.float64)), axis=1),
            "candidate_clip_fraction": np.mean(
                np.abs(replay["raw_sampled_knots"] - replay["sampled_knots"])
                > 1e-7,
                axis=(1, 2, 3),
            ),
        }
    )

    if np.any(replay["teacher_gain_vs_warm"] < -1e-12):
        raise AssertionError("teacher regresses relative to warm")
    present_folds = sorted(np.unique(replay["fold_id"]).tolist())
    if split_strategy == "targeted-train-only-fit-folds-v1":
        if present_folds != [2, 3, 4]:
            raise AssertionError(
                f"targeted train-only folds must be [2,3,4], got {present_folds}"
            )
        expected_counts = {
            int(fold): sum(int(value) for value in cells.values())
            for fold, cells in source_manifest["fold_cell_counts"].items()
        }
        for fold_id in present_folds:
            episodes = np.unique(replay["episode_id"][replay["fold_id"] == fold_id])
            if len(episodes) != expected_counts[fold_id]:
                raise AssertionError(
                    f"fold {fold_id} has {len(episodes)} episodes, expected {expected_counts[fold_id]}"
                )
    else:
        expected_episodes_per_fold = len(source_manifest["episodes"]) // 5
        for fold_id in range(5):
            episodes = np.unique(replay["episode_id"][replay["fold_id"] == fold_id])
            if len(episodes) != expected_episodes_per_fold:
                raise AssertionError(
                    f"fold {fold_id} does not contain {expected_episodes_per_fold} episodes"
                )

    fold_records = []
    for fold_id in present_folds:
        mask = replay["fold_id"] == fold_id
        fold_records.append(
            {
                "fold_id": fold_id,
                "role": "out_of_fold_holdout",
                "episode_ids": sorted(np.unique(replay["episode_id"][mask]).tolist()),
                "row_count": int(mask.sum()),
                "speed_kph": sorted(np.unique(replay["speed_kph"][mask]).tolist()),
                "variant_indices": sorted(
                    np.unique(replay["variant_index"][mask]).tolist()
                ),
            }
        )
    if split_strategy == "targeted-train-only-fit-folds-v1":
        split_formula = "fold_id = explicit episode fold_id in {2,3,4}"
        split_note = (
            "Targeted coverage episodes are train-only additions. They augment fit "
            "folds 2/3/4 and never enter inner fold 1 or outer fold 0."
        )
    elif split_strategy == "cell-repeat-grouped-5fold-v2":
        split_formula = "fold_id = repeat_index"
        split_note = (
            "Every holdout fold contains one independent episode from every selected "
            "speed-road cell; fit, selection, and OOF therefore all cover every cell."
        )
    else:
        split_formula = "fold_id = (speed_index + variant_index) % 5"
        split_note = (
            "Each holdout fold contains four whole episodes, one per road variant; "
            "because there are four episodes per fold and five speeds, each individual "
            "fold omits one speed while the five OOF folds are jointly balanced."
        )
    splits = {
        "strategy": split_strategy,
        "formula": split_formula,
        "note": split_note,
        "folds": fold_records,
        "formal_validation_or_test_consumed": False,
        "query_analytic_gradient_consumed": False,
    }

    all_mask = np.ones(row_count, dtype=bool)
    summary = {
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "row_count": row_count,
        "episode_count": len(source_manifest["episodes"]),
        "candidate_count_per_row": int(replay["cost"].shape[1]),
        "overall": grouped_metrics(replay, all_mask),
        "by_speed_kph": {
            str(speed): grouped_metrics(replay, replay["speed_kph"] == speed)
            for speed in sorted(np.unique(replay["speed_kph"]))
        },
        "by_variant_index": {
            str(variant): grouped_metrics(
                replay, replay["variant_index"] == variant
            )
            for variant in sorted(np.unique(replay["variant_index"]))
        },
        "by_fold": {
            str(fold): grouped_metrics(replay, replay["fold_id"] == fold)
            for fold in present_folds
        },
        "formal_validation_or_test_consumed": False,
        "query_analytic_gradient_consumed": False,
    }

    output.mkdir(parents=True)
    replay_path = output / "replay.npz"
    np.savez_compressed(replay_path, **replay)
    dump_json(output / "splits.json", splits)
    dump_json(output / "summary.json", summary)
    with (output / "rows.csv").open("w", newline="") as stream:
        fieldnames = list(row_records[0]) + [
            "best_candidate_index",
            "warm_direct_cost",
            "best_sampled_direct_cost",
            "weighted_output_direct_cost",
            "teacher_source_name",
            "teacher_direct_cost",
            "teacher_gain_vs_warm",
            "teacher_delta_sigma_rms",
            "effective_sample_size",
            "candidate_clip_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(row_records):
            writer.writerow(
                record
                | {
                    name: replay[name][index].item()
                    for name in fieldnames
                    if name not in record
                }
            )

    manifest = {
        "schema_version": "query-expected-road-t0-replay-v1",
        "dataset_type": "pure-query-replay-with-t0-teacher",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_collection": str(source),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_validation_sha256": sha256(source_validation_path),
        "source_snapshot_sha256": source_snapshot_hashes,
        "query_checkpoint": source_manifest["query_checkpoint"],
        "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
        "source_fields_consolidated": list(source_keys),
        "replay_sha256": sha256(replay_path),
        "splits_sha256": sha256(output / "splits.json"),
        "rows_sha256": sha256(output / "rows.csv"),
        "row_count": row_count,
        "episode_count": len(source_manifest["episodes"]),
        "teacher_contract": {
            "candidate_order": TEACHER_SOURCE_NAMES.tolist(),
            "selection": "argmin deterministic Query direct cost; first index wins ties",
            "warm_floor": "warm is candidate 0, so teacher_direct_cost <= warm_direct_cost",
            "warm_definition": (
                "control-call entry mean_knots_before, interpolated to 50 actions and "
                "evaluated by a fresh deterministic frozen-Query rollout"
            ),
            "saved_candidate_zero_semantics": (
                "candidate 0 retains sampling_mean_knots of the last of two MPPI "
                "iterations; it is not the control-call entry warm center"
            ),
            "weighted_knots": "optimized_action_sequence at indices 0,7,14,21,28,35,42,49",
            "primary_supervision_target": "teacher_delta_knots",
            "noise_sigma": sigma.tolist(),
        },
        "split_contract": splits,
        "dbm_fields_or_labels_consumed": [],
        "new_query_rollouts_generated": row_count,
        "formal_validation_or_test_consumed": False,
        "query_analytic_gradient_consumed": False,
        "limitations": [
            "T0 searches only warm, the 256 saved Gaussian candidates, and the saved weighted output.",
            "This is a train-only OOF sidecar, not a formal generalization or deployment result.",
            "The 40-kph collection contains documented left-turn substitutions for both nominal-right and recovery-right slots.",
        ],
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
