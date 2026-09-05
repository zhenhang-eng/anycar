#!/usr/bin/env python3
"""Independently validate the pure-Query Replay and T0 teacher sidecar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from car_foundation.query_deployment import (  # noqa: E402
    OnnxQueryRolloutBackend,
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)


DEFAULT_SIDECAR = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "query_expected_road_t0_20260901_v1"
)
DEFAULT_ONNX = REPO_ROOT / "outputs/query_mppi/anycar_query.onnx"
TEACHER_SOURCE_NAMES = np.asarray(
    ("warm", "best_sampled", "weighted_output"), dtype="<U32"
)
KNOT_INDICES = np.asarray((0, 7, 14, 21, 28, 35, 42, 49), dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path, nargs="?", default=DEFAULT_SIDECAR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--query-replay-rows", type=int, default=20)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--onnx-replay-rows", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum(values: list[float]) -> float:
    return float(max(values, default=0.0))


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


def numeric_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float("inf")
    if actual.dtype.kind in "US" or expected.dtype.kind in "US":
        return 0.0 if np.array_equal(actual, expected) else 1.0
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def interpolate_knots(knots: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(knots, dtype=torch.float32)
    flat = tensor.reshape(-1, 8, 2)
    sequence = F.interpolate(
        flat.transpose(1, 2), size=50, mode="linear", align_corners=True
    ).transpose(1, 2)
    return sequence.reshape(*tensor.shape[:-2], 50, 2).numpy()


def stratified_rows(replay: np.lib.npyio.NpzFile, requested: int) -> np.ndarray:
    if requested <= 0:
        return np.empty(0, dtype=np.int64)
    episodes = np.unique(replay["episode_index"])
    selected = []
    for episode in episodes:
        rows = np.flatnonzero(replay["episode_index"] == episode)
        selected.append(int(rows[len(rows) // 2]))
    if requested < len(selected):
        selected = np.asarray(selected)[
            np.linspace(0, len(selected) - 1, requested, dtype=int)
        ].tolist()
    elif requested > len(selected):
        extra = np.linspace(0, len(replay["state"]) - 1, requested, dtype=int)
        selected = sorted(set(selected).union(extra.tolist()))[:requested]
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    args = parse_args()
    sidecar = args.sidecar.resolve()
    manifest_path = sidecar / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    splits = json.loads((sidecar / "splits.json").read_text())
    source = Path(manifest["source_collection"])
    source_manifest_path = source / "manifest.json"
    source_validation_path = source / "validation.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_validation = json.loads(source_validation_path.read_text())
    replay_path = sidecar / "replay.npz"

    hash_checks = {
        "source_manifest": sha256(source_manifest_path)
        == manifest["source_manifest_sha256"],
        "source_validation": sha256(source_validation_path)
        == manifest["source_validation_sha256"],
        "query_checkpoint": sha256(Path(manifest["query_checkpoint"]))
        == manifest["query_checkpoint_sha256"],
        "replay": sha256(replay_path) == manifest["replay_sha256"],
        "splits": sha256(sidecar / "splits.json") == manifest["splits_sha256"],
        "rows": sha256(sidecar / "rows.csv") == manifest["rows_sha256"],
    }
    if source_validation["qualification"] not in {
        "QUERY_EXPECTED_ROAD_CLOSED_LOOP_PASS",
        "QUERY_EXPECTED_ROAD_TARGET_COVERAGE_PASS",
    }:
        raise AssertionError("source validation no longer passes")
    if manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("sidecar reports consuming formal validation/test")
    if manifest.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("pure Query Replay must not consume DBM fields or labels")

    source_errors: list[float] = []
    metadata_errors: list[float] = []
    source_hash_errors: list[float] = []
    offset = 0
    with np.load(replay_path, allow_pickle=False) as replay:
        row_count = len(replay["state"])
        for episode in source_manifest["episodes"]:
            episode_id = str(episode["episode_id"])
            source_path = source / episode_id / "snapshots.npz"
            source_hash_errors.append(
                0.0
                if sha256(source_path)
                == manifest["source_snapshot_sha256"][episode_id]
                else 1.0
            )
            with np.load(source_path, allow_pickle=False) as source_rows:
                count = len(source_rows["state"])
                target = slice(offset, offset + count)
                for name in manifest["source_fields_consolidated"]:
                    source_errors.append(
                        numeric_error(replay[name][target], source_rows[name])
                    )
                speed_index = list(
                    source_manifest["collection"]["speed_bins_kph"]
                ).index(int(episode["speed_kph"]))
                expected_fold = int(
                    episode.get(
                        "fold_id",
                        episode.get(
                            "repeat_index",
                            (speed_index + int(episode["variant_index"])) % 5,
                        ),
                    )
                )
                expected_metadata = {
                    "episode_id": np.full(count, episode_id),
                    "episode_index": np.full(count, int(episode["episode_index"])),
                    "row_in_episode": np.arange(count),
                    "speed_kph": np.full(count, int(episode["speed_kph"])),
                    "speed_index": np.full(count, speed_index),
                    "variant_index": np.full(count, int(episode["variant_index"])),
                    "fold_id": np.full(count, expected_fold),
                    "seed_row": np.full(count, int(episode["seed_row"])),
                }
                if "repeat_index" in replay.files:
                    expected_metadata["repeat_index"] = np.full(
                        count, int(episode.get("repeat_index", -1))
                    )
                for name, expected in expected_metadata.items():
                    metadata_errors.append(numeric_error(replay[name][target], expected))
                offset += count
        if offset != row_count or row_count != manifest["row_count"]:
            raise AssertionError("row count mismatch")

        fold_errors: list[float] = []
        episode_fold = {}
        for episode in np.unique(replay["episode_id"]):
            folds = np.unique(replay["fold_id"][replay["episode_id"] == episode])
            fold_errors.append(0.0 if len(folds) == 1 else 1.0)
            episode_fold[str(episode)] = int(folds[0])
        targeted_train_only = splits["strategy"] == "targeted-train-only-fit-folds-v1"
        expected_targeted_counts = {
            int(fold): sum(int(value) for value in cells.values())
            for fold, cells in source_manifest.get("fold_cell_counts", {}).items()
        }
        for fold in splits["folds"]:
            fold_id = int(fold["fold_id"])
            mask = replay["fold_id"] == fold_id
            episodes = sorted(np.unique(replay["episode_id"][mask]).tolist())
            fold_errors.extend(
                (
                    0.0 if episodes == fold["episode_ids"] else 1.0,
                    0.0
                    if (
                        targeted_train_only
                        and len(episodes) == expected_targeted_counts.get(fold_id, -1)
                    )
                    or (
                        not targeted_train_only
                        and len(episodes) == manifest["episode_count"] // 5
                    )
                    else 1.0,
                    0.0
                    if targeted_train_only
                    or len(np.unique(replay["variant_index"][mask])) == 4
                    else 1.0,
                    0.0
                    if targeted_train_only
                    or splits["strategy"] != "cell-repeat-grouped-5fold-v2"
                    or len(np.unique(replay["speed_kph"][mask])) == 5
                    else 1.0,
                    0.0 if int(mask.sum()) == int(fold["row_count"]) else 1.0,
                )
            )
        if targeted_train_only:
            fold_errors.extend(
                (
                    0.0 if sorted(episode_fold.values()) else 1.0,
                    0.0 if sorted(set(episode_fold.values())) == [2, 3, 4] else 1.0,
                    0.0 if len(splits["folds"]) == 3 else 1.0,
                )
            )

        row = np.arange(row_count)
        cost = replay["cost"].astype(np.float64)
        best_index = np.argmin(cost, axis=1).astype(np.int64)
        warm_action = interpolate_knots(replay["mean_knots_before"])
        warm_cost = replay["warm_direct_cost"].astype(np.float64)
        best_cost = cost[row, best_index]
        weighted_cost = replay["optimized_cost"].astype(np.float64)
        weighted_knots = replay["optimized_action_sequence"][:, KNOT_INDICES]
        center_cost = np.stack((warm_cost, best_cost, weighted_cost), axis=1)
        teacher_source = np.argmin(center_cost, axis=1).astype(np.int8)
        center_knots = np.stack(
            (
                replay["mean_knots_before"],
                replay["sampled_knots"][row, best_index],
                weighted_knots,
            ),
            axis=1,
        )
        center_action = np.stack(
            (
                warm_action,
                replay["sampled_action_sequences"][row, best_index],
                replay["optimized_action_sequence"],
            ),
            axis=1,
        )
        center_trajectory = np.stack(
            (
                replay["warm_trajectory"],
                replay["predicted_trajectories"][row, best_index],
                replay["optimized_trajectory"],
            ),
            axis=1,
        )
        teacher_knots = center_knots[row, teacher_source]
        teacher_action = center_action[row, teacher_source]
        teacher_trajectory = center_trajectory[row, teacher_source]
        teacher_cost = center_cost[row, teacher_source]
        teacher_delta = teacher_knots - replay["mean_knots_before"]
        sigma = np.asarray(manifest["teacher_contract"]["noise_sigma"])
        teacher_sigma_rms = np.sqrt(
            np.mean(np.square(teacher_delta / sigma), axis=(1, 2))
        )
        ess = 1.0 / np.sum(np.square(replay["weight"].astype(np.float64)), axis=1)
        clip_fraction = np.mean(
            np.abs(replay["raw_sampled_knots"] - replay["sampled_knots"]) > 1e-7,
            axis=(1, 2, 3),
        )
        derived_expected = {
            "best_candidate_index": best_index,
            "warm_action_sequence": warm_action,
            "last_iteration_candidate_zero_direct_cost": cost[:, 0],
            "candidate_center_knots": center_knots,
            "candidate_center_action_sequences": center_action,
            "candidate_center_trajectories": center_trajectory,
            "candidate_center_direct_cost": center_cost,
            "warm_direct_cost": warm_cost,
            "best_sampled_direct_cost": best_cost,
            "weighted_output_direct_cost": weighted_cost,
            "best_sampled_gain_vs_warm": warm_cost - best_cost,
            "weighted_output_gain_vs_warm": warm_cost - weighted_cost,
            "teacher_source_index": teacher_source,
            "teacher_source_name": TEACHER_SOURCE_NAMES[teacher_source],
            "teacher_knots": teacher_knots,
            "teacher_action_sequence": teacher_action,
            "teacher_trajectory": teacher_trajectory,
            "teacher_direct_cost": teacher_cost,
            "teacher_delta_knots": teacher_delta,
            "teacher_gain_vs_warm": warm_cost - teacher_cost,
            "teacher_delta_sigma_rms": teacher_sigma_rms,
            "effective_sample_size": ess,
            "candidate_clip_fraction": clip_fraction,
        }
        derived_errors = {
            name: numeric_error(replay[name], expected)
            for name, expected in derived_expected.items()
        }
        interpolation_errors = {
            "warm": numeric_error(
                interpolate_knots(replay["candidate_center_knots"][:, 0]),
                replay["candidate_center_action_sequences"][:, 0],
            ),
            "best_sampled": numeric_error(
                interpolate_knots(replay["candidate_center_knots"][:, 1]),
                replay["candidate_center_action_sequences"][:, 1],
            ),
            "weighted_output": numeric_error(
                interpolate_knots(replay["candidate_center_knots"][:, 2]),
                replay["candidate_center_action_sequences"][:, 2],
            ),
            "teacher": numeric_error(
                interpolate_knots(replay["teacher_knots"]),
                replay["teacher_action_sequence"],
            ),
        }

        component_sum = np.zeros_like(cost)
        for name in (
            "position",
            "yaw",
            "vx",
            "acceleration_rate",
            "steering_rate",
        ):
            component_sum += replay[f"cost_component_{name}"]
        weights = TorchMPPICostWeights()
        feature_cost = (
            weights.position * replay["feature_position_error_sq"].sum(axis=2)
            + weights.yaw * replay["feature_yaw_error_sq"].sum(axis=2)
            + weights.vx * replay["feature_vx_error_sq"].sum(axis=2)
            + weights.acceleration_rate
            * replay["feature_action_rate_sq"][:, :, :, 0].sum(axis=2)
            + weights.steering_rate
            * replay["feature_action_rate_sq"][:, :, :, 1].sum(axis=2)
        )
        cost_component_error = numeric_error(component_sum, cost)
        raw_feature_cost_error = numeric_error(feature_cost, cost)
        warm_floor_violation = float(np.max(teacher_cost - warm_cost))

        with (sidecar / "rows.csv").open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        csv_errors = [0.0 if len(csv_rows) == row_count else 1.0]
        for index in np.linspace(0, row_count - 1, min(30, row_count), dtype=int):
            record = csv_rows[int(index)]
            csv_errors.extend(
                (
                    0.0 if int(record["row_index"]) == int(index) else 1.0,
                    abs(float(record["teacher_direct_cost"]) - teacher_cost[index]),
                    0.0
                    if record["teacher_source_name"]
                    == str(TEACHER_SOURCE_NAMES[teacher_source[index]])
                    else 1.0,
                )
            )

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        params = TorchMPPIParams(**source_manifest["collection"]["mppi"])
        model = QueryDeploymentModel.from_checkpoint(
            Path(manifest["query_checkpoint"]), args.device
        )
        torch_controller = TorchMPPIController(
            TorchQueryRolloutBackend(model), params, device=args.device
        )
        query_trajectory_errors: list[float] = []
        query_cost_errors: list[float] = []
        warm_query_trajectory_errors: list[float] = []
        warm_query_cost_errors: list[float] = []
        for index in range(row_count):
            result = torch_controller.evaluate_action_sequences(
                replay["state"][index],
                replay["current_action"][index],
                replay["history"][index : index + 1],
                replay["reference"][index],
                warm_action[index : index + 1],
            )
            warm_query_trajectory_errors.append(
                numeric_error(
                    result["trajectories"][0].cpu().numpy(),
                    replay["warm_trajectory"][index],
                )
            )
            warm_query_cost_errors.append(
                abs(float(result["cost"][0].cpu()) - warm_cost[index])
            )
        query_rows = stratified_rows(replay, args.query_replay_rows)
        for index in query_rows:
            result = torch_controller.evaluate_action_sequences(
                replay["state"][index],
                replay["current_action"][index],
                replay["history"][index : index + 1],
                replay["reference"][index],
                replay["sampled_action_sequences"][index],
            )
            query_trajectory_errors.append(
                numeric_error(
                    result["trajectories"].cpu().numpy(),
                    replay["predicted_trajectories"][index],
                )
            )
            query_cost_errors.append(
                numeric_error(result["cost"].cpu().numpy(), replay["cost"][index])
            )
            optimized = torch_controller.evaluate_action_sequences(
                replay["state"][index],
                replay["current_action"][index],
                replay["history"][index : index + 1],
                replay["reference"][index],
                replay["optimized_action_sequence"][index : index + 1],
            )
            query_trajectory_errors.append(
                numeric_error(
                    optimized["trajectories"][0].cpu().numpy(),
                    replay["optimized_trajectory"][index],
                )
            )
            query_cost_errors.append(
                abs(float(optimized["cost"][0].cpu()) - weighted_cost[index])
            )

        onnx_trajectory_errors: list[float] = []
        onnx_cost_errors: list[float] = []
        onnx_cost_relative_errors: list[float] = []
        onnx_best_index_errors: list[float] = []
        onnx_best_cost_errors: list[float] = []
        onnx_rows = np.empty(0, dtype=np.int64)
        onnx_hash = None
        if args.onnx_replay_rows > 0:
            onnx_path = args.onnx.resolve()
            onnx_hash = sha256(onnx_path)
            onnx_controller = TorchMPPIController(
                OnnxQueryRolloutBackend(onnx_path, provider=args.device, output_device=args.device),
                params,
                device=args.device,
            )
            onnx_rows = stratified_rows(replay, args.onnx_replay_rows)
            for index in onnx_rows:
                result = onnx_controller.evaluate_action_sequences(
                    replay["state"][index],
                    replay["current_action"][index],
                    replay["history"][index : index + 1],
                    replay["reference"][index],
                    replay["sampled_action_sequences"][index],
                )
                onnx_trajectory_errors.append(
                    numeric_error(
                        result["trajectories"].cpu().numpy(),
                        replay["predicted_trajectories"][index],
                    )
                )
                onnx_cost_errors.append(
                    numeric_error(result["cost"].cpu().numpy(), replay["cost"][index])
                )
                predicted_cost = result["cost"].cpu().numpy()
                stored_cost = replay["cost"][index]
                onnx_cost_relative_errors.append(
                    float(
                        np.max(
                            np.abs(predicted_cost - stored_cost)
                            / np.maximum(np.abs(stored_cost), 1.0)
                        )
                    )
                )
                predicted_best = int(np.argmin(predicted_cost))
                stored_best = int(np.argmin(stored_cost))
                onnx_best_index_errors.append(
                    0.0 if predicted_best == stored_best else 1.0
                )
                onnx_best_cost_errors.append(
                    abs(
                        float(predicted_cost[predicted_best])
                        - float(stored_cost[stored_best])
                    )
                )

        checks = {
            "all_artifact_hashes": all(hash_checks.values())
            and maximum(source_hash_errors) == 0.0,
            "exact_source_consolidation": maximum(source_errors) == 0.0,
            "exact_row_metadata": maximum(metadata_errors) == 0.0,
            "episode_grouped_5fold": maximum(fold_errors) == 0.0
            and len(episode_fold) == manifest["episode_count"],
            "teacher_argmin_reconstruction": maximum(list(derived_errors.values()))
            <= 1e-7,
            "teacher_never_regresses_warm": warm_floor_violation <= 1e-12,
            "teacher_knot_interpolation": maximum(list(interpolation_errors.values()))
            <= 2e-6,
            "candidate_cost_component_replay": cost_component_error <= 0.25,
            "candidate_raw_feature_replay": raw_feature_cost_error <= 0.25,
            "rows_csv_reconstruction": maximum(csv_errors) <= 1e-9,
            "pytorch_query_replay": maximum(query_trajectory_errors) <= 5e-5
            and maximum(query_cost_errors) <= 0.5
            and maximum(warm_query_trajectory_errors) <= 5e-5
            and maximum(warm_query_cost_errors) <= 0.5,
            "onnx_query_shadow_replay": args.onnx_replay_rows <= 0
            or (
                maximum(onnx_trajectory_errors) <= 2e-3
                and maximum(onnx_cost_errors) <= 1.0
                and maximum(onnx_cost_relative_errors) <= 5e-3
                and maximum(onnx_best_index_errors) == 0.0
                and maximum(onnx_best_cost_errors) <= 0.01
            ),
            "formal_validation_test_sealed": not manifest[
                "formal_validation_or_test_consumed"
            ]
            and not splits["formal_validation_or_test_consumed"],
            "query_analytic_gradient_not_consumed": not manifest.get(
                "query_analytic_gradient_consumed", False
            )
            and not splits.get("query_analytic_gradient_consumed", False),
        }
        passed = all(checks.values())
        output = {
            "qualification": (
                "QUERY_EXPECTED_ROAD_T0_REPLAY_PASS"
                if passed
                else "QUERY_EXPECTED_ROAD_T0_REPLAY_FAIL"
            ),
            "sidecar": str(sidecar),
            "manifest_sha256": sha256(manifest_path),
            "checks": checks,
            "hash_checks": hash_checks,
            "maximum_errors": {
                "source_consolidation": maximum(source_errors),
                "row_metadata": maximum(metadata_errors),
                "fold": maximum(fold_errors),
                "teacher_derived": maximum(list(derived_errors.values())),
                "warm_floor_violation": warm_floor_violation,
                "teacher_interpolation": maximum(
                    list(interpolation_errors.values())
                ),
                "candidate_cost_component_replay": cost_component_error,
                "candidate_raw_feature_replay": raw_feature_cost_error,
                "rows_csv": maximum(csv_errors),
                "pytorch_query_trajectory": maximum(query_trajectory_errors),
                "pytorch_query_cost": maximum(query_cost_errors),
                "all_warm_query_trajectory": maximum(
                    warm_query_trajectory_errors
                ),
                "all_warm_query_cost": maximum(warm_query_cost_errors),
                "onnx_query_trajectory": maximum(onnx_trajectory_errors),
                "onnx_query_cost": maximum(onnx_cost_errors),
                "onnx_query_cost_relative": maximum(
                    onnx_cost_relative_errors
                ),
                "onnx_best_index": maximum(onnx_best_index_errors),
                "onnx_best_cost": maximum(onnx_best_cost_errors),
            },
            "teacher": {
                "source_counts": {
                    str(name): int(np.sum(teacher_source == index))
                    for index, name in enumerate(TEACHER_SOURCE_NAMES.tolist())
                },
                "gain_vs_warm": stats(warm_cost - teacher_cost),
                "delta_sigma_rms": stats(teacher_sigma_rms),
                "warm_floor_violation_count": int(
                    np.sum(teacher_cost > warm_cost + 1e-12)
                ),
            },
            "split": {
                "episode_count": len(episode_fold),
                "fold_row_counts": {
                    str(fold): int(np.sum(replay["fold_id"] == fold))
                    for fold in range(5)
                },
            },
            "pytorch_query_replay_rows": query_rows.tolist(),
            "all_warm_query_replay_row_count": row_count,
            "onnx_shadow": {
                "path": str(args.onnx.resolve()),
                "sha256": onnx_hash,
                "rows": onnx_rows.tolist(),
                "note": (
                    "Behavioral parity is checked against frozen PyTorch Query and saved "
                    "rollouts; the ONNX file itself carries no embedded checkpoint metadata."
                ),
                "gates": {
                    "trajectory_max_abs": 0.002,
                    "cost_max_abs": 1.0,
                    "cost_max_relative_with_unit_floor": 0.005,
                    "best_candidate_index_mismatch": 0,
                    "best_cost_max_abs": 0.01,
                },
            },
            "formal_validation_or_test_consumed": False,
        }

    validation_path = sidecar / "validation.json"
    validation_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
