#!/usr/bin/env python3
"""Independently validate the T0-centered full-rank Query sidecar."""

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
    "query_expected_road_fullrank_20260901_v1"
)
DEFAULT_ONNX = REPO_ROOT / "outputs/query_mppi/anycar_query.onnx"


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


def maximum(values) -> float:
    values = list(values)
    return float(max(values, default=0.0))


def error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float("inf")
    if actual.dtype.kind in "US" or expected.dtype.kind in "US":
        return 0.0 if np.array_equal(actual, expected) else 1.0
    return float(
        np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
    )


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


def hadamard_16() -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float32)
    while len(matrix) < 16:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix.reshape(16, 8, 2)


def interpolate(knots: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(knots, dtype=torch.float32)
    flat = tensor.reshape(-1, 8, 2)
    actions = F.interpolate(
        flat.transpose(1, 2), size=50, mode="linear", align_corners=True
    ).transpose(1, 2)
    return actions.reshape(*tensor.shape[:-2], 50, 2).numpy()


def stratified_rows(data: np.lib.npyio.NpzFile, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    episode_rows = []
    for episode in np.unique(data["episode_index"]):
        rows = np.flatnonzero(data["episode_index"] == episode)
        episode_rows.append(int(rows[len(rows) // 2]))
    if count <= len(episode_rows):
        return np.asarray(episode_rows)[
            np.linspace(0, len(episode_rows) - 1, count, dtype=int)
        ]
    extra = np.linspace(0, len(data["state"]) - 1, count, dtype=int)
    return np.asarray(sorted(set(episode_rows).union(extra.tolist()))[:count])


def main() -> None:
    args = parse_args()
    sidecar = args.sidecar.resolve()
    manifest_path = sidecar / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((sidecar / "config.json").read_text())
    parent = Path(manifest["parent_t0"])
    parent_manifest_path = parent / "manifest.json"
    parent_validation_path = parent / "validation.json"
    parent_replay_path = parent / "replay.npz"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_validation = json.loads(parent_validation_path.read_text())
    bank_path = sidecar / "bank.npz"

    hash_checks = {
        "config": sha256(sidecar / "config.json") == manifest["config_sha256"],
        "parent_manifest": sha256(parent_manifest_path)
        == manifest["parent_manifest_sha256"],
        "parent_validation": sha256(parent_validation_path)
        == manifest["parent_validation_sha256"],
        "parent_replay": sha256(parent_replay_path)
        == manifest["parent_replay_sha256"],
        "checkpoint": sha256(Path(manifest["query_checkpoint"]))
        == manifest["query_checkpoint_sha256"],
        "bank": sha256(bank_path) == manifest["bank_sha256"],
        "rows": sha256(sidecar / "rows.csv") == manifest["rows_sha256"],
        "splits": sha256(sidecar / "splits.json") == manifest["splits_sha256"],
    }
    if parent_validation["qualification"] != "QUERY_EXPECTED_ROAD_T0_REPLAY_PASS":
        raise AssertionError("parent T0 no longer passes")
    if manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("sidecar consumed formal validation/test")
    if manifest.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("pure Query sidecar may not consume DBM data")
    if manifest.get("query_analytic_gradient_consumed", False):
        raise AssertionError("pure Query sidecar may not consume Query analytic gradients")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    sigma = np.asarray(config["noise_sigma"], dtype=np.float32)
    directions = hadamard_16() * sigma.reshape(1, 1, 2)
    candidate_count = int(config["total_candidates"])
    context_errors = []
    contract_errors = []
    bank_errors = []
    derived_errors = []
    rank_errors = []
    baseline_trajectory_errors = []
    baseline_cost_errors = []
    baseline_cost_relative_errors = []
    baseline_argmin_errors = []
    with np.load(parent_replay_path, allow_pickle=False) as parent_data, np.load(
        bank_path, allow_pickle=False
    ) as data:
        contract_errors.append(0.0 if manifest["bank_contract"] == config else 1.0)
        row_count = len(data["state"])
        if row_count != manifest["row_count"]:
            raise AssertionError("row count mismatch")
        for name in (
            "row_index",
            "episode_id",
            "episode_index",
            "row_in_episode",
            "control_step",
            "speed_kph",
            "speed_index",
            "variant_index",
            "road_name",
            "fold_id",
            "seed_row",
            "seed_source",
            "seed_source_file",
            "seed_source_window_index",
            "source_snapshot_sha256",
            "state",
            "current_action",
            "history",
            "reference",
            "reference_ego",
            "mean_knots_before",
            "teacher_knots",
            "teacher_direct_cost",
        ):
            context_errors.append(error(data[name], parent_data[name]))

        candidate_names = list(config["base_candidates"])
        groups = ["baseline"] * 4
        radii = [0.0] * 4
        direction_indices = [-1] * 4
        signs = [0] * 4
        pair_indices = np.empty((4, 16, 2), dtype=np.int64)
        for radius_index, radius in enumerate(config["probe_radii_sigma"]):
            for direction_index in range(16):
                plus = len(candidate_names)
                pair_indices[radius_index, direction_index] = (plus, plus + 1)
                tag = str(radius).replace(".", "p")
                candidate_names.extend(
                    (
                        f"prox_r{tag}_d{direction_index:02d}_plus",
                        f"prox_r{tag}_d{direction_index:02d}_minus",
                    )
                )
                groups.extend(("proximal", "proximal"))
                radii.extend((radius, radius))
                direction_indices.extend((direction_index, direction_index))
                signs.extend((1, -1))
        expected_contract = {
            "candidate_name": np.asarray(candidate_names),
            "candidate_group": np.asarray(groups),
            "candidate_radius_sigma": np.asarray(radii, np.float32),
            "candidate_direction_index": np.asarray(direction_indices, np.int16),
            "candidate_sign": np.asarray(signs, np.int8),
            "pair_candidate_indices": pair_indices,
        }
        for name, expected in expected_contract.items():
            contract_errors.append(error(data[name], expected))
            contract_errors.append(
                0.0
                if manifest["candidate_contract"][name] == expected.tolist()
                else 1.0
            )

        expected_raw = np.empty_like(data["raw_candidate_knots"])
        for row in range(row_count):
            expected_raw[row, :4] = np.stack(
                (
                    parent_data["mean_knots_before"][row],
                    parent_data["teacher_knots"][row],
                    parent_data["candidate_center_knots"][row, 1],
                    parent_data["candidate_center_knots"][row, 2],
                )
            )
            index = 4
            anchor = parent_data["teacher_knots"][row]
            for radius in config["probe_radii_sigma"]:
                for direction in directions:
                    for sign in (1.0, -1.0):
                        expected_raw[row, index] = anchor + sign * radius * direction
                        index += 1
        expected_knots = np.clip(
            expected_raw,
            config["action_bounds"]["minimum"],
            config["action_bounds"]["maximum"],
        )
        expected_actions = interpolate(expected_knots)
        bank_errors.extend(
            (
                error(data["raw_candidate_knots"], expected_raw),
                error(data["candidate_knots"], expected_knots),
                error(data["candidate_action_sequences"], expected_actions),
                error(
                    data["candidate_clipped_mask"],
                    np.abs(expected_raw - expected_knots) > 1e-7,
                ),
            )
        )

        expected_baseline_trajectory = np.stack(
            (
                parent_data["warm_trajectory"],
                parent_data["teacher_trajectory"],
                parent_data["candidate_center_trajectories"][:, 1],
                parent_data["candidate_center_trajectories"][:, 2],
            ),
            axis=1,
        )
        expected_baseline_cost = np.stack(
            (
                parent_data["warm_direct_cost"],
                parent_data["teacher_direct_cost"],
                parent_data["best_sampled_direct_cost"],
                parent_data["weighted_output_direct_cost"],
            ),
            axis=1,
        )
        baseline_trajectory_errors.append(
            error(data["candidate_trajectories"][:, :4], expected_baseline_trajectory)
        )
        baseline_cost_errors.append(
            error(data["candidate_cost"][:, :4], expected_baseline_cost)
        )
        baseline_cost_relative_errors.append(
            float(
                np.max(
                    np.abs(data["candidate_cost"][:, :4] - expected_baseline_cost)
                    / np.maximum(np.abs(expected_baseline_cost), 1.0)
                )
            )
        )
        baseline_argmin_errors.append(
            float(
                np.sum(
                    np.argmin(data["candidate_cost"][:, :4], axis=1)
                    != np.argmin(expected_baseline_cost, axis=1)
                )
            )
        )

        component_sum = np.zeros_like(data["candidate_cost"], dtype=np.float64)
        for name in ("position", "yaw", "vx", "acceleration_rate", "steering_rate"):
            component_sum += data[f"cost_component_{name}"]
        weights = TorchMPPICostWeights()
        feature_cost = (
            weights.position * data["feature_position_error_sq"].sum(axis=2)
            + weights.yaw * data["feature_yaw_error_sq"].sum(axis=2)
            + weights.vx * data["feature_vx_error_sq"].sum(axis=2)
            + weights.acceleration_rate
            * data["feature_action_rate_sq"][:, :, :, 0].sum(axis=2)
            + weights.steering_rate
            * data["feature_action_rate_sq"][:, :, :, 1].sum(axis=2)
        )
        cost_component_error = error(component_sum, data["candidate_cost"])
        feature_cost_error = error(feature_cost, data["candidate_cost"])

        pair_cost = data["candidate_cost"][:, pair_indices]
        pair_clipped = data["candidate_clipped_mask"][:, pair_indices]
        expected_pair_symmetric = ~np.any(pair_clipped, axis=(3, 4, 5))
        t0_cost = data["candidate_cost"][:, 1].astype(np.float64)
        expected_gain = t0_cost[:, None] - data["candidate_cost"].astype(np.float64)
        expected_clip_fraction = np.mean(
            data["candidate_clipped_mask"], axis=(1, 2, 3)
        )
        derived_errors.extend(
            (
                error(data["pair_cost"], pair_cost),
                error(
                    data["pair_cost_difference_minus_minus_plus"],
                    pair_cost[..., 1] - pair_cost[..., 0],
                ),
                error(data["pair_symmetric_mask"], expected_pair_symmetric),
                error(data["candidate_gain_vs_t0"], expected_gain),
                error(data["candidate_clip_fraction"], expected_clip_fraction),
            )
        )

        expected_rank = np.empty_like(data["local_rank_by_radius"])
        expected_condition = np.empty_like(data["local_condition_by_radius"])
        for row in range(row_count):
            for radius_index in range(4):
                indices = pair_indices[radius_index].reshape(-1)
                matrix = (
                    (data["candidate_knots"][row, indices] - data["candidate_knots"][row, 1])
                    / sigma
                ).reshape(32, 16)
                singular = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
                expected_rank[row, radius_index] = np.linalg.matrix_rank(matrix, tol=1e-7)
                expected_condition[row, radius_index] = (
                    singular[0] / singular[-1] if singular[-1] > 1e-12 else np.inf
                )
        rank_errors.extend(
            (
                error(data["local_rank_by_radius"], expected_rank),
                error(data["local_condition_by_radius"], expected_condition),
            )
        )

        teacher_index = np.argmin(data["candidate_cost"].astype(np.float64), axis=1)
        row = np.arange(row_count)
        teacher_knots = data["candidate_knots"][row, teacher_index]
        teacher_cost = data["candidate_cost"][row, teacher_index].astype(np.float64)
        delta_t0 = teacher_knots - data["candidate_knots"][:, 1]
        delta_rms = np.sqrt(np.mean(np.square(delta_t0 / sigma), axis=(1, 2)))
        teacher_expected = {
            "fullrank_teacher_index": teacher_index,
            "fullrank_teacher_name": data["candidate_name"][teacher_index],
            "fullrank_teacher_knots": teacher_knots,
            "fullrank_teacher_action_sequence": data["candidate_action_sequences"][row, teacher_index],
            "fullrank_teacher_trajectory": data["candidate_trajectories"][row, teacher_index],
            "fullrank_teacher_direct_cost": teacher_cost,
            "fullrank_gain_vs_t0": t0_cost - teacher_cost,
            "fullrank_gain_vs_warm": data["candidate_cost"][:, 0] - teacher_cost,
            "fullrank_delta_from_t0_knots": delta_t0,
            "fullrank_delta_from_t0_sigma_rms": delta_rms,
            "fullrank_delta_from_warm_knots": teacher_knots - data["candidate_knots"][:, 0],
        }
        for name, expected in teacher_expected.items():
            derived_errors.append(error(data[name], expected))
        warm_floor_violation = float(
            np.max(teacher_cost - data["candidate_cost"][:, 0])
        )
        t0_floor_violation = float(np.max(teacher_cost - t0_cost))

        with (sidecar / "rows.csv").open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        csv_errors = [0.0 if len(csv_rows) == row_count else 1.0]
        for index in np.linspace(0, row_count - 1, 30, dtype=int):
            csv_errors.extend(
                (
                    0.0 if int(csv_rows[index]["row_index"]) == index else 1.0,
                    abs(
                        float(csv_rows[index]["fullrank_teacher_direct_cost"])
                        - teacher_cost[index]
                    ),
                    0.0
                    if csv_rows[index]["fullrank_teacher_name"]
                    == str(data["candidate_name"][teacher_index[index]])
                    else 1.0,
                )
            )

        source_manifest = json.loads(
            (Path(parent_manifest["source_collection"]) / "manifest.json").read_text()
        )
        params = TorchMPPIParams(**source_manifest["collection"]["mppi"])
        model = QueryDeploymentModel.from_checkpoint(
            Path(manifest["query_checkpoint"]), args.device
        )
        torch_controller = TorchMPPIController(
            TorchQueryRolloutBackend(model), params, device=args.device
        )
        query_rows = stratified_rows(data, args.query_replay_rows)
        query_trajectory_errors = []
        query_cost_errors = []
        for index in query_rows:
            result = torch_controller.evaluate_action_sequences(
                data["state"][index],
                data["current_action"][index],
                data["history"][index : index + 1],
                data["reference"][index],
                data["candidate_action_sequences"][index],
            )
            query_trajectory_errors.append(
                error(result["trajectories"].cpu().numpy(), data["candidate_trajectories"][index])
            )
            query_cost_errors.append(
                error(result["cost"].cpu().numpy(), data["candidate_cost"][index])
            )

        onnx_rows = stratified_rows(data, args.onnx_replay_rows)
        onnx_trajectory_errors = []
        onnx_cost_errors = []
        onnx_relative_errors = []
        onnx_best_index_errors = []
        onnx_best_cost_errors = []
        onnx_hash = None
        if len(onnx_rows):
            onnx_path = args.onnx.resolve()
            onnx_hash = sha256(onnx_path)
            onnx_controller = TorchMPPIController(
                OnnxQueryRolloutBackend(
                    onnx_path, provider=args.device, output_device=args.device
                ),
                params,
                device=args.device,
            )
            for index in onnx_rows:
                result = onnx_controller.evaluate_action_sequences(
                    data["state"][index],
                    data["current_action"][index],
                    data["history"][index : index + 1],
                    data["reference"][index],
                    data["candidate_action_sequences"][index],
                )
                predicted_trajectory = result["trajectories"].cpu().numpy()
                predicted_cost = result["cost"].cpu().numpy()
                stored_cost = data["candidate_cost"][index]
                onnx_trajectory_errors.append(
                    error(predicted_trajectory, data["candidate_trajectories"][index])
                )
                absolute = np.abs(predicted_cost - stored_cost)
                onnx_cost_errors.append(float(np.max(absolute)))
                onnx_relative_errors.append(
                    float(np.max(absolute / np.maximum(np.abs(stored_cost), 1.0)))
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

        finite_errors = []
        for name in data.files:
            value = data[name]
            if np.issubdtype(value.dtype, np.number):
                finite_errors.append(0.0 if np.isfinite(value).all() else 1.0)
        folds = [
            len(np.unique(data["fold_id"][data["episode_id"] == episode]))
            for episode in np.unique(data["episode_id"])
        ]
        checks = {
            "artifact_hashes": all(hash_checks.values()),
            "exact_parent_context": maximum(context_errors) == 0.0,
            "candidate_contract": maximum(contract_errors) == 0.0,
            "bank_reconstruction": maximum(bank_errors) <= 6e-7,
            "baseline_parent_replay": maximum(baseline_trajectory_errors) <= 5e-4
            and maximum(baseline_cost_errors) <= 0.01
            and maximum(baseline_cost_relative_errors) <= 2e-3
            and maximum(baseline_argmin_errors) == 0.0,
            "cost_component_replay": cost_component_error <= 0.25,
            "raw_feature_replay": feature_cost_error <= 0.25,
            "pair_teacher_reconstruction": maximum(derived_errors) <= 1e-6,
            "fullrank_geometry": maximum(rank_errors) <= 1e-5
            and int(data["local_rank_by_radius"].min()) == 16,
            "warm_and_t0_floor": warm_floor_violation <= 1e-12
            and t0_floor_violation <= 1e-12,
            "episode_grouping": max(folds) == 1
            and len(np.unique(data["episode_id"]))
            == int(parent_manifest["episode_count"]),
            "rows_csv": maximum(csv_errors) <= 1e-9,
            "finite": maximum(finite_errors) == 0.0,
            "pytorch_query_replay": maximum(query_trajectory_errors) <= 5e-5
            and maximum(query_cost_errors) <= 0.5,
            "onnx_shadow_replay": len(onnx_rows) == 0
            or (
                maximum(onnx_trajectory_errors) <= 2e-3
                and maximum(onnx_cost_errors) <= 1.0
                and maximum(onnx_relative_errors) <= 1e-2
                and maximum(onnx_best_index_errors) == 0.0
                and maximum(onnx_best_cost_errors) <= 0.01
            ),
            "formal_validation_test_sealed": not manifest[
                "formal_validation_or_test_consumed"
            ],
            "query_analytic_gradient_not_consumed": not manifest.get(
                "query_analytic_gradient_consumed", False
            ),
        }
        passed = all(checks.values())
        output = {
            "qualification": (
                "QUERY_EXPECTED_ROAD_FULLRANK_PASS"
                if passed
                else "QUERY_EXPECTED_ROAD_FULLRANK_FAIL"
            ),
            "sidecar": str(sidecar),
            "manifest_sha256": sha256(manifest_path),
            "checks": checks,
            "hash_checks": hash_checks,
            "maximum_errors": {
                "parent_context": maximum(context_errors),
                "candidate_contract": maximum(contract_errors),
                "bank_reconstruction": maximum(bank_errors),
                "baseline_parent_trajectory": maximum(
                    baseline_trajectory_errors
                ),
                "baseline_parent_cost": maximum(baseline_cost_errors),
                "baseline_parent_cost_relative": maximum(
                    baseline_cost_relative_errors
                ),
                "baseline_parent_argmin_mismatch_count": maximum(
                    baseline_argmin_errors
                ),
                "cost_component": cost_component_error,
                "raw_feature": feature_cost_error,
                "derived": maximum(derived_errors),
                "rank_condition": maximum(rank_errors),
                "warm_floor_violation": warm_floor_violation,
                "t0_floor_violation": t0_floor_violation,
                "rows_csv": maximum(csv_errors),
                "pytorch_query_trajectory": maximum(query_trajectory_errors),
                "pytorch_query_cost": maximum(query_cost_errors),
                "onnx_trajectory": maximum(onnx_trajectory_errors),
                "onnx_cost": maximum(onnx_cost_errors),
                "onnx_cost_relative": maximum(onnx_relative_errors),
                "onnx_best_index": maximum(onnx_best_index_errors),
                "onnx_best_cost": maximum(onnx_best_cost_errors),
            },
            "geometry": {
                "minimum_rank": int(data["local_rank_by_radius"].min()),
                "condition": stats(data["local_condition_by_radius"]),
                "symmetric_pair_fraction": float(np.mean(data["pair_symmetric_mask"])),
            },
            "teacher": {
                "gain_vs_t0": stats(data["fullrank_gain_vs_t0"]),
                "gain_vs_warm": stats(data["fullrank_gain_vs_warm"]),
                "strict_improvement_vs_t0": int(
                    np.sum(data["fullrank_gain_vs_t0"] > 0)
                ),
                "source_counts": {
                    str(name): int(np.sum(data["fullrank_teacher_name"] == name))
                    for name in np.unique(data["fullrank_teacher_name"])
                },
            },
            "pytorch_query_replay_rows": query_rows.tolist(),
            "onnx_shadow": {
                "path": str(args.onnx.resolve()),
                "sha256": onnx_hash,
                "rows": onnx_rows.tolist(),
                "note": "Behavioral parity only; the ONNX file has no embedded checkpoint provenance.",
            },
            "formal_validation_or_test_consumed": False,
            "query_analytic_gradient_consumed": False,
        }

    validation_path = sidecar / "validation.json"
    validation_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
