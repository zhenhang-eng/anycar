#!/usr/bin/env python3
"""Independently validate a frozen-Query expected-road closed-loop collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


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
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--replay-snapshots", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def wrapped(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def append_history(history, state, next_state, action):
    dx_world = float(next_state[0] - state[0])
    dy_world = float(next_state[1] - state[1])
    cosine = math.cos(float(state[2]))
    sine = math.sin(float(state[2]))
    token = np.asarray(
        (
            dx_world * cosine + dy_world * sine,
            -dx_world * sine + dy_world * cosine,
            wrapped(float(next_state[2] - state[2])),
            float(next_state[3] - state[3]),
            float(next_state[4] - state[4]),
            float(action[0]),
            float(action[1]),
        ),
        np.float32,
    )
    return np.concatenate((history[1:], token[None]), axis=0)


def maximum(error_values: list[float]) -> float:
    return float(max(error_values, default=0.0))


def main() -> None:
    args = parse_args()
    root = args.collection.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_type"] != "anycar-query-expected-road-closed-loop":
        raise ValueError("not a Query expected-road closed-loop collection")
    if manifest["formal_validation_or_test_consumed"]:
        raise AssertionError("formal validation/test must remain sealed")
    checkpoint_path = Path(manifest["query_checkpoint"])
    if sha256(checkpoint_path) != manifest["query_checkpoint_sha256"]:
        raise AssertionError("checkpoint hash mismatch")
    if sha256(Path(manifest["seed_replay"])) != manifest["seed_replay_sha256"]:
        raise AssertionError("seed replay hash mismatch")
    if set(manifest["seed_replay_fields_used"]).intersection(
        manifest["seed_replay_fields_explicitly_not_used"]
    ):
        raise AssertionError("used and prohibited seed fields overlap")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history_mean, history_std = [
        np.asarray(value, np.float64) for value in checkpoint["stats"]["history"]
    ]
    context_mean, context_std = [
        np.asarray(value, np.float64) for value in checkpoint["stats"]["context"]
    ]
    params_dict = manifest["collection"]["mppi"]
    params = TorchMPPIParams(**params_dict)
    cost_weights = TorchMPPICostWeights()

    shape_errors = []
    hash_errors = []
    finite_errors = []
    state_chain_errors = []
    action_chain_errors = []
    warm_cold_errors = []
    warm_recede_errors = []
    snapshot_trace_errors = []
    history_reconstruction_errors = []
    reference_progress_ratio = []
    cost_replay_errors = []
    raw_feature_errors = []
    weight_sum_errors = []
    candidate_zero_errors = []
    optimized_next_errors = []
    history_abs_z = []
    history_tail_fraction = []
    current_state_abs_z = []
    candidate_context_abs_z = []
    snapshot_speed_error_kph = []
    snapshot_position_error_m = []
    episode_max_abs_speed_error_kph = []
    episode_max_position_error_m = []
    snapshot_records = []

    episodes = manifest["episodes"]
    speed_bins = [int(value) for value in manifest["collection"]["speed_bins_kph"]]
    variant_indices = [
        int(value) for value in manifest["collection"]["selected_variant_indices"]
    ]
    repeats_per_cell = int(manifest["collection"].get("repeats_per_cell", 1))
    expected_cells = {
        (speed, variant, repeat)
        for speed in speed_bins
        for variant in variant_indices
        for repeat in range(repeats_per_cell)
    }
    actual_cells = {
        (
            int(episode["speed_kph"]),
            int(episode["variant_index"]),
            int(episode.get("repeat_index", 0)),
        )
        for episode in episodes
    }
    episode_distribution_checks = {
        "unique_episode_ids": len({str(row["episode_id"]) for row in episodes})
        == len(episodes),
        "exact_speed_variant_repeat_grid": actual_cells == expected_cells,
        "unique_seed_rows": len({int(row["seed_row"]) for row in episodes})
        == len(episodes),
        "repeat_folds_cover_all_cells": all(
            {
                (int(row["speed_kph"]), int(row["variant_index"]))
                for row in episodes
                if int(row.get("repeat_index", 0)) == repeat
            }
            == {(speed, variant) for speed in speed_bins for variant in variant_indices}
            for repeat in range(repeats_per_cell)
        ),
    }

    for episode in manifest["episodes"]:
        episode_dir = root / episode["episode_id"]
        summary = json.loads((episode_dir / "summary.json").read_text())
        for name in ("road.npz", "trace.npz", "snapshots.npz"):
            expected = summary["artifacts"][name]["sha256"]
            hash_errors.append(0.0 if sha256(episode_dir / name) == expected else 1.0)
        road = np.load(episode_dir / "road.npz", allow_pickle=False)
        trace = np.load(episode_dir / "trace.npz", allow_pickle=False)
        snapshots = np.load(episode_dir / "snapshots.npz", allow_pickle=False)
        step_count = len(trace["control_step"])
        snapshot_count = len(snapshots["control_step"])
        sample_count = snapshots["sampled_action_sequences"].shape[1]
        expected_shapes = {
            "trace_state": ((step_count, 5), trace["state"].shape),
            "trace_reference": ((step_count, 51, 4), trace["reference"].shape),
            "snapshot_history": ((snapshot_count, 250, 7), snapshots["history"].shape),
            "snapshot_actions": (
                (snapshot_count, sample_count, 50, 2),
                snapshots["sampled_action_sequences"].shape,
            ),
            "snapshot_trajectory": (
                (snapshot_count, sample_count, 50, 5),
                snapshots["predicted_trajectories"].shape,
            ),
        }
        shape_errors.extend(
            0.0 if expected == actual else 1.0
            for expected, actual in expected_shapes.values()
        )
        for archive in (road, trace, snapshots):
            for name in archive.files:
                value = archive[name]
                if np.issubdtype(value.dtype, np.number):
                    finite_errors.append(0.0 if np.isfinite(value).all() else 1.0)

        state_chain_errors.append(
            float(np.max(np.abs(trace["state"][1:] - trace["next_state"][:-1])))
        )
        action_chain_errors.append(
            float(
                np.max(
                    np.abs(
                        trace["current_action"][1:]
                        - trace["executed_action"][:-1]
                    )
                )
            )
        )
        warm_cold_errors.append(
            float(
                np.max(
                    np.abs(
                        trace["mean_knots_before"][0]
                        - trace["current_action"][0][None]
                    )
                )
            )
        )
        warm_recede_errors.append(
            float(
                np.max(
                    np.abs(
                        trace["mean_knots_before"][1:]
                        - trace["mean_knots_after"][:-1]
                    )
                )
            )
        )

        reference_step = np.linalg.norm(
            np.diff(trace["reference"][:, :, :2], axis=1), axis=2
        )
        expected_step = trace["reference"][:, :-1, 3] * params.dt
        reference_progress_ratio.append(reference_step / expected_step)

        snapshot_lookup = {
            int(step): index
            for index, step in enumerate(snapshots["control_step"])
        }
        history = np.asarray(trace["seed_history"], np.float32).copy()
        for step in range(step_count):
            if step in snapshot_lookup:
                index = snapshot_lookup[step]
                for snapshot_name, trace_name in (
                    ("state", "state"),
                    ("current_action", "current_action"),
                    ("reference", "reference"),
                    ("mean_knots_before", "mean_knots_before"),
                    ("mean_knots_after", "mean_knots_after"),
                ):
                    snapshot_trace_errors.append(
                        float(
                            np.max(
                                np.abs(
                                    snapshots[snapshot_name][index]
                                    - trace[trace_name][step]
                                )
                            )
                        )
                    )
                history_reconstruction_errors.append(
                    float(np.max(np.abs(snapshots["history"][index] - history)))
                )
                snapshot_records.append((episode_dir, index))
            history = append_history(
                history,
                trace["state"][step],
                trace["next_state"][step],
                trace["executed_action"][step],
            )

        weights = snapshots["weight"]
        weight_sum_errors.append(float(np.max(np.abs(weights.sum(axis=1) - 1.0))))
        candidate_zero_errors.append(
            float(
                max(
                    np.max(np.abs(snapshots["sampling_noise_knots"][:, 0])),
                    np.max(
                        np.abs(
                            snapshots["raw_sampled_knots"][:, 0]
                            - snapshots["sampling_mean_knots"]
                        )
                    ),
                )
            )
        )
        component_sum = np.zeros_like(snapshots["cost"], dtype=np.float64)
        for name in (
            "position",
            "yaw",
            "vx",
            "acceleration_rate",
            "steering_rate",
        ):
            component_sum += snapshots[f"cost_component_{name}"]
        cost_replay_errors.append(
            float(np.max(np.abs(component_sum - snapshots["cost"])))
        )
        feature_cost = (
            cost_weights.position
            * snapshots["feature_position_error_sq"].sum(axis=2)
            + cost_weights.yaw
            * snapshots["feature_yaw_error_sq"].sum(axis=2)
            + cost_weights.vx
            * snapshots["feature_vx_error_sq"].sum(axis=2)
            + cost_weights.acceleration_rate
            * snapshots["feature_action_rate_sq"][:, :, :, 0].sum(axis=2)
            + cost_weights.steering_rate
            * snapshots["feature_action_rate_sq"][:, :, :, 1].sum(axis=2)
        )
        raw_feature_errors.append(
            float(np.max(np.abs(feature_cost - snapshots["cost"])))
        )
        for index, step in enumerate(snapshots["control_step"]):
            optimized_next_errors.append(
                float(
                    np.max(
                        np.abs(
                            snapshots["optimized_trajectory"][index, 0]
                            - trace["next_state"][int(step)]
                        )
                    )
                )
            )

        z = np.abs(
            (snapshots["history"][:, :, :5].astype(np.float64) - history_mean)
            / history_std
        )
        history_abs_z.append(z)
        history_tail_fraction.extend(np.mean(z > 3.0, axis=(1, 2)).tolist())
        state_z = np.abs(
            (snapshots["state"][:, 3:5].astype(np.float64) - context_mean[:2])
            / context_std[:2]
        )
        current_state_abs_z.append(state_z)
        speed_error = (
            snapshots["state"][:, 3] - snapshots["reference"][:, 0, 3]
        ) * 3.6
        position_error = np.linalg.norm(
            snapshots["state"][:, :2] - snapshots["reference"][:, 0, :2],
            axis=1,
        )
        snapshot_speed_error_kph.append(speed_error)
        snapshot_position_error_m.append(position_error)
        episode_max_abs_speed_error_kph.append(float(np.max(np.abs(speed_error))))
        episode_max_position_error_m.append(float(np.max(position_error)))
        context = np.stack(
            (
                np.broadcast_to(snapshots["state"][:, None, 3], snapshots["cost"].shape),
                np.broadcast_to(snapshots["state"][:, None, 4], snapshots["cost"].shape),
                snapshots["sampled_action_sequences"][:, :, 0, 0],
                np.broadcast_to(
                    snapshots["current_action"][:, None, 1], snapshots["cost"].shape
                ),
            ),
            axis=-1,
        )
        candidate_context_abs_z.append(
            np.abs((context - context_mean) / context_std)
        )

    replay_trajectory_errors = []
    replay_cost_errors = []
    first_state_future_leak_errors = []
    replay_count = min(max(args.replay_snapshots, 0), len(snapshot_records))
    if replay_count:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        model = QueryDeploymentModel.from_checkpoint(checkpoint_path, args.device)
        controller = TorchMPPIController(
            TorchQueryRolloutBackend(model), params, device=args.device
        )
        chosen = np.linspace(0, len(snapshot_records) - 1, replay_count, dtype=int)
        for record_index in chosen:
            episode_dir, index = snapshot_records[int(record_index)]
            snapshots = np.load(episode_dir / "snapshots.npz", allow_pickle=False)
            result = controller.evaluate_action_sequences(
                snapshots["state"][index],
                snapshots["current_action"][index],
                snapshots["history"][index:index + 1],
                snapshots["reference"][index],
                snapshots["sampled_action_sequences"][index],
            )
            replay_trajectory_errors.append(
                float(
                    np.max(
                        np.abs(
                            result["trajectories"].cpu().numpy()
                            - snapshots["predicted_trajectories"][index]
                        )
                    )
                )
            )
            replay_cost_errors.append(
                float(
                    np.max(
                        np.abs(
                            result["cost"].cpu().numpy()
                            - snapshots["cost"][index]
                        )
                    )
                )
            )
            original = snapshots["optimized_action_sequence"][index].copy()
            changed = original.copy()
            changed[1:] = np.clip(-changed[1:] + 0.137, -1.0, 1.0)
            pair = controller.evaluate_action_sequences(
                snapshots["state"][index],
                snapshots["current_action"][index],
                snapshots["history"][index:index + 1],
                snapshots["reference"][index],
                np.stack((original, changed)),
            )
            first_state_future_leak_errors.append(
                float(
                    np.max(
                        np.abs(
                            pair["trajectories"][0, 0].cpu().numpy()
                            - pair["trajectories"][1, 0].cpu().numpy()
                        )
                    )
                )
            )

    history_z = np.concatenate(history_abs_z)
    current_z = np.concatenate(current_state_abs_z)
    candidate_z = np.concatenate(candidate_context_abs_z)
    speed_error_kph = np.concatenate(snapshot_speed_error_kph)
    position_error_m = np.concatenate(snapshot_position_error_m)
    reference_ratio = np.concatenate(reference_progress_ratio)
    checks = {
        "artifact_hash": maximum(hash_errors) == 0.0,
        "shape": maximum(shape_errors) == 0.0,
        "finite": maximum(finite_errors) == 0.0,
        "state_chain": maximum(state_chain_errors) <= 1e-6,
        "action_chain": maximum(action_chain_errors) <= 1e-7,
        "cold_start_is_current_hold": maximum(warm_cold_errors) <= 1e-7,
        "warm_is_strictly_receded": maximum(warm_recede_errors) <= 1e-7,
        "snapshot_matches_trace": maximum(snapshot_trace_errors) <= 1e-7,
        "history_is_causally_reconstructed": maximum(history_reconstruction_errors) <= 1e-6,
        "reference_arc_length_progress": float(np.max(np.abs(reference_ratio - 1.0))) <= 2e-3,
        "weights_normalized": maximum(weight_sum_errors) <= 2e-6,
        "candidate_zero_retains_warm": maximum(candidate_zero_errors) <= 1e-7,
        "cost_components_replay": maximum(cost_replay_errors) <= 0.25,
        "raw_features_replay": maximum(raw_feature_errors) <= 0.25,
        "optimized_next_state": maximum(optimized_next_errors) <= 1e-7,
        "history_domain_tail": max(history_tail_fraction, default=0.0) <= 0.05,
        "current_state_domain": float(np.max(current_z)) <= 3.0,
        "candidate_context_domain": float(np.quantile(candidate_z, 0.99)) <= 3.0,
        "snapshot_speed_tracks_requested_bin": maximum(
            episode_max_abs_speed_error_kph
        ) <= 5.0,
        "snapshot_position_tracks_desired_road": maximum(
            episode_max_position_error_m
        ) <= 3.0,
        "query_replay": maximum(replay_trajectory_errors) <= 5e-5
        and maximum(replay_cost_errors) <= 0.5,
        "first_state_has_no_future_action_leak": maximum(first_state_future_leak_errors) <= 5e-5,
        "episode_distribution_contract": all(episode_distribution_checks.values()),
    }
    passed = all(checks.values())
    output = {
        "qualification": (
            "QUERY_EXPECTED_ROAD_CLOSED_LOOP_PASS"
            if passed
            else "QUERY_EXPECTED_ROAD_CLOSED_LOOP_FAIL"
        ),
        "collection": str(root),
        "manifest_sha256": sha256(manifest_path),
        "checks": checks,
        "maximum_errors": {
            "state_chain": maximum(state_chain_errors),
            "action_chain": maximum(action_chain_errors),
            "warm_cold": maximum(warm_cold_errors),
            "warm_recede": maximum(warm_recede_errors),
            "snapshot_trace": maximum(snapshot_trace_errors),
            "history_reconstruction": maximum(history_reconstruction_errors),
            "reference_progress_ratio": float(np.max(np.abs(reference_ratio - 1.0))),
            "weight_sum": maximum(weight_sum_errors),
            "candidate_zero": maximum(candidate_zero_errors),
            "cost_component_replay": maximum(cost_replay_errors),
            "raw_feature_replay": maximum(raw_feature_errors),
            "optimized_next": maximum(optimized_next_errors),
            "query_trajectory_replay": maximum(replay_trajectory_errors),
            "query_cost_replay": maximum(replay_cost_errors),
            "first_state_future_action_leak": maximum(first_state_future_leak_errors),
        },
        "domain": {
            "history_abs_z": stats(history_z),
            "history_fraction_abs_z_gt_3": stats(np.asarray(history_tail_fraction)),
            "current_vx_yawrate_abs_z": stats(current_z),
            "candidate_context_abs_z": stats(candidate_z),
            "snapshot_speed_error_kph": stats(speed_error_kph),
            "snapshot_position_error_m": stats(position_error_m),
            "episode_max_abs_speed_error_kph": stats(
                np.asarray(episode_max_abs_speed_error_kph)
            ),
            "episode_max_position_error_m": stats(
                np.asarray(episode_max_position_error_m)
            ),
        },
        "replayed_snapshot_count": replay_count,
        "episode_distribution_checks": episode_distribution_checks,
        "episode_count": len(episodes),
        "repeats_per_cell": repeats_per_cell,
        "formal_validation_or_test_consumed": False,
    }
    output_path = args.output or (root / "validation.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
