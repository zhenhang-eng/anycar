#!/usr/bin/env python3
"""Validate cost-relabelable MPPI closed-loop snapshot episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = {
    "format_version",
    "control_step",
    "initial_state",
    "initial_state_six",
    "current_action",
    "history",
    "reference",
    "reference_ego",
    "frenet_pose",
    "mean_knots_before",
    "mean_knots_after",
    "sampling_mean_knots",
    "sampling_noise_knots",
    "raw_sampled_knots",
    "sampled_knots",
    "sampled_knots_clipped",
    "sampled_action_sequences",
    "predicted_trajectories",
    "predicted_trajectories_full",
    "feature_position_error_sq",
    "feature_yaw_error_sq",
    "feature_vx_error_sq",
    "feature_action_rate_sq",
    "cost",
    "weight",
    "optimized_action",
    "optimized_action_sequence",
    "simulator_metadata_json",
    "mppi_params_json",
    "cost_weights_json",
    "dbm_params_json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "episode_dir", type=Path, help="Episode directory containing manifest.json"
    )
    return parser.parse_args()


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.allclose(actual, expected, rtol=2e-4, atol=2e-5):
        maximum = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name} mismatch; maximum absolute error={maximum}")


def prepare_reference(reference: np.ndarray, horizon: int) -> np.ndarray:
    if len(reference) == horizon + 1:
        return reference[1:]
    if len(reference) != horizon:
        raise AssertionError(
            f"reference length {len(reference)} does not match horizon {horizon}"
        )
    return reference


def validate_snapshot(path: Path, manifest: dict) -> dict:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_ARRAYS.difference(data.files))
        if missing:
            raise AssertionError(f"{path}: missing arrays {missing}")
        if int(data["format_version"]) != 2:
            raise AssertionError(f"{path}: unsupported snapshot format")

        params = json.loads(str(data["mppi_params_json"]))
        weights = json.loads(str(data["cost_weights_json"]))
        simulator = json.loads(str(data["simulator_metadata_json"]))
        dbm_params = json.loads(str(data["dbm_params_json"]))
        if int(manifest.get("format_version", 1)) >= 2:
            for name in ("history_valid_steps", "history_is_fully_observed"):
                if name not in data.files:
                    raise AssertionError(f"{path}: missing schema-v2 array {name}")
            expected_valid_steps = min(
                int(data["control_step"]), params["history_length"]
            )
            if int(data["history_valid_steps"]) != expected_valid_steps:
                raise AssertionError(f"{path}: history_valid_steps mismatch")
            if bool(data["history_is_fully_observed"]) != (
                expected_valid_steps == params["history_length"]
            ):
                raise AssertionError(
                    f"{path}: history_is_fully_observed mismatch"
                )
        if not simulator:
            raise AssertionError(f"{path}: simulator metadata is empty")
        if simulator != manifest["simulator"]:
            raise AssertionError(f"{path}: simulator parameters differ from manifest")
        if dbm_params != manifest["rollout_model"]["dbm_params"]:
            raise AssertionError(f"{path}: DBM parameters differ from manifest")

        count = params["num_samples"]
        horizon = params["horizon"]
        knots = params["num_knots"]
        expected_shapes = {
            "initial_state": (5,),
            "initial_state_six": (6,),
            "current_action": (2,),
            "history": (1, params["history_length"], 7),
            "frenet_pose": (3,),
            "sampling_mean_knots": (knots, 2),
            "sampling_noise_knots": (count, knots, 2),
            "raw_sampled_knots": (count, knots, 2),
            "sampled_knots": (count, knots, 2),
            "sampled_action_sequences": (count, horizon, 2),
            "predicted_trajectories": (count, horizon, 5),
            "predicted_trajectories_full": (count, horizon, 6),
            "feature_position_error_sq": (count, horizon),
            "feature_yaw_error_sq": (count, horizon),
            "feature_vx_error_sq": (count, horizon),
            "feature_action_rate_sq": (count, horizon, 2),
            "cost": (count,),
            "weight": (count,),
            "optimized_action_sequence": (horizon, 2),
        }
        for name, expected_shape in expected_shapes.items():
            if data[name].shape != expected_shape:
                raise AssertionError(
                    f"{path}: {name} shape {data[name].shape}, "
                    f"expected {expected_shape}"
                )
            if not np.isfinite(data[name]).all():
                raise AssertionError(f"{path}: {name} contains NaN/Inf")
        for name in ("reference", "reference_ego"):
            valid_length = data[name].shape[0] in (horizon, horizon + 1)
            valid_width = data[name].shape[1] in (4, 5)
            if not (valid_length and valid_width):
                raise AssertionError(
                    f"{path}: {name} shape {data[name].shape} is invalid"
                )
            if not np.isfinite(data[name]).all():
                raise AssertionError(f"{path}: {name} contains NaN/Inf")

        if "reference_speed_override_mps" in data.files:
            speed_override = float(data["reference_speed_override_mps"])
            manifest_override = float(
                manifest["collection"]["reference_speed_override_mps"]
            )
            if not np.isclose(speed_override, manifest_override, atol=1e-6):
                raise AssertionError(
                    f"{path}: reference speed override differs from manifest"
                )
            if speed_override >= 0.0 and not np.allclose(
                data["reference"][:, 3], speed_override, atol=1e-5
            ):
                raise AssertionError(
                    f"{path}: reference does not use the configured speed override"
                )
            if speed_override > 0.0:
                progress_speed = float(
                    np.linalg.norm(
                        np.diff(data["reference"][:, :2], axis=0), axis=1
                    ).mean()
                    / float(params["dt"])
                )
                if not np.isclose(
                    progress_speed,
                    speed_override,
                    # The planner's periodic spline parameterization is only
                    # approximately arc length, so Cartesian progress varies
                    # by track phase even with a constant ds/dt command.
                    rtol=0.20,
                    atol=0.05,
                ):
                    raise AssertionError(
                        f"{path}: reference position progression implies "
                        f"{progress_speed:.3f} m/s, expected {speed_override:.3f}"
                    )

        assert_close(
            "raw knots",
            data["raw_sampled_knots"],
            data["sampling_mean_knots"][None] + data["sampling_noise_knots"],
        )
        clipped = np.clip(
            data["raw_sampled_knots"],
            np.asarray(params["action_min"]),
            np.asarray(params["action_max"]),
        )
        assert_close("clipped knots", data["sampled_knots"], clipped)
        expected_clip_mask = np.any(
            data["raw_sampled_knots"] != data["sampled_knots"], axis=-1
        )
        if not np.array_equal(data["sampled_knots_clipped"], expected_clip_mask):
            raise AssertionError(f"{path}: knot clipping mask mismatch")

        trajectory = data["predicted_trajectories"]
        full_trajectory = data["predicted_trajectories_full"]
        assert_close(
            "five/six-state trajectory projection",
            trajectory,
            full_trajectory[..., [0, 1, 2, 3, 5]],
        )
        reference = prepare_reference(data["reference"], horizon)
        position_error = trajectory[..., :2] - reference[None, :, :2]
        yaw_delta = trajectory[..., 2] - reference[None, :, 2]
        expected_position = np.square(position_error).sum(axis=-1)
        expected_yaw = np.square(
            np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
        )
        expected_vx = np.square(trajectory[..., 3] - reference[None, :, 3])
        actions = data["sampled_action_sequences"]
        previous = np.concatenate(
            (
                np.broadcast_to(data["current_action"], (count, 1, 2)),
                actions[:, :-1],
            ),
            axis=1,
        )
        expected_action_rate = np.square(actions - previous)
        assert_close(
            "position feature", data["feature_position_error_sq"], expected_position
        )
        assert_close("yaw feature", data["feature_yaw_error_sq"], expected_yaw)
        assert_close("vx feature", data["feature_vx_error_sq"], expected_vx)
        assert_close(
            "action-rate feature",
            data["feature_action_rate_sq"],
            expected_action_rate,
        )

        expected_components = {
            "position": weights["position"] * expected_position.sum(axis=1),
            "yaw": weights["yaw"] * expected_yaw.sum(axis=1),
            "vx": weights["vx"] * expected_vx.sum(axis=1),
            "acceleration_rate": weights["acceleration_rate"]
            * expected_action_rate[..., 0].sum(axis=1),
            "steering_rate": weights["steering_rate"]
            * expected_action_rate[..., 1].sum(axis=1),
        }
        for name, expected in expected_components.items():
            assert_close(f"cost_{name}", data[f"cost_{name}"], expected)
        expected_total = sum(expected_components.values())
        assert_close("total cost", data["cost"], expected_total)
        if not np.isclose(float(data["weight"].sum()), 1.0, atol=2e-5):
            raise AssertionError(f"{path}: MPPI weights do not sum to one")

        return {
            "step": int(data["control_step"]),
            "state": data["initial_state_six"].copy(),
            "current_action": data["current_action"].copy(),
            "candidate_count": count,
            "clip_fraction": float(expected_clip_mask.mean()),
            "best_cost": float(data["cost"].min()),
            "median_cost": float(np.median(data["cost"])),
        }


def validate_track(episode_dir: Path, manifest: dict) -> None:
    track = manifest.get("track")
    if not isinstance(track, dict):
        raise AssertionError("schema-v2 manifest track metadata must be an object")
    artifact = episode_dir / track.get("artifact", "")
    with np.load(artifact, allow_pickle=False) as data:
        if set(data.files) != {"source_track", "planner_waypoints"}:
            raise AssertionError(f"{artifact}: unexpected track artifact arrays")
        source_track = data["source_track"]
        planner_waypoints = data["planner_waypoints"]
        if source_track.ndim != 2 or source_track.shape[1] < 4:
            raise AssertionError(f"{artifact}: source_track shape is invalid")
        if planner_waypoints.ndim != 2 or planner_waypoints.shape[1] < 18:
            raise AssertionError(
                f"{artifact}: planner_waypoints shape is invalid"
            )
        if not np.isfinite(source_track).all() or not np.isfinite(
            planner_waypoints
        ).all():
            raise AssertionError(f"{artifact}: track data contains NaN/Inf")

    source_path = Path(track.get("source_path", ""))
    if source_path.is_file():
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != track.get("source_sha256"):
            raise AssertionError("source track SHA256 differs from manifest")


def validate_trace(episode_dir: Path, manifest: dict, results: list[dict]) -> int:
    artifact = manifest.get("artifacts", {}).get("closed_loop_trace")
    if not artifact:
        raise AssertionError("schema-v2 manifest omits closed-loop trace artifact")
    trace_path = episode_dir / artifact
    with trace_path.open() as stream:
        trace = [json.loads(line) for line in stream if line.strip()]
    if not trace:
        raise AssertionError(f"{trace_path}: trace is empty")

    steps = [int(record["control_step"]) for record in trace]
    expected_start = int(
        manifest.get("collection", {}).get(
            "trace_start_step", results[0]["step"]
        )
    )
    expected_steps = list(range(expected_start, results[-1]["step"] + 1))
    if steps != expected_steps:
        raise AssertionError(
            f"{trace_path}: expected continuous steps "
            f"{expected_steps[0]}..{expected_steps[-1]}, got "
            f"{steps[0]}..{steps[-1]} ({len(steps)} records)"
        )

    snapshots_by_step = {result["step"]: result for result in results}
    params = manifest["mppi_params"]
    previous_executed_action = None
    for record in trace:
        step = int(record["control_step"])
        for name, width in (
            ("state", 6),
            ("current_action", 2),
            ("controller_action", 2),
            ("executed_action", 2),
            ("frenet_pose", 3),
        ):
            value = np.asarray(record[name])
            if value.shape != (width,) or not np.isfinite(value).all():
                raise AssertionError(
                    f"{trace_path}: step {step} has invalid {name}"
                )
        if not np.isclose(
            float(record["simulated_time_s"]), step * params["dt"]
        ):
            raise AssertionError(
                f"{trace_path}: step {step} simulated_time_s mismatch"
            )
        expected_history_steps = min(step, params["history_length"])
        if int(record["history_valid_steps"]) != expected_history_steps:
            raise AssertionError(
                f"{trace_path}: step {step} history_valid_steps mismatch"
            )
        if previous_executed_action is not None:
            assert_close(
                f"trace action link at step {step}",
                np.asarray(record["current_action"]),
                previous_executed_action,
            )
        previous_executed_action = np.asarray(record["executed_action"])
        reference = np.asarray(record["reference"])
        if (
            reference.ndim != 2
            or reference.shape[0] not in (
                params["horizon"], params["horizon"] + 1
            )
            or reference.shape[1] not in (4, 5)
            or not np.isfinite(reference).all()
        ):
            raise AssertionError(
                f"{trace_path}: step {step} has invalid reference"
            )
        for name, shape in (
            ("mean_knots_before", (params["num_knots"], 2)),
            ("mean_knots_after", (params["num_knots"], 2)),
            ("optimized_action_sequence", (params["horizon"], 2)),
        ):
            value = np.asarray(record[name])
            if value.shape != shape or not np.isfinite(value).all():
                raise AssertionError(
                    f"{trace_path}: step {step} has invalid {name}"
                )
        cost_summary = np.asarray(
            list(record["cost_summary_at_collection"].values()), dtype=float
        )
        if not np.isfinite(cost_summary).all():
            raise AssertionError(
                f"{trace_path}: step {step} has invalid cost summary"
            )
        if not np.isfinite(record["controller_duration_s"]) or float(
            record["controller_duration_s"]
        ) < 0:
            raise AssertionError(
                f"{trace_path}: step {step} has invalid controller duration"
            )
        snapshot = snapshots_by_step.get(step)
        if snapshot is not None:
            assert_close(
                f"trace state at step {step}",
                np.asarray(record["state"]),
                snapshot["state"],
            )
            assert_close(
                f"trace current action at step {step}",
                np.asarray(record["current_action"]),
                snapshot["current_action"],
            )
    return len(trace)


def main() -> None:
    args = parse_args()
    episode_dir = args.episode_dir.expanduser().resolve()
    manifest_path = episode_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("fixed_dbm_parameters"):
        raise AssertionError("manifest does not declare fixed DBM parameters")
    records = manifest.get("snapshots", [])
    if len(records) != manifest.get("snapshot_count"):
        raise AssertionError("manifest snapshot count mismatch")
    if not records:
        raise AssertionError("episode contains no snapshots")

    results = [
        validate_snapshot(episode_dir / record["snapshot"], manifest)
        for record in records
    ]
    steps = [result["step"] for result in results]
    if steps != sorted(set(steps)):
        raise AssertionError("snapshot steps must be unique and increasing")

    trace_count = None
    if int(manifest.get("format_version", 1)) >= 2:
        validate_track(episode_dir, manifest)
        trace_count = validate_trace(episode_dir, manifest, results)

    output = {
        "status": "ok",
        "episode_dir": str(episode_dir),
        "snapshot_count": len(results),
        "first_step": steps[0],
        "last_step": steps[-1],
        "candidate_rollout_count": int(
            sum(result["candidate_count"] for result in results)
        ),
        "closed_loop_trace_count": trace_count,
        "mean_knot_clip_fraction": float(
            np.mean([result["clip_fraction"] for result in results])
        ),
        "best_cost_range": [
            float(min(result["best_cost"] for result in results)),
            float(max(result["best_cost"] for result in results)),
        ],
        "median_cost_range": [
            float(min(result["median_cost"] for result in results)),
            float(max(result["median_cost"] for result in results)),
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
