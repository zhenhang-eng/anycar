#!/usr/bin/env python3
"""Compare aligned Query-MPPI and DBM-MPPI Quick Start ROS bags."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPICS = (
    "/odometry",
    "/odometry_ground_truth",
    "/ackermann_command",
    "/lateral_error",
    "/mppi_time",
)


def read_bag(path: Path) -> dict[str, list]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    output = {topic: [] for topic in TOPICS}
    while reader.has_next():
        topic, payload, timestamp = reader.read_next()
        if topic not in output:
            continue
        message = deserialize_message(payload, get_message(topic_types[topic]))
        output[topic].append((timestamp * 1e-9, message))
    return output


def yaw_from_odometry(message) -> float:
    quaternion = message.pose.pose.orientation
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2),
    )


def arrays_from_bag(
    records: dict[str, list], state_topic: str = "/odometry"
) -> dict[str, np.ndarray]:
    odometry = records[state_topic]
    states = np.asarray(
        [
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                yaw_from_odometry(message),
                message.twist.twist.linear.x,
                message.twist.twist.linear.y,
                message.twist.twist.angular.z,
            ]
            for _, message in odometry
        ],
        dtype=np.float64,
    )
    actions = np.asarray(
        [
            [message.drive.speed, message.drive.steering_angle]
            for _, message in records["/ackermann_command"]
        ],
        dtype=np.float64,
    )
    return {
        "state": states,
        "state_time": np.asarray([time for time, _ in odometry]),
        "action": actions,
        "lateral_error": np.asarray(
            [message.data for _, message in records["/lateral_error"]],
            dtype=np.float64,
        ),
        "mppi_time": np.asarray(
            [message.data for _, message in records["/mppi_time"]],
            dtype=np.float64,
        ),
    }


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def controller_metrics(data: dict[str, np.ndarray], steps: int, deadline: float) -> dict:
    state = data["state"][: steps + 1]
    action = data["action"][:steps]
    lateral = data["lateral_error"][:steps]
    timing = data["mppi_time"][:steps]
    abs_lateral = np.abs(lateral)
    longitudinal_speed_error = state[:-1, 3] - 2.0
    scalar_speed = np.linalg.norm(state[:-1, 3:5], axis=1)
    scalar_speed_error = scalar_speed - 2.0
    increments = np.linalg.norm(np.diff(state[:, :2], axis=0), axis=1)
    action_delta = np.diff(np.vstack((np.zeros((1, 2)), action)), axis=0)
    return {
        "steps": steps,
        "lateral_error": {
            "mae_m": float(abs_lateral.mean()),
            "rmse_m": float(np.sqrt(np.mean(lateral**2))),
            "p95_abs_m": percentile(abs_lateral, 95),
            "max_abs_m": float(abs_lateral.max()),
            "final_abs_m": float(abs_lateral[-1]),
        },
        "longitudinal_speed_error": {
            "mae_mps": float(np.abs(longitudinal_speed_error).mean()),
            "rmse_mps": float(np.sqrt(np.mean(longitudinal_speed_error**2))),
            "final_mps": float(state[steps, 3]),
        },
        "scalar_speed_error": {
            "mae_mps": float(np.abs(scalar_speed_error).mean()),
            "rmse_mps": float(np.sqrt(np.mean(scalar_speed_error**2))),
            "final_mps": float(np.linalg.norm(state[steps, 3:5])),
        },
        "distance_travelled_m": float(increments.sum()),
        "net_displacement_m": float(np.linalg.norm(state[-1, :2] - state[0, :2])),
        "mppi_time": {
            "mean_ms": float(timing.mean() * 1000.0),
            "median_ms": percentile(timing * 1000.0, 50),
            "p95_ms": percentile(timing * 1000.0, 95),
            "max_ms": float(timing.max() * 1000.0),
            "deadline_miss_rate": float(np.mean(timing > deadline)),
        },
        "action": {
            "acceleration_mean": float(action[:, 0].mean()),
            "acceleration_abs_mean": float(np.abs(action[:, 0]).mean()),
            "steering_mean": float(action[:, 1].mean()),
            "steering_abs_mean": float(np.abs(action[:, 1]).mean()),
            "acceleration_rate_abs_mean": float(np.abs(action_delta[:, 0]).mean()),
            "steering_rate_abs_mean": float(np.abs(action_delta[:, 1]).mean()),
            "saturation_rate": float(np.mean(np.abs(action) >= 0.999)),
        },
        "initial_state": state[0].tolist(),
        "final_state": state[-1].tolist(),
        "all_finite": bool(
            np.isfinite(state).all()
            and np.isfinite(action).all()
            and np.isfinite(lateral).all()
            and np.isfinite(timing).all()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-bag", type=Path, required=True)
    parser.add_argument("--dbm-bag", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-dt", type=float, default=0.02)
    parser.add_argument("--model-dt", type=float, default=0.05)
    args = parser.parse_args()

    query = arrays_from_bag(read_bag(args.query_bag))
    dbm = arrays_from_bag(read_bag(args.dbm_bag))
    steps = min(
        len(query["state"]) - 1,
        len(query["action"]),
        len(query["lateral_error"]),
        len(query["mppi_time"]),
        len(dbm["state"]) - 1,
        len(dbm["action"]),
        len(dbm["lateral_error"]),
        len(dbm["mppi_time"]),
    )
    if steps < 1:
        raise RuntimeError("bags do not contain an aligned control transition")

    query_state = query["state"][: steps + 1]
    dbm_state = dbm["state"][: steps + 1]
    position_difference = np.linalg.norm(
        query_state[:, :2] - dbm_state[:, :2], axis=1
    )
    yaw_difference = np.arctan2(
        np.sin(query_state[:, 2] - dbm_state[:, 2]),
        np.cos(query_state[:, 2] - dbm_state[:, 2]),
    )
    summary = {
        "protocol": {
            "aligned_steps": steps,
            "simulated_duration_s": steps * args.sim_dt,
            "simulator_dt_s": args.sim_dt,
            "mppi_model_dt_s": args.model_dt,
            "timebase_aligned": bool(args.sim_dt == args.model_dt),
            "target_speed_mps": 2.0,
        },
        "query_pytorch": controller_metrics(query, steps, args.model_dt),
        "dbm_torch": controller_metrics(dbm, steps, args.model_dt),
        "closed_loop_difference": {
            "position_rmse_m": float(np.sqrt(np.mean(position_difference**2))),
            "position_p95_m": percentile(position_difference, 95),
            "final_position_separation_m": float(position_difference[-1]),
            "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_difference**2))),
            "final_yaw_difference_rad": float(abs(yaw_difference[-1])),
            "action_rmse": np.sqrt(
                np.mean((query["action"][:steps] - dbm["action"][:steps]) ** 2, axis=0)
            ).tolist(),
        },
    }
    query_lateral = summary["query_pytorch"]["lateral_error"]
    dbm_lateral = summary["dbm_torch"]["lateral_error"]
    summary["query_minus_dbm"] = {
        "lateral_mae_m": query_lateral["mae_m"] - dbm_lateral["mae_m"],
        "lateral_rmse_m": query_lateral["rmse_m"] - dbm_lateral["rmse_m"],
        "lateral_mae_ratio": (
            query_lateral["mae_m"] / dbm_lateral["mae_m"]
            if dbm_lateral["mae_m"] > 0
            else None
        ),
        "mppi_mean_ms": (
            summary["query_pytorch"]["mppi_time"]["mean_ms"]
            - summary["dbm_torch"]["mppi_time"]["mean_ms"]
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = args.output_dir / "aligned_steps.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "sim_time_s",
                "query_x",
                "query_y",
                "query_lateral_error",
                "query_acceleration",
                "query_steering",
                "query_mppi_time_s",
                "dbm_x",
                "dbm_y",
                "dbm_lateral_error",
                "dbm_acceleration",
                "dbm_steering",
                "dbm_mppi_time_s",
                "position_separation_m",
            ]
        )
        for step in range(steps):
            writer.writerow(
                [
                    step,
                    step * args.sim_dt,
                    *query["state"][step, :2],
                    query["lateral_error"][step],
                    *query["action"][step],
                    query["mppi_time"][step],
                    *dbm["state"][step, :2],
                    dbm["lateral_error"][step],
                    *dbm["action"][step],
                    dbm["mppi_time"][step],
                    position_difference[step],
                ]
            )
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
