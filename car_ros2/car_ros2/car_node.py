from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path as FilePath
import subprocess
import torch
from termcolor import colored
import rclpy
from rclpy.node import Node
import time

from car_ros2 import CAR_ROS2_TMP
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32, Pose
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
from visualization_msgs.msg import MarkerArray, Marker
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Joy

from tf_transformations import quaternion_from_euler, euler_matrix, euler_from_quaternion

from car_planner import CAR_PLANNER_ASSETS_DIR
from car_dynamics.controllers_torch import (
    PurePersuitParams,
    PurePersuitController,
    TorchMPPIController,
    TorchMPPIParams,
)
from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_planner.global_trajectory import GlobalTrajectory, generate_circle_trajectory, generate_oval_trajectory, generate_rectangle_trajectory, generate_raceline_trajectory
from car_foundation.query_deployment import (
    OnnxQueryRolloutBackend,
    QueryDeploymentModel,
    QueryHistoryBuffer,
    TorchQueryRolloutBackend,
)
import numpy as np
import tf2_geometry_msgs
import datetime


print("DEVICE", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

import threading
from multiprocessing.pool import ThreadPool


unique_prefix = datetime.datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:-3]

SPEED = 1.
SAFE_SPEED_MAX = 10.0

# TELEOP = True
TELEOP = False
USE_KEYBOARD = False


if USE_KEYBOARD:
    from pynput import keyboard

import os
os.environ["OMP_NUM_THREADS"] = "1"


@dataclass(frozen=True)
class QueryVehicleParams:
    """Only the geometry/timing values used by the Query-model ROS node."""

    LF: float = 1.95
    LR: float = 1.95
    DT: float = 0.05

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QueryRuntimeEnvironment:
    name: str = "query-model"
    mass: float = 0.0
    friction: float = 0.0
    delay: int = 0


class CarNode(Node):
    def __init__(self):
        super().__init__('car_node')
        
        print("Car node start")
        self.env_params = QueryRuntimeEnvironment()
        self.model_params = QueryVehicleParams()
        self.control_dt = 0.05

        # NOTE: Can choose either 'mppi' or 'pure_persuit'
        # self.controller_type = 'pure_persuit'
        self.controller_type = 'mppi'
        
        self._counter = 0
        if self.controller_type == 'mppi':
            repository_root = os.environ.get("CAR_PATH", "/home/plusai/anycar")
            default_checkpoint = os.path.join(
                repository_root,
                "outputs/formal_real_finetune_query_baseline_split/"
                "20260728T143256/query_best.pt",
            )
            default_onnx = os.path.join(
                repository_root, "outputs/query_mppi/anycar_query.onnx"
            )
            backend_name = self.declare_parameter(
                "mppi_backend", "pytorch"
            ).value
            checkpoint_path = self.declare_parameter(
                "query_checkpoint", default_checkpoint
            ).value
            onnx_path = self.declare_parameter(
                "query_onnx_path", default_onnx
            ).value
            self.mppi_snapshot_step = int(
                self.declare_parameter("mppi_snapshot_step", -1).value
            )
            self.mppi_snapshot_dir = str(
                self.declare_parameter("mppi_snapshot_dir", "").value
            )
            self.mppi_snapshot_written = False
            self.mppi_dataset_dir = str(
                self.declare_parameter("mppi_dataset_dir", "").value
            )
            self.mppi_dataset_start_step = int(
                self.declare_parameter("mppi_dataset_start_step", -1).value
            )
            self.mppi_dataset_stop_step = int(
                self.declare_parameter("mppi_dataset_stop_step", -1).value
            )
            self.mppi_dataset_stride = int(
                self.declare_parameter("mppi_dataset_stride", 10).value
            )
            self.mppi_dataset_max_snapshots = int(
                self.declare_parameter("mppi_dataset_max_snapshots", 0).value
            )
            self.mppi_dataset_shutdown_on_complete = bool(
                self.declare_parameter(
                    "mppi_dataset_shutdown_on_complete", False
                ).value
            )
            requested_episode_id = str(
                self.declare_parameter("mppi_dataset_episode_id", "").value
            ).strip()
            self.mppi_dataset_episode_id = requested_episode_id or unique_prefix
            if self.mppi_dataset_stride < 1:
                raise ValueError("mppi_dataset_stride must be positive")
            if self.mppi_dataset_max_snapshots < 0:
                raise ValueError("mppi_dataset_max_snapshots cannot be negative")
            self.mppi_dataset_snapshot_count = 0
            self.mppi_dataset_records = []
            self.mppi_dataset_shutdown_requested = False
            self.simulator_metadata = None
            self.mppi_backend_name = backend_name
            self.query_checkpoint_path = checkpoint_path
            num_samples = int(
                self.declare_parameter("mppi_num_samples", 256).value
            )
            num_iterations = int(
                self.declare_parameter("mppi_num_iterations", 1).value
            )
            mppi_seed = int(
                self.declare_parameter("mppi_seed", 3407).value
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if backend_name == "pytorch":
                deployment_model = QueryDeploymentModel.from_checkpoint(
                    checkpoint_path, device
                )
                rollout_backend = TorchQueryRolloutBackend(deployment_model)
                self.control_dt = deployment_model.dt
                self.model_params = QueryVehicleParams(
                    LF=0.5 * deployment_model.wheelbase,
                    LR=0.5 * deployment_model.wheelbase,
                    DT=deployment_model.dt,
                )
            elif backend_name == "onnx":
                if not os.path.isfile(onnx_path):
                    raise FileNotFoundError(
                        f"Query ONNX model not found: {onnx_path}. "
                        "Export it before starting car_node."
                    )
                rollout_backend = OnnxQueryRolloutBackend(
                    onnx_path, provider="cuda", output_device=device
                )
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                query_params = checkpoint["params"]
                self.control_dt = float(query_params["dt"])
                wheelbase = float(query_params["wheelbase"])
                self.model_params = QueryVehicleParams(
                    LF=0.5 * wheelbase,
                    LR=0.5 * wheelbase,
                    DT=self.control_dt,
                )
            elif backend_name == "dbm":
                rollout_backend = TorchDynamicBicycleRolloutBackend()
                self.model_params = QueryVehicleParams(
                    LF=0.1008, LR=0.1092, DT=self.control_dt
                )
            else:
                raise ValueError(
                    "mppi_backend must be 'pytorch', 'onnx', or 'dbm', "
                    f"got {backend_name!r}"
                )
            if self.mppi_dataset_dir and backend_name != "dbm":
                raise ValueError(
                    "Closed-loop MPPI dataset collection currently requires "
                    "mppi_backend=dbm so six-state trajectories are available"
                )
            self.mppi_params = TorchMPPIParams(
                num_samples=num_samples,
                num_iterations=num_iterations,
                seed=mppi_seed,
            )
            if abs(self.control_dt - self.mppi_params.dt) > 1e-9:
                raise ValueError(
                    f"Query checkpoint dt={self.control_dt} does not match "
                    f"MPPI dt={self.mppi_params.dt}"
                )
            self.mppi = TorchMPPIController(
                rollout_backend, self.mppi_params, device=device
            )
            self.mppi_running_params = self.mppi.get_init_state()
            self.query_history = QueryHistoryBuffer(dt=self.control_dt)
            self.get_logger().info(
                f"Query MPPI backend={backend_name}, device={device}, "
                f"dt={self.control_dt}, wheelbase="
                f"{self.model_params.LF + self.model_params.LR}, "
                f"samples={num_samples}"
            )
        elif self.controller_type == 'pure_persuit':
            ## Pure pursuit controller
            pure_persuit_params = PurePersuitParams()
            if 'numeric' in self.env_params.name or \
                'mujoco' in self.env_params.name or \
                    'unity' in self.env_params.name or \
                        'isaac' in self.env_params.name:
                pure_persuit_params.mode = 'throttle'
                pure_persuit_params.target_vel = 0.8
                pure_persuit_params.wheelbase = 0.2
                pure_persuit_params.kp = 3.
            self.pure_pursuit = PurePersuitController(pure_persuit_params)

        self.L = self.model_params.LF + self.model_params.LR

        pure_persuit_params = PurePersuitParams()
        if 'numeric' in self.env_params.name or \
                'mujoco' in self.env_params.name or\
                    'unity' in self.env_params.name or \
                        'isaac' in self.env_params.name:
            pure_persuit_params.mode = 'throttle'
            pure_persuit_params.target_vel = 0.8
            pure_persuit_params.wheelbase = 0.2
            pure_persuit_params.kp = 3.
        self.recover_controller = PurePersuitController(pure_persuit_params)

        # Load pre-computed track file
        # Here are three examples of tracks
        # 1. .txt
        # 2. gnerate_fn
        # 3. .csv
        # track = np.loadtxt(os.path.join(CAR_PLANNER_ASSETS_DIR, "math_park_v2.txt"), delimiter=',', skiprows=1)
        # track = generate_oval_trajectory((0., -20.0), 20.0, 20.0, direction=-1)
        self.track_path = os.path.join(CAR_PLANNER_ASSETS_DIR, "cuc_inside.csv")
        track = np.loadtxt(self.track_path, delimiter=',', skiprows=1)
        self.track_array = np.asarray(track, dtype=np.float32)

        self.global_planner = GlobalTrajectory(track)
        self.step_mode_ = self.declare_parameter('step_mode', False).value
        ## ROS2 publishers and subscribers
        self.path_pub_ = self.create_publisher(Path, 'path', 1)
        self.waypoint_list_pub_ = self.create_publisher(Path, 'waypoint_list', 1)
        self.ref_trajectory_pub_ = self.create_publisher(Path, 'ref_trajectory', 1)
        self.pose_pub_ = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
        # if not self.step_mode_:
        #     self.timer_ = self.create_timer(self.control_dt, self.timer_callback)
        self.slow_timer_ = self.create_timer(1.0, self.slow_timer_callback)
        self.throttle_pub_ = self.create_publisher(Float64, 'speed', 1)
        self.steer_pub_ = self.create_publisher(Float64, 'steering', 1)
        self.trajectory_array_pub_ = self.create_publisher(MarkerArray, 'trajectory_array', 1)
        self.body_pub_ = self.create_publisher(PolygonStamped, 'body', 1)
        self.vehicle_cmd_pub_ = self.create_publisher(AckermannDriveStamped, 'ackermann_command', 1)
        self.odom_sub_ = self.create_subscription(Odometry, 'odometry', self.odom_callback, 1)
        self.odom_copy_pub_ = self.create_publisher(Odometry, 'odometry_copy', 1)
        self.action_rate_pub = self.create_publisher(Float64, 'debug/action_rate', 1)
        self.lateral_error_pub_ = self.create_publisher(Float64, 'lateral_error', 1)
        self.ref_vel_pub_ = self.create_publisher(Odometry, 'tracking/ref_vel', 1)
        self.loss_pub_ = self.create_publisher(Float64, 'adapt/loss', 1)
        self.misc_pub_ = self.create_publisher(String, 'misc_message', 1)
        self.mppi_time_pub_ = self.create_publisher(Float64, 'mppi_time', 1)
        self.simulator_metadata_sub_ = self.create_subscription(
            String,
            "simulator_metadata",
            self.simulator_metadata_callback,
            1,
        )
        if TELEOP:
            self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 1)
            self.joy = None
        self.odom = None
        self.prev_action = np.zeros(2)
        self.debug_buffer = dict(timestamp=[], obs=[], action=[], action_canidate=[], sampled_traj=[])

        self.params_pub_list = []
        for param in self.model_params.to_dict().keys():
            self.params_pub_list.append(self.create_publisher(Float64, f"param/{param}", 1))


        # self.is_recover_mode = False
        self.is_recover_mode = False
        self.emergency_stop = False
        if USE_KEYBOARD:
            listener = keyboard.Listener(on_press=self.on_press_key)
            listener.start()

        if not self.step_mode_:
            timer_thread = threading.Thread(target=self.timer_thread_fn)
            timer_thread.start()


    def on_press_key(self, key):
        # print(key, type(key), dir(key), key.char)
        if hasattr(key, 'char') and key.char == 'r':
            self.is_recover_mode = not self.is_recover_mode
            print(colored(f"[INFO] Recover mode: {self.is_recover_mode}", "blue"))
        if hasattr(key, 'char') and key.char == 'q':
            self.emergency_stop = not self.emergency_stop
            if self.emergency_stop:
                print(colored(f"[INFO] Emergency stop", "red"))


    def timer_thread_fn(self):
        while True:
            self.timer_callback()

    def simulator_metadata_callback(self, msg):
        try:
            self.simulator_metadata = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid simulator metadata JSON: {exc}")

    def _dataset_snapshot_due(self):
        if not self.mppi_dataset_dir:
            return False
        start_step = max(0, self.mppi_dataset_start_step)
        if self._counter < start_step:
            return False
        if (
            self.mppi_dataset_stop_step >= 0
            and self._counter > self.mppi_dataset_stop_step
        ):
            return False
        if (self._counter - start_step) % self.mppi_dataset_stride != 0:
            return False
        return not (
            self.mppi_dataset_max_snapshots
            and self.mppi_dataset_snapshot_count
            >= self.mppi_dataset_max_snapshots
        )

    def _dataset_collection_complete(self):
        return bool(
            self.mppi_dataset_dir
            and self.mppi_dataset_max_snapshots
            and self.mppi_dataset_snapshot_count
            >= self.mppi_dataset_max_snapshots
        )

    @staticmethod
    def _shutdown_after_dataset_callback():
        # rclpy.shutdown() waits for active executor callbacks. Calling it from
        # the odometry callback itself deadlocks, so let that callback return.
        time.sleep(0.05)
        if rclpy.ok():
            rclpy.shutdown()

    def _dataset_trace_due(self):
        if not self.mppi_dataset_dir or self._dataset_collection_complete():
            return False
        return not (
            self.mppi_dataset_stop_step >= 0
            and self._counter > self.mppi_dataset_stop_step
        )

    def _snapshot_targets(self):
        targets = []
        if (
            not self.mppi_snapshot_written
            and self.mppi_snapshot_step == self._counter
            and bool(self.mppi_snapshot_dir)
        ):
            output_dir = FilePath(self.mppi_snapshot_dir).expanduser().resolve()
            targets.append(
                {
                    "mode": "legacy-single",
                    "snapshot": output_dir / "snapshot.npz",
                    "summary": output_dir / "summary.json",
                    "candidate_csv": output_dir / "candidate_costs.csv",
                }
            )
        if self._dataset_snapshot_due():
            episode_root = (
                FilePath(self.mppi_dataset_dir).expanduser().resolve()
                / self.mppi_dataset_episode_id
            )
            if (
                not self.mppi_dataset_records
                and episode_root.exists()
                and any(
                    path.name != "closed_loop_trace.jsonl"
                    for path in episode_root.iterdir()
                )
            ):
                raise FileExistsError(
                    f"Refusing to overwrite existing dataset episode: {episode_root}"
                )
            stem = f"step_{self._counter:06d}"
            targets.append(
                {
                    "mode": "closed-loop-dataset",
                    "episode_root": episode_root,
                    "snapshot": episode_root / "snapshots" / f"{stem}.npz",
                    "summary": episode_root / "snapshots" / f"{stem}.json",
                    "candidate_csv": None,
                }
            )
        return targets

    @staticmethod
    def _reference_in_ego_frame(reference, full_state):
        local_reference = np.asarray(reference, dtype=np.float32).copy()
        delta = local_reference[:, :2] - np.asarray(full_state[:2])
        yaw = float(full_state[2])
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        local_reference[:, 0] = cosine * delta[:, 0] + sine * delta[:, 1]
        local_reference[:, 1] = -sine * delta[:, 0] + cosine * delta[:, 1]
        yaw_delta = local_reference[:, 2] - yaw
        local_reference[:, 2] = np.arctan2(
            np.sin(yaw_delta), np.cos(yaw_delta)
        )
        return local_reference

    @staticmethod
    def _raw_cost_features(trajectory, action, reference, current_action):
        cost_reference = np.asarray(reference, dtype=np.float32)
        if len(cost_reference) == trajectory.shape[1] + 1:
            cost_reference = cost_reference[1:]
        if len(cost_reference) != trajectory.shape[1]:
            raise ValueError("reference length does not match predicted trajectory")
        position_error = trajectory[..., :2] - cost_reference[None, :, :2]
        yaw_delta = trajectory[..., 2] - cost_reference[None, :, 2]
        previous_action = np.concatenate(
            (
                np.broadcast_to(
                    np.asarray(current_action, dtype=np.float32),
                    (action.shape[0], 1, action.shape[2]),
                ),
                action[:, :-1],
            ),
            axis=1,
        )
        features = {
            "feature_position_error_sq": np.square(position_error).sum(axis=-1),
            "feature_yaw_error_sq": np.square(
                np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
            ),
            "feature_vx_error_sq": np.square(
                trajectory[..., 3] - cost_reference[None, :, 3]
            ),
            "feature_action_rate_sq": np.square(action - previous_action),
        }
        if cost_reference.shape[1] >= 5:
            features["feature_yawrate_error_sq"] = np.square(
                trajectory[..., 4] - cost_reference[None, :, 4]
            )
        return features

    @staticmethod
    def _git_metadata(repository_root):
        def run_git(*arguments):
            result = subprocess.run(
                ["git", "-C", repository_root, *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.rstrip()

        try:
            status = run_git("status", "--porcelain")
            return {
                "commit": run_git("rev-parse", "HEAD"),
                "branch": run_git("branch", "--show-current"),
                "dirty": bool(status),
                "dirty_files": status.splitlines(),
            }
        except (OSError, subprocess.CalledProcessError) as exc:
            return {"error": str(exc)}

    def _append_dataset_trace(
        self,
        control_step,
        full_state,
        current_action,
        controller_action,
        executed_action,
        reference,
        frenet_pose,
        mean_knots_before,
        running_state_after,
        mppi_info,
        duration_sec,
    ):
        episode_root = (
            FilePath(self.mppi_dataset_dir).expanduser().resolve()
            / self.mppi_dataset_episode_id
        )
        if control_step == 0 and episode_root.exists() and any(
            episode_root.iterdir()
        ):
            raise FileExistsError(
                f"Refusing to overwrite existing dataset episode: {episode_root}"
            )
        episode_root.mkdir(parents=True, exist_ok=True)
        trace_path = episode_root / "closed_loop_trace.jsonl"
        cost = mppi_info["cost"].cpu().numpy()
        record = {
            "control_step": int(control_step),
            "simulated_time_s": float(control_step * self.control_dt),
            "history_valid_steps": int(
                min(control_step, self.mppi_params.history_length)
            ),
            "state": np.asarray(full_state).tolist(),
            "current_action": np.asarray(current_action).tolist(),
            "controller_action": np.asarray(controller_action).tolist(),
            "executed_action": np.asarray(executed_action).tolist(),
            "reference": np.asarray(reference).tolist(),
            "frenet_pose": [
                float(frenet_pose.s),
                float(frenet_pose.t),
                float(frenet_pose.xi),
            ],
            "mean_knots_before": mean_knots_before.cpu().numpy().tolist(),
            "mean_knots_after": (
                running_state_after.mean_knots.cpu().numpy().tolist()
            ),
            "optimized_action_sequence": (
                mppi_info["optimized_action_sequence"].cpu().numpy().tolist()
            ),
            "cost_summary_at_collection": {
                "best": float(cost.min()),
                "mean": float(cost.mean()),
                "median": float(np.median(cost)),
                "p95": float(np.percentile(cost, 95)),
                "effective_sample_size": float(
                    mppi_info["effective_sample_size"].cpu()
                ),
            },
            "controller_duration_s": float(duration_sec),
        }
        with trace_path.open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _write_dataset_manifest(self, episode_root, snapshot_summary):
        episode_root.mkdir(parents=True, exist_ok=True)
        manifest_path = episode_root / "manifest.json"
        repository_root = os.environ.get("CAR_PATH", "/home/plusai/anycar")
        if not self.mppi_dataset_records:
            if manifest_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing dataset episode: {episode_root}"
                )
            track_artifact = episode_root / "track.npz"
            np.savez_compressed(
                track_artifact,
                source_track=self.track_array,
                planner_waypoints=np.asarray(
                    self.global_planner.waypoints.T, dtype=np.float32
                ),
            )
            track_sha256 = hashlib.sha256(
                FilePath(self.track_path).read_bytes()
            ).hexdigest()
            manifest = {
                "format_version": 2,
                "dataset_type": "anycar-mppi-closed-loop-snapshots",
                "episode_id": self.mppi_dataset_episode_id,
                "created_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "fixed_dbm_parameters": True,
                "repository": self._git_metadata(repository_root),
                "track": {
                    "source_path": str(FilePath(self.track_path).resolve()),
                    "source_sha256": track_sha256,
                    "artifact": "track.npz",
                },
                "controller_backend": self.mppi_backend_name,
                "rollout_model": snapshot_summary["model"],
                "simulator": snapshot_summary["simulator"],
                "mppi_params": asdict(self.mppi_params),
                "cost_weights_at_collection": asdict(self.mppi.cost_weights),
                "collection": {
                    "trace_start_step": 0,
                    "start_step": max(0, self.mppi_dataset_start_step),
                    "stop_step": self.mppi_dataset_stop_step,
                    "stride": self.mppi_dataset_stride,
                    "max_snapshots": self.mppi_dataset_max_snapshots,
                    "cost_independent_payload": (
                        "candidate actions, raw/clipped knots, sampling noise, "
                        "DBM trajectories, and unweighted per-step error features"
                    ),
                },
                "artifacts": {
                    "closed_loop_trace": "closed_loop_trace.jsonl",
                    "track": "track.npz",
                },
                "snapshot_count": 0,
                "snapshots": [],
            }
        else:
            manifest = json.loads(manifest_path.read_text())
        record = {
            "control_step": snapshot_summary["scenario"]["control_step"],
            "simulated_time_s": snapshot_summary["scenario"]["simulated_time_s"],
            "frenet_s_m": snapshot_summary["scenario"]["frenet_s_m"],
            "frenet_lateral_m": snapshot_summary["scenario"]["frenet_lateral_m"],
            "frenet_heading_error_rad": snapshot_summary["scenario"][
                "frenet_heading_error_rad"
            ],
            "snapshot": os.path.relpath(
                snapshot_summary["artifacts"]["snapshot"], episode_root
            ),
            "summary": os.path.relpath(
                snapshot_summary["artifacts"]["summary"], episode_root
            ),
        }
        self.mppi_dataset_records.append(record)
        manifest["snapshots"] = self.mppi_dataset_records
        manifest["snapshot_count"] = len(self.mppi_dataset_records)
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_path.replace(manifest_path)

    def _write_mppi_snapshot(
        self,
        query_state,
        full_state,
        current_action,
        history,
        reference,
        frenet_pose,
        mean_knots_before,
        rng_state_before,
        action,
        running_state_after,
        mppi_info,
        target,
    ):
        """Persist one controller call with cost-independent rollout details."""
        snapshot_path = target["snapshot"]
        summary_path = target["summary"]
        candidate_csv_path = target["candidate_csv"]
        output_dir = snapshot_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        sampled_action = mppi_info["sampled_action_sequences"].cpu().numpy()
        sampled_trajectory = mppi_info["sampled_trajectories"].cpu().numpy()
        sampled_trajectory_full = mppi_info.get("sampled_trajectories_full")
        if sampled_trajectory_full is not None:
            sampled_trajectory_full = sampled_trajectory_full.cpu().numpy()
        cost = mppi_info["cost"].cpu().numpy()
        weight = mppi_info["weight"].cpu().numpy()
        components = {
            name: value.cpu().numpy()
            for name, value in mppi_info["cost_components"].items()
        }
        sampled_knots = mppi_info["sampled_knots"].cpu().numpy()
        raw_sampled_knots = mppi_info["raw_sampled_knots"].cpu().numpy()
        sampling_noise_knots = mppi_info["sampling_noise_knots"].cpu().numpy()
        sampling_mean_knots = mppi_info["sampling_mean_knots"].cpu().numpy()
        raw_features = self._raw_cost_features(
            sampled_trajectory,
            sampled_action,
            reference,
            current_action,
        )
        best_index = int(np.argmin(cost))
        rng_state_after = self.mppi._generator.get_state().cpu().numpy()
        reference_local = self._reference_in_ego_frame(reference, full_state)
        simulator_metadata_json = json.dumps(
            self.simulator_metadata or {}, sort_keys=True
        )

        snapshot_payload = dict(
            format_version=np.asarray(2, dtype=np.int32),
            collection_mode=np.asarray(target["mode"]),
            control_step=np.asarray(self._counter, dtype=np.int64),
            simulated_time_s=np.asarray(
                self._counter * self.control_dt, dtype=np.float64
            ),
            history_valid_steps=np.asarray(
                min(self._counter, self.mppi_params.history_length),
                dtype=np.int32,
            ),
            history_is_fully_observed=np.asarray(
                self._counter >= self.mppi_params.history_length,
                dtype=np.bool_,
            ),
            initial_state=np.asarray(query_state, dtype=np.float32),
            initial_state_six=np.asarray(full_state, dtype=np.float32),
            initial_lateral_velocity=np.asarray(full_state[4], dtype=np.float32),
            current_action=np.asarray(current_action, dtype=np.float32),
            history=history.cpu().numpy(),
            reference=np.asarray(reference, dtype=np.float32),
            reference_ego=reference_local,
            frenet_pose=np.asarray(
                [frenet_pose.s, frenet_pose.t, frenet_pose.xi], dtype=np.float32
            ),
            mean_knots_before=mean_knots_before.cpu().numpy(),
            mean_knots_after=running_state_after.mean_knots.cpu().numpy(),
            rng_state_before=rng_state_before.cpu().numpy(),
            rng_state_after=rng_state_after,
            sampling_noise_knots=sampling_noise_knots,
            sampling_mean_knots=sampling_mean_knots,
            raw_sampled_knots=raw_sampled_knots,
            sampled_knots=sampled_knots,
            sampled_knots_clipped=np.any(
                np.not_equal(raw_sampled_knots, sampled_knots), axis=-1
            ),
            sampled_action_sequences=sampled_action,
            predicted_trajectories=sampled_trajectory,
            cost=cost,
            weight=weight,
            optimized_action=np.asarray(action, dtype=np.float32),
            optimized_action_sequence=(
                mppi_info["optimized_action_sequence"].cpu().numpy()
            ),
            simulator_metadata_json=np.asarray(simulator_metadata_json),
            mppi_params_json=np.asarray(json.dumps(asdict(self.mppi_params))),
            cost_weights_json=np.asarray(
                json.dumps(asdict(self.mppi.cost_weights))
            ),
            dbm_params_json=np.asarray(
                json.dumps(asdict(self.mppi.rollout_backend.params))
                if self.mppi_backend_name == "dbm"
                else "{}"
            ),
            **{f"cost_{name}": value for name, value in components.items()},
            **raw_features,
        )
        if sampled_trajectory_full is not None:
            snapshot_payload["predicted_trajectories_full"] = (
                sampled_trajectory_full
            )
        np.savez_compressed(snapshot_path, **snapshot_payload)

        component_names = list(components)
        if candidate_csv_path is not None:
            with candidate_csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "candidate_index",
                        "total_cost",
                        *[f"cost_{name}" for name in component_names],
                        "mppi_weight",
                        "first_acceleration",
                        "first_steering",
                    ]
                )
                for index in range(len(cost)):
                    writer.writerow(
                        [
                            index,
                            cost[index],
                            *[components[name][index] for name in component_names],
                            weight[index],
                            sampled_action[index, 0, 0],
                            sampled_action[index, 0, 1],
                        ]
                    )

        model_summary = (
            {
                "backend": "dbm",
                "checkpoint": None,
                "dbm_params": asdict(self.mppi.rollout_backend.params),
                "uses_observed_lateral_velocity": True,
                "integration": "torch-rk4",
            }
            if self.mppi_backend_name == "dbm"
            else {
                "backend": self.mppi_backend_name,
                "checkpoint": str(
                    FilePath(self.query_checkpoint_path).resolve()
                ),
            }
        )
        ranking = np.argsort(cost)
        summary = {
            "format_version": 2,
            "collection_mode": target["mode"],
            "scenario": {
                "source": "live numeric closed-loop simulation",
                "observation_noise": False,
                "control_step": self._counter,
                "simulated_time_s": self._counter * self.control_dt,
                "history_valid_steps": min(
                    self._counter, self.mppi_params.history_length
                ),
                "history_is_fully_observed": (
                    self._counter >= self.mppi_params.history_length
                ),
                "track": str(FilePath(self.track_path).resolve()),
                "frenet_s_m": float(frenet_pose.s),
                "frenet_lateral_m": float(frenet_pose.t),
                "frenet_heading_error_rad": float(frenet_pose.xi),
                "query_state": np.asarray(query_state).tolist(),
                "full_state": np.asarray(full_state).tolist(),
                "current_action": np.asarray(current_action).tolist(),
            },
            "simulator": self.simulator_metadata or {
                "status": "metadata topic not received"
            },
            "model": model_summary,
            "mppi_params": asdict(self.mppi_params),
            "cost_weights": asdict(self.mppi.cost_weights),
            "cost_weights_at_collection": asdict(self.mppi.cost_weights),
            "cost_relabeling": {
                "supported": True,
                "raw_per_step_features": sorted(raw_features),
                "note": (
                    "Saved total cost and weights describe collection-time MPPI "
                    "only; use raw trajectories/actions/features for future labels."
                ),
            },
            "baseline": {
                "candidate_count": len(cost),
                "best_candidate_index": best_index,
                "best_cost": float(cost[best_index]),
                "mean_cost": float(cost.mean()),
                "median_cost": float(np.median(cost)),
                "p95_cost": float(np.percentile(cost, 95)),
                "max_cost": float(cost.max()),
                "effective_sample_size": float(
                    mppi_info["effective_sample_size"].cpu()
                ),
                "knot_clip_fraction": float(
                    np.not_equal(raw_sampled_knots, sampled_knots).mean()
                ),
                "optimized_action": np.asarray(action).tolist(),
                "best_cost_components": {
                    name: float(values[best_index])
                    for name, values in components.items()
                },
                "top_10_candidate_indices": ranking[:10].tolist(),
                "top_10_costs": cost[ranking[:10]].tolist(),
            },
            "artifacts": {
                "snapshot": str(snapshot_path),
                "summary": str(summary_path),
                "candidate_costs_csv": (
                    str(candidate_csv_path) if candidate_csv_path else None
                ),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if target["mode"] == "legacy-single":
            self.mppi_snapshot_written = True
        else:
            self._write_dataset_manifest(target["episode_root"], summary)
            self.mppi_dataset_snapshot_count += 1
        self.get_logger().info(f"MPPI snapshot written to {snapshot_path}")

    def timer_callback(self):
        if getattr(self, "mppi_dataset_shutdown_requested", False):
            return

        control_step = self._counter
        start_time = self.get_clock().now()
        # print("here")
        if self.odom is None:
            print("ODOM NOT FOUND!")
            # time.sleep(self.control_dt)
            return

        if TELEOP and self.joy is None:
            print("TELEOP NOT FOUND!")
            return


        odom_copy = deepcopy(self.odom)

        rpy = euler_from_quaternion([
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w,
        ])

        pose_car = np.array([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ], dtype=np.float32)

        lin_vel_car = np.array([
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ], dtype=np.float32)

        quat_car = np.array([
            self.odom.pose.pose.orientation.w,
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
        ], dtype=np.float32)
        
        state = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, rpy[2], self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.angular.z], dtype=np.float32)
        query_state = state[[0, 1, 2, 3, 5]]
        if self._counter == 0 and self.controller_type == 'mppi':
            self.query_history.prime_constant_motion(
                query_state, self.prev_action
            )
        
        if np.any(np.isnan(state)):
            return
        
        # print("State", state)
        ## Generate reference trajectory
        target_pos_arr, frenet_pose = self.global_planner.generate(
            state[:5], self.control_dt, self.mppi_params.horizon + 1, True
        )
        target_pos_arr[:, 3] = np.clip(target_pos_arr[:, 3], 0.0, SAFE_SPEED_MAX)
        target_pos_list = np.array(target_pos_arr)


        ref_vel = target_pos_arr[0][3]
        action_candidate_np = None
        sampled_traj = None
        if self.controller_type == 'mppi':
            _ctr_start = time.time()
            if self.mppi_backend_name == "dbm":
                self.mppi.rollout_backend.set_initial_lateral_velocity(state[4])
            mppi_history = self.query_history.tensor()
            snapshot_targets = self._snapshot_targets()
            snapshot_this_step = bool(snapshot_targets)
            trace_this_step = self._dataset_trace_due()
            if snapshot_this_step or trace_this_step:
                mean_knots_before = self.mppi_running_params.mean_knots.detach().clone()
                current_action_before = self.prev_action.copy()
            if snapshot_this_step:
                rng_state_before = self.mppi._generator.get_state().detach().clone()
            action, self.mppi_running_params, mppi_info = self.mppi(
                query_state,
                self.prev_action,
                mppi_history,
                target_pos_arr,
                self.mppi_running_params,
            )
            action = action.cpu().numpy().astype(np.float32)
            controller_action = action.copy()
            if snapshot_this_step:
                for snapshot_target in snapshot_targets:
                    self._write_mppi_snapshot(
                        query_state,
                        state,
                        current_action_before,
                        mppi_history,
                        target_pos_arr,
                        frenet_pose,
                        mean_knots_before,
                        rng_state_before,
                        action,
                        self.mppi_running_params,
                        mppi_info,
                        snapshot_target,
                    )
            action_candidate_np = (
                mppi_info["optimized_action_sequence"].cpu().numpy()
            )
            sampled_traj = mppi_info["trajectory"][:, :2].cpu().numpy()
            controller_duration_sec = time.time() - _ctr_start
            print("ctr time", controller_duration_sec)

        elif self.controller_type == 'pure_persuit':
            # print(colored("Pure Persuit Controller", "green"))
            action  = self.pure_pursuit.step(pose_car, lin_vel_car, quat_car, target_pos_list)
        else:
            raise ValueError(f"Invalid controller type: {self.controller_type}")        
        
        if self.is_recover_mode: # Override the action with recover controller
            action = self.recover_controller.step(pose_car, lin_vel_car, quat_car, target_pos_list)
        
        if TELEOP:
            # Map joystick to action
            steer_joy = self.joy.axes[0]

            if self.joy.axes[2] <= 0.98:
                speed_joy = (self.joy.axes[2] - 1.0) / 2.0
            else:
                speed_joy = -1.0 * (self.joy.axes[5] - 1.0) / 2.0
            action = np.array([speed_joy, steer_joy])
        
        if self.emergency_stop:
            print(colored("Emergency stop", "red"))
            action = np.array([0., 0.])

        if self.controller_type == 'mppi' and trace_this_step:
            self._append_dataset_trace(
                control_step,
                state,
                current_action_before,
                controller_action,
                action,
                target_pos_arr,
                frenet_pose,
                mean_knots_before,
                self.mppi_running_params,
                mppi_info,
                controller_duration_sec,
            )
            
        
        # Feed history to MPPI
        #  append history here because we sometimes wants to overwirte the mppi action 
        #  with other controllers (e.g. pure pursuit)
        if self.controller_type == 'mppi':
            self.query_history.append(query_state, action)
    

        action_rate = action - self.prev_action
        self.prev_action = action
        
        self._counter += 1
        
        # px, py, psi, vx, vy, omega = env.obs_state().tolist()
        px, py, psi, vx, vy, omega = state.tolist()
        
        q = quaternion_from_euler(0, 0, psi)
        now = self.get_clock().now().to_msg()
        cmd = AckermannDriveStamped()
        cmd.header.stamp = now
        cmd.drive.speed = float(action[0])
        cmd.drive.steering_angle = float(action[1])
        
        odom_copy.header.stamp = now

        pose_with_covariance_stamped = PoseWithCovarianceStamped()
        pose_with_covariance_stamped.header.frame_id = 'map'
        pose_with_covariance_stamped.header.stamp = now
        pose_with_covariance_stamped.pose.pose.position.x = px
        pose_with_covariance_stamped.pose.pose.position.y = py
        pose_with_covariance_stamped.pose.pose.orientation.x = q[0]
        pose_with_covariance_stamped.pose.pose.orientation.y = q[1]
        pose_with_covariance_stamped.pose.pose.orientation.z = q[2]
        pose_with_covariance_stamped.pose.pose.orientation.w = q[3]
        
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = now
        for i in range(target_pos_list.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(target_pos_list[i][0])
            pose.pose.position.y = float(target_pos_list[i][1])
            path.poses.append(pose)
        
        if self.controller_type == 'mppi':
            mppi_path = Path()
            mppi_path.header.frame_id = 'map'
            mppi_path.header.stamp = now
            for i in range(len(sampled_traj)):
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.pose.position.x = float(sampled_traj[i, 0])
                pose.pose.position.y = float(sampled_traj[i, 1])
                mppi_path.poses.append(pose)
        
        throttle = Float64()
        throttle.data = float(action[0]) * 3905.9 * 2
        
        steer = Float64()
        steer.data = float(action[1] * -1.0 / 2 + 0.5)
        
        action_rate_msg = Float64()
        action_rate_msg.data = float(np.linalg.norm(action_rate))
        self.action_rate_pub.publish(action_rate_msg)
        
        lateral_error = Float64()
        lateral_error.data = frenet_pose.t
        
        ref_vel_msg = deepcopy(self.odom)
        ref_vel_msg.twist.twist.linear.x = ref_vel
        
        # body polygon
        pts = np.array([
            [self.model_params.LF, self.L/3],
            [self.model_params.LF, -self.L/3],
            [-self.model_params.LR, -self.L/3],
            [-self.model_params.LR, self.L/3],
        ])
        # transform to world frame
        R = euler_matrix(0, 0, psi)[:2, :2]
        pts = np.dot(R, pts.T).T
        pts += np.array([px, py])
        body = PolygonStamped()
        body.header.frame_id = 'map'
        body.header.stamp = now
        for i in range(pts.shape[0]):
            p = Point32()
            p.x = float(pts[i, 0])
            p.y = float(pts[i, 1])
            p.z = 0.
            body.polygon.points.append(p)
        self.body_pub_.publish(body)

        end_time = self.get_clock().now()
        duration_sec = (end_time - start_time).nanoseconds / 1e9
        print(colored(f"duration: {duration_sec:.3f}", "red"))
        if duration_sec > self.control_dt:
            # print("Out of time!")
            self.get_logger().warn(f"MPPI took {duration_sec} seconds which is longer than DT {self.control_dt} seconds.", throttle_duration_sec=1.0)
        else:
            sleep_time = self.control_dt - duration_sec
            time.sleep(sleep_time)

        self.vehicle_cmd_pub_.publish(cmd)
        self.odom_copy_pub_.publish(odom_copy)
        # print(cmd.header.stamp, odom_copy.header.stamp)
        self.pose_pub_.publish(pose_with_covariance_stamped)
        self.ref_trajectory_pub_.publish(path)
        if self.controller_type == 'mppi':
            self.path_pub_.publish(mppi_path)
        self.throttle_pub_.publish(throttle)
        self.steer_pub_.publish(steer)
        self.lateral_error_pub_.publish(lateral_error)
        self.ref_vel_pub_.publish(ref_vel_msg)
        for i, (param, val) in enumerate(self.model_params.to_dict().items()):
            msg = Float64()
            msg.data = float(val)
            self.params_pub_list[i].publish(msg)
            
        if self.is_recover_mode:
            controller_type = "recover: pure pursuit"
        else:
            controller_type = self.controller_type
        misc_msg = String()
        misc_msg.data = f"env: {self.env_params.name}\ncontroller: {controller_type}\nEnv:\n- mass:{self.env_params.mass}\n- friction:{self.env_params.friction}\n- delay:{self.env_params.delay}\n- step:{self._counter}\n"
        self.misc_pub_.publish(misc_msg)

        #publish mppi time
        mppi_time_msg = Float64()
        mppi_time_msg.data = duration_sec
        self.mppi_time_pub_.publish(mppi_time_msg)

        if (
            self.mppi_dataset_shutdown_on_complete
            and self._dataset_collection_complete()
            and not self.mppi_dataset_shutdown_requested
        ):
            self.mppi_dataset_shutdown_requested = True
            self.get_logger().info(
                "MPPI dataset collection complete; shutting down car_node"
            )
            threading.Thread(
                target=self._shutdown_after_dataset_callback,
                daemon=True,
            ).start()
 
           
    def slow_timer_callback(self):
        # publish waypoint_list as path
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(self.global_planner.waypoints.shape[1]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(self.global_planner.waypoints[0][i])
            pose.pose.position.y = float(self.global_planner.waypoints[1][i])
            path.poses.append(pose)
        self.waypoint_list_pub_.publish(path)

    def odom_callback(self, msg:Odometry):
        self.odom = msg
        if self.step_mode_:
            self.timer_callback()

    def joy_callback(self, msg):
        self.joy = msg
        if self.step_mode_:
            self.timer_callback()

def main():
    rclpy.init()
    car_node = CarNode()
    try:
        rclpy.spin(car_node)
    finally:
        car_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
