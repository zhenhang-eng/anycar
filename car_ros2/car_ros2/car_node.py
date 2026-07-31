from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path as FilePath
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


unique_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')

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
            self.mppi_backend_name = backend_name
            self.query_checkpoint_path = checkpoint_path
            num_samples = int(
                self.declare_parameter("mppi_num_samples", 256).value
            )
            num_iterations = int(
                self.declare_parameter("mppi_num_iterations", 1).value
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
            self.mppi_params = TorchMPPIParams(
                num_samples=num_samples,
                num_iterations=num_iterations,
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
    ):
        """Persist one atomic controller call for sampling/cost experiments."""
        output_dir = FilePath(self.mppi_snapshot_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        sampled_action = mppi_info["sampled_action_sequences"].cpu().numpy()
        sampled_trajectory = mppi_info["sampled_trajectories"].cpu().numpy()
        cost = mppi_info["cost"].cpu().numpy()
        weight = mppi_info["weight"].cpu().numpy()
        components = {
            name: value.cpu().numpy()
            for name, value in mppi_info["cost_components"].items()
        }
        sampled_knots = self.mppi._sequence_to_knots(
            mppi_info["sampled_action_sequences"]
        ).cpu().numpy()
        best_index = int(np.argmin(cost))
        rng_state_after = self.mppi._generator.get_state().cpu().numpy()

        snapshot_path = output_dir / "snapshot.npz"
        np.savez_compressed(
            snapshot_path,
            initial_state=np.asarray(query_state, dtype=np.float32),
            initial_state_six=np.asarray(full_state, dtype=np.float32),
            initial_lateral_velocity=np.asarray(full_state[4], dtype=np.float32),
            current_action=np.asarray(current_action, dtype=np.float32),
            history=history.cpu().numpy(),
            reference=np.asarray(reference, dtype=np.float32),
            mean_knots_before=mean_knots_before.cpu().numpy(),
            mean_knots_after=running_state_after.mean_knots.cpu().numpy(),
            rng_state_before=rng_state_before.cpu().numpy(),
            rng_state_after=rng_state_after,
            sampled_knots=sampled_knots,
            sampled_action_sequences=sampled_action,
            predicted_trajectories=sampled_trajectory,
            cost=cost,
            weight=weight,
            optimized_action=np.asarray(action, dtype=np.float32),
            optimized_action_sequence=(
                mppi_info["optimized_action_sequence"].cpu().numpy()
            ),
            **{f"cost_{name}": value for name, value in components.items()},
        )

        component_names = list(components)
        with (output_dir / "candidate_costs.csv").open("w", newline="") as stream:
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

        ranking = np.argsort(cost)
        summary = {
            "format_version": 1,
            "scenario": {
                "source": "live clean Quick Start simulation",
                "observation_noise": False,
                "control_step": self._counter,
                "simulated_time_s": self._counter * self.control_dt,
                "track": str(FilePath(self.track_path).resolve()),
                "frenet_s_m": float(frenet_pose.s),
                "frenet_lateral_m": float(frenet_pose.t),
                "frenet_heading_error_rad": float(frenet_pose.xi),
                "query_state": np.asarray(query_state).tolist(),
                "full_state": np.asarray(full_state).tolist(),
                "current_action": np.asarray(current_action).tolist(),
            },
            "model": (
                {
                    "backend": "dbm",
                    "checkpoint": None,
                    "dbm_params": asdict(self.mppi.rollout_backend.params),
                    "uses_observed_lateral_velocity": True,
                }
                if self.mppi_backend_name == "dbm"
                else {
                    "backend": self.mppi_backend_name,
                    "checkpoint": str(
                        FilePath(self.query_checkpoint_path).resolve()
                    ),
                }
            ),
            "mppi_params": asdict(self.mppi_params),
            "cost_weights": asdict(self.mppi.cost_weights),
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
                "candidate_costs_csv": str(output_dir / "candidate_costs.csv"),
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        self.mppi_snapshot_written = True
        self.get_logger().info(f"MPPI snapshot written to {output_dir}")

    def timer_callback(self):

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
            snapshot_this_step = (
                not self.mppi_snapshot_written
                and self.mppi_snapshot_step == self._counter
                and bool(self.mppi_snapshot_dir)
            )
            if snapshot_this_step:
                mean_knots_before = self.mppi_running_params.mean_knots.detach().clone()
                rng_state_before = self.mppi._generator.get_state().detach().clone()
                current_action_before = self.prev_action.copy()
            action, self.mppi_running_params, mppi_info = self.mppi(
                query_state,
                self.prev_action,
                mppi_history,
                target_pos_arr,
                self.mppi_running_params,
            )
            action = action.cpu().numpy().astype(np.float32)
            if snapshot_this_step:
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
                )
            action_candidate_np = (
                mppi_info["optimized_action_sequence"].cpu().numpy()
            )
            sampled_traj = mppi_info["trajectory"][:, :2].cpu().numpy()
            print("ctr time", time.time() - _ctr_start)

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
    rclpy.spin(car_node)
    car_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
