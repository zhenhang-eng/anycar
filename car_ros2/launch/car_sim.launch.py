import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

def get_share_file(package_name, *args):
    return os.path.join(get_package_share_directory(package_name), *args)


def get_sim_time_launch_arg():
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time", default_value="False", description="Use simulation clock if True"
    )

    return declare_use_sim_time_cmd, {"use_sim_time": use_sim_time}


def generate_launch_description():
    declare_use_sim_time_cmd, use_sim_time = get_sim_time_launch_arg()
    mppi_backend = LaunchConfiguration("mppi_backend")
    query_checkpoint = LaunchConfiguration("query_checkpoint")
    query_onnx_path = LaunchConfiguration("query_onnx_path")
    mppi_snapshot_step = LaunchConfiguration("mppi_snapshot_step")
    mppi_snapshot_dir = LaunchConfiguration("mppi_snapshot_dir")
    mppi_dataset_dir = LaunchConfiguration("mppi_dataset_dir")
    mppi_dataset_start_step = LaunchConfiguration("mppi_dataset_start_step")
    mppi_dataset_stop_step = LaunchConfiguration("mppi_dataset_stop_step")
    mppi_dataset_stride = LaunchConfiguration("mppi_dataset_stride")
    mppi_dataset_max_snapshots = LaunchConfiguration(
        "mppi_dataset_max_snapshots"
    )
    mppi_dataset_episode_id = LaunchConfiguration("mppi_dataset_episode_id")
    mppi_dataset_shutdown_on_complete = LaunchConfiguration(
        "mppi_dataset_shutdown_on_complete"
    )
    mppi_seed = LaunchConfiguration("mppi_seed")
    mppi_num_samples = LaunchConfiguration("mppi_num_samples")
    mppi_sampling_mode = LaunchConfiguration("mppi_sampling_mode")
    mppi_reference_speed = LaunchConfiguration("mppi_reference_speed")
    mppi_reference_speed_max = LaunchConfiguration("mppi_reference_speed_max")
    mppi_hard_guard_checkpoint = LaunchConfiguration(
        "mppi_hard_guard_checkpoint"
    )
    mppi_hard_guard_first_seed = LaunchConfiguration(
        "mppi_hard_guard_first_seed"
    )
    mppi_hard_guard_switch_margin = LaunchConfiguration(
        "mppi_hard_guard_switch_margin"
    )
    mppi_hard_guard_min_dwell = LaunchConfiguration(
        "mppi_hard_guard_min_dwell"
    )
    mppi_hard_guard_warm_hard_return = LaunchConfiguration(
        "mppi_hard_guard_warm_hard_return"
    )
    sim_initial_state = LaunchConfiguration("sim_initial_state")
    repository_root = os.environ.get("CAR_PATH", "/home/plusai/anycar")
    declare_mppi_backend_cmd = DeclareLaunchArgument(
        "mppi_backend",
        default_value="pytorch",
        description="MPPI rollout backend: pytorch, onnx, or dbm",
    )
    declare_query_checkpoint_cmd = DeclareLaunchArgument(
        "query_checkpoint",
        default_value=os.path.join(
            repository_root,
            "outputs/formal_real_finetune_query_baseline_split/"
            "20260728T143256/query_best.pt",
        ),
        description="Deterministic Query PyTorch checkpoint",
    )
    declare_query_onnx_path_cmd = DeclareLaunchArgument(
        "query_onnx_path",
        default_value=os.path.join(
            repository_root, "outputs/query_mppi/anycar_query.onnx"
        ),
        description="Exported Query ONNX graph",
    )
    declare_mppi_snapshot_step_cmd = DeclareLaunchArgument(
        "mppi_snapshot_step",
        default_value="-1",
        description="Control step to dump a fixed MPPI sampling/cost snapshot",
    )
    declare_mppi_snapshot_dir_cmd = DeclareLaunchArgument(
        "mppi_snapshot_dir",
        default_value="",
        description="Output directory for the optional MPPI snapshot",
    )
    declare_mppi_dataset_dir_cmd = DeclareLaunchArgument(
        "mppi_dataset_dir",
        default_value="",
        description="Root directory for multi-step closed-loop MPPI snapshots",
    )
    declare_mppi_dataset_start_step_cmd = DeclareLaunchArgument(
        "mppi_dataset_start_step",
        default_value="-1",
        description="First control step to collect; -1 means step zero",
    )
    declare_mppi_dataset_stop_step_cmd = DeclareLaunchArgument(
        "mppi_dataset_stop_step",
        default_value="-1",
        description="Last control step to collect; -1 disables the upper bound",
    )
    declare_mppi_dataset_stride_cmd = DeclareLaunchArgument(
        "mppi_dataset_stride",
        default_value="10",
        description="Control-step interval between dataset snapshots",
    )
    declare_mppi_dataset_max_snapshots_cmd = DeclareLaunchArgument(
        "mppi_dataset_max_snapshots",
        default_value="0",
        description="Maximum snapshots in this episode; zero is unlimited",
    )
    declare_mppi_dataset_episode_id_cmd = DeclareLaunchArgument(
        "mppi_dataset_episode_id",
        default_value="",
        description="Stable episode id; empty creates a timestamp id",
    )
    declare_mppi_dataset_shutdown_on_complete_cmd = DeclareLaunchArgument(
        "mppi_dataset_shutdown_on_complete",
        default_value="False",
        description="Stop the launch after the requested snapshot count",
    )
    declare_mppi_seed_cmd = DeclareLaunchArgument(
        "mppi_seed",
        default_value="3407",
        description="Reproducible Torch MPPI sampling seed",
    )
    declare_mppi_num_samples_cmd = DeclareLaunchArgument(
        "mppi_num_samples",
        default_value="256",
        description="MPPI candidate count per controller step",
    )
    declare_mppi_sampling_mode_cmd = DeclareLaunchArgument(
        "mppi_sampling_mode",
        default_value="gaussian",
        description=(
            "MPPI knot sampling: gaussian or fixed_hadamard_64; the fixed "
            "mode requires mppi_num_samples=64"
        ),
    )
    declare_mppi_reference_speed_cmd = DeclareLaunchArgument(
        "mppi_reference_speed",
        default_value="-1.0",
        description=(
            "Reference speed override in m/s; a negative value preserves the "
            "track speed"
        ),
    )
    declare_mppi_reference_speed_max_cmd = DeclareLaunchArgument(
        "mppi_reference_speed_max",
        default_value="10.0",
        description=(
            "Explicit reference-speed ceiling in m/s. The default preserves "
            "the normal runtime limit; isolated synthetic high-speed data "
            "collection must opt in to a larger value."
        ),
    )
    declare_mppi_hard_guard_checkpoint_cmd = DeclareLaunchArgument(
        "mppi_hard_guard_checkpoint",
        default_value="",
        description=(
            "Optional residual Actor checkpoint for the DBM baseline-preserving "
            "hard guard; empty keeps the original controller"
        ),
    )
    declare_mppi_hard_guard_first_seed_cmd = DeclareLaunchArgument(
        "mppi_hard_guard_first_seed",
        default_value="24001",
        description="Frozen 129-rollout Actor-context first-pass seed",
    )
    declare_mppi_hard_guard_switch_margin_cmd = DeclareLaunchArgument(
        "mppi_hard_guard_switch_margin",
        default_value="0.0",
        description=(
            "Guard hysteresis: challenger must undercut the incumbent branch "
            "model cost by more than this margin before a switch is allowed"
        ),
    )
    declare_mppi_hard_guard_min_dwell_cmd = DeclareLaunchArgument(
        "mppi_hard_guard_min_dwell",
        default_value="0",
        description="Guard hysteresis: minimum incumbent dwell steps per branch",
    )
    declare_mppi_hard_guard_warm_hard_return_cmd = DeclareLaunchArgument(
        "mppi_hard_guard_warm_hard_return",
        default_value="false",
        description=(
            "Asymmetric guard hysteresis: returning to the warm branch "
            "bypasses margin/dwell so the warm hard floor holds step-by-step"
        ),
    )
    declare_sim_initial_state_cmd = DeclareLaunchArgument(
        "sim_initial_state",
        default_value="0,0,0,0,0,0",
        description="Numeric DBM initial x,y,yaw,vx,vy,yawrate",
    )
    car_node = Node(
        package="car_ros2",
        executable="car_node",
        name="car_node",
        output="screen",
        parameters=[
            use_sim_time,
            {
                "step_mode": True,
                "mppi_backend": mppi_backend,
                "mppi_seed": ParameterValue(mppi_seed, value_type=int),
                "mppi_num_samples": ParameterValue(
                    mppi_num_samples, value_type=int
                ),
                "mppi_sampling_mode": mppi_sampling_mode,
                "mppi_reference_speed": ParameterValue(
                    mppi_reference_speed, value_type=float
                ),
                "mppi_reference_speed_max": ParameterValue(
                    mppi_reference_speed_max, value_type=float
                ),
                "mppi_hard_guard_checkpoint": mppi_hard_guard_checkpoint,
                "mppi_hard_guard_first_seed": ParameterValue(
                    mppi_hard_guard_first_seed, value_type=int
                ),
                "mppi_hard_guard_switch_margin": ParameterValue(
                    mppi_hard_guard_switch_margin, value_type=float
                ),
                "mppi_hard_guard_min_dwell": ParameterValue(
                    mppi_hard_guard_min_dwell, value_type=int
                ),
                "mppi_hard_guard_warm_hard_return": ParameterValue(
                    mppi_hard_guard_warm_hard_return, value_type=bool
                ),
                "query_checkpoint": query_checkpoint,
                "query_onnx_path": query_onnx_path,
                "mppi_snapshot_step": ParameterValue(
                    mppi_snapshot_step, value_type=int
                ),
                "mppi_snapshot_dir": mppi_snapshot_dir,
                "mppi_dataset_dir": mppi_dataset_dir,
                "mppi_dataset_start_step": ParameterValue(
                    mppi_dataset_start_step, value_type=int
                ),
                "mppi_dataset_stop_step": ParameterValue(
                    mppi_dataset_stop_step, value_type=int
                ),
                "mppi_dataset_stride": ParameterValue(
                    mppi_dataset_stride, value_type=int
                ),
                "mppi_dataset_max_snapshots": ParameterValue(
                    mppi_dataset_max_snapshots, value_type=int
                ),
                "mppi_dataset_episode_id": mppi_dataset_episode_id,
                "mppi_dataset_shutdown_on_complete": ParameterValue(
                    mppi_dataset_shutdown_on_complete, value_type=bool
                ),
            },
        ],
        emulate_tty=True,
    )
    simulator_node = Node(
        package="car_ros2",
        executable="car_simulator_node",
        name="car_simulator_node",
        output="screen",
        parameters=[use_sim_time, {"initial_state": sim_initial_state}],
        emulate_tty=True,
    )
    shutdown_when_controller_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=car_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="car_node exited")
                )
            ],
        )
    )
    return LaunchDescription(
        [
            declare_use_sim_time_cmd,
            declare_mppi_backend_cmd,
            declare_mppi_seed_cmd,
            declare_mppi_num_samples_cmd,
            declare_mppi_sampling_mode_cmd,
            declare_mppi_reference_speed_cmd,
            declare_mppi_reference_speed_max_cmd,
            declare_mppi_hard_guard_checkpoint_cmd,
            declare_mppi_hard_guard_first_seed_cmd,
            declare_mppi_hard_guard_switch_margin_cmd,
            declare_mppi_hard_guard_min_dwell_cmd,
            declare_mppi_hard_guard_warm_hard_return_cmd,
            declare_query_checkpoint_cmd,
            declare_query_onnx_path_cmd,
            declare_mppi_snapshot_step_cmd,
            declare_mppi_snapshot_dir_cmd,
            declare_mppi_dataset_dir_cmd,
            declare_mppi_dataset_start_step_cmd,
            declare_mppi_dataset_stop_step_cmd,
            declare_mppi_dataset_stride_cmd,
            declare_mppi_dataset_max_snapshots_cmd,
            declare_mppi_dataset_episode_id_cmd,
            declare_mppi_dataset_shutdown_on_complete_cmd,
            declare_sim_initial_state_cmd,
            car_node,
            simulator_node,
            shutdown_when_controller_exits,
        ]
    )
