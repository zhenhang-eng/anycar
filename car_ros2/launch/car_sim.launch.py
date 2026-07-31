import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

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
    return LaunchDescription(
        [
            declare_use_sim_time_cmd,
            declare_mppi_backend_cmd,
            declare_query_checkpoint_cmd,
            declare_query_onnx_path_cmd,
            declare_mppi_snapshot_step_cmd,
            declare_mppi_snapshot_dir_cmd,
            Node(
                package="car_ros2",
                executable="car_node",
                name="car_node",
                output="screen",
                parameters=[
                    use_sim_time,
                    {
                        "step_mode": True,
                        "mppi_backend": mppi_backend,
                        "query_checkpoint": query_checkpoint,
                        "query_onnx_path": query_onnx_path,
                        "mppi_snapshot_step": ParameterValue(
                            mppi_snapshot_step, value_type=int
                        ),
                        "mppi_snapshot_dir": mppi_snapshot_dir,
                    }
                ],
                remappings=[
                ],
                emulate_tty=True,
            ),
            Node(
                package="car_ros2",
                executable="car_simulator_node",
                name="car_simulator_node",
                output="screen",
                parameters=[
                    use_sim_time
                ],
                remappings=[
                ],
                emulate_tty=True,
            ),
        ]
    )
