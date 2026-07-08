"""Launch the configurable visual tracking node."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("befr_visual_tracking")
    config_dir = Path(package_share) / "config"

    use_sim_time = LaunchConfiguration("use_sim_time")
    filter_type = LaunchConfiguration("filter_type")
    launch_camera_simulator = LaunchConfiguration("launch_camera_simulator")
    ground_truth_topic = LaunchConfiguration("ground_truth_topic")
    cameras_config_path = LaunchConfiguration("cameras_config_path")

    camera_simulator_node = Node(
        package="befr_visual_tracking",
        executable="camera_simulator",
        name="camera_simulator",
        output="screen",
        parameters=[
            ParameterFile(str(config_dir / "sensor_simulator.yaml"), allow_substs=True),
            {"use_sim_time": use_sim_time},
            {"ground_truth_topic": ground_truth_topic},
            {"cameras_config_path": cameras_config_path},
        ],
        condition=IfCondition(launch_camera_simulator),
    )

    tracker_node = Node(
        package="befr_visual_tracking",
        executable="visual_tracker",
        name="visual_tracker",
        output="screen",
        parameters=[
            ParameterFile(str(config_dir / "tracking.yaml"), allow_substs=True),
            {"use_sim_time": use_sim_time},
            {"filter_type": filter_type},
            {"cameras_config_path": cameras_config_path},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "filter_type",
                default_value="ekf",
                description="Tracker implementation: kf, ekf, or ukf",
            ),
            DeclareLaunchArgument(
                "launch_camera_simulator",
                default_value="false",
                description="Also launch the simulated camera sensor node",
            ),
            DeclareLaunchArgument(
                "ground_truth_topic",
                default_value="/model/rrbot/odometry",
            ),
            DeclareLaunchArgument(
                "cameras_config_path",
                default_value=str(config_dir / "cameras_gazebo.yaml"),
                description="Camera layout YAML (use cameras.yaml for offline z=1.5 m)",
            ),
            camera_simulator_node,
            tracker_node,
        ]
    )
