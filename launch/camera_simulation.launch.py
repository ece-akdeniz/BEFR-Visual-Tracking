"""Launch the synchronized multi-camera sensor simulator."""

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
    rviz_config = Path(package_share) / "rviz" / "camera_simulation.rviz"

    use_sim_time = LaunchConfiguration("use_sim_time")
    ground_truth_topic = LaunchConfiguration("ground_truth_topic")
    launch_rviz = LaunchConfiguration("launch_rviz")
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
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", str(rviz_config)],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation clock from Gazebo",
            ),
            DeclareLaunchArgument(
                "ground_truth_topic",
                default_value="/model/rrbot/odometry",
                description="Ground-truth odometry topic to verify in Gazebo before use",
            ),
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="true",
                description="Start RViz with the camera sensor visualization",
            ),
            DeclareLaunchArgument(
                "cameras_config_path",
                default_value=str(config_dir / "cameras_gazebo.yaml"),
                description="Camera layout YAML (use cameras.yaml for offline z=1.5 m)",
            ),
            camera_simulator_node,
            rviz_node,
        ]
    )
