"""Launch camera simulation, dataset recording, and rosbag capture."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("befr_visual_tracking"))
    config_dir = package_share / "config"

    use_sim_time = LaunchConfiguration("use_sim_time")
    ground_truth_topic = LaunchConfiguration("ground_truth_topic")
    record_bag = LaunchConfiguration("record_bag")
    output_dir = LaunchConfiguration("output_dir")
    record_duration_s = LaunchConfiguration("record_duration_s")
    bag_output = LaunchConfiguration("bag_output")
    cameras_config_path = LaunchConfiguration("cameras_config_path")

    camera_simulator = Node(
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

    dataset_recorder = Node(
        package="befr_visual_tracking",
        executable="dataset_recorder",
        name="dataset_recorder",
        output="screen",
        parameters=[
            ParameterFile(str(config_dir / "dataset_recorder.yaml"), allow_substs=True),
            {"use_sim_time": use_sim_time},
            {"ground_truth_topic": ground_truth_topic},
            {"output_dir": output_dir},
            {"record_duration_s": record_duration_s},
            {"experiment_name": "live_slow_circle"},
            {"trajectory_name": "slow_circle"},
        ],
    )

    bag_process = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "-o",
            bag_output,
            ground_truth_topic,
            "/visual_tracking/camera_0/detection",
            "/visual_tracking/camera_1/detection",
            "/visual_tracking/camera_2/detection",
            "/visual_tracking/camera_3/detection",
            "/visual_tracking/visualization",
        ],
        output="screen",
        condition=IfCondition(record_bag),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "ground_truth_topic",
                default_value="/model/rrbot/odometry",
            ),
            DeclareLaunchArgument("record_bag", default_value="true"),
            DeclareLaunchArgument(
                "output_dir",
                default_value="results/live_slow_circle",
            ),
            DeclareLaunchArgument(
                "bag_output",
                default_value="results/live_slow_circle/rosbag",
            ),
            DeclareLaunchArgument(
                "record_duration_s",
                default_value="21.0",
                description="Recording duration for one slow circle (~2*pi/0.3 s)",
            ),
            DeclareLaunchArgument(
                "cameras_config_path",
                default_value=str(config_dir / "cameras_gazebo.yaml"),
                description="Camera layout YAML (use cameras.yaml for offline z=1.5 m)",
            ),
            camera_simulator,
            dataset_recorder,
            bag_process,
        ]
    )
