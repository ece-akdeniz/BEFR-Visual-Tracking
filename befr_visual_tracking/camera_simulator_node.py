"""ROS 2 node that simulates synchronized multi-camera pixel detections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import CameraMeasurement, cameras_from_yaml
from befr_visual_tracking.sensor_visualizer import build_sensor_markers


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class CameraSimulatorNode(Node):
    """Publish noisy pixel detections from fixed cameras using Gazebo ground truth."""

    def __init__(self) -> None:
        super().__init__("camera_simulator")

        self.declare_parameter("ground_truth_topic", "/model/rrbot/odometry")
        self.declare_parameter("cameras_config_path", "")
        self.declare_parameter("sensor_rate_hz", 30.0)
        self.declare_parameter("detection_topic_prefix", "/visual_tracking")
        self.declare_parameter("publish_visualization", True)
        self.declare_parameter("visualization_topic", "/visual_tracking/visualization")
        self.declare_parameter("ray_length", 6.0)
        self.declare_parameter("axis_length", 1.0)
        self.declare_parameter("random_seed", 42)

        config_path = self._resolve_config_path(
            self.get_parameter("cameras_config_path").get_parameter_value().string_value
        )
        self._system_config, self._cameras = cameras_from_yaml(config_path)
        self._world_frame_id = self._system_config.world.frame_id

        seed = self.get_parameter("random_seed").get_parameter_value().integer_value
        self._rng = np.random.default_rng(seed)

        self._latest_position: np.ndarray | None = None
        self._has_ground_truth = False

        prefix = self.get_parameter("detection_topic_prefix").get_parameter_value().string_value
        self._detection_publishers = {
            camera.name: self.create_publisher(
                PointStamped,
                f"{prefix}/{camera.name}/detection",
                10,
            )
            for camera in self._cameras
        }

        self._visualization_publisher = None
        if self.get_parameter("publish_visualization").get_parameter_value().bool_value:
            visualization_topic = (
                self.get_parameter("visualization_topic").get_parameter_value().string_value
            )
            self._visualization_publisher = self.create_publisher(
                MarkerArray,
                visualization_topic,
                10,
            )

        ground_truth_topic = (
            self.get_parameter("ground_truth_topic").get_parameter_value().string_value
        )
        self._ground_truth_subscription = self.create_subscription(
            Odometry,
            ground_truth_topic,
            self._ground_truth_callback,
            10,
        )

        rate_hz = self.get_parameter("sensor_rate_hz").get_parameter_value().double_value
        timer_period = 1.0 / max(rate_hz, 1e-3)
        self._sensor_timer = self.create_timer(timer_period, self._sensor_timer_callback)

        camera_names = ", ".join(camera.name for camera in self._cameras)
        self.get_logger().info(
            f"Camera simulator ready: {len(self._cameras)} cameras [{camera_names}], "
            f"ground truth on '{ground_truth_topic}', config '{config_path}'"
        )

    def _resolve_config_path(self, configured_path: str) -> Path:
        if configured_path:
            return Path(configured_path).expanduser()

        try:
            share_dir = Path(get_package_share_directory("befr_visual_tracking"))
            installed = share_dir / "config" / "cameras.yaml"
            if installed.is_file():
                return installed
        except PackageNotFoundError:
            pass

        return default_cameras_yaml_path()

    def _ground_truth_callback(self, message: Odometry) -> None:
        position = message.pose.pose.position
        self._latest_position = np.array([position.x, position.y, position.z], dtype=float)
        self._has_ground_truth = True

    def _sensor_timer_callback(self) -> None:
        if not self._has_ground_truth or self._latest_position is None:
            return

        stamp = self.get_clock().now().to_msg()
        stamp_seconds = stamp_to_seconds(stamp)
        target_position = self._latest_position
        measurements: dict[str, CameraMeasurement] = {}

        for camera in self._cameras:
            measurement = camera.measure(target_position, stamp_seconds, rng=self._rng)
            if measurement is None:
                continue

            measurements[camera.name] = measurement
            message = PointStamped()
            message.header.stamp = stamp
            message.header.frame_id = camera.frame_id
            message.point.x = float(measurement.u)
            message.point.y = float(measurement.v)
            message.point.z = 0.0
            self._detection_publishers[camera.name].publish(message)

        if self._visualization_publisher is not None:
            markers = build_sensor_markers(
                world_frame_id=self._world_frame_id,
                stamp=stamp,
                cameras=self._cameras,
                target_position=target_position,
                measurements=measurements,
                ray_length=self.get_parameter("ray_length").get_parameter_value().double_value,
                axis_length=self.get_parameter("axis_length").get_parameter_value().double_value,
            )
            self._visualization_publisher.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraSimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
