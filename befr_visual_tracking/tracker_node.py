"""Configurable ROS tracking node for KF, EKF, and UKF."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import CameraNoiseModel, cameras_from_yaml
from befr_visual_tracking.odometry_publisher import odometry_from_tracker
from befr_visual_tracking.tracker_base import TrackerBase, TrackerDetection
from befr_visual_tracking.tracker_factory import create_tracker


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class TrackerNode(Node):
    """Subscribe to pixel detections, run the selected filter, publish state estimates."""

    def __init__(self) -> None:
        super().__init__("visual_tracker")

        self.declare_parameter("filter_type", "ekf")
        self.declare_parameter("cameras_config_path", "")
        self.declare_parameter("detection_topic_prefix", "/visual_tracking")
        self.declare_parameter("state_estimate_topic", "/visual_tracking/state_estimate")
        self.declare_parameter("active_cameras", ["camera_0", "camera_1", "camera_2"])
        self.declare_parameter("acceleration_noise_std", 0.5)
        self.declare_parameter("pixel_noise_std_u", 1.0)
        self.declare_parameter("pixel_noise_std_v", 1.0)
        self.declare_parameter("innovation_gate", 9.21)
        self.declare_parameter("monte_carlo_samples", 500)
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("sync_slop_sec", 0.002)

        config_path = self._resolve_config_path(
            self.get_parameter("cameras_config_path").get_parameter_value().string_value
        )
        self._system_config, cameras = cameras_from_yaml(config_path)
        self._world_frame_id = self._system_config.world.frame_id

        pixel_noise_std_u = (
            self.get_parameter("pixel_noise_std_u").get_parameter_value().double_value
        )
        pixel_noise_std_v = (
            self.get_parameter("pixel_noise_std_v").get_parameter_value().double_value
        )
        for camera in cameras:
            camera.noise_model = CameraNoiseModel(
                pixel_noise_std_u=pixel_noise_std_u,
                pixel_noise_std_v=pixel_noise_std_v,
                discretize=camera.noise_model.discretize,
                detection_probability=camera.noise_model.detection_probability,
            )

        self._cameras_by_id = {camera.name: camera for camera in cameras}
        self._active_cameras = list(
            self.get_parameter("active_cameras").get_parameter_value().string_array_value
        )
        self._sync_slop_sec = (
            self.get_parameter("sync_slop_sec").get_parameter_value().double_value
        )

        filter_type = self.get_parameter("filter_type").get_parameter_value().string_value
        normalized_filter = filter_type.strip().lower()
        monte_carlo_noise = 0.5 * (pixel_noise_std_u + pixel_noise_std_v)
        self._tracker: TrackerBase = create_tracker(
            filter_type,
            self._cameras_by_id,
            active_cameras=self._active_cameras,
            acceleration_noise_std=self.get_parameter("acceleration_noise_std")
            .get_parameter_value()
            .double_value,
            pixel_noise_std=monte_carlo_noise if normalized_filter == "kf" else None,
            innovation_gate=self.get_parameter("innovation_gate")
            .get_parameter_value()
            .double_value,
            monte_carlo_samples=self.get_parameter("monte_carlo_samples")
            .get_parameter_value()
            .integer_value,
            random_seed=self.get_parameter("random_seed").get_parameter_value().integer_value,
        )

        self._detection_buffer: dict[float, dict[str, TrackerDetection]] = defaultdict(dict)
        self._pending_timers: dict[float, object] = {}

        state_topic = (
            self.get_parameter("state_estimate_topic").get_parameter_value().string_value
        )
        self._state_publisher = self.create_publisher(Odometry, state_topic, 10)

        prefix = self.get_parameter("detection_topic_prefix").get_parameter_value().string_value
        self._detection_subscriptions = []
        for camera_id in self._active_cameras:
            topic = f"{prefix}/{camera_id}/detection"
            self._detection_subscriptions.append(
                self.create_subscription(
                    PointStamped,
                    topic,
                    lambda msg, cam_id=camera_id: self._detection_callback(msg, cam_id),
                    10,
                )
            )

        self.get_logger().info(
            f"Visual tracker ready: filter={filter_type}, active_cameras={self._active_cameras}, "
            f"publishing to '{state_topic}'"
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

    def _detection_callback(self, message: PointStamped, camera_id: str) -> None:
        timestamp = stamp_to_seconds(message.header.stamp)
        detection = TrackerDetection(
            timestamp=timestamp,
            camera_id=camera_id,
            u=float(message.point.x),
            v=float(message.point.y),
        )
        self._detection_buffer[timestamp][camera_id] = detection

        if timestamp not in self._pending_timers:
            self._pending_timers[timestamp] = self.create_timer(
                self._sync_slop_sec,
                lambda ts=timestamp: self._process_buffered_timestamp(ts),
            )

    def _process_buffered_timestamp(self, timestamp: float) -> None:
        timer = self._pending_timers.pop(timestamp, None)
        if timer is not None:
            timer.cancel()

        buffered = self._detection_buffer.pop(timestamp, {})
        detections = [
            buffered[camera_id]
            for camera_id in self._active_cameras
            if camera_id in buffered
        ]
        # Skip partial sets: single-camera EKF updates on a loaded VM cause divergence.
        if len(detections) < 2:
            return

        self._tracker.process_timestep(timestamp, detections)
        if not self._tracker.is_initialized:
            return

        stamp_msg = self.get_clock().now().to_msg()
        odometry = odometry_from_tracker(
            self._tracker,
            frame_id=self._world_frame_id,
            child_frame_id="target/base_link",
            stamp=stamp_msg,
        )
        self._state_publisher.publish(odometry)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
