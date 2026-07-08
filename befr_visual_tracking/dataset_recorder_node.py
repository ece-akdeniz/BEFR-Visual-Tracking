"""ROS node that records canonical datasets during simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.dataset_generator import CanonicalDataset, DetectionRow, GroundTruthRow
from befr_visual_tracking.result_writer import save_canonical_dataset
from befr_visual_tracking.trajectory_design import TrajectorySample


class DatasetRecorderNode(Node):
    """Record synchronized detections and ground truth to the standard CSV format."""

    def __init__(self) -> None:
        super().__init__("dataset_recorder")

        self.declare_parameter("ground_truth_topic", "/model/rrbot/odometry")
        self.declare_parameter("cameras_config_path", "")
        self.declare_parameter("sensor_rate_hz", 30.0)
        self.declare_parameter("record_duration_s", 0.0)
        self.declare_parameter("output_dir", "")
        self.declare_parameter("experiment_name", "live_recording")
        self.declare_parameter("trajectory_name", "unknown")
        self.declare_parameter("random_seed", 42)

        config_path = self._resolve_config_path(
            self.get_parameter("cameras_config_path").get_parameter_value().string_value
        )
        self._system_config, self._cameras = cameras_from_yaml(config_path)
        self._camera_config_path = config_path

        seed = self.get_parameter("random_seed").get_parameter_value().integer_value
        self._rng = np.random.default_rng(seed)
        self._random_seed = seed

        self._latest_position: np.ndarray | None = None
        self._latest_velocity: np.ndarray | None = None
        self._has_ground_truth = False

        self._detection_rows: list[DetectionRow] = []
        self._ground_truth_rows: list[GroundTruthRow] = []
        self._seen_timestamps: set[float] = set()
        self._recording_started = False
        self._elapsed = 0.0
        self._dataset_saved = False

        output_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        if not output_dir:
            output_dir = str(Path.cwd() / "results" / "live_recording")
        self._output_dir = Path(output_dir)

        self._experiment_name = (
            self.get_parameter("experiment_name").get_parameter_value().string_value
        )
        self._trajectory_name = (
            self.get_parameter("trajectory_name").get_parameter_value().string_value
        )
        self._record_duration = (
            self.get_parameter("record_duration_s").get_parameter_value().double_value
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
        self._sensor_period = 1.0 / max(rate_hz, 1e-3)
        self._sensor_timer = self.create_timer(self._sensor_period, self._sensor_timer_callback)

        self.get_logger().info(
            f"Dataset recorder ready. Output directory: {self._output_dir}. "
            f"Ground truth topic: {ground_truth_topic}"
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
        velocity = message.twist.twist.linear
        self._latest_position = np.array([position.x, position.y, position.z], dtype=float)
        self._latest_velocity = np.array([velocity.x, velocity.y, velocity.z], dtype=float)
        self._has_ground_truth = True

    def _sensor_timer_callback(self) -> None:
        if not self._has_ground_truth:
            return
        if self._latest_position is None or self._latest_velocity is None:
            return

        if self._record_duration > 0.0 and self._elapsed >= self._record_duration:
            self._finalize_and_shutdown()
            return

        stamp = self.get_clock().now()
        timestamp = float(stamp.nanoseconds) * 1e-9
        if timestamp in self._seen_timestamps:
            return
        self._seen_timestamps.add(timestamp)

        sample = TrajectorySample(
            timestamp=timestamp,
            position=self._latest_position.copy(),
            velocity=self._latest_velocity.copy(),
        )
        self._append_sample(sample)
        self._recording_started = True
        self._elapsed += self._sensor_period

    def _append_sample(self, sample: TrajectorySample) -> None:
        position = sample.position
        velocity = sample.velocity
        self._ground_truth_rows.append(
            GroundTruthRow(
                timestamp=sample.timestamp,
                gt_x=float(position[0]),
                gt_y=float(position[1]),
                gt_z=float(position[2]),
                gt_vx=float(velocity[0]),
                gt_vy=float(velocity[1]),
                gt_vz=float(velocity[2]),
            )
        )

        for camera in self._cameras:
            ideal = camera.project_world_point(position)
            visible = ideal is not None
            u_ideal = float(ideal[0]) if ideal is not None else None
            v_ideal = float(ideal[1]) if ideal is not None else None
            measurement = camera.measure(position, sample.timestamp, rng=self._rng)
            u_noisy = float(measurement.u) if measurement is not None else None
            v_noisy = float(measurement.v) if measurement is not None else None

            self._detection_rows.append(
                DetectionRow(
                    timestamp=sample.timestamp,
                    camera_id=camera.name,
                    u=u_noisy,
                    v=v_noisy,
                    u_ideal=u_ideal,
                    v_ideal=v_ideal,
                    visible=visible,
                    gt_x=float(position[0]),
                    gt_y=float(position[1]),
                    gt_z=float(position[2]),
                    gt_vx=float(velocity[0]),
                    gt_vy=float(velocity[1]),
                    gt_vz=float(velocity[2]),
                )
            )

    def _save_dataset(self) -> Path | None:
        if self._dataset_saved or not self._detection_rows:
            return None

        dataset = CanonicalDataset(
            detections=tuple(self._detection_rows),
            ground_truth=tuple(self._ground_truth_rows),
            random_seed=self._random_seed,
        )
        metadata = {
            "experiment": self._experiment_name,
            "trajectory": self._trajectory_name,
            "source": "ros_live_recorder",
            "active_cameras": [camera.name for camera in self._cameras],
            "pixel_noise_std": self._system_config.noise.sigma_u,
            "dropout_scenario": "none",
            "record_duration_s": self._elapsed,
        }
        output_path = save_canonical_dataset(
            self._output_dir,
            dataset,
            metadata=metadata,
            camera_config_path=self._camera_config_path,
        )
        self._dataset_saved = True
        return output_path

    def _finalize_and_shutdown(self) -> None:
        if not self._recording_started:
            self.get_logger().warning("No samples recorded; skipping dataset write.")
            rclpy.shutdown()
            return

        output_path = self._save_dataset()
        if output_path is not None:
            self.get_logger().info(f"Dataset saved to {output_path}")
        rclpy.shutdown()

    def destroy_node(self) -> bool:
        if self._recording_started and not self._dataset_saved:
            output_path = self._save_dataset()
            if output_path is not None:
                self.get_logger().info(f"Dataset saved to {output_path}")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
