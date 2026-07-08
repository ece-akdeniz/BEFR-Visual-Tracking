"""Tests for the raw-pixel Extended Kalman Filter."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml, make_camera
from befr_visual_tracking.ekf_tracker import (
    EkfTracker,
    compute_position_rmse,
    compute_velocity_rmse,
    ekf_pixel_update,
    predicted_pixel_measurement,
    run_ekf_tracker_on_dataset,
)
from befr_visual_tracking.motion_model import state_from_position_velocity
from befr_visual_tracking.tracker_base import TrackerDetection


class TestEkfProjection(unittest.TestCase):
    def test_filter_projection_allows_pixels_outside_image(self) -> None:
        camera = make_camera(position=(0.0, 0.0, 0.0))
        point_far_right = np.array([20.0, 0.0, 1.0])
        sensor_projection = camera.project_world_point(point_far_right)
        filter_projection = camera.project_world_point_for_filter(point_far_right)

        self.assertIsNone(sensor_projection)
        self.assertIsNotNone(filter_projection)
        assert filter_projection is not None
        self.assertGreater(filter_projection[0], camera.intrinsics.width_px)

    def test_filter_projection_rejects_point_behind_camera(self) -> None:
        camera = make_camera()
        self.assertIsNone(camera.project_world_point_for_filter(np.array([0.0, 0.0, -1.0])))


class TestEkfUpdate(unittest.TestCase):
    def test_ekf_update_moves_toward_measurement(self) -> None:
        state = state_from_position_velocity([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
        covariance = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        camera = make_camera()

        prediction = predicted_pixel_measurement(camera, state)
        assert prediction is not None
        predicted_measurement, jacobian = prediction
        measurement = predicted_measurement + np.array([5.0, -3.0])

        updated_state, _, accepted, _ = ekf_pixel_update(
            state,
            covariance,
            measurement,
            predicted_measurement,
            jacobian,
            camera.noise_model.measurement_noise_covariance,
        )
        self.assertTrue(accepted)
        self.assertFalse(np.allclose(updated_state[:3], state[:3]))

    def test_innovation_gate_rejects_outlier(self) -> None:
        state = state_from_position_velocity([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
        covariance = 0.01 * np.eye(6)
        camera = make_camera()
        prediction = predicted_pixel_measurement(camera, state)
        assert prediction is not None
        predicted_measurement, jacobian = prediction
        measurement = predicted_measurement + np.array([500.0, 500.0])

        _, _, accepted, nis = ekf_pixel_update(
            state,
            covariance,
            measurement,
            predicted_measurement,
            jacobian,
            camera.noise_model.measurement_noise_covariance,
            innovation_gate=9.21,
        )
        self.assertFalse(accepted)
        self.assertGreater(nis, 9.21)


class TestEkfTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.cameras = cameras_from_yaml(default_cameras_yaml_path())
        cls.cameras_by_id = {camera.name: camera for camera in cls.cameras}
        cls.canonical_dir = (
            Path(__file__).resolve().parents[1] / "results" / "canonical_slow_circle"
        )

    def _detections_for_target(self, target: np.ndarray, timestamp: float = 0.0):
        detections: list[TrackerDetection] = []
        for camera in self.cameras:
            uv = camera.project_world_point(target)
            assert uv is not None
            detections.append(
                TrackerDetection(
                    timestamp=timestamp,
                    camera_id=camera.name,
                    u=uv[0],
                    v=uv[1],
                )
            )
        return detections

    def test_single_camera_update_supported(self) -> None:
        tracker = EkfTracker(
            self.cameras_by_id,
            monte_carlo_samples=50,
            rng=np.random.default_rng(1),
        )
        target = np.array([0.2, -0.1, 1.6])
        tracker.try_initialize(0.0, self._detections_for_target(target))
        updates_before = tracker.stats.num_camera_updates

        one_camera = self._detections_for_target(target + np.array([0.05, 0.0, 0.0]), 0.1)
        tracker.process_timestep(0.1, one_camera[:1])
        self.assertGreater(tracker.stats.num_camera_updates, updates_before)

    def test_sequential_multi_camera_updates(self) -> None:
        tracker = EkfTracker(
            self.cameras_by_id,
            monte_carlo_samples=50,
            rng=np.random.default_rng(2),
        )
        target = np.array([0.0, 0.0, 1.5])
        detections = self._detections_for_target(target)
        tracker.process_timestep(0.0, detections)
        self.assertGreaterEqual(tracker.stats.num_camera_updates, len(detections))

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "results" / "canonical_slow_circle").is_dir(),
        "canonical dataset not generated",
    )
    def test_tracks_canonical_slow_circle(self) -> None:
        result = run_ekf_tracker_on_dataset(
            experiment_dir=self.canonical_dir,
            cameras_by_id=self.cameras_by_id,
            monte_carlo_samples=80,
            rng=np.random.default_rng(42),
        )
        position_rmse = compute_position_rmse(result)
        velocity_rmse = compute_velocity_rmse(result)

        self.assertGreater(len(result.states), 100)
        self.assertLess(position_rmse, 0.15)
        self.assertLess(velocity_rmse, 0.15)


if __name__ == "__main__":
    unittest.main()
