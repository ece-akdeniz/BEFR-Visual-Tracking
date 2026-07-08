"""Tests for the raw-pixel Unscented Kalman Filter."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml, make_camera
from befr_visual_tracking.ekf_tracker import run_ekf_tracker_on_dataset, compute_position_rmse as ekf_position_rmse
from befr_visual_tracking.kalman_tracker import run_kalman_tracker_on_dataset, compute_position_rmse as kf_position_rmse
from befr_visual_tracking.motion_model import state_from_position_velocity
from befr_visual_tracking.tracker_base import TrackerDetection
from befr_visual_tracking.ukf_core import generate_sigma_points
from befr_visual_tracking.ukf_tracker import (
    UkfTracker,
    compute_position_rmse,
    compute_velocity_rmse,
    run_ukf_tracker_on_dataset,
    ukf_pixel_update,
)


class TestUkfCore(unittest.TestCase):
    def test_generates_thirteen_sigma_points(self) -> None:
        state = state_from_position_velocity([0.0, 0.0, 1.5], [0.1, 0.0, 0.0])
        covariance = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        sigma_points, _, _, _ = generate_sigma_points(state, covariance)
        self.assertEqual(sigma_points.shape, (13, 6))
        np.testing.assert_allclose(sigma_points[0], state)


class TestUkfUpdate(unittest.TestCase):
    def test_ukf_update_accepts_valid_measurement(self) -> None:
        state = state_from_position_velocity([0.0, 0.0, 2.0], [0.0, 0.0, 0.0])
        covariance = np.diag([0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
        camera = make_camera()
        measurement = np.array([325.0, 235.0])

        updated_state, _, accepted, _, invalid_count = ukf_pixel_update(
            state,
            covariance,
            measurement,
            camera,
            camera.noise_model.measurement_noise_covariance,
        )
        self.assertTrue(accepted)
        self.assertEqual(invalid_count, 0)
        self.assertFalse(np.allclose(updated_state, state))

    def test_ukf_update_rejects_when_too_many_invalid_sigma_points(self) -> None:
        state = state_from_position_velocity([0.0, 0.0, -0.5], [0.0, 0.0, 0.0])
        covariance = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        camera = make_camera()
        measurement = np.array([320.0, 240.0])

        _, _, accepted, _, invalid_count = ukf_pixel_update(
            state,
            covariance,
            measurement,
            camera,
            camera.noise_model.measurement_noise_covariance,
            max_invalid_sigma_points=2,
        )
        self.assertFalse(accepted)
        self.assertGreater(invalid_count, 2)


class TestUkfTracker(unittest.TestCase):
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

    def test_sequential_multi_camera_updates(self) -> None:
        tracker = UkfTracker(
            self.cameras_by_id,
            monte_carlo_samples=50,
            rng=np.random.default_rng(1),
        )
        target = np.array([0.0, 0.0, 1.5])
        tracker.process_timestep(0.0, self._detections_for_target(target))
        self.assertGreaterEqual(tracker.stats.num_camera_updates, len(self.cameras))

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "results" / "canonical_slow_circle").is_dir(),
        "canonical dataset not generated",
    )
    def test_tracks_canonical_slow_circle(self) -> None:
        result = run_ukf_tracker_on_dataset(
            experiment_dir=self.canonical_dir,
            cameras_by_id=self.cameras_by_id,
            monte_carlo_samples=80,
            rng=np.random.default_rng(42),
        )
        self.assertGreater(len(result.states), 100)
        self.assertLess(compute_position_rmse(result), 0.15)
        self.assertLess(compute_velocity_rmse(result), 0.15)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "results" / "canonical_slow_circle").is_dir(),
        "canonical dataset not generated",
    )
    def test_all_filters_comparable_on_same_dataset(self) -> None:
        kf_result = run_kalman_tracker_on_dataset(
            experiment_dir=self.canonical_dir,
            cameras_by_id=self.cameras_by_id,
            monte_carlo_samples=80,
            rng=np.random.default_rng(42),
        )
        ekf_result = run_ekf_tracker_on_dataset(
            experiment_dir=self.canonical_dir,
            cameras_by_id=self.cameras_by_id,
            monte_carlo_samples=80,
            rng=np.random.default_rng(42),
        )
        ukf_result = run_ukf_tracker_on_dataset(
            experiment_dir=self.canonical_dir,
            cameras_by_id=self.cameras_by_id,
            monte_carlo_samples=80,
            rng=np.random.default_rng(42),
        )

        self.assertEqual(len(kf_result.states), len(ekf_result.states))
        self.assertEqual(len(ekf_result.states), len(ukf_result.states))
        self.assertLess(kf_position_rmse(kf_result), 0.35)
        self.assertLess(ekf_position_rmse(ekf_result), 0.15)
        self.assertLess(compute_position_rmse(ukf_result), 0.15)


if __name__ == "__main__":
    unittest.main()
