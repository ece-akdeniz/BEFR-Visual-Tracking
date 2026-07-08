"""Tests for the triangulation + linear Kalman baseline."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.kalman_tracker import (
    KalmanTracker,
    compute_position_rmse,
    compute_velocity_rmse,
    linear_position_update,
    run_kalman_tracker_on_dataset,
)
from befr_visual_tracking.motion_model import state_from_position_velocity
from befr_visual_tracking.tracker_base import TrackerDetection


class TestLinearKalmanUpdate(unittest.TestCase):
    def test_update_moves_state_toward_measurement(self) -> None:
        state = state_from_position_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        covariance = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        measurement = np.array([1.0, 0.0, 0.0])
        measurement_covariance = 0.1 * np.eye(3)

        updated_state, _ = linear_position_update(
            state, covariance, measurement, measurement_covariance
        )
        self.assertGreater(updated_state[0], state[0])
        self.assertLess(updated_state[0], measurement[0])


class TestKalmanTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.cameras = cameras_from_yaml(default_cameras_yaml_path())
        cls.cameras_by_id = {camera.name: camera for camera in cls.cameras}
        cls.canonical_dir = (
            Path(__file__).resolve().parents[1] / "results" / "canonical_slow_circle"
        )

    def _synthetic_detections(self, target: np.ndarray, timestamp: float = 0.0):
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

    def test_initializes_from_triangulation(self) -> None:
        tracker = KalmanTracker(
            self.cameras_by_id,
            monte_carlo_samples=50,
            rng=np.random.default_rng(1),
        )
        target = np.array([0.2, -0.1, 1.6])
        initialized = tracker.try_initialize(0.0, self._synthetic_detections(target))
        self.assertTrue(initialized)
        np.testing.assert_allclose(tracker.get_position(), target, atol=0.05)

    def test_prediction_only_with_one_camera(self) -> None:
        tracker = KalmanTracker(
            self.cameras_by_id,
            monte_carlo_samples=50,
            rng=np.random.default_rng(2),
        )
        tracker.initialize(
            np.array([0.0, 0.0, 1.5]),
            np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5]),
            timestamp=0.0,
            velocity=[0.3, 0.0, 0.0],
        )
        position_before = tracker.get_position().copy()
        covariance_before = tracker.get_covariance().copy()

        one_camera = self._synthetic_detections(np.array([0.03, 0.0, 1.5]), timestamp=0.1)[:1]
        tracker.process_timestep(0.1, one_camera)

        self.assertGreater(tracker.stats.num_prediction_only_steps, 0)
        self.assertGreater(tracker.get_position()[0], position_before[0])
        self.assertGreater(tracker.get_covariance()[0, 0], covariance_before[0, 0])

    def test_rejects_single_camera_update(self) -> None:
        tracker = KalmanTracker(
            self.cameras_by_id,
            monte_carlo_samples=50,
            rng=np.random.default_rng(3),
        )
        target = np.array([0.0, 0.0, 1.5])
        tracker.try_initialize(0.0, self._synthetic_detections(target))
        updates_before = tracker.stats.num_updates
        tracker.update(self._synthetic_detections(target)[:1])
        self.assertEqual(tracker.stats.num_updates, updates_before)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "results" / "canonical_slow_circle").is_dir(),
        "canonical dataset not generated",
    )
    def test_tracks_canonical_slow_circle(self) -> None:
        result = run_kalman_tracker_on_dataset(
            experiment_dir=self.canonical_dir,
            cameras_by_id=self.cameras_by_id,
            monte_carlo_samples=80,
            rng=np.random.default_rng(42),
        )
        position_rmse = compute_position_rmse(result)
        velocity_rmse = compute_velocity_rmse(result)

        self.assertGreater(len(result.states), 100)
        self.assertLess(position_rmse, 0.35)
        self.assertLess(velocity_rmse, 0.25)


if __name__ == "__main__":
    unittest.main()
