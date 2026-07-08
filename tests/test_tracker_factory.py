"""Tests for tracker factory and odometry publishing."""

from __future__ import annotations

import unittest

import numpy as np

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.ekf_tracker import EkfTracker
from befr_visual_tracking.kalman_tracker import KalmanTracker
from befr_visual_tracking.state_estimate_export import state_estimate_from_tracker
from befr_visual_tracking.tracker_factory import create_tracker
from befr_visual_tracking.ukf_tracker import UkfTracker


class TestTrackerFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.cameras = cameras_from_yaml(default_cameras_yaml_path())
        cls.cameras_by_id = {camera.name: camera for camera in cls.cameras}

    def test_create_kf(self) -> None:
        tracker = create_tracker("kf", self.cameras_by_id)
        self.assertIsInstance(tracker, KalmanTracker)

    def test_create_ekf(self) -> None:
        tracker = create_tracker("ekf", self.cameras_by_id)
        self.assertIsInstance(tracker, EkfTracker)

    def test_create_ukf(self) -> None:
        tracker = create_tracker("ukf", self.cameras_by_id)
        self.assertIsInstance(tracker, UkfTracker)

    def test_invalid_filter_type(self) -> None:
        with self.assertRaises(ValueError):
            create_tracker("particle", self.cameras_by_id)


class TestStateEstimateExport(unittest.TestCase):
    def test_exports_tracker_state(self) -> None:
        _, cameras = cameras_from_yaml(default_cameras_yaml_path())
        tracker = create_tracker("ekf", {camera.name: camera for camera in cameras})
        tracker.initialize(
            np.array([1.0, 2.0, 3.0]),
            np.diag([0.5, 0.5, 0.5, 0.2, 0.2, 0.2]),
            timestamp=0.0,
            velocity=[0.1, -0.2, 0.0],
        )

        export = state_estimate_from_tracker(tracker)
        self.assertAlmostEqual(export.position[0], 1.0)
        self.assertAlmostEqual(export.velocity[0], 0.1)
        self.assertAlmostEqual(export.pose_covariance[0], 0.5)
        self.assertGreater(export.pose_covariance[21], 1e5)


if __name__ == "__main__":
    unittest.main()
