"""Tests for the shared tracker interface."""

from __future__ import annotations

import unittest

import numpy as np

from befr_visual_tracking.tracker_base import TrackerBase, TrackerDetection


class DummyTracker(TrackerBase):
    """Concrete tracker used to test the shared predict/initialize path."""

    def update(self, detections: list[TrackerDetection]) -> None:
        _ = detections


class TestTrackerBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = DummyTracker(acceleration_noise_std=0.5)

    def test_initialize_and_get_state(self) -> None:
        position = np.array([0.0, 0.0, 1.5])
        covariance = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        self.tracker.initialize(position, covariance, timestamp=0.0, velocity=[0.1, 0.0, 0.0])

        state = self.tracker.get_state()
        np.testing.assert_allclose(state[:3], position)
        np.testing.assert_allclose(state[3:], [0.1, 0.0, 0.0])
        np.testing.assert_allclose(self.tracker.get_covariance(), covariance)

    def test_predict_advances_state(self) -> None:
        self.tracker.initialize(
            np.array([0.0, 0.0, 1.5]),
            np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            timestamp=0.0,
            velocity=[0.3, 0.0, 0.0],
        )
        self.tracker.predict(0.5)
        np.testing.assert_allclose(self.tracker.get_position(), [0.15, 0.0, 1.5], atol=1e-12)
        self.assertEqual(self.tracker.timestamp, 0.5)

    def test_predict_rejects_backwards_time(self) -> None:
        self.tracker.initialize(
            np.array([0.0, 0.0, 1.5]),
            np.eye(3),
            timestamp=1.0,
        )
        with self.assertRaises(ValueError):
            self.tracker.predict(0.5)

    def test_uninitialized_access_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.tracker.get_state()


if __name__ == "__main__":
    unittest.main()
