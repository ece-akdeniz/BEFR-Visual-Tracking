"""Tests for the analytical camera projection Jacobian."""

from __future__ import annotations

import unittest

import numpy as np

from befr_visual_tracking.camera_config import look_at_orientation_xyzw
from befr_visual_tracking.camera_model import make_camera


def numerical_projection_jacobian(
    camera,
    point_world: np.ndarray,
    delta: float = 1e-5,
) -> np.ndarray:
    """Central-difference approximation of d(u, v) / d(p_world)."""
    point_world = np.asarray(point_world, dtype=float).reshape(3)
    base = camera.project_world_point(point_world)
    if base is None:
        raise ValueError("Base point must be visible for numerical Jacobian")

    jacobian = np.zeros((2, 3), dtype=float)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = delta
        plus = camera.project_world_point(point_world + offset)
        minus = camera.project_world_point(point_world - offset)
        if plus is None or minus is None:
            raise ValueError("Perturbed points must remain visible")
        jacobian[0, axis] = (plus[0] - minus[0]) / (2.0 * delta)
        jacobian[1, axis] = (plus[1] - minus[1]) / (2.0 * delta)
    return jacobian


class TestProjectionJacobian(unittest.TestCase):
    def test_identity_camera_matches_finite_differences(self) -> None:
        camera = make_camera()
        point_world = np.array([0.4, -0.3, 2.5])
        analytical = camera.projection_jacobian(point_world)
        numerical = numerical_projection_jacobian(camera, point_world)
        np.testing.assert_allclose(analytical, numerical, atol=1e-4)

    def test_rotated_camera_matches_finite_differences(self) -> None:
        orientation = look_at_orientation_xyzw((3.0, 3.0, 3.5), (0.0, 0.0, 1.5))
        camera = make_camera(position=(3.0, 3.0, 3.5), orientation_xyzw=orientation)
        point_world = np.array([0.2, 0.1, 1.7])
        analytical = camera.projection_jacobian(point_world)
        numerical = numerical_projection_jacobian(camera, point_world)
        np.testing.assert_allclose(analytical, numerical, atol=1e-4)

    def test_measurement_jacobian_has_zero_velocity_block(self) -> None:
        camera = make_camera()
        H = camera.measurement_jacobian(np.array([0.1, 0.2, 3.0]))
        self.assertEqual(H.shape, (2, 6))
        np.testing.assert_allclose(H[:, 3:], 0.0)

    def test_jacobian_raises_for_point_behind_camera(self) -> None:
        camera = make_camera()
        with self.assertRaises(ValueError):
            camera.projection_jacobian(np.array([0.0, 0.0, -1.0]))


if __name__ == "__main__":
    unittest.main()
