"""Tests for multi-ray triangulation."""

from __future__ import annotations

import unittest

import numpy as np

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.triangulation import (
    PixelObservation,
    build_camera_ray,
    estimate_triangulation_covariance_monte_carlo,
    minimum_ray_angle_deg,
    triangulate_from_cameras,
    triangulate_from_observations,
    triangulate_rays,
)


class TestTriangulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.cameras = cameras_from_yaml(default_cameras_yaml_path())
        cls.cameras_by_id = {camera.name: camera for camera in cls.cameras}

    def test_recovers_known_point_from_four_cameras(self) -> None:
        target = np.array([0.2, -0.1, 1.6])
        detections = {}
        for camera in self.cameras:
            uv = camera.project_world_point(target)
            self.assertIsNotNone(uv)
            assert uv is not None
            detections[camera.name] = uv

        result = triangulate_from_cameras(self.cameras, detections)
        self.assertTrue(result.quality.is_valid)
        assert result.point is not None
        np.testing.assert_allclose(result.point, target, atol=1e-6)
        self.assertAlmostEqual(result.quality.reprojection_rmse, 0.0, places=6)
        self.assertLess(result.quality.ray_intersection_residual, 1e-6)

    def test_two_cameras_minimum(self) -> None:
        target = np.array([0.0, 0.0, 1.5])
        selected = self.cameras[:2]
        detections = {}
        for camera in selected:
            uv = camera.project_world_point(target)
            assert uv is not None
            detections[camera.name] = uv

        result = triangulate_from_cameras(selected, detections)
        self.assertTrue(result.quality.is_valid)
        assert result.point is not None
        np.testing.assert_allclose(result.point, target, atol=1e-5)

    def test_rejects_parallel_rays(self) -> None:
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        rays = [
            build_camera_ray(self.cameras[0], 320.0, 240.0),
            build_camera_ray(self.cameras[0], 321.0, 240.0),
        ]
        rays[0] = type(rays[0])("camera_a", origin, direction)
        rays[1] = type(rays[1])("camera_b", origin + np.array([0.1, 0.0, 0.0]), direction)

        result = triangulate_rays(rays, min_angle_deg=2.0)
        self.assertFalse(result.quality.is_valid)
        self.assertIsNone(result.point)
        self.assertLess(result.quality.min_ray_angle_deg, 2.0)

    def test_minimum_ray_angle(self) -> None:
        angle = minimum_ray_angle_deg([np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])])
        self.assertAlmostEqual(angle, 90.0)

    def test_monte_carlo_covariance_is_positive_semidefinite(self) -> None:
        target = np.array([0.3, 0.2, 1.55])
        observations: list[PixelObservation] = []
        for camera in self.cameras:
            uv = camera.project_world_point(target)
            assert uv is not None
            observations.append(PixelObservation(camera.name, uv[0], uv[1]))

        covariance = estimate_triangulation_covariance_monte_carlo(
            cameras_by_id=self.cameras_by_id,
            observations=observations,
            pixel_noise_std=1.0,
            num_samples=300,
            rng=np.random.default_rng(42),
        )
        self.assertEqual(covariance.shape, (3, 3))
        eigenvalues = np.linalg.eigvalsh(covariance)
        self.assertTrue(np.all(eigenvalues >= -1e-9))
        self.assertGreater(np.trace(covariance), 0.0)

    def test_triangulation_with_covariance_wrapper(self) -> None:
        target = np.array([-0.2, 0.4, 1.5])
        observations: list[PixelObservation] = []
        for camera in self.cameras:
            uv = camera.project_world_point(target)
            assert uv is not None
            observations.append(PixelObservation(camera.name, uv[0], uv[1]))

        result = triangulate_from_observations(
            self.cameras_by_id,
            observations,
            estimate_covariance=True,
            pixel_noise_std=1.0,
            monte_carlo_samples=200,
            rng=np.random.default_rng(7),
        )
        self.assertTrue(result.quality.is_valid)
        assert result.point is not None
        assert result.covariance is not None
        np.testing.assert_allclose(result.point, target, atol=0.05)


if __name__ == "__main__":
    unittest.main()
