"""Tests for the pinhole camera model."""

from __future__ import annotations

import unittest

import numpy as np

from befr_visual_tracking.camera_config import (
    default_cameras_yaml_path,
    load_camera_system_config,
    look_at_orientation_xyzw,
)
from befr_visual_tracking.camera_model import Camera, cameras_from_yaml, make_camera


def camera_from_system_config(camera_id: str) -> Camera:
    _, cameras = cameras_from_yaml(default_cameras_yaml_path())
    for camera in cameras:
        if camera.name == camera_id:
            return camera
    raise KeyError(f"Unknown camera id: {camera_id!r}")


class TestIdentityCameraProjection(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = make_camera(
            position=(0.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        self.cx, self.cy = self.camera.intrinsics.principal_point

    def test_projection_to_image_centre(self) -> None:
        uv = self.camera.project_world_point(np.array([0.0, 0.0, 2.0]))
        self.assertIsNotNone(uv)
        assert uv is not None
        u, v = uv
        self.assertAlmostEqual(u, self.cx)
        self.assertAlmostEqual(v, self.cy)

    def test_movement_right_increases_u(self) -> None:
        centre = self.camera.project_world_point(np.array([0.0, 0.0, 2.0]))
        right = self.camera.project_world_point(np.array([0.2, 0.0, 2.0]))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(right)
        assert centre is not None and right is not None
        self.assertGreater(right[0], centre[0])
        self.assertAlmostEqual(right[1], centre[1])

    def test_movement_down_increases_v(self) -> None:
        centre = self.camera.project_world_point(np.array([0.0, 0.0, 2.0]))
        down = self.camera.project_world_point(np.array([0.0, 0.2, 2.0]))
        self.assertIsNotNone(centre)
        self.assertIsNotNone(down)
        assert centre is not None and down is not None
        self.assertGreater(down[1], centre[1])
        self.assertAlmostEqual(down[0], centre[0])

    def test_negative_z_is_invisible(self) -> None:
        self.assertFalse(self.camera.is_visible(np.array([0.0, 0.0, -2.0])))
        self.assertIsNone(self.camera.project_world_point(np.array([0.0, 0.0, -2.0])))

    def test_points_outside_image_are_invisible(self) -> None:
        far_right = self.camera.project_camera_point(np.array([10.0, 0.0, 1.0]))
        far_down = self.camera.project_camera_point(np.array([0.0, 10.0, 1.0]))
        too_left = self.camera.project_camera_point(np.array([-10.0, 0.0, 1.0]))
        too_up = self.camera.project_camera_point(np.array([0.0, -10.0, 1.0]))

        self.assertIsNone(far_right)
        self.assertIsNone(far_down)
        self.assertIsNone(too_left)
        self.assertIsNone(too_up)

        self.assertFalse(self.camera.is_visible(np.array([10.0, 0.0, 1.0])))
        self.assertFalse(self.camera.is_visible(np.array([0.0, -10.0, 1.0])))


class TestRotatedCameraProjection(unittest.TestCase):
    def test_rotated_camera_projects_flight_volume_centre(self) -> None:
        config = load_camera_system_config(default_cameras_yaml_path())
        target = np.array(config.world.flight_volume_center)

        for camera_id in ("camera_0", "camera_1", "camera_2", "camera_3"):
            camera = camera_from_system_config(camera_id)
            uv = camera.project_world_point(target)
            self.assertIsNotNone(uv, msg=camera_id)
            assert uv is not None
            u, v = uv
            self.assertGreaterEqual(u, 0.0)
            self.assertLess(u, camera.intrinsics.width_px)
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, camera.intrinsics.height_px)

    def test_look_at_camera_points_optical_axis_at_target(self) -> None:
        position = (4.0, 0.0, 1.5)
        target = (0.0, 0.0, 1.5)
        orientation = look_at_orientation_xyzw(position, target)
        camera = make_camera(position=position, orientation_xyzw=orientation)

        uv = camera.project_world_point(np.array(target))
        cx, cy = camera.intrinsics.principal_point
        self.assertIsNotNone(uv)
        assert uv is not None
        self.assertAlmostEqual(uv[0], cx, places=5)
        self.assertAlmostEqual(uv[1], cy, places=5)


class TestCameraRays(unittest.TestCase):
    def test_camera_ray_projects_back_to_original_pixel(self) -> None:
        camera = camera_from_system_config("camera_1")
        point_world = np.array([0.3, -0.2, 1.6])
        uv = camera.project_world_point(point_world)
        self.assertIsNotNone(uv)
        assert uv is not None
        u, v = uv

        origin, direction = camera.pixel_to_world_ray(u, v)
        for distance in (2.0, 4.0, 6.0):
            sample = origin + distance * direction
            reprojected = camera.project_world_point(sample)
            self.assertIsNotNone(reprojected)
            assert reprojected is not None
            self.assertAlmostEqual(reprojected[0], u, places=6)
            self.assertAlmostEqual(reprojected[1], v, places=6)

    def test_pixel_to_camera_ray_is_unit_length(self) -> None:
        camera = make_camera()
        direction = camera.pixel_to_camera_ray(400.0, 200.0)
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0)

    def test_world_ray_direction_matches_camera_frame(self) -> None:
        camera = camera_from_system_config("camera_2")
        u, v = 350.0, 260.0
        direction_camera = camera.pixel_to_camera_ray(u, v)
        _, direction_world = camera.pixel_to_world_ray(u, v)
        expected = camera.extrinsics.R_camera_to_world @ direction_camera
        np.testing.assert_allclose(direction_world, expected, atol=1e-9)


class TestCameraValidation(unittest.TestCase):
    def test_validate_frame_convention(self) -> None:
        make_camera().validate_frame_convention()


if __name__ == "__main__":
    unittest.main()
