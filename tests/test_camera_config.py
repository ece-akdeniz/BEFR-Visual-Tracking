"""Tests for camera configuration and look-at pose generation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from befr_visual_tracking.camera_config import (
    default_cameras_yaml_path,
    load_camera_system_config,
    look_at_orientation_xyzw,
    rotation_matrix_from_quaternion_xyzw,
)


class TestLookAtOrientation(unittest.TestCase):
    def test_optical_z_points_at_target(self) -> None:
        position = (0.0, -4.0, 1.5)
        target = (0.0, 0.0, 1.5)
        q = look_at_orientation_xyzw(position, target)
        R = rotation_matrix_from_quaternion_xyzw(q)

        forward_world = R @ np.array([0.0, 0.0, 1.0])
        expected = np.array(target) - np.array(position)
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(forward_world, expected, atol=1e-9)

    def test_right_handed_optical_frame(self) -> None:
        q = look_at_orientation_xyzw((4.0, 0.0, 1.5), (0.0, 0.0, 1.5))
        R = rotation_matrix_from_quaternion_xyzw(q)
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        np.testing.assert_allclose(np.cross(x_axis, y_axis), z_axis, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(x_axis), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(y_axis), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(z_axis), 1.0, atol=1e-9)

    def test_flight_volume_center_in_front_of_all_cameras(self) -> None:
        config = load_camera_system_config(default_cameras_yaml_path())
        target = np.array(config.world.flight_volume_center)

        for camera in config.cameras:
            R = rotation_matrix_from_quaternion_xyzw(np.array(camera.orientation_xyzw))
            t = np.array(camera.position)
            R_wc = R.T
            t_wc = -R_wc @ t
            point_camera = R_wc @ target + t_wc
            self.assertGreater(
                point_camera[2],
                0.1,
                msg=f"{camera.id} does not see flight volume centre",
            )


class TestCameraYamlConfig(unittest.TestCase):
    def test_load_default_config(self) -> None:
        config = load_camera_system_config(default_cameras_yaml_path())

        self.assertEqual(config.image.width, 640)
        self.assertEqual(config.image.height, 480)
        self.assertEqual(config.intrinsics.fx, 320.0)
        self.assertEqual(config.noise.sigma_u, 1.0)
        self.assertTrue(config.noise.quantize)
        self.assertEqual(config.world.frame_id, "world")
        self.assertEqual(len(config.cameras), 4)

    def test_camera_ids_and_frames(self) -> None:
        config = load_camera_system_config(default_cameras_yaml_path())
        ids = [camera.id for camera in config.cameras]
        self.assertEqual(ids, ["camera_0", "camera_1", "camera_2", "camera_3"])

        for camera in config.cameras:
            self.assertEqual(camera.frame_id, f"{camera.id}_optical_frame")

    def test_stored_orientations_match_look_at(self) -> None:
        config = load_camera_system_config(default_cameras_yaml_path())
        target = config.world.flight_volume_center
        up = config.world.up

        for camera in config.cameras:
            expected = look_at_orientation_xyzw(camera.position, target, up)
            stored = np.array(camera.orientation_xyzw)
            # Quaternions q and -q represent the same rotation.
            dot = abs(float(np.dot(expected, stored)))
            self.assertGreater(dot, 1.0 - 1e-6, msg=camera.id)

    def test_cameras_are_not_collinear(self) -> None:
        config = load_camera_system_config(default_cameras_yaml_path())
        positions = np.array([camera.position for camera in config.cameras])

        # No three cameras should lie on a line through the flight volume.
        centre = np.array(config.world.flight_volume_center)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                for k in range(j + 1, len(positions)):
                    v1 = positions[i] - centre
                    v2 = positions[j] - centre
                    v3 = positions[k] - centre
                    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
                    self.assertGreater(area2, 0.5)

    def test_minimum_baseline_for_depth(self) -> None:
        config = load_camera_system_config(default_cameras_yaml_path())
        centre = np.array(config.world.flight_volume_center)
        baselines = [
            np.linalg.norm(np.array(camera.position) - centre)
            for camera in config.cameras
        ]
        self.assertTrue(all(b >= 3.0 for b in baselines))

    def test_viewing_directions_are_diverse(self) -> None:
        config = load_camera_system_config(default_cameras_yaml_path())
        forwards = []
        for camera in config.cameras:
            R = rotation_matrix_from_quaternion_xyzw(np.array(camera.orientation_xyzw))
            forwards.append(R @ np.array([0.0, 0.0, 1.0]))

        min_angle_deg = math.inf
        for i in range(len(forwards)):
            for j in range(i + 1, len(forwards)):
                cos_angle = np.clip(np.dot(forwards[i], forwards[j]), -1.0, 1.0)
                angle_deg = math.degrees(math.acos(cos_angle))
                min_angle_deg = min(min_angle_deg, angle_deg)

        self.assertGreater(min_angle_deg, 15.0)


if __name__ == "__main__":
    unittest.main()
