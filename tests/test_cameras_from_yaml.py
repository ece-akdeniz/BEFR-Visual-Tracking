"""Tests for building camera models from YAML configuration."""

from __future__ import annotations

import unittest

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml


class TestCamerasFromYaml(unittest.TestCase):
    def test_builds_four_cameras_from_default_config(self) -> None:
        config, cameras = cameras_from_yaml(default_cameras_yaml_path())
        self.assertEqual(len(config.cameras), 4)
        self.assertEqual(len(cameras), 4)
        self.assertEqual([camera.name for camera in cameras], [cfg.id for cfg in config.cameras])

    def test_shared_intrinsics_and_noise(self) -> None:
        _, cameras = cameras_from_yaml(default_cameras_yaml_path())
        intrinsics = cameras[0].intrinsics
        noise = cameras[0].noise_model
        for camera in cameras[1:]:
            self.assertEqual(camera.intrinsics.fx, intrinsics.fx)
            self.assertEqual(camera.noise_model.pixel_noise_std_u, noise.pixel_noise_std_u)


if __name__ == "__main__":
    unittest.main()
