"""Tests for advanced calibration (Phase 11)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from befr_visual_tracking.advanced.calibration_pipeline import run_calibration_pipeline
from befr_visual_tracking.advanced.evaluation import umeyama_similarity
from befr_visual_tracking.advanced.gauge import apply_gauge_fix_to_cameras
from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.generate_canonical_dataset import generate_canonical_dataset
from befr_visual_tracking.trajectory_design import (
    build_trajectory,
    sample_trajectory,
    stationary_hover_trajectory,
)


class TestGaugeFixing(unittest.TestCase):
    def test_camera_zero_becomes_identity(self) -> None:
        _, cameras = cameras_from_yaml(default_cameras_yaml_path())
        cameras_by_id = {camera.name: camera for camera in cameras}
        fixed = apply_gauge_fix_to_cameras(cameras_by_id, "camera_0")
        reference = fixed["camera_0"]
        np.testing.assert_allclose(reference.extrinsics.t_camera_to_world, 0.0, atol=1e-9)
        np.testing.assert_allclose(reference.extrinsics.R_camera_to_world, np.eye(3), atol=1e-9)


class TestSimilarityAlignment(unittest.TestCase):
    def test_recovers_scaled_trajectory(self) -> None:
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=float)
        transform = umeyama_similarity(2.5 * source + np.array([1.0, -2.0, 0.5]), source)
        self.assertAlmostEqual(transform.scale, 1.0 / 2.5, places=5)


class TestCalibrationTrajectories(unittest.TestCase):
    def test_stationary_hover_is_constant(self) -> None:
        definition = stationary_hover_trajectory(duration=1.0, sample_rate_hz=10.0)
        samples = sample_trajectory(definition)
        positions = np.array([sample.position for sample in samples])
        for position in positions:
            np.testing.assert_allclose(position, positions[0])

    def test_build_all_calibration_trajectories(self) -> None:
        for name in (
            "stationary_hover",
            "straight_line",
            "planar_figure_eight",
            "calibration_figure_eight_3d",
        ):
            definition = build_trajectory(name)
            samples = sample_trajectory(definition)
            self.assertGreater(len(samples), 10)


class TestCalibrationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.experiment_dir = generate_canonical_dataset(
            experiment_name="canonical_slow_circle",
            output_dir=Path(cls.temp_dir.name),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_pipeline_improves_or_maintains_alignment(self) -> None:
        result = run_calibration_pipeline(
            self.experiment_dir,
            active_cameras=["camera_0", "camera_1", "camera_2"],
            use_opencv_init=False,
            use_metric_scale_from_ground_truth=True,
            use_ideal_detections=True,
            subsample_stride=10,
            max_timesteps=40,
            max_nfev=50,
        )
        self.assertTrue(result.calibrated_config_path.is_file())
        self.assertLess(result.after_optimization.reprojection_rmse, 5.0)
        self.assertLess(result.after_optimization.trajectory_rmse_aligned, 0.5)

    def test_two_camera_configuration_runs(self) -> None:
        result = run_calibration_pipeline(
            self.experiment_dir,
            active_cameras=["camera_0", "camera_1"],
            use_opencv_init=False,
            use_metric_scale_from_ground_truth=True,
            use_ideal_detections=True,
            subsample_stride=10,
            max_timesteps=30,
            max_nfev=30,
        )
        self.assertEqual(result.active_cameras, ["camera_0", "camera_1"])


if __name__ == "__main__":
    unittest.main()
