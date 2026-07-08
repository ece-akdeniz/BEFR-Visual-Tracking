"""Tests for canonical dataset generation and saving."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from befr_visual_tracking.camera_config import default_cameras_yaml_path
from befr_visual_tracking.camera_model import cameras_from_yaml
from befr_visual_tracking.dataset_generator import generate_detection_rows
from befr_visual_tracking.generate_canonical_dataset import generate_canonical_dataset
from befr_visual_tracking.result_writer import (
    load_detections_csv,
    load_ground_truth_csv,
    load_metadata_json,
)
from befr_visual_tracking.trajectory_design import sample_trajectory, slow_circle_trajectory


class TestCanonicalDatasetGeneration(unittest.TestCase):
    def test_slow_circle_visible_from_all_cameras(self) -> None:
        _, cameras = cameras_from_yaml(default_cameras_yaml_path())
        trajectory = slow_circle_trajectory()
        samples = sample_trajectory(trajectory)
        dataset = generate_detection_rows(
            cameras=cameras,
            trajectory_samples=samples,
            random_seed=42,
        )

        for row in dataset.detections:
            if not row.visible:
                self.fail(
                    f"Camera {row.camera_id} lost visibility at t={row.timestamp} "
                    "on the canonical slow circle"
                )
            self.assertIsNotNone(row.u)
            self.assertIsNotNone(row.v)
            self.assertIsNotNone(row.u_ideal)
            self.assertIsNotNone(row.v_ideal)

    def test_detection_rows_include_ground_truth_fields(self) -> None:
        _, cameras = cameras_from_yaml(default_cameras_yaml_path())
        samples = sample_trajectory(slow_circle_trajectory(duration=1.0))
        dataset = generate_detection_rows(
            cameras=cameras,
            trajectory_samples=samples,
            random_seed=7,
        )
        row = dataset.detections[0]
        self.assertAlmostEqual(row.gt_z, 1.5)
        self.assertEqual(len(dataset.ground_truth), len(samples))

    def test_save_and_reload_canonical_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = generate_canonical_dataset(
                experiment_name="canonical_slow_circle",
                output_dir=tmp_dir,
            )

            metadata = load_metadata_json(output_path / "metadata.json")
            detections = load_detections_csv(output_path / "detections.csv")
            ground_truth = load_ground_truth_csv(output_path / "ground_truth.csv")

            self.assertEqual(metadata["trajectory"], "slow_circle")
            self.assertEqual(metadata["active_cameras"], [
                "camera_0",
                "camera_1",
                "camera_2",
                "camera_3",
            ])
            self.assertGreater(len(detections), 0)
            self.assertGreater(len(ground_truth), 0)
            self.assertTrue((output_path / "camera_config.yaml").is_file())
            self.assertTrue((output_path / "detections_ideal.csv").is_file())

            first = detections[0]
            expected_fields = {
                "timestamp",
                "camera_id",
                "u",
                "v",
                "visible",
                "gt_x",
                "gt_y",
                "gt_z",
                "gt_vx",
                "gt_vy",
                "gt_vz",
            }
            self.assertEqual(set(first.keys()), expected_fields)


if __name__ == "__main__":
    unittest.main()
